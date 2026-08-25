//! Optional native GPU kernels backed by wgpu.
//!
//! The GPU device and compiled pipelines are initialized lazily and reused for
//! the lifetime of the loaded NIF. Exact Flat indexes additionally retain a
//! generation-aware matrix and bounded query scratch pool on-device. CPU/SIMD
//! remains the caller-controlled fallback for unavailable or unsafe workloads.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use wgpu::util::DeviceExt;

use crate::distances::{self, Metric};
use crate::gpu_math::{
    finish_prepared_metric, metric_code, prepare_mean_pool, prepare_metric, prepare_normalization,
};

const WORKGROUP_SIZE: u32 = 256;
pub(crate) const MAX_RESIDENT_TOP_K: u32 = 64;
const FLAT_TOP_K_CHUNK_ROWS: u32 = 8_192;
const MAX_RESIDENT_SCRATCHES: usize = 4;
const DEFAULT_GPU_POLL_TIMEOUT: Duration = Duration::from_secs(10);
const GPU_INIT_RETRY_DELAY: Duration = Duration::from_secs(10);
const GPU_NOT_DETECTED: &str = "gpu not detected";

const METRIC_SHADER: &str = r#"
struct Params {
    values: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> scratch_a: array<f32, 256>;
var<workgroup> scratch_b: array<f32, 256>;
var<workgroup> scratch_c: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let index = global_id.x;
    let lane = local_id.x;
    let length = params.values.x;
    let operation = params.values.y;

    var a = 0.0;
    var b = 0.0;
    var c = 0.0;

    if (index < length) {
        let x = left[index];
        let y = right[index];
        let difference = x - y;

        if (operation == 0u || operation == 1u) {
            a = difference * difference;
        } else if (operation == 2u) {
            a = x * y;
            b = x * x;
            c = y * y;
        } else if (operation == 3u || operation == 4u) {
            a = x * y;
        } else if (operation == 5u || operation == 6u) {
            a = abs(difference);
        } else if (operation == 7u) {
            a = select(0.0, 1.0, (x != 0.0) != (y != 0.0));
        } else if (operation == 8u) {
            let x_set = x != 0.0;
            let y_set = y != 0.0;
            a = select(0.0, 1.0, x_set && y_set);
            b = select(0.0, 1.0, x_set || y_set);
        }
    }

    scratch_a[lane] = a;
    scratch_b[lane] = b;
    scratch_c[lane] = c;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (lane < stride) {
            if (operation == 6u) {
                scratch_a[lane] = max(scratch_a[lane], scratch_a[lane + stride]);
            } else {
                scratch_a[lane] = scratch_a[lane] + scratch_a[lane + stride];
            }

            scratch_b[lane] = scratch_b[lane] + scratch_b[lane + stride];

            scratch_c[lane] = scratch_c[lane] + scratch_c[lane + stride];
        }

        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lane == 0u) {
        output[workgroup_id.x] = vec4<f32>(
            scratch_a[0], scratch_b[0], scratch_c[0], 0.0
        );
    }
}
"#;

const NORMALIZE_SHADER: &str = r#"
struct Params {
    values: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= params.values.x) {
        return;
    }

    let operation = params.values.y;
    let first = bitcast<f32>(params.values.z);
    let second = bitcast<f32>(params.values.w);
    let value = input[index];

    if (operation == 1u) {
        output[index] = select(value / first, 0.0, first == 0.0);
    } else if (operation == 2u) {
        output[index] = select((value - first) / second, 0.0, second == 0.0);
    } else if (operation == 3u) {
        output[index] = select((value - first) / second, 0.0, second == 0.0);
    } else {
        output[index] = value;
    }
}
"#;

const MEAN_POOL_SHADER: &str = r#"
struct Params {
    values: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let column = workgroup_id.x;
    let lane = local_id.x;
    let dimensions = params.values.x;
    let selected_count = params.values.y;

    if (column >= dimensions) {
        return;
    }

    var sum = 0.0;
    var compensation = 0.0;
    var selected = lane;
    loop {
        if (selected >= selected_count) {
            break;
        }
        let value = matrix[selected * dimensions + column] - compensation;
        let next = sum + value;
        compensation = (next - sum) - value;
        sum = next;
        selected = selected + 256u;
    }

    scratch[lane] = sum;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (lane < stride) {
            scratch[lane] = scratch[lane] + scratch[lane + stride];
        }
        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lane == 0u) {
        output[column] = scratch[0] / f32(selected_count);
    }
}
"#;

const FLAT_SCORE_SHADER: &str = r#"
struct Score {
    raw: f32,
    valid: u32,
}

struct Params {
    shape: vec4<u32>,
    scales: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> matrix: array<f32>;
@group(0) @binding(1) var<storage, read> row_metadata: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> query: array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<Score>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> scratch_a: array<f32, 256>;
var<workgroup> scratch_b: array<f32, 256>;

fn scaled_product(value: f32, left_scale: f32, right_scale: f32) -> f32 {
    if (value == 0.0 || left_scale == 0.0 || right_scale == 0.0) {
        return 0.0;
    }
    let value_parts = frexp(value);
    let left_parts = frexp(left_scale);
    let right_parts = frexp(right_scale);
    let fraction = value_parts.fract * left_parts.fract * right_parts.fract;
    return ldexp(fraction, value_parts.exp + left_parts.exp + right_parts.exp);
}

fn scale_ratio(value: f32, scale: f32) -> f32 {
    if (value == 0.0 || scale == 0.0) {
        return 0.0;
    }
    if (value == scale) {
        return 1.0;
    }
    let value_parts = frexp(value);
    let scale_parts = frexp(scale);
    return ldexp(value_parts.fract / scale_parts.fract, value_parts.exp - scale_parts.exp);
}

fn checked_nonnegative_product(left: f32, right: f32) -> vec2<f32> {
    if (left == 0.0 || right == 0.0) {
        return vec2<f32>(0.0, 1.0);
    }
    let maximum = bitcast<f32>(0x7f7fffffu);
    if (left > scale_ratio(maximum, right)) {
        return vec2<f32>(0.0, 0.0);
    }
    return vec2<f32>(left * right, 1.0);
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let dimensions = params.shape.x;
    let row_count = params.shape.y;
    let operation = params.shape.z;
    let groups_x = params.shape.w;
    let row = workgroup_id.y * groups_x + workgroup_id.x;
    let lane = local_id.x;

    if (row >= row_count) {
        return;
    }

    let query_scale = bitcast<f32>(params.scales.x);
    let query_norm_squared = bitcast<f32>(params.scales.y);
    let metadata = row_metadata[row];
    let row_scale = metadata.x;
    let row_norm_squared = metadata.y;
    let pair_scale = max(query_scale, row_scale);
    let query_factor = scale_ratio(query_scale, pair_scale);
    let row_factor = scale_ratio(row_scale, pair_scale);

    var a = 0.0;
    var b = 0.0;
    var column = lane;
    loop {
        if (column >= dimensions) {
            break;
        }

        let x = query[column];
        let y = matrix[row * dimensions + column];

        if (operation == 0u || operation == 1u) {
            let difference = x * query_factor - y * row_factor;
            a = a + difference * difference;
        } else if (operation == 2u || operation == 3u || operation == 4u) {
            a = a + x * y;
        } else if (operation == 5u) {
            a = a + abs(x * query_factor - y * row_factor);
        } else if (operation == 6u) {
            a = max(a, abs(x * query_factor - y * row_factor));
        } else if (operation == 7u) {
            a = a + select(0.0, 1.0, (x != 0.0) != (y != 0.0));
        } else if (operation == 8u) {
            let x_set = x != 0.0;
            let y_set = y != 0.0;
            a = a + select(0.0, 1.0, x_set && y_set);
            b = b + select(0.0, 1.0, x_set || y_set);
        }

        column = column + 256u;
    }

    scratch_a[lane] = a;
    scratch_b[lane] = b;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (lane < stride) {
            if (operation == 6u) {
                scratch_a[lane] = max(scratch_a[lane], scratch_a[lane + stride]);
            } else {
                scratch_a[lane] = scratch_a[lane] + scratch_a[lane + stride];
            }
            scratch_b[lane] = scratch_b[lane] + scratch_b[lane + stride];
        }

        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lane == 0u) {
        var raw = scratch_a[0];
        var valid = 1u;
        if (operation == 0u) {
            let checked = checked_nonnegative_product(sqrt(raw), pair_scale);
            raw = checked.x;
            valid = u32(checked.y);
        } else if (operation == 1u) {
            let distance = checked_nonnegative_product(sqrt(raw), pair_scale);
            let squared = checked_nonnegative_product(distance.x, distance.x);
            raw = squared.x;
            valid = u32(distance.y * squared.y);
        } else if (operation == 2u) {
            let denominator = sqrt(query_norm_squared * row_norm_squared);
            raw = select(clamp(raw / denominator, -1.0, 1.0), 0.0, denominator == 0.0);
        } else if (operation == 3u) {
            raw = scaled_product(raw, query_scale, row_scale);
        } else if (operation == 4u) {
            raw = -scaled_product(raw, query_scale, row_scale);
        } else if (operation == 5u || operation == 6u) {
            let checked = checked_nonnegative_product(raw, pair_scale);
            raw = checked.x;
            valid = u32(checked.y);
        } else if (operation == 8u) {
            raw = select(1.0 - raw / scratch_b[0], 0.0, scratch_b[0] == 0.0);
        }
        let maximum = bitcast<f32>(0x7f7fffffu);
        if (!(raw >= -maximum && raw <= maximum)) {
            raw = 0.0;
            valid = 0u;
        }
        scores[row].raw = raw;
        scores[row].valid = valid;
    }
}
"#;

const FLAT_TOP_K_SHADER: &str = r#"
struct Score {
    raw: f32,
    valid: u32,
}

struct Candidate {
    raw: f32,
    index: u32,
}

struct Params {
    values: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> scores: array<Score>;
@group(0) @binding(1) var<storage, read_write> candidates: array<Candidate>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> lane_best_raw: array<f32, 1024>;
var<workgroup> lane_best_index: array<u32, 1024>;

fn rank_value(operation: u32, raw: f32) -> f32 {
    if (operation == 2u) {
        return 1.0 - raw;
    }
    if (operation == 3u) {
        return -raw;
    }
    return raw;
}

fn precedes(rank: f32, index: u32, other_rank: f32, other_index: u32) -> bool {
    return rank < other_rank || (rank == other_rank && index < other_index);
}

@compute @workgroup_size(16)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let row_count = params.values.x;
    let limit = params.values.y;
    let operation = params.values.z;
    let chunk_rows = params.values.w;
    let chunk = workgroup_id.x;
    let lane = local_id.x;
    let start = chunk * chunk_rows;
    let end = min(start + chunk_rows, row_count);

    var best_rank: array<f32, 64>;
    var best_raw: array<f32, 64>;
    var best_index: array<u32, 64>;

    var slot = 0u;
    loop {
        if (slot >= 64u) {
            break;
        }
        best_rank[slot] = bitcast<f32>(0x7f800000u);
        best_raw[slot] = 0.0;
        best_index[slot] = 0xffffffffu;
        slot = slot + 1u;
    }

    var row = start + lane;
    loop {
        if (row >= end) {
            break;
        }

        let score = scores[row];
        let raw = score.raw;
        if (score.valid != 0u && abs(raw) <= bitcast<f32>(0x7f7fffffu)) {
            let rank = rank_value(operation, raw);
            var insertion = limit;
            var position = 0u;
            loop {
                if (position >= limit) {
                    break;
                }
                if (precedes(rank, row, best_rank[position], best_index[position])) {
                    insertion = position;
                    break;
                }
                position = position + 1u;
            }

            if (insertion < limit) {
                var cursor = limit - 1u;
                loop {
                    if (cursor <= insertion) {
                        break;
                    }
                    best_rank[cursor] = best_rank[cursor - 1u];
                    best_raw[cursor] = best_raw[cursor - 1u];
                    best_index[cursor] = best_index[cursor - 1u];
                    cursor = cursor - 1u;
                }
                best_rank[insertion] = rank;
                best_raw[insertion] = raw;
                best_index[insertion] = row;
            }
        }
        row = row + 16u;
    }

    slot = 0u;
    loop {
        if (slot >= limit) {
            break;
        }
        let lane_offset = lane * limit + slot;
        lane_best_raw[lane_offset] = best_raw[slot];
        lane_best_index[lane_offset] = best_index[slot];
        slot = slot + 1u;
    }

    workgroupBarrier();
    if (lane != 0u) {
        return;
    }

    slot = 0u;
    loop {
        if (slot >= 64u) {
            break;
        }
        best_rank[slot] = bitcast<f32>(0x7f800000u);
        best_raw[slot] = 0.0;
        best_index[slot] = 0xffffffffu;
        slot = slot + 1u;
    }

    var candidate_index = 0u;
    let candidate_count = 16u * limit;
    loop {
        if (candidate_index >= candidate_count) {
            break;
        }
        let raw = lane_best_raw[candidate_index];
        let index = lane_best_index[candidate_index];
        if (index != 0xffffffffu && abs(raw) <= bitcast<f32>(0x7f7fffffu)) {
            let rank = rank_value(operation, raw);
            var insertion = limit;
            var position = 0u;
            loop {
                if (position >= limit) {
                    break;
                }
                if (precedes(rank, index, best_rank[position], best_index[position])) {
                    insertion = position;
                    break;
                }
                position = position + 1u;
            }

            if (insertion < limit) {
                var cursor = limit - 1u;
                loop {
                    if (cursor <= insertion) {
                        break;
                    }
                    best_rank[cursor] = best_rank[cursor - 1u];
                    best_raw[cursor] = best_raw[cursor - 1u];
                    best_index[cursor] = best_index[cursor - 1u];
                    cursor = cursor - 1u;
                }
                best_rank[insertion] = rank;
                best_raw[insertion] = raw;
                best_index[insertion] = index;
            }
        }
        candidate_index = candidate_index + 1u;
    }

    slot = 0u;
    loop {
        if (slot >= limit) {
            break;
        }
        let output_index = chunk * limit + slot;
        candidates[output_index].raw = best_raw[slot];
        candidates[output_index].index = best_index[slot];
        slot = slot + 1u;
    }
}
"#;

const FLAT_FINAL_TOP_K_SHADER: &str = r#"
struct Candidate {
    raw: f32,
    index: u32,
}

struct Params {
    values: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> input_candidates: array<Candidate>;
@group(0) @binding(1) var<storage, read_write> output_candidates: array<Candidate>;
@group(0) @binding(2) var<uniform> params: Params;

fn rank_value(operation: u32, raw: f32) -> f32 {
    if (operation == 2u) {
        return 1.0 - raw;
    }
    if (operation == 3u) {
        return -raw;
    }
    return raw;
}

fn precedes(rank: f32, index: u32, other_rank: f32, other_index: u32) -> bool {
    return rank < other_rank || (rank == other_rank && index < other_index);
}

@compute @workgroup_size(1)
fn main() {
    let candidate_count = params.values.x;
    let limit = params.values.y;
    let operation = params.values.z;
    var best_rank: array<f32, 64>;
    var best_raw: array<f32, 64>;
    var best_index: array<u32, 64>;

    var slot = 0u;
    loop {
        if (slot >= 64u) {
            break;
        }
        best_rank[slot] = bitcast<f32>(0x7f800000u);
        best_raw[slot] = 0.0;
        best_index[slot] = 0xffffffffu;
        slot = slot + 1u;
    }

    var candidate_index = 0u;
    loop {
        if (candidate_index >= candidate_count) {
            break;
        }
        let candidate = input_candidates[candidate_index];
        if (candidate.index != 0xffffffffu && abs(candidate.raw) <= bitcast<f32>(0x7f7fffffu)) {
            let rank = rank_value(operation, candidate.raw);
            var insertion = limit;
            var position = 0u;
            loop {
                if (position >= limit) {
                    break;
                }
                if (precedes(rank, candidate.index, best_rank[position], best_index[position])) {
                    insertion = position;
                    break;
                }
                position = position + 1u;
            }

            if (insertion < limit) {
                var cursor = limit - 1u;
                loop {
                    if (cursor <= insertion) {
                        break;
                    }
                    best_rank[cursor] = best_rank[cursor - 1u];
                    best_raw[cursor] = best_raw[cursor - 1u];
                    best_index[cursor] = best_index[cursor - 1u];
                    cursor = cursor - 1u;
                }
                best_rank[insertion] = rank;
                best_raw[insertion] = candidate.raw;
                best_index[insertion] = candidate.index;
            }
        }
        candidate_index = candidate_index + 1u;
    }

    slot = 0u;
    loop {
        if (slot >= limit) {
            break;
        }
        output_candidates[slot].raw = best_raw[slot];
        output_candidates[slot].index = best_index[slot];
        slot = slot + 1u;
    }
}
"#;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuInfo {
    pub name: String,
    pub backend: String,
    pub device_type: String,
}

struct GpuRuntime {
    device: wgpu::Device,
    queue: wgpu::Queue,
    metric_pipeline: wgpu::ComputePipeline,
    normalize_pipeline: wgpu::ComputePipeline,
    mean_pool_pipeline: wgpu::ComputePipeline,
    flat_score_pipeline: wgpu::ComputePipeline,
    flat_top_k_pipeline: wgpu::ComputePipeline,
    flat_final_top_k_pipeline: wgpu::ComputePipeline,
    limits: wgpu::Limits,
    info: GpuInfo,
}

pub(crate) struct ResidentMatrix {
    inner: Arc<ResidentMatrixInner>,
}

struct ResidentMatrixInner {
    runtime: Arc<GpuRuntime>,
    metric: Metric,
    dimensions: u32,
    rows: u32,
    score_groups_x: u32,
    score_groups_y: u32,
    top_k_groups: u32,
    matrix: wgpu::Buffer,
    row_metadata: wgpu::Buffer,
    scratch_pool: Mutex<Vec<FlatSearchScratch>>,
}

struct FlatSearchScratch {
    query: wgpu::Buffer,
    final_candidates: wgpu::Buffer,
    staging: wgpu::Buffer,
    score_params: wgpu::Buffer,
    top_k_params: wgpu::Buffer,
    final_top_k_params: wgpu::Buffer,
    score_bind_group: wgpu::BindGroup,
    top_k_bind_group: wgpu::BindGroup,
    final_top_k_bind_group: wgpu::BindGroup,
}

enum RuntimeCache {
    Empty,
    Ready(Arc<GpuRuntime>),
    Failed {
        error: String,
        allow_software: bool,
        retry_at: Instant,
    },
}

static GPU_RUNTIME: OnceLock<Mutex<RuntimeCache>> = OnceLock::new();
static GPU_POLL_TIMEOUT: OnceLock<Duration> = OnceLock::new();

pub fn detected() -> bool {
    runtime().is_ok()
}

pub fn info() -> Result<GpuInfo, String> {
    Ok(runtime()?.info.clone())
}

pub fn metric(left: &[f32], right: &[f32], metric: Metric) -> Result<f32, String> {
    distances::validate_finite_vector(left)?;
    distances::validate_finite_vector(right)?;

    if left.len() != right.len() {
        return Err("dimension mismatch".to_string());
    }

    if left.is_empty() {
        return distances::compute(metric, left, right);
    }

    let prepared = prepare_metric(left, right, metric);
    with_runtime(|runtime| {
        let partials = runtime.metric_partials(
            prepared.left.as_ref(),
            prepared.right.as_ref(),
            metric_code(metric),
        )?;
        finish_prepared_metric(metric, &partials, &prepared)
    })
}

pub fn normalize(vector: Vec<f32>, method: u8) -> Result<Vec<f32>, String> {
    distances::validate_finite_vector(&vector)?;

    if vector.is_empty() {
        return match method {
            0..=3 => Ok(vector),
            _ => Err("unknown normalization".to_string()),
        };
    }

    if method == 0 {
        return Ok(vector);
    }

    if method > 3 {
        return Err("unknown normalization".to_string());
    }

    let prepared = prepare_normalization(&vector, method)?;
    with_runtime(|runtime| {
        let normalized =
            runtime.transform(&prepared.vector, method, prepared.first, prepared.second)?;

        if normalized.iter().all(|value| value.is_finite()) {
            Ok(normalized)
        } else {
            Err("gpu normalization produced a non-finite value".to_string())
        }
    })
}

pub fn mean_pool_f32_le(
    matrix: &[u8],
    dimensions: usize,
    row_indices: &[usize],
) -> Result<Vec<f32>, String> {
    let dimensions_u32 = u32::try_from(dimensions).map_err(|_| "invalid dimensions".to_string())?;
    let selected_count =
        u32::try_from(row_indices.len()).map_err(|_| "too many row indices".to_string())?;
    let prepared = prepare_mean_pool(matrix, dimensions, row_indices)?;

    with_runtime(|runtime| {
        let selected_size = prepared
            .values
            .len()
            .checked_mul(4)
            .ok_or_else(|| "gpu workload too large".to_string())?;
        runtime.validate_storage_size(selected_size as u64)?;
        let pooled = runtime.mean_pool(&prepared.values, dimensions_u32, selected_count)?;
        let pooled = pooled
            .into_iter()
            .zip(&prepared.column_scales)
            .map(|(value, scale)| {
                let value = value.clamp(-1.0, 1.0);
                let value = f64::from(value) * f64::from(*scale);
                if value.is_finite() && value >= f64::from(f32::MIN) && value <= f64::from(f32::MAX)
                {
                    Ok(value as f32)
                } else {
                    Err("metric overflow".to_string())
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        if pooled.iter().all(|value| value.is_finite()) {
            Ok(pooled)
        } else {
            Err("metric overflow".to_string())
        }
    })
}

/// Uploads a row-major flat-index snapshot once and keeps it resident until the
/// owning index generation changes. Rows must already be in stable id order.
pub(crate) fn resident_matrix(
    vectors: Vec<f32>,
    rows: usize,
    dimensions: usize,
    metric: Metric,
) -> Result<ResidentMatrix, String> {
    if rows == 0 || dimensions == 0 {
        return Err("gpu resident matrix must not be empty".to_string());
    }
    let expected = rows
        .checked_mul(dimensions)
        .ok_or_else(|| "gpu workload too large".to_string())?;
    if vectors.len() != expected {
        return Err("dimension mismatch".to_string());
    }
    distances::validate_finite_vector(&vectors)?;
    if !matches!(metric, Metric::Hamming | Metric::Jaccard)
        && vectors
            .chunks_exact(dimensions)
            .any(unsafe_gpu_dynamic_range)
    {
        return Err("gpu numeric range unsupported".to_string());
    }

    let rows = u32::try_from(rows).map_err(|_| "gpu workload too large".to_string())?;
    let dimensions = u32::try_from(dimensions).map_err(|_| "gpu workload too large".to_string())?;
    let (prepared, row_metadata) = prepare_resident_matrix(vectors, dimensions as usize, metric);
    if metric_uses_absolute_scale(metric)
        && row_metadata
            .iter()
            .any(|metadata| unsafe_gpu_scale(metadata[0]))
    {
        return Err("gpu numeric range unsupported".to_string());
    }
    let runtime = runtime()?;

    let result = catch_unwind(AssertUnwindSafe(|| {
        ResidentMatrixInner::new(
            Arc::clone(&runtime),
            metric,
            dimensions,
            rows,
            &prepared,
            &row_metadata,
        )
    }))
    .map_err(|_| "gpu resident matrix allocation panicked".to_string())?;

    match result {
        Ok(inner) => Ok(ResidentMatrix {
            inner: Arc::new(inner),
        }),
        Err(error) => {
            if runtime_error_requires_reinitialization(&error) {
                invalidate_runtime(&runtime);
            }
            Err(error)
        }
    }
}

pub(crate) fn vector_top_k(
    mut vectors: Vec<(String, Vec<f32>)>,
    query: &[f32],
    metric: Metric,
    dimensions: usize,
    limit: usize,
) -> Result<Vec<(String, f32)>, String> {
    if dimensions == 0 || dimensions > query.len() {
        return Err("invalid prefix dimensions".to_string());
    }
    distances::validate_finite_vector(&query[..dimensions])?;
    for (_, vector) in &vectors {
        if dimensions > vector.len() {
            return Err("dimension mismatch".to_string());
        }
        distances::validate_finite_vector(&vector[..dimensions])?;
    }
    if vectors.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    if usize::min(limit, vectors.len()) > MAX_RESIDENT_TOP_K as usize {
        return Err(format!(
            "gpu flat top-k supports at most {MAX_RESIDENT_TOP_K} results"
        ));
    }
    validate_resident_query(&query[..dimensions], metric)?;

    vectors.sort_by(|left, right| left.0.cmp(&right.0));
    let mut ids = Vec::with_capacity(vectors.len());
    let mut matrix = Vec::with_capacity(vectors.len().saturating_mul(dimensions));
    for (id, vector) in vectors {
        ids.push(id);
        matrix.extend_from_slice(&vector[..dimensions]);
    }

    let resident = resident_matrix(matrix, ids.len(), dimensions, metric)?;
    Ok(resident
        .search(&query[..dimensions], limit)?
        .into_iter()
        .map(|(row, raw)| (ids[row].clone(), raw))
        .collect())
}

pub(crate) fn validate_resident_query(query: &[f32], metric: Metric) -> Result<(), String> {
    distances::validate_finite_vector(query)?;
    if !matches!(metric, Metric::Hamming | Metric::Jaccard) && unsafe_gpu_dynamic_range(query) {
        return Err("gpu numeric range unsupported".to_string());
    }

    let scale = resident_scale(query, metric);
    if metric_uses_absolute_scale(metric) && unsafe_gpu_scale(scale) {
        return Err("gpu numeric range unsupported".to_string());
    }
    Ok(())
}

impl ResidentMatrix {
    pub(crate) fn is_current(&self) -> bool {
        cached_runtime()
            .as_ref()
            .is_some_and(|runtime| Arc::ptr_eq(runtime, &self.inner.runtime))
    }

    /// Scores one query against all resident rows, performs an exact per-chunk
    /// top-k reduction on the device, and merges only those compact candidates
    /// on the host.
    pub(crate) fn search(&self, query: &[f32], limit: usize) -> Result<Vec<(usize, f32)>, String> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        if query.len() != self.inner.dimensions as usize {
            return Err("dimension mismatch".to_string());
        }
        validate_resident_query(query, self.inner.metric)?;

        let effective_limit = usize::min(limit, self.inner.rows as usize);
        if effective_limit > MAX_RESIDENT_TOP_K as usize {
            return Err(format!(
                "gpu flat top-k supports at most {MAX_RESIDENT_TOP_K} results"
            ));
        }

        if !self.is_current() {
            return Err("gpu resident matrix expired".to_string());
        }

        let (prepared_query, query_scale, query_norm_squared) =
            prepare_resident_query(query, self.inner.metric);
        let mut scratch = self.inner.take_scratch()?;
        let runtime = Arc::clone(&self.inner.runtime);
        let result = catch_unwind(AssertUnwindSafe(|| {
            self.inner.execute_search(
                &mut scratch,
                &prepared_query,
                query_scale,
                query_norm_squared,
                effective_limit as u32,
            )
        }))
        .unwrap_or_else(|_| Err("gpu operation panicked".to_string()));
        self.inner.return_scratch(scratch);

        if let Err(error) = &result {
            if runtime_error_requires_reinitialization(error) {
                invalidate_runtime(&runtime);
            }
        }

        result
    }
}

impl ResidentMatrixInner {
    fn new(
        runtime: Arc<GpuRuntime>,
        metric: Metric,
        dimensions: u32,
        rows: u32,
        matrix: &[f32],
        row_metadata: &[[f32; 2]],
    ) -> Result<Self, String> {
        let matrix_size = matrix
            .len()
            .checked_mul(4)
            .and_then(|size| u64::try_from(size).ok())
            .ok_or_else(|| "gpu workload too large".to_string())?;
        let metadata_size = row_metadata
            .len()
            .checked_mul(8)
            .and_then(|size| u64::try_from(size).ok())
            .ok_or_else(|| "gpu workload too large".to_string())?;
        runtime.validate_storage_size(matrix_size)?;
        runtime.validate_storage_size(metadata_size)?;

        let max_groups = runtime.limits.max_compute_workgroups_per_dimension;
        let score_groups_x = rows.min(max_groups);
        let score_groups_y = rows.div_ceil(score_groups_x);
        if score_groups_y > max_groups {
            return Err("gpu workload too large".to_string());
        }

        let top_k_groups = rows.div_ceil(FLAT_TOP_K_CHUNK_ROWS);
        if top_k_groups > max_groups {
            return Err("gpu workload too large".to_string());
        }

        Ok(Self {
            matrix: storage_buffer(&runtime.device, "vettore flat resident matrix", matrix),
            row_metadata: storage_buffer(
                &runtime.device,
                "vettore flat resident row metadata",
                row_metadata,
            ),
            runtime,
            metric,
            dimensions,
            rows,
            score_groups_x,
            score_groups_y,
            top_k_groups,
            scratch_pool: Mutex::new(Vec::new()),
        })
    }

    fn take_scratch(&self) -> Result<FlatSearchScratch, String> {
        if let Some(scratch) = self
            .scratch_pool
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pop()
        {
            return Ok(scratch);
        }

        self.create_scratch()
    }

    fn return_scratch(&self, scratch: FlatSearchScratch) {
        let mut pool = self
            .scratch_pool
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if pool.len() < MAX_RESIDENT_SCRATCHES {
            pool.push(scratch);
        }
    }

    fn create_scratch(&self) -> Result<FlatSearchScratch, String> {
        let query_size = u64::from(self.dimensions) * 4;
        let scores_size = u64::from(self.rows) * 8;
        let candidates_size = u64::from(self.top_k_groups) * u64::from(MAX_RESIDENT_TOP_K) * 8;
        let final_candidates_size = u64::from(MAX_RESIDENT_TOP_K) * 8;
        self.runtime.validate_storage_size(query_size)?;
        self.runtime.validate_storage_size(scores_size)?;
        self.runtime.validate_storage_size(candidates_size)?;
        self.runtime.validate_storage_size(final_candidates_size)?;

        let device = &self.runtime.device;
        let query = reusable_buffer(
            device,
            "vettore flat query",
            query_size,
            wgpu::BufferUsages::STORAGE,
        );
        let scores = reusable_buffer(
            device,
            "vettore flat scores",
            scores_size,
            wgpu::BufferUsages::STORAGE,
        );
        let candidates = reusable_buffer(
            device,
            "vettore flat chunk top-k candidates",
            candidates_size,
            wgpu::BufferUsages::STORAGE,
        );
        let final_candidates = reusable_buffer(
            device,
            "vettore flat final top-k candidates",
            final_candidates_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vettore flat top-k staging"),
            size: final_candidates_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let score_params = reusable_buffer(
            device,
            "vettore flat score parameters",
            32,
            wgpu::BufferUsages::UNIFORM,
        );
        let top_k_params = reusable_buffer(
            device,
            "vettore flat top-k parameters",
            16,
            wgpu::BufferUsages::UNIFORM,
        );
        let final_top_k_params = reusable_buffer(
            device,
            "vettore flat final top-k parameters",
            16,
            wgpu::BufferUsages::UNIFORM,
        );

        let score_layout = self.runtime.flat_score_pipeline.get_bind_group_layout(0);
        let score_bind_group = bind_group(
            device,
            "vettore flat score bindings",
            &score_layout,
            &[
                &self.matrix,
                &self.row_metadata,
                &query,
                &scores,
                &score_params,
            ],
        );
        let top_k_layout = self.runtime.flat_top_k_pipeline.get_bind_group_layout(0);
        let top_k_bind_group = bind_group(
            device,
            "vettore flat top-k bindings",
            &top_k_layout,
            &[&scores, &candidates, &top_k_params],
        );
        let final_top_k_layout = self
            .runtime
            .flat_final_top_k_pipeline
            .get_bind_group_layout(0);
        let final_top_k_bind_group = bind_group(
            device,
            "vettore flat final top-k bindings",
            &final_top_k_layout,
            &[&candidates, &final_candidates, &final_top_k_params],
        );

        Ok(FlatSearchScratch {
            query,
            final_candidates,
            staging,
            score_params,
            top_k_params,
            final_top_k_params,
            score_bind_group,
            top_k_bind_group,
            final_top_k_bind_group,
        })
    }

    fn execute_search(
        &self,
        scratch: &mut FlatSearchScratch,
        query: &[f32],
        query_scale: f32,
        query_norm_squared: f32,
        limit: u32,
    ) -> Result<Vec<(usize, f32)>, String> {
        let queue = &self.runtime.queue;
        queue.write_buffer(&scratch.query, 0, bytemuck::cast_slice(query));
        queue.write_buffer(
            &scratch.score_params,
            0,
            bytemuck::cast_slice(&[
                self.dimensions,
                self.rows,
                metric_code(self.metric),
                self.score_groups_x,
                query_scale.to_bits(),
                query_norm_squared.to_bits(),
                0,
                0,
            ]),
        );
        queue.write_buffer(
            &scratch.top_k_params,
            0,
            bytemuck::cast_slice(&[
                self.rows,
                limit,
                metric_code(self.metric),
                FLAT_TOP_K_CHUNK_ROWS,
            ]),
        );
        queue.write_buffer(
            &scratch.final_top_k_params,
            0,
            bytemuck::cast_slice(&[
                self.top_k_groups * limit,
                limit,
                metric_code(self.metric),
                0,
            ]),
        );

        let final_candidate_size = u64::from(limit) * 8;
        let mut encoder =
            self.runtime
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vettore resident flat search"),
                });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vettore resident flat scoring"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.runtime.flat_score_pipeline);
            pass.set_bind_group(0, &scratch.score_bind_group, &[]);
            pass.dispatch_workgroups(self.score_groups_x, self.score_groups_y, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vettore resident flat top-k"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.runtime.flat_top_k_pipeline);
            pass.set_bind_group(0, &scratch.top_k_bind_group, &[]);
            pass.dispatch_workgroups(self.top_k_groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vettore resident flat final top-k"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.runtime.flat_final_top_k_pipeline);
            pass.set_bind_group(0, &scratch.final_top_k_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &scratch.final_candidates,
            0,
            &scratch.staging,
            0,
            final_candidate_size,
        );
        let submission = queue.submit([encoder.finish()]);
        let bytes =
            self.runtime
                .map_readback(&scratch.staging, final_candidate_size, Some(submission))?;
        Ok(decode_resident_candidates(&bytes, self.rows as usize))
    }
}

fn prepare_resident_matrix(
    mut vectors: Vec<f32>,
    dimensions: usize,
    metric: Metric,
) -> (Vec<f32>, Vec<[f32; 2]>) {
    let mut metadata = Vec::with_capacity(vectors.len() / dimensions);

    for row in vectors.chunks_exact_mut(dimensions) {
        let scale = resident_scale(row, metric);
        if matches!(metric, Metric::Hamming | Metric::Jaccard) {
            for value in row.iter_mut() {
                *value = if *value == 0.0 { 0.0 } else { 1.0 };
            }
        } else if scale != 0.0 {
            for value in row.iter_mut() {
                *value /= scale;
            }
        }
        let norm_squared = row
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>() as f32;
        metadata.push([scale, norm_squared]);
    }

    (vectors, metadata)
}

fn prepare_resident_query(query: &[f32], metric: Metric) -> (Vec<f32>, f32, f32) {
    prepare_resident_values(query, metric)
}

fn prepare_resident_values(values: &[f32], metric: Metric) -> (Vec<f32>, f32, f32) {
    if matches!(metric, Metric::Hamming | Metric::Jaccard) {
        let prepared = values
            .iter()
            .map(|value| if *value == 0.0 { 0.0 } else { 1.0 })
            .collect::<Vec<_>>();
        let norm_squared = prepared.iter().map(|value| value * value).sum();
        return (prepared, 1.0, norm_squared);
    }

    let scale = resident_scale(values, metric);
    if scale == 0.0 {
        return (vec![0.0; values.len()], 0.0, 0.0);
    }

    let prepared = values
        .iter()
        .map(|value| *value / scale)
        .collect::<Vec<_>>();
    let norm_squared = prepared
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>() as f32;
    (prepared, scale, norm_squared)
}

fn resident_scale(values: &[f32], metric: Metric) -> f32 {
    if matches!(metric, Metric::Hamming | Metric::Jaccard) {
        1.0
    } else {
        values
            .iter()
            .map(|value| value.abs())
            .fold(0.0f32, f32::max)
    }
}

fn metric_uses_absolute_scale(metric: Metric) -> bool {
    !matches!(metric, Metric::Cosine | Metric::Hamming | Metric::Jaccard)
}

fn unsafe_gpu_scale(scale: f32) -> bool {
    scale != 0.0 && !scale.is_normal()
}

fn unsafe_gpu_dynamic_range(values: &[f32]) -> bool {
    let mut minimum = f32::INFINITY;
    let mut maximum = 0.0f32;
    for value in values {
        let absolute = value.abs();
        if absolute != 0.0 {
            minimum = minimum.min(absolute);
            maximum = maximum.max(absolute);
        }
    }

    if maximum == 0.0 {
        return false;
    }
    let normalized_minimum = minimum / maximum;
    normalized_minimum == 0.0 || !normalized_minimum.is_normal()
}

fn decode_resident_candidates(bytes: &[u8], row_count: usize) -> Vec<(usize, f32)> {
    let mut hits = Vec::with_capacity(bytes.len() / 8);
    for candidate in bytes.chunks_exact(8) {
        let raw = f32::from_ne_bytes(candidate[..4].try_into().expect("four-byte score"));
        let index = u32::from_ne_bytes(candidate[4..].try_into().expect("four-byte index"));
        if index == u32::MAX || index as usize >= row_count || !raw.is_finite() {
            continue;
        }
        hits.push((index as usize, raw));
    }
    hits
}

impl GpuRuntime {
    fn new_with_software_adapter(allow_software: bool) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let mut adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));

        adapters.retain(|adapter| {
            let info = adapter.get_info();
            hardware_gpu(&info)
                || (allow_software
                    && info.backend != wgpu::Backend::Noop
                    && info.device_type == wgpu::DeviceType::Cpu)
        });
        adapters.sort_by_key(|adapter| adapter_priority(adapter.get_info().device_type));

        let adapter = adapters
            .into_iter()
            .next()
            .ok_or_else(|| GPU_NOT_DETECTED.to_string())?;
        let adapter_info = adapter.get_info();
        let descriptor = wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..wgpu::DeviceDescriptor::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&descriptor))
            .map_err(|error| format!("gpu device initialization failed: {error}"))?;

        let metric_pipeline = create_pipeline(&device, "vettore metric", METRIC_SHADER);
        let normalize_pipeline = create_pipeline(&device, "vettore normalize", NORMALIZE_SHADER);
        let mean_pool_pipeline = create_pipeline(&device, "vettore mean pool", MEAN_POOL_SHADER);
        let flat_score_pipeline =
            create_pipeline(&device, "vettore flat batched score", FLAT_SCORE_SHADER);
        let flat_top_k_pipeline = create_pipeline(&device, "vettore flat top k", FLAT_TOP_K_SHADER);
        let flat_final_top_k_pipeline =
            create_pipeline(&device, "vettore flat final top k", FLAT_FINAL_TOP_K_SHADER);
        let limits = device.limits();

        Ok(Self {
            device,
            queue,
            metric_pipeline,
            normalize_pipeline,
            mean_pool_pipeline,
            flat_score_pipeline,
            flat_top_k_pipeline,
            flat_final_top_k_pipeline,
            limits,
            info: GpuInfo {
                name: adapter_info.name,
                backend: adapter_info.backend.to_str().to_string(),
                device_type: device_type_name(adapter_info.device_type).to_string(),
            },
        })
    }

    fn metric_partials(
        &self,
        left: &[f32],
        right: &[f32],
        operation: u32,
    ) -> Result<Vec<[f32; 4]>, String> {
        let (length, groups) = self.dispatch_dimensions(left.len())?;
        let vector_size = u64::from(length) * 4;
        self.validate_storage_size(vector_size)?;
        let output_size = u64::from(groups) * 16;
        self.validate_storage_size(output_size)?;

        let left_buffer = storage_buffer(&self.device, "vettore metric left", left);
        let right_buffer = storage_buffer(&self.device, "vettore metric right", right);
        let output = output_buffer(&self.device, "vettore metric output", output_size);
        let params = uniform_buffer(&self.device, [length, operation, 0, 0]);
        let layout = self.metric_pipeline.get_bind_group_layout(0);
        let bind_group = bind_group(
            &self.device,
            "vettore metric bindings",
            &layout,
            &[&left_buffer, &right_buffer, &output, &params],
        );

        let bytes = self.dispatch_and_read(
            &self.metric_pipeline,
            &bind_group,
            groups,
            &output,
            output_size,
        )?;

        Ok(bytes
            .as_chunks::<16>()
            .0
            .iter()
            .map(|chunk| {
                let values = bytemuck::cast_slice::<u8, f32>(chunk);
                [values[0], values[1], values[2], values[3]]
            })
            .collect())
    }

    fn transform(
        &self,
        vector: &[f32],
        method: u8,
        first: f32,
        second: f32,
    ) -> Result<Vec<f32>, String> {
        let (length, groups) = self.dispatch_dimensions(vector.len())?;
        let logical_output_size = u64::from(length) * 4;
        let output_size = aligned_readback_size(logical_output_size);
        self.validate_storage_size(output_size)?;

        let input = storage_buffer(&self.device, "vettore normalize input", vector);
        let output = output_buffer(&self.device, "vettore normalize output", output_size);
        let params = uniform_buffer(
            &self.device,
            [length, u32::from(method), first.to_bits(), second.to_bits()],
        );
        let layout = self.normalize_pipeline.get_bind_group_layout(0);
        let bind_group = bind_group(
            &self.device,
            "vettore normalize bindings",
            &layout,
            &[&input, &output, &params],
        );

        let bytes = self.dispatch_and_read(
            &self.normalize_pipeline,
            &bind_group,
            groups,
            &output,
            output_size,
        )?;
        Ok(bytemuck::cast_slice::<u8, f32>(&bytes[..logical_output_size as usize]).to_vec())
    }

    fn mean_pool(
        &self,
        matrix: &[f32],
        dimensions: u32,
        selected_count: u32,
    ) -> Result<Vec<f32>, String> {
        if dimensions > self.limits.max_compute_workgroups_per_dimension {
            return Err("gpu workload too large".to_string());
        }
        let matrix_size = matrix
            .len()
            .checked_mul(4)
            .and_then(|size| u64::try_from(size).ok())
            .ok_or_else(|| "gpu workload too large".to_string())?;
        self.validate_storage_size(matrix_size)?;
        let logical_output_size = u64::from(dimensions) * 4;
        let output_size = aligned_readback_size(logical_output_size);
        self.validate_storage_size(output_size)?;

        let matrix_buffer = storage_buffer(&self.device, "vettore mean matrix", matrix);
        let output = output_buffer(&self.device, "vettore mean output", output_size);
        let params = uniform_buffer(&self.device, [dimensions, selected_count, 0, 0]);
        let layout = self.mean_pool_pipeline.get_bind_group_layout(0);
        let bind_group = bind_group(
            &self.device,
            "vettore mean bindings",
            &layout,
            &[&matrix_buffer, &output, &params],
        );

        let bytes = self.dispatch_and_read(
            &self.mean_pool_pipeline,
            &bind_group,
            dimensions,
            &output,
            output_size,
        )?;
        Ok(bytemuck::cast_slice::<u8, f32>(&bytes[..logical_output_size as usize]).to_vec())
    }

    fn dispatch_dimensions(&self, length: usize) -> Result<(u32, u32), String> {
        let length = u32::try_from(length).map_err(|_| "gpu workload too large".to_string())?;
        let workgroups = length.div_ceil(WORKGROUP_SIZE);
        if workgroups > self.limits.max_compute_workgroups_per_dimension {
            return Err("gpu workload too large".to_string());
        }

        Ok((length, workgroups))
    }

    fn validate_storage_size(&self, size: u64) -> Result<(), String> {
        if size > self.limits.max_storage_buffer_binding_size || size > self.limits.max_buffer_size
        {
            Err("gpu workload too large".to_string())
        } else {
            Ok(())
        }
    }

    fn dispatch_and_read(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: u32,
        output: &wgpu::Buffer,
        output_size: u64,
    ) -> Result<Vec<u8>, String> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vettore gpu staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vettore gpu encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vettore gpu pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(output, 0, &staging, 0, output_size);
        let submission_index = self.queue.submit([encoder.finish()]);

        self.map_readback(&staging, output_size, Some(submission_index))
    }

    fn map_readback(
        &self,
        staging: &wgpu::Buffer,
        output_size: u64,
        submission_index: Option<wgpu::SubmissionIndex>,
    ) -> Result<Vec<u8>, String> {
        let timeout = gpu_poll_timeout();
        let slice = staging.slice(..output_size);
        let (sender, receiver) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ignored = sender.send(result);
        });
        let started = Instant::now();
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index,
                timeout: Some(timeout),
            })
            .map_err(|error| format!("gpu synchronization failed: {error}"))?;
        receiver
            .recv_timeout(timeout.saturating_sub(started.elapsed()))
            .map_err(|_| "gpu mapping callback failed".to_string())?
            .map_err(|error| format!("gpu readback failed: {error}"))?;

        let mapped = slice
            .get_mapped_range()
            .map_err(|error| format!("gpu mapped range failed: {error}"))?;
        let bytes = mapped.to_vec();
        drop(mapped);
        staging.unmap();
        Ok(bytes)
    }
}

fn runtime() -> Result<Arc<GpuRuntime>, String> {
    let allow_software = allow_software_adapter();
    let mut cached = runtime_slot()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    match &*cached {
        RuntimeCache::Ready(runtime) => return Ok(Arc::clone(runtime)),
        RuntimeCache::Failed {
            error,
            allow_software: cached_allow_software,
            retry_at,
        } if *cached_allow_software == allow_software && Instant::now() < *retry_at => {
            return Err(error.clone());
        }
        RuntimeCache::Empty | RuntimeCache::Failed { .. } => {}
    }

    let initialized = catch_unwind(|| GpuRuntime::new_with_software_adapter(allow_software))
        .map_err(|_| "gpu initialization panicked".to_string())
        .and_then(|result| result);

    match initialized {
        Ok(runtime) => {
            let runtime = Arc::new(runtime);
            *cached = RuntimeCache::Ready(Arc::clone(&runtime));
            Ok(runtime)
        }
        Err(error) => {
            *cached = RuntimeCache::Failed {
                error: error.clone(),
                allow_software,
                retry_at: Instant::now() + GPU_INIT_RETRY_DELAY,
            };
            Err(error)
        }
    }
}

fn with_runtime<T>(operation: impl FnOnce(&GpuRuntime) -> Result<T, String>) -> Result<T, String> {
    let runtime = runtime()?;
    let result = match catch_unwind(AssertUnwindSafe(|| operation(&runtime))) {
        Ok(result) => result,
        Err(_panic) => Err("gpu operation panicked".to_string()),
    };

    if let Err(error) = &result {
        if runtime_error_requires_reinitialization(error) {
            invalidate_runtime(&runtime);
        }
    }

    result
}

fn runtime_slot() -> &'static Mutex<RuntimeCache> {
    GPU_RUNTIME.get_or_init(|| Mutex::new(RuntimeCache::Empty))
}

fn cached_runtime() -> Option<Arc<GpuRuntime>> {
    let cached = runtime_slot()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match &*cached {
        RuntimeCache::Ready(runtime) => Some(Arc::clone(runtime)),
        RuntimeCache::Empty | RuntimeCache::Failed { .. } => None,
    }
}

fn invalidate_runtime(runtime: &Arc<GpuRuntime>) {
    let mut cached = runtime_slot()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if matches!(&*cached, RuntimeCache::Ready(candidate) if Arc::ptr_eq(candidate, runtime)) {
        *cached = RuntimeCache::Empty;
    }
}

fn allow_software_adapter() -> bool {
    std::env::var("VETTORE_GPU_ALLOW_SOFTWARE")
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

fn gpu_poll_timeout() -> Duration {
    *GPU_POLL_TIMEOUT.get_or_init(|| {
        std::env::var("VETTORE_GPU_TIMEOUT_MS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|milliseconds| (100..=120_000).contains(milliseconds))
            .map(Duration::from_millis)
            .unwrap_or(DEFAULT_GPU_POLL_TIMEOUT)
    })
}

fn runtime_error_requires_reinitialization(error: &str) -> bool {
    error.starts_with("gpu synchronization")
        || error.starts_with("gpu mapping")
        || error.starts_with("gpu readback")
        || error == "gpu operation panicked"
}

fn aligned_readback_size(size: u64) -> u64 {
    size.div_ceil(wgpu::MAP_ALIGNMENT) * wgpu::MAP_ALIGNMENT
}

fn create_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    source: &'static str,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn storage_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    label: &'static str,
    values: &[T],
) -> wgpu::Buffer {
    storage_bytes(device, label, bytemuck::cast_slice(values))
}

fn storage_bytes(device: &wgpu::Device, label: &'static str, bytes: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn output_buffer(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn reusable_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn uniform_buffer(device: &wgpu::Device, values: [u32; 4]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vettore gpu parameters"),
        contents: bytemuck::cast_slice(&values),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

fn bind_group(
    device: &wgpu::Device,
    label: &'static str,
    layout: &wgpu::BindGroupLayout,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect::<Vec<_>>();

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    })
}

fn hardware_gpu(info: &wgpu::AdapterInfo) -> bool {
    info.backend != wgpu::Backend::Noop && info.device_type != wgpu::DeviceType::Cpu
}

fn adapter_priority(device_type: wgpu::DeviceType) -> u8 {
    match device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::Cpu => 4,
    }
}

fn device_type_name(device_type: wgpu::DeviceType) -> &'static str {
    match device_type {
        wgpu::DeviceType::DiscreteGpu => "discrete_gpu",
        wgpu::DeviceType::IntegratedGpu => "integrated_gpu",
        wgpu::DeviceType::VirtualGpu => "virtual_gpu",
        wgpu::DeviceType::Other => "other",
        wgpu::DeviceType::Cpu => "cpu",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn adapter_selection_prefers_hardware_and_reports_stable_names() {
        assert_eq!(adapter_priority(wgpu::DeviceType::DiscreteGpu), 0);
        assert_eq!(adapter_priority(wgpu::DeviceType::IntegratedGpu), 1);
        assert_eq!(adapter_priority(wgpu::DeviceType::Cpu), 4);
        assert_eq!(
            device_type_name(wgpu::DeviceType::VirtualGpu),
            "virtual_gpu"
        );
    }

    #[test]
    fn every_wgsl_shader_parses_and_validates_without_hardware() {
        for source in [
            METRIC_SHADER,
            NORMALIZE_SHADER,
            MEAN_POOL_SHADER,
            FLAT_SCORE_SHADER,
            FLAT_TOP_K_SHADER,
            FLAT_FINAL_TOP_K_SHADER,
        ] {
            let module = naga::front::wgsl::parse_str(source).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
        }
    }

    #[test]
    fn resident_preparation_guards_dynamic_range_and_decodes_final_candidates() {
        assert!(!unsafe_gpu_dynamic_range(&[0.0, 1.0, -2.0]));
        assert!(unsafe_gpu_dynamic_range(&[1.0e38, 1.0e-7]));
        assert!(unsafe_gpu_scale(f32::from_bits(1)));
        assert!(!unsafe_gpu_scale(1.0));

        let (binary, scale, norm) = prepare_resident_values(&[0.0, -2.0, 3.0], Metric::Hamming);
        assert_eq!(binary, [0.0, 1.0, 1.0]);
        assert_eq!(scale, 1.0);
        assert_eq!(norm, 2.0);

        let candidates = [(1.0f32, 0u32), (1.0, 1), (2.0, 2)];
        let bytes = candidates
            .into_iter()
            .flat_map(|(raw, index)| raw.to_ne_bytes().into_iter().chain(index.to_ne_bytes()))
            .collect::<Vec<_>>();
        assert_eq!(
            decode_resident_candidates(&bytes, 4),
            [(0, 1.0), (1, 1.0), (2, 2.0)]
        );

        let similarity_candidates = [(2.0f32, 0u32), (0.5, 1)];
        let bytes = similarity_candidates
            .into_iter()
            .flat_map(|(raw, index)| raw.to_ne_bytes().into_iter().chain(index.to_ne_bytes()))
            .collect::<Vec<_>>();
        assert_eq!(decode_resident_candidates(&bytes, 3), [(0, 2.0), (1, 0.5)]);
    }

    #[test]
    fn available_wgpu_adapter_executes_every_pipeline() {
        let runtime = Arc::new(match GpuRuntime::new_with_software_adapter(true) {
            Ok(runtime) => runtime,
            Err(error) if std::env::var("VETTORE_REQUIRE_GPU").as_deref() == Ok("1") => {
                panic!("a GPU adapter is required for this test: {error}")
            }
            Err(_error) => return,
        });

        let maximum_elements =
            runtime.limits.max_compute_workgroups_per_dimension as usize * WORKGROUP_SIZE as usize;
        assert!(runtime.dispatch_dimensions(maximum_elements).is_ok());
        assert!(runtime.dispatch_dimensions(maximum_elements + 1).is_err());
        assert!(runtime
            .validate_storage_size(runtime.limits.max_storage_buffer_binding_size + 1)
            .is_err());

        let left = [1.0, 0.0, 2.0];
        let right = [0.0, 3.0, 2.0];
        let expected = [
            (Metric::L2, 3.162_277_7),
            (Metric::L2Squared, 10.0),
            (Metric::Cosine, 0.496_138_93),
            (Metric::InnerProduct, 4.0),
            (Metric::NegativeInnerProduct, -4.0),
            (Metric::Manhattan, 4.0),
            (Metric::Chebyshev, 3.0),
            (Metric::Hamming, 2.0),
            (Metric::Jaccard, 2.0 / 3.0),
        ];

        for (metric, expected) in expected {
            let prepared = prepare_metric(&left, &right, metric);
            let partials = runtime
                .metric_partials(
                    prepared.left.as_ref(),
                    prepared.right.as_ref(),
                    metric_code(metric),
                )
                .unwrap();
            let actual = finish_prepared_metric(metric, &partials, &prepared).unwrap();
            assert!((actual - expected).abs() < 1.0e-5, "metric {metric:?}");
        }

        let prepared = prepare_normalization(&[3.0, 4.0], 1).unwrap();
        let normalized = runtime
            .transform(&prepared.vector, 1, prepared.first, prepared.second)
            .unwrap();
        assert!((normalized[0] - 0.6).abs() < 1.0e-5);
        assert!((normalized[1] - 0.8).abs() < 1.0e-5);

        for method in [2, 3] {
            let values = [1.0, 2.0, 3.0, 4.0];
            let prepared = prepare_normalization(&values, method).unwrap();
            let actual = runtime
                .transform(&prepared.vector, method, prepared.first, prepared.second)
                .unwrap();
            let expected = match method {
                2 => distances::normalize_zscore(values.to_vec()).unwrap(),
                3 => distances::normalize_minmax(values.to_vec()).unwrap(),
                _ => unreachable!(),
            };
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1.0e-5);
            }
        }

        for method in [1, 2, 3] {
            let values = [4.0, 4.0];
            let prepared = prepare_normalization(&values, method).unwrap();
            let actual = runtime
                .transform(&prepared.vector, method, prepared.first, prepared.second)
                .unwrap();
            assert!(actual.iter().all(|value| value.is_finite()));
        }

        let matrix = [1.0f32, 2.0, 3.0, 4.0];
        let pooled = runtime.mean_pool(&matrix, 2, 2).unwrap();
        assert!((pooled[0] - 2.0).abs() < 1.0e-5);
        assert!((pooled[1] - 3.0).abs() < 1.0e-5);

        let repeated_matrix = [3.0f32, 4.0, 3.0, 4.0, 1.0, 2.0];
        let repeated = runtime.mean_pool(&repeated_matrix, 2, 3).unwrap();
        assert!((repeated[0] - 7.0 / 3.0).abs() < 1.0e-5);
        assert!((repeated[1] - 10.0 / 3.0).abs() < 1.0e-5);

        let rows = 137usize;
        let dimensions = 5usize;
        let vectors = (0..rows)
            .flat_map(|row| {
                (0..dimensions).map(move |column| {
                    if column == 4 {
                        if (row + column) % 3 == 0 {
                            0.0
                        } else {
                            1.0
                        }
                    } else {
                        ((row * 17 + column * 11) % 97) as f32 / 19.0 - 2.5
                    }
                })
            })
            .collect::<Vec<_>>();
        let query = [0.25, -1.5, 2.0, 0.75, 1.0];

        for metric in [
            Metric::L2,
            Metric::L2Squared,
            Metric::Cosine,
            Metric::InnerProduct,
            Metric::NegativeInnerProduct,
            Metric::Manhattan,
            Metric::Chebyshev,
            Metric::Hamming,
            Metric::Jaccard,
        ] {
            let (prepared, metadata) = prepare_resident_matrix(vectors.clone(), dimensions, metric);
            let resident = ResidentMatrixInner::new(
                Arc::clone(&runtime),
                metric,
                dimensions as u32,
                rows as u32,
                &prepared,
                &metadata,
            )
            .unwrap();
            let (prepared_query, query_scale, query_norm_squared) =
                prepare_resident_query(&query, metric);
            let mut scratch = resident.take_scratch().unwrap();
            let actual = resident
                .execute_search(
                    &mut scratch,
                    &prepared_query,
                    query_scale,
                    query_norm_squared,
                    7,
                )
                .unwrap();

            let mut expected = vectors
                .chunks_exact(dimensions)
                .enumerate()
                .map(|(row, vector)| {
                    let raw = distances::compute(metric, &query, vector).unwrap();
                    (row, raw, distances::rank_value(metric, raw))
                })
                .collect::<Vec<_>>();
            expected.sort_by(|left, right| {
                left.2
                    .total_cmp(&right.2)
                    .then_with(|| left.0.cmp(&right.0))
            });
            let cpu_scores = expected
                .iter()
                .map(|(row, raw, _rank)| (*row, *raw))
                .collect::<std::collections::HashMap<_, _>>();
            expected.truncate(7);

            assert_eq!(actual.len(), expected.len(), "resident flat hit count");
            let mut seen = HashSet::new();
            for ((row, actual), (_, _expected, expected_rank)) in actual.iter().zip(expected) {
                assert!(seen.insert(*row), "duplicate resident row for {metric:?}");
                let cpu_score = cpu_scores[row];
                let score_tolerance = (cpu_score.abs() * 1.0e-4).max(1.0e-5);
                assert!(
                    (actual - cpu_score).abs() <= score_tolerance,
                    "resident flat score for {metric:?}, row {row}: {actual} vs {cpu_score}"
                );

                let cpu_rank = distances::rank_value(metric, cpu_score);
                let rank_tolerance = (expected_rank.abs() * 1.0e-4).max(1.0e-5);
                assert!(
                    (cpu_rank - expected_rank).abs() <= rank_tolerance,
                    "resident flat rank for {metric:?}, row {row}: {cpu_rank} vs {expected_rank}"
                );
            }
        }

        let rows = FLAT_TOP_K_CHUNK_ROWS as usize + 4;
        let mut vectors = vec![1_000.0f32; rows];
        for offset in 0..4usize {
            vectors[FLAT_TOP_K_CHUNK_ROWS as usize - 4 + offset] = (offset * 2) as f32;
            vectors[FLAT_TOP_K_CHUNK_ROWS as usize + offset] = (offset * 2 + 1) as f32;
        }
        let (prepared, metadata) = prepare_resident_matrix(vectors, 1, Metric::L2);
        let resident = ResidentMatrixInner::new(
            Arc::clone(&runtime),
            Metric::L2,
            1,
            rows as u32,
            &prepared,
            &metadata,
        )
        .unwrap();
        let (query, scale, norm_squared) = prepare_resident_query(&[0.0], Metric::L2);
        let mut scratch = resident.take_scratch().unwrap();
        let actual = resident
            .execute_search(&mut scratch, &query, scale, norm_squared, 7)
            .unwrap();
        let boundary = FLAT_TOP_K_CHUNK_ROWS as usize;
        assert_eq!(
            actual.iter().map(|hit| hit.0).collect::<Vec<_>>(),
            [
                boundary - 4,
                boundary,
                boundary - 3,
                boundary + 1,
                boundary - 2,
                boundary + 2,
                boundary - 1,
            ]
        );

        let overflow_vectors = vec![f32::MAX, -f32::MAX];
        let (prepared, metadata) = prepare_resident_matrix(overflow_vectors, 1, Metric::L2);
        let resident =
            ResidentMatrixInner::new(Arc::clone(&runtime), Metric::L2, 1, 2, &prepared, &metadata)
                .unwrap();
        let (query, scale, norm_squared) = prepare_resident_query(&[f32::MAX], Metric::L2);
        let mut scratch = resident.take_scratch().unwrap();
        let actual = resident
            .execute_search(&mut scratch, &query, scale, norm_squared, 2)
            .unwrap();
        assert_eq!(actual, vec![(0, 0.0)]);
    }
}
