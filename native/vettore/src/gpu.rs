//! Optional native GPU kernels backed by wgpu.
//!
//! The GPU device and compiled pipelines are initialized lazily and reused for
//! the lifetime of the loaded NIF. CPU/SIMD remains the caller-controlled
//! fallback for unavailable devices and small workloads.

use std::sync::{mpsc, Mutex, OnceLock};

use wgpu::util::DeviceExt;

use crate::distances::{self, Metric};
use crate::gpu_math::{
    cpu_normalize, finish_metric, metric_code, normalization_parameters, selected_matrix_rows,
};

const WORKGROUP_SIZE: u32 = 256;
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
var<workgroup> scratch_d: array<f32, 256>;

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
    var d = 0.0;

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
        } else if (operation == 9u) {
            a = x;
            b = x * x;
            c = x;
            d = x;
        }
    } else if (operation == 9u) {
        c = 3.402823466e38;
        d = -3.402823466e38;
    }

    scratch_a[lane] = a;
    scratch_b[lane] = b;
    scratch_c[lane] = c;
    scratch_d[lane] = d;
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

            if (operation == 9u) {
                scratch_c[lane] = min(scratch_c[lane], scratch_c[lane + stride]);
                scratch_d[lane] = max(scratch_d[lane], scratch_d[lane + stride]);
            } else {
                scratch_c[lane] = scratch_c[lane] + scratch_c[lane + stride];
                scratch_d[lane] = scratch_d[lane] + scratch_d[lane + stride];
            }
        }

        workgroupBarrier();
        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    if (lane == 0u) {
        output[workgroup_id.x] = vec4<f32>(
            scratch_a[0], scratch_b[0], scratch_c[0], scratch_d[0]
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
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let column = global_id.x;
    let dimensions = params.values.x;
    let selected_count = params.values.y;

    if (column >= dimensions) {
        return;
    }

    var sum = 0.0;
    var selected = 0u;
    loop {
        if (selected >= selected_count) {
            break;
        }
        sum = sum + matrix[indices[selected] * dimensions + column];
        selected = selected + 1u;
    }

    output[column] = sum / f32(selected_count);
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
    info: GpuInfo,
}

static GPU_RUNTIME: OnceLock<Result<Mutex<GpuRuntime>, String>> = OnceLock::new();

pub fn detected() -> bool {
    runtime().is_ok()
}

pub fn info() -> Result<GpuInfo, String> {
    let runtime = lock_runtime()?;
    Ok(runtime.info.clone())
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

    let mut runtime = lock_runtime()?;
    let partials = runtime.metric_partials(left, right, metric_code(metric))?;
    finish_metric(metric, &partials).or_else(|_| distances::compute(metric, left, right))
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

    let mut runtime = lock_runtime()?;
    let partials = runtime.metric_partials(&vector, &vector, 9)?;
    let (first, second) = normalization_parameters(method, vector.len(), &partials)?;

    if !first.is_finite() || !second.is_finite() {
        return cpu_normalize(vector, method);
    }

    let normalized = runtime.transform(&vector, method, first, second)?;

    if normalized.iter().all(|value| value.is_finite()) {
        Ok(normalized)
    } else {
        cpu_normalize(vector, method)
    }
}

pub fn mean_pool_f32_le(
    matrix: &[u8],
    dimensions: usize,
    row_indices: &[usize],
) -> Result<Vec<f32>, String> {
    let selected_matrix = selected_matrix_rows(matrix, dimensions, row_indices)?;

    let dimensions_u32 = u32::try_from(dimensions).map_err(|_| "invalid dimensions".to_string())?;
    let indices = (0..row_indices.len())
        .map(|index| u32::try_from(index).map_err(|_| "too many row indices".to_string()))
        .collect::<Result<Vec<_>, _>>()?;

    let mut runtime = lock_runtime()?;
    let pooled = runtime.mean_pool(&selected_matrix, dimensions_u32, &indices)?;

    if pooled.iter().all(|value| value.is_finite()) {
        Ok(pooled)
    } else {
        crate::dense::mean_pool_f32_le(matrix, dimensions, row_indices)
    }
}

impl GpuRuntime {
    fn new() -> Result<Self, String> {
        Self::new_with_software_adapter(false)
    }

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
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .map_err(|error| format!("gpu device initialization failed: {error}"))?;

        let metric_pipeline = create_pipeline(&device, "vettore metric", METRIC_SHADER);
        let normalize_pipeline = create_pipeline(&device, "vettore normalize", NORMALIZE_SHADER);
        let mean_pool_pipeline = create_pipeline(&device, "vettore mean pool", MEAN_POOL_SHADER);

        Ok(Self {
            device,
            queue,
            metric_pipeline,
            normalize_pipeline,
            mean_pool_pipeline,
            info: GpuInfo {
                name: adapter_info.name,
                backend: adapter_info.backend.to_str().to_string(),
                device_type: device_type_name(adapter_info.device_type).to_string(),
            },
        })
    }

    fn metric_partials(
        &mut self,
        left: &[f32],
        right: &[f32],
        operation: u32,
    ) -> Result<Vec<[f32; 4]>, String> {
        let length = u32::try_from(left.len()).map_err(|_| "vector too large".to_string())?;
        let groups = length.div_ceil(WORKGROUP_SIZE);
        let left_buffer = storage_buffer(&self.device, "vettore metric left", left);
        let right_buffer = storage_buffer(&self.device, "vettore metric right", right);
        let output_size = u64::from(groups) * 16;
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
            .chunks_exact(16)
            .map(|chunk| {
                let values = bytemuck::cast_slice::<u8, f32>(chunk);
                [values[0], values[1], values[2], values[3]]
            })
            .collect())
    }

    fn transform(
        &mut self,
        vector: &[f32],
        method: u8,
        first: f32,
        second: f32,
    ) -> Result<Vec<f32>, String> {
        let length = u32::try_from(vector.len()).map_err(|_| "vector too large".to_string())?;
        let groups = length.div_ceil(WORKGROUP_SIZE);
        let input = storage_buffer(&self.device, "vettore normalize input", vector);
        let output_size = u64::from(length) * 4;
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
        Ok(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
    }

    fn mean_pool(
        &mut self,
        matrix: &[u8],
        dimensions: u32,
        indices: &[u32],
    ) -> Result<Vec<f32>, String> {
        let groups = dimensions.div_ceil(WORKGROUP_SIZE);
        let matrix_buffer = storage_bytes(&self.device, "vettore mean matrix", matrix);
        let indices_buffer = storage_buffer(&self.device, "vettore mean indices", indices);
        let output_size = u64::from(dimensions) * 4;
        let output = output_buffer(&self.device, "vettore mean output", output_size);
        let params = uniform_buffer(
            &self.device,
            [
                dimensions,
                u32::try_from(indices.len()).map_err(|_| "too many row indices".to_string())?,
                0,
                0,
            ],
        );
        let layout = self.mean_pool_pipeline.get_bind_group_layout(0);
        let bind_group = bind_group(
            &self.device,
            "vettore mean bindings",
            &layout,
            &[&matrix_buffer, &indices_buffer, &output, &params],
        );

        let bytes = self.dispatch_and_read(
            &self.mean_pool_pipeline,
            &bind_group,
            groups,
            &output,
            output_size,
        )?;
        Ok(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
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
        self.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ignored = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| format!("gpu synchronization failed: {error}"))?;
        receiver
            .recv()
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

fn runtime() -> Result<&'static Mutex<GpuRuntime>, String> {
    let initialized = GPU_RUNTIME.get_or_init(|| {
        std::panic::catch_unwind(GpuRuntime::new)
            .map_err(|_| "gpu initialization panicked".to_string())
            .and_then(|result| result)
            .map(Mutex::new)
    });

    initialized.as_ref().map_err(Clone::clone)
}

fn lock_runtime() -> Result<std::sync::MutexGuard<'static, GpuRuntime>, String> {
    runtime()?
        .lock()
        .map_err(|_| "gpu runtime lock poisoned".to_string())
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
        for source in [METRIC_SHADER, NORMALIZE_SHADER, MEAN_POOL_SHADER] {
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
    fn available_wgpu_adapter_executes_every_pipeline() {
        let Ok(mut runtime) = GpuRuntime::new_with_software_adapter(true) else {
            return;
        };

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
            let partials = runtime
                .metric_partials(&left, &right, metric_code(metric))
                .unwrap();
            let actual = finish_metric(metric, &partials).unwrap();
            assert!((actual - expected).abs() < 1.0e-5, "metric {metric:?}");
        }

        let stats = runtime
            .metric_partials(&[3.0, 4.0], &[3.0, 4.0], 9)
            .unwrap();
        let (norm, unused) = normalization_parameters(1, 2, &stats).unwrap();
        let normalized = runtime.transform(&[3.0, 4.0], 1, norm, unused).unwrap();
        assert!((normalized[0] - 0.6).abs() < 1.0e-5);
        assert!((normalized[1] - 0.8).abs() < 1.0e-5);

        for method in [2, 3] {
            let values = [1.0, 2.0, 3.0, 4.0];
            let stats = runtime.metric_partials(&values, &values, 9).unwrap();
            let (first, second) = normalization_parameters(method, values.len(), &stats).unwrap();
            let actual = runtime.transform(&values, method, first, second).unwrap();
            let expected = cpu_normalize(values.to_vec(), method).unwrap();
            for (actual, expected) in actual.iter().zip(expected) {
                assert!((actual - expected).abs() < 1.0e-5);
            }
        }

        for method in [1, 2, 3] {
            let values = [4.0, 4.0];
            let stats = runtime.metric_partials(&values, &values, 9).unwrap();
            let (first, second) = normalization_parameters(method, values.len(), &stats).unwrap();
            let actual = runtime.transform(&values, method, first, second).unwrap();
            assert!(actual.iter().all(|value| value.is_finite()));
        }

        let matrix = [1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let pooled = runtime.mean_pool(&matrix, 2, &[0, 1]).unwrap();
        assert!((pooled[0] - 2.0).abs() < 1.0e-5);
        assert!((pooled[1] - 3.0).abs() < 1.0e-5);

        let repeated = runtime.mean_pool(&matrix, 2, &[1, 1, 0]).unwrap();
        assert!((repeated[0] - 7.0 / 3.0).abs() < 1.0e-5);
        assert!((repeated[1] - 10.0 / 3.0).abs() < 1.0e-5);
    }
}
