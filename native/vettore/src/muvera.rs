//! MuVERA (Fixed Dimensional Encodings) — minimal and dependency-light.
//! Works with Elixir NIFs by taking Vec<Vec<f32>> as token embeddings.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProjectionType {
    Identity,
    AmsSketch,
}

#[derive(Clone, Debug)]
pub struct MuveraConfig {
    pub dimension: usize,                          // token embedding dimension
    pub num_repetitions: usize,                    // R
    pub num_simhash_projections: u32,              // p in [0,31] => 2^p partitions
    pub seed: u64,                                 // determinism
    pub projection_type: ProjectionType,           // Identity (default) or AmsSketch
    pub projection_dimension: Option<usize>,       // inner dim for AmsSketch; None=>Identity
    pub fill_empty_partitions: bool,               // only used for Avg/doc
    pub final_projection_dimension: Option<usize>, // optional CountSketch
}

impl Default for MuveraConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            num_repetitions: 2,
            num_simhash_projections: 6,
            seed: 42,
            projection_type: ProjectionType::Identity,
            projection_dimension: None,
            fill_empty_partitions: false,
            final_projection_dimension: None,
        }
    }
}

pub enum Aggregation {
    Sum, // for queries
    Avg, // for documents
}

pub struct MuveraEncoder {
    cfg: MuveraConfig,
}

struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_bool(&mut self) -> bool {
        (self.next_u64() & 1) == 1
    }
    fn gen_range(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.next_u64() as usize) % upper
        }
    }
}
fn splitmix64_once(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/* ---- small helpers ---- */
#[inline]
fn clamp_p(p: u32) -> u32 {
    if p > 31 {
        31
    } else {
        p
    }
}
#[inline]
fn partitions(p: u32) -> usize {
    1usize << p
}
#[inline]
fn append_gray(g: u32, bit: u32) -> u32 {
    (g << 1) + (bit ^ (g & 1))
}
#[inline]
fn gray_to_binary(g: u32) -> u32 {
    let mut n = g;
    let mut m = n >> 1;
    while m != 0 {
        n ^= m;
        m >>= 1;
    }
    n
}

impl MuveraEncoder {
    pub fn new(mut cfg: MuveraConfig) -> Self {
        cfg.num_simhash_projections = clamp_p(cfg.num_simhash_projections);
        if let ProjectionType::AmsSketch = cfg.projection_type {
            if cfg.projection_dimension.unwrap_or(0) == 0 {
                // invalid AMS config -> fallback to Identity
                cfg.projection_type = ProjectionType::Identity;
                cfg.projection_dimension = None;
            }
        }
        Self { cfg }
    }

    #[inline]
    fn inner_dim(&self) -> usize {
        match self.cfg.projection_type {
            ProjectionType::Identity => self.cfg.dimension,
            ProjectionType::AmsSketch => {
                self.cfg.projection_dimension.unwrap_or(self.cfg.dimension)
            }
        }
    }
    pub fn fde_dim_raw(&self) -> usize {
        self.cfg.num_repetitions * partitions(self.cfg.num_simhash_projections) * self.inner_dim()
    }
    pub fn fde_dim(&self) -> usize {
        self.cfg
            .final_projection_dimension
            .unwrap_or_else(|| self.fde_dim_raw())
    }

    pub fn encode_query(&self, tokens: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        self.encode(tokens, Aggregation::Sum)
    }
    pub fn encode_doc(&self, tokens: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        self.encode(tokens, Aggregation::Avg)
    }

    fn encode(&self, tokens: &[Vec<f32>], mode: Aggregation) -> Result<Vec<f32>, String> {
        let n = tokens.len();
        let d = self.cfg.dimension;
        for (i, row) in tokens.iter().enumerate() {
            if row.len() != d {
                return Err(format!("row {} has dim {}, expected {}", i, row.len(), d));
            }
        }

        // early out: empty input => zero vector (after optional final projection)
        if n == 0 {
            let mut v = vec![0.0f32; self.fde_dim_raw()];
            if let Some(fd) = self.cfg.final_projection_dimension {
                v = self.count_sketch(&v, fd, self.cfg.seed);
            }
            return Ok(v);
        }

        let p = self.cfg.num_simhash_projections;
        let parts = partitions(p);
        let inner = self.inner_dim();
        let mask = if p == 32 { u32::MAX } else { (1u32 << p) - 1 };
        let mut out = vec![0.0f32; self.fde_dim_raw()];

        for rep in 0..self.cfg.num_repetitions {
            // SimHash (dim x p) with ±1 entries
            let mut rng =
                SplitMix64::new(self.cfg.seed ^ ((rep as u64).wrapping_mul(0xD6E8FEB86659FD93)));
            let mut simhash = vec![0.0f32; d * (p as usize)];
            for j in 0..d {
                for b in 0..(p as usize) {
                    simhash[j * (p as usize) + b] = if rng.next_bool() { 1.0 } else { -1.0 };
                }
            }

            // AMS one-sparse maps (if enabled)
            let (mut ams_idx, mut ams_sign);
            if let ProjectionType::AmsSketch = self.cfg.projection_type {
                ams_idx = vec![0usize; d];
                ams_sign = vec![0.0f32; d];
                let mut prng = SplitMix64::new(self.cfg.seed ^ 0x9E3779B97F4A7C15u64 ^ rep as u64);
                for j in 0..d {
                    ams_idx[j] = prng.gen_range(inner);
                    ams_sign[j] = if prng.next_bool() { 1.0 } else { -1.0 };
                }
            } else {
                ams_idx = Vec::new();
                ams_sign = Vec::new();
            }

            // per-token: SimHash bits+Gray, plus inner projection
            let mut token_bits = vec![0u32; n];
            let mut token_gray = vec![0u32; n];
            let mut projected = vec![0.0f32; n * inner];

            for i in 0..n {
                let row = &tokens[i];

                // SimHash sign bits and Gray-code index
                let mut bits: u32 = 0;
                let mut gray: u32 = 0;
                for b in 0..(p as usize) {
                    let mut dp = 0.0f32;
                    let base = b;
                    for j in 0..d {
                        dp += row[j] * simhash[j * (p as usize) + base];
                    }
                    let bit = if dp > 0.0 { 1u32 } else { 0u32 };
                    bits = (bits << 1) | bit;
                    gray = append_gray(gray, bit);
                }
                token_bits[i] = bits & mask;
                token_gray[i] = gray & mask;

                // Project row into inner-dim
                let dst = &mut projected[i * inner..(i + 1) * inner];
                match self.cfg.projection_type {
                    ProjectionType::Identity => {
                        for j in 0..d {
                            dst[j] = row[j];
                        }
                    }
                    ProjectionType::AmsSketch => {
                        for j in 0..d {
                            let idx = ams_idx[j];
                            dst[idx] += ams_sign[j] * row[j];
                        }
                    }
                }
            }

            // aggregate per partition
            let rep_base = rep * parts * inner;
            let mut counts = vec![0u32; parts];
            for i in 0..n {
                let pidx = token_gray[i] as usize;
                let dst = &mut out[rep_base + pidx * inner..rep_base + (pidx + 1) * inner];
                let src = &projected[i * inner..(i + 1) * inner];
                for k in 0..inner {
                    dst[k] += src[k];
                }
                counts[pidx] = counts[pidx].saturating_add(1);
            }

            // AVG normalization + optional fill-empty
            if let Aggregation::Avg = mode {
                for part in 0..parts {
                    let c = counts[part];
                    let dst = &mut out[rep_base + part * inner..rep_base + (part + 1) * inner];
                    if c > 0 {
                        let inv = 1.0f32 / (c as f32);
                        for k in 0..inner {
                            dst[k] *= inv;
                        }
                    } else if self.cfg.fill_empty_partitions && n > 0 {
                        let tgt_bin = gray_to_binary(part as u32) & mask;
                        let mut best_i = 0usize;
                        let mut best_dist = u32::MAX;
                        for i in 0..n {
                            let dist = ((token_bits[i] ^ tgt_bin) & mask).count_ones();
                            if dist < best_dist {
                                best_dist = dist;
                                best_i = i;
                                if dist == 0 {
                                    break;
                                }
                            }
                        }
                        let src = &projected[best_i * inner..(best_i + 1) * inner];
                        for k in 0..inner {
                            dst[k] = src[k];
                        }
                    }
                }
            }
        }

        // optional final CountSketch compression
        if let Some(fd) = self.cfg.final_projection_dimension {
            Ok(self.count_sketch(&out, fd, self.cfg.seed))
        } else {
            Ok(out)
        }
    }

    fn count_sketch(&self, input: &[f32], final_dim: usize, seed: u64) -> Vec<f32> {
        let mut out = vec![0.0f32; final_dim];
        for (i, &v) in input.iter().enumerate() {
            let h = splitmix64_once(seed ^ (i as u64).wrapping_mul(0x9E3779B97F4A7C15));
            let j = (h as usize) % final_dim;
            let sign = if (h >> 62) & 1 == 0 { 1.0f32 } else { -1.0f32 };
            out[j] += sign * v;
        }
        out
    }
}

pub fn muvera_encode_query(tokens: &[Vec<f32>], cfg: &MuveraConfig) -> Result<Vec<f32>, String> {
    MuveraEncoder::new(cfg.clone()).encode_query(tokens)
}
pub fn muvera_encode_doc(tokens: &[Vec<f32>], cfg: &MuveraConfig) -> Result<Vec<f32>, String> {
    MuveraEncoder::new(cfg.clone()).encode_doc(tokens)
}

/// Build config from simple primitives; string PT: "identity" | "ams"
pub fn make_config(
    dimension: usize,
    num_repetitions: usize,
    num_simhash_projections: u32,
    seed: u64,
    projection_type: &str,
    projection_dimension: Option<usize>,
    fill_empty_partitions: bool,
    final_projection_dimension: Option<usize>,
) -> MuveraConfig {
    let pt = match projection_type.to_ascii_lowercase().as_str() {
        "ams" | "amssketch" | "ams_sketch" => ProjectionType::AmsSketch,
        _ => ProjectionType::Identity,
    };
    MuveraConfig {
        dimension,
        num_repetitions,
        num_simhash_projections,
        seed,
        projection_type: pt,
        projection_dimension,
        fill_empty_partitions,
        final_projection_dimension,
    }
}
