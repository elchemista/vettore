//! Native MUVERA fixed-dimensional encoding implementation.
//!
//! Query encodings sum projected vectors per partition. Document encodings
//! average projected vectors per partition. Both use deterministic hashing from
//! the shared config seed so query and document FDEs are comparable.

pub enum Mode {
    Query,
    Document,
}

pub struct Config {
    pub dimension: usize,
    pub num_repetitions: usize,
    pub num_simhash_projections: usize,
    pub seed: u64,
    pub projection_dimension: usize,
    pub final_projection_dimension: Option<usize>,
}

/// Builds a fixed-dimensional encoding for query or document multi-vectors.
pub fn encode(vectors: Vec<Vec<f32>>, config: Config, mode: Mode) -> Result<Vec<f32>, String> {
    validate(&vectors, &config)?;

    let partitions = 1usize
        .checked_shl(config.num_simhash_projections as u32)
        .ok_or_else(|| "invalid simhash projection count".to_string())?;
    let repetition_size = partitions
        .checked_mul(config.projection_dimension)
        .ok_or_else(|| "fde dimension overflow".to_string())?;
    let output_size = config
        .num_repetitions
        .checked_mul(repetition_size)
        .ok_or_else(|| "fde dimension overflow".to_string())?;

    let mut out = vec![0.0f32; output_size];
    let mut counts = vec![0usize; config.num_repetitions * partitions];

    for repetition in 0..config.num_repetitions {
        for vector in &vectors {
            let partition = partition_index(vector, &config, repetition);
            let count_index = repetition * partitions + partition;
            counts[count_index] += 1;

            let base = repetition * repetition_size + partition * config.projection_dimension;
            add_projected(&mut out, base, vector, &config, repetition);
        }
    }

    if matches!(mode, Mode::Document) {
        for repetition in 0..config.num_repetitions {
            for partition in 0..partitions {
                let count = counts[repetition * partitions + partition];
                if count == 0 {
                    continue;
                }

                let base = repetition * repetition_size + partition * config.projection_dimension;
                let scale = count as f32;
                for offset in 0..config.projection_dimension {
                    out[base + offset] /= scale;
                }
            }
        }
    }

    match config.final_projection_dimension {
        Some(final_dimension) => count_sketch(&out, final_dimension, config.seed),
        None => Ok(out),
    }
}

/// Validates shape and configuration before allocating the FDE output.
fn validate(vectors: &[Vec<f32>], config: &Config) -> Result<(), String> {
    if vectors.is_empty() {
        return Err("empty vectors".to_string());
    }
    if config.dimension == 0 {
        return Err("dimension must be positive".to_string());
    }
    if config.num_repetitions == 0 {
        return Err("num_repetitions must be positive".to_string());
    }
    if config.num_simhash_projections >= 31 {
        return Err("num_simhash_projections must be < 31".to_string());
    }
    if config.projection_dimension == 0 {
        return Err("projection_dimension must be positive".to_string());
    }
    if config.final_projection_dimension == Some(0) {
        return Err("final_projection_dimension must be positive".to_string());
    }
    if vectors
        .iter()
        .any(|vector| vector.len() != config.dimension)
    {
        return Err("dimension mismatch".to_string());
    }
    Ok(())
}

/// Assigns one vector to a SimHash partition for a repetition.
fn partition_index(vector: &[f32], config: &Config, repetition: usize) -> usize {
    if config.num_simhash_projections == 0 {
        return 0;
    }

    let mut partition = 0usize;
    for projection in 0..config.num_simhash_projections {
        let mut dot = 0.0f32;
        for (dimension, value) in vector.iter().enumerate() {
            dot += *value * random_weight(config.seed, repetition, projection, dimension);
        }
        partition = (partition << 1) + usize::from(dot >= 0.0);
    }
    partition
}

/// Adds either identity-projected or signed-projected coordinates into output.
fn add_projected(out: &mut [f32], base: usize, vector: &[f32], config: &Config, repetition: usize) {
    if config.projection_dimension == config.dimension {
        for (offset, value) in vector.iter().enumerate() {
            out[base + offset] += *value;
        }
        return;
    }

    for projection in 0..config.projection_dimension {
        let mut value = 0.0f32;
        for (dimension, coordinate) in vector.iter().enumerate() {
            value += *coordinate
                * random_sign(
                    config.seed.wrapping_add(17),
                    repetition,
                    projection,
                    dimension,
                );
        }
        out[base + projection] += value;
    }
}

/// Compresses the full FDE with a count-sketch style signed hash projection.
fn count_sketch(input: &[f32], final_dimension: usize, seed: u64) -> Result<Vec<f32>, String> {
    if final_dimension == 0 {
        return Err("final_projection_dimension must be positive".to_string());
    }

    let mut out = vec![0.0f32; final_dimension];
    for (index, value) in input.iter().enumerate() {
        let slot = (hash4(seed, 0x9E37_79B9_7F4A_7C15, index as u64, 0) as usize) % final_dimension;
        let sign = if hash4(seed, 0xD1B5_4A32_D192_ED03, index as u64, slot as u64) & 1 == 0 {
            1.0
        } else {
            -1.0
        };
        out[slot] += sign * *value;
    }
    Ok(out)
}

/// Generates a deterministic pseudo-random projection weight in `[-1.0, 1.0]`.
fn random_weight(seed: u64, repetition: usize, projection: usize, dimension: usize) -> f32 {
    let hash = hash4(seed, repetition as u64, projection as u64, dimension as u64);
    let unit = (hash as f64 / u64::MAX as f64) as f32;
    unit * 2.0 - 1.0
}

/// Generates a deterministic Rademacher sign for sparse random projection.
fn random_sign(seed: u64, repetition: usize, projection: usize, dimension: usize) -> f32 {
    if hash4(seed, repetition as u64, projection as u64, dimension as u64) & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Mixes four integer coordinates into one stable 64-bit hash.
fn hash4(a: u64, b: u64, c: u64, d: u64) -> u64 {
    let mut x = a ^ b.rotate_left(17) ^ c.rotate_left(31) ^ d.rotate_left(47);
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}
