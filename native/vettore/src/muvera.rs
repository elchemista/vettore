//! Native MUVERA fixed-dimensional encoding implementation.
//!
//! Query encodings sum projected vectors per partition. Document encodings
//! average projected vectors per partition. Both use deterministic hashing from
//! the shared config seed so query and document FDEs are comparable.

#[derive(Clone, Copy)]
pub enum Mode {
    Query,
    Document,
}

#[derive(Clone, Copy)]
pub struct Config {
    pub dimension: usize,
    pub num_repetitions: usize,
    pub num_simhash_projections: usize,
    pub seed: u64,
    pub projection_dimension: usize,
    pub final_projection_dimension: Option<usize>,
}

const MAX_OUTPUT_DIMENSIONS: usize = 16_777_216;

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
    let final_size = config.final_projection_dimension.unwrap_or(output_size);
    if output_size > MAX_OUTPUT_DIMENSIONS || final_size > MAX_OUTPUT_DIMENSIONS {
        return Err("fde dimension exceeds safety limit".to_string());
    }

    let mut out = vec![0.0f32; output_size];
    let counts_size = config
        .num_repetitions
        .checked_mul(partitions)
        .ok_or_else(|| "fde dimension overflow".to_string())?;
    let mut counts = vec![0usize; counts_size];

    for repetition in 0..config.num_repetitions {
        for vector in &vectors {
            let partition = partition_index(vector, &config, repetition);
            let count_index = repetition * partitions + partition;
            counts[count_index] += 1;

            let base = repetition * repetition_size + partition * config.projection_dimension;
            add_projected(
                &mut out,
                base,
                vector,
                &config,
                repetition,
                mode,
                counts[count_index],
            )?;
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
    for vector in vectors {
        crate::distances::validate_finite_vector(vector)?;
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
        let mut dot = 0.0f64;
        for (dimension, value) in vector.iter().enumerate() {
            dot += f64::from(*value)
                * f64::from(random_weight(
                    config.seed,
                    repetition,
                    projection,
                    dimension,
                ));
        }
        partition = (partition << 1) + usize::from(dot >= 0.0);
    }
    partition
}

/// Adds either identity-projected or signed-projected coordinates into output.
fn add_projected(
    out: &mut [f32],
    base: usize,
    vector: &[f32],
    config: &Config,
    repetition: usize,
    mode: Mode,
    count: usize,
) -> Result<(), String> {
    if config.projection_dimension == config.dimension {
        for (offset, value) in vector.iter().enumerate() {
            accumulate(&mut out[base + offset], f64::from(*value), mode, count)?;
        }
        return Ok(());
    }

    for projection in 0..config.projection_dimension {
        let mut value = 0.0f64;
        for (dimension, coordinate) in vector.iter().enumerate() {
            value += f64::from(*coordinate)
                * f64::from(random_sign(
                    config.seed.wrapping_add(17),
                    repetition,
                    projection,
                    dimension,
                ));
        }
        accumulate(&mut out[base + projection], value, mode, count)?;
    }
    Ok(())
}

fn accumulate(slot: &mut f32, value: f64, mode: Mode, count: usize) -> Result<(), String> {
    let current = f64::from(*slot);
    let next = match mode {
        Mode::Query => current + value,
        Mode::Document => current + (value - current) / count as f64,
    };

    if next.is_finite() && next >= f64::from(f32::MIN) && next <= f64::from(f32::MAX) {
        *slot = next as f32;
        Ok(())
    } else {
        Err("encoding overflow".to_string())
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
        let next = f64::from(out[slot]) + f64::from(sign * *value);
        if !next.is_finite() || next < f64::from(f32::MIN) || next > f64::from(f32::MAX) {
            return Err("encoding overflow".to_string());
        }
        out[slot] = next as f32;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        Config {
            dimension: 2,
            num_repetitions: 2,
            num_simhash_projections: 1,
            seed: 42,
            projection_dimension: 2,
            final_projection_dimension: None,
        }
    }

    #[test]
    fn query_and_document_encodings_are_deterministic_and_asymmetric() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let query = encode(vectors.clone(), config(), Mode::Query).unwrap();
        let repeated = encode(vectors.clone(), config(), Mode::Query).unwrap();
        let document = encode(vectors, config(), Mode::Document).unwrap();
        assert_eq!(query, repeated);
        assert_ne!(query, document);
        assert_eq!(query.len(), 8);
    }

    #[test]
    fn supports_projection_and_count_sketch() {
        let mut cfg = config();
        cfg.projection_dimension = 3;
        cfg.final_projection_dimension = Some(5);
        let encoded = encode(vec![vec![1.0, 2.0]], cfg, Mode::Query).unwrap();
        assert_eq!(encoded.len(), 5);
    }

    #[test]
    fn rejects_invalid_shapes_values_and_unsafe_allocations() {
        assert!(encode(vec![], config(), Mode::Query).is_err());
        assert!(encode(vec![vec![1.0]], config(), Mode::Query).is_err());
        assert!(encode(vec![vec![f32::NAN, 0.0]], config(), Mode::Query).is_err());

        let mut cfg = config();
        cfg.num_simhash_projections = 30;
        assert!(encode(vec![vec![1.0, 0.0]], cfg, Mode::Query).is_err());

        let mut cfg = config();
        cfg.final_projection_dimension = Some(0);
        assert!(encode(vec![vec![1.0, 0.0]], cfg, Mode::Query).is_err());

        let max = f32::MAX;
        let mut cfg = config();
        cfg.dimension = 1;
        cfg.projection_dimension = 1;
        cfg.num_repetitions = 1;
        cfg.num_simhash_projections = 0;
        assert!(encode(vec![vec![max], vec![max]], cfg, Mode::Query).is_err());

        let mut cfg = config();
        cfg.dimension = 1;
        cfg.projection_dimension = 1;
        cfg.num_repetitions = 1;
        cfg.num_simhash_projections = 0;
        assert_eq!(
            encode(vec![vec![max], vec![max]], cfg, Mode::Document),
            Ok(vec![max])
        );
    }

    #[test]
    fn validates_every_configuration_boundary() {
        let vectors = vec![vec![1.0, 0.0]];

        let mut invalid_configs = Vec::new();
        let mut cfg = config();
        cfg.dimension = 0;
        invalid_configs.push(cfg);
        let mut cfg = config();
        cfg.num_repetitions = 0;
        invalid_configs.push(cfg);
        let mut cfg = config();
        cfg.num_simhash_projections = 31;
        invalid_configs.push(cfg);
        let mut cfg = config();
        cfg.projection_dimension = 0;
        invalid_configs.push(cfg);
        let mut cfg = config();
        cfg.final_projection_dimension = Some(0);
        invalid_configs.push(cfg);

        for cfg in invalid_configs {
            assert!(encode(vectors.clone(), cfg, Mode::Query).is_err());
        }

        let mut cfg = config();
        cfg.num_repetitions = MAX_OUTPUT_DIMENSIONS + 1;
        cfg.num_simhash_projections = 0;
        cfg.projection_dimension = 1;
        assert_eq!(
            encode(vectors.clone(), cfg, Mode::Query),
            Err("fde dimension exceeds safety limit".into())
        );

        let mut cfg = config();
        cfg.final_projection_dimension = Some(MAX_OUTPUT_DIMENSIONS + 1);
        assert_eq!(
            encode(vectors, cfg, Mode::Query),
            Err("fde dimension exceeds safety limit".into())
        );
    }

    #[test]
    fn identity_projection_has_exact_sum_and_online_average_semantics() {
        let cfg = Config {
            dimension: 2,
            num_repetitions: 1,
            num_simhash_projections: 0,
            seed: 0,
            projection_dimension: 2,
            final_projection_dimension: None,
        };
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![-2.0, 0.0]];

        assert_eq!(
            encode(vectors.clone(), cfg, Mode::Query),
            Ok(vec![2.0, 6.0])
        );
        assert_eq!(
            encode(vectors, cfg, Mode::Document),
            Ok(vec![2.0 / 3.0, 2.0])
        );
    }

    #[test]
    fn encodings_are_permutation_invariant_and_seed_sensitive() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![-1.0, 0.5]];
        let mut reversed = vectors.clone();
        reversed.reverse();

        let query = encode(vectors.clone(), config(), Mode::Query).unwrap();
        let reversed_query = encode(reversed.clone(), config(), Mode::Query).unwrap();
        assert_eq!(query, reversed_query);

        let document = encode(vectors.clone(), config(), Mode::Document).unwrap();
        let reversed_document = encode(reversed, config(), Mode::Document).unwrap();
        for (left, right) in document.iter().zip(reversed_document) {
            assert!((left - right).abs() <= 1.0e-6);
        }

        let mut another_seed = config();
        another_seed.seed += 1;
        assert_ne!(query, encode(vectors, another_seed, Mode::Query).unwrap());
    }

    #[test]
    fn projection_sizes_hashes_and_random_weights_stay_in_range() {
        for projections in 0..=4 {
            let mut cfg = config();
            cfg.num_repetitions = 3;
            cfg.num_simhash_projections = projections;
            cfg.projection_dimension = 5;
            let encoded = encode(vec![vec![1.0, -2.0]], cfg, Mode::Query).unwrap();
            assert_eq!(encoded.len(), 3 * (1usize << projections) * 5);
        }

        for seed in [0, 1, 42, u64::MAX] {
            for coordinate in 0..100 {
                let weight = random_weight(seed, 3, 7, coordinate);
                assert!((-1.0..=1.0).contains(&weight));
                assert!(matches!(random_sign(seed, 3, 7, coordinate), -1.0 | 1.0));
                assert_eq!(
                    hash4(seed, 3, 7, coordinate as u64),
                    hash4(seed, 3, 7, coordinate as u64)
                );
            }
        }
    }

    #[test]
    fn count_sketch_detects_colliding_accumulation_overflow() {
        let overflow_seed = (0..10_000u64)
            .find(|seed| count_sketch(&[f32::MAX, f32::MAX], 1, *seed).is_err())
            .expect("at least one seed must assign equal signs to both inputs");
        assert_eq!(
            count_sketch(&[f32::MAX, f32::MAX], 1, overflow_seed),
            Err("encoding overflow".into())
        );
        assert_eq!(
            count_sketch(&[1.0, 2.0], 0, 0),
            Err("final_projection_dimension must be positive".into())
        );
    }
}
