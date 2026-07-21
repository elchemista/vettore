//! Native distance, similarity, normalization, and compression kernels.
//!
//! This module is intentionally algorithm-only. It owns no collection records
//! and no database state; Elixir/ETS remains the canonical store. Hot dense
//! vector kernels use portable SIMD through `wide` where it is useful and fall
//! back to scalar tails for arbitrary dimensions.

use wide::f32x8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Metric {
    L2,
    L2Squared,
    Cosine,
    InnerProduct,
    NegativeInnerProduct,
    Manhattan,
    Chebyshev,
    Hamming,
    Jaccard,
}

impl Metric {
    /// Decodes the compact metric identifier used by batched NIF helpers.
    pub fn from_code(code: u8) -> Result<Self, String> {
        match code {
            0 => Ok(Self::L2),
            1 => Ok(Self::L2Squared),
            2 => Ok(Self::Cosine),
            3 => Ok(Self::InnerProduct),
            4 => Ok(Self::NegativeInnerProduct),
            5 => Ok(Self::Manhattan),
            6 => Ok(Self::Chebyshev),
            7 => Ok(Self::Hamming),
            8 => Ok(Self::Jaccard),
            _ => Err("unknown metric".to_string()),
        }
    }
}

/// Dispatches a named metric to its native kernel after checking dimensions.
pub fn compute(metric: Metric, left: &[f32], right: &[f32]) -> Result<f32, String> {
    if left.len() != right.len() {
        return Err("dimension mismatch".to_string());
    }

    let value = match metric {
        Metric::L2 => l2(left, right),
        Metric::L2Squared => l2_squared(left, right),
        Metric::Cosine => dot(left, right),
        Metric::InnerProduct => dot(left, right),
        Metric::NegativeInnerProduct => -dot(left, right),
        Metric::Manhattan => manhattan(left, right),
        Metric::Chebyshev => chebyshev(left, right),
        Metric::Hamming => hamming(left, right),
        Metric::Jaccard => jaccard(left, right),
    };

    if value.is_finite() {
        return Ok(value);
    }

    // SIMD f32 intermediates can overflow even when the final mathematical
    // result is representable (for example, a large dot product whose terms
    // cancel). Recompute only that exceptional path in f64 before reporting a
    // genuine output overflow.
    recover_metric_overflow(metric, left, right).ok_or_else(|| "metric overflow".to_string())
}

fn recover_metric_overflow(metric: Metric, left: &[f32], right: &[f32]) -> Option<f32> {
    let recovered = match metric {
        Metric::L2 => f64_l2_squared(left, right).sqrt(),
        Metric::L2Squared => f64_l2_squared(left, right),
        Metric::Cosine | Metric::InnerProduct => f64_dot(left, right),
        Metric::NegativeInnerProduct => -f64_dot(left, right),
        Metric::Manhattan => left
            .iter()
            .zip(right)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
            .sum(),
        Metric::Chebyshev => left
            .iter()
            .zip(right)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
            .fold(0.0f64, f64::max),
        Metric::Hamming | Metric::Jaccard => return None,
    };

    f64_to_f32(recovered)
}

fn f64_to_f32(value: f64) -> Option<f32> {
    if value.is_finite() && value >= f64::from(f32::MIN) && value <= f64::from(f32::MAX) {
        Some(value as f32)
    } else {
        None
    }
}

/// Validates direct NIF inputs before dispatching to a metric kernel.
pub fn compute_checked(metric: Metric, left: &[f32], right: &[f32]) -> Result<f32, String> {
    validate_finite_vector(left)?;
    validate_finite_vector(right)?;
    compute(metric, left, right)
}

/// Converts similarity metrics into ascending rank distances for indexes.
pub fn rank_distance(metric: Metric, left: &[f32], right: &[f32]) -> Result<f32, String> {
    Ok(rank_value(metric, compute(metric, left, right)?))
}

/// Converts one already-computed raw metric value into ascending rank order.
pub fn rank_value(metric: Metric, raw: f32) -> f32 {
    match metric {
        Metric::Cosine => 1.0 - raw,
        Metric::InnerProduct => -raw,
        _ => raw,
    }
}

/// Converts one raw metric value into a higher-is-better similarity.
pub fn similarity_value(metric: Metric, raw: f32) -> f32 {
    match metric {
        Metric::Cosine | Metric::InnerProduct => raw,
        Metric::NegativeInnerProduct => -raw,
        _ => 1.0 / (1.0 + raw),
    }
}

/// Validates values before they enter long-lived native indexes.
pub fn validate_finite_vector(vector: &[f32]) -> Result<(), String> {
    if vector.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err("vector contains a non-finite value".to_string())
    }
}

/// Euclidean distance derived from the SIMD squared L2 kernel.
pub fn l2(left: &[f32], right: &[f32]) -> f32 {
    let squared = l2_squared(left, right);
    if squared.is_finite() {
        squared.sqrt()
    } else {
        f64_l2_squared(left, right).sqrt() as f32
    }
}

/// Squared L2 distance using portable SIMD for dense f32 chunks.
pub fn l2_squared(left: &[f32], right: &[f32]) -> f32 {
    simd_l2_squared(left, right)
}

/// Inner product using portable SIMD for dense f32 chunks.
pub fn dot(left: &[f32], right: &[f32]) -> f32 {
    simd_dot(left, right)
}

/// Cosine similarity for vectors that have not already been normalized.
pub fn cosine(left: &[f32], right: &[f32]) -> Result<f32, String> {
    if left.len() != right.len() {
        return Err("dimension mismatch".to_string());
    }

    let left_norm = f64_dot(left, left).sqrt();
    let right_norm = f64_dot(right, right).sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        Ok(0.0)
    } else {
        let similarity = f64_dot(left, right) / (left_norm * right_norm);
        if similarity.is_finite() {
            Ok(similarity.clamp(-1.0, 1.0) as f32)
        } else {
            Err("metric overflow".to_string())
        }
    }
}

fn f64_dot(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| f64::from(*left) * f64::from(*right))
        .sum()
}

fn f64_l2_squared(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let difference = f64::from(*left) - f64::from(*right);
            difference * difference
        })
        .sum()
}

/// Accumulates squared coordinate differences in 8-lane SIMD chunks.
fn simd_l2_squared(left: &[f32], right: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut i = 0usize;

    while i + 8 <= left.len() {
        let va = f32x8::from([
            left[i],
            left[i + 1],
            left[i + 2],
            left[i + 3],
            left[i + 4],
            left[i + 5],
            left[i + 6],
            left[i + 7],
        ]);
        let vb = f32x8::from([
            right[i],
            right[i + 1],
            right[i + 2],
            right[i + 3],
            right[i + 4],
            right[i + 5],
            right[i + 6],
            right[i + 7],
        ]);
        let diff = va - vb;
        acc += (diff * diff).reduce_add();
        i += 8;
    }

    while i < left.len() {
        let diff = left[i] - right[i];
        acc += diff * diff;
        i += 1;
    }
    acc
}

/// Accumulates products in 8-lane SIMD chunks.
fn simd_dot(left: &[f32], right: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut i = 0usize;

    while i + 8 <= left.len() {
        let va = f32x8::from([
            left[i],
            left[i + 1],
            left[i + 2],
            left[i + 3],
            left[i + 4],
            left[i + 5],
            left[i + 6],
            left[i + 7],
        ]);
        let vb = f32x8::from([
            right[i],
            right[i + 1],
            right[i + 2],
            right[i + 3],
            right[i + 4],
            right[i + 5],
            right[i + 6],
            right[i + 7],
        ]);
        acc += (va * vb).reduce_add();
        i += 8;
    }

    while i < left.len() {
        acc += left[i] * right[i];
        i += 1;
    }
    acc
}

/// Manhattan/L1 distance using SIMD absolute differences.
fn manhattan(left: &[f32], right: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    let mut i = 0usize;

    while i + 8 <= left.len() {
        let va = f32x8::from([
            left[i],
            left[i + 1],
            left[i + 2],
            left[i + 3],
            left[i + 4],
            left[i + 5],
            left[i + 6],
            left[i + 7],
        ]);
        let vb = f32x8::from([
            right[i],
            right[i + 1],
            right[i + 2],
            right[i + 3],
            right[i + 4],
            right[i + 5],
            right[i + 6],
            right[i + 7],
        ]);
        acc += (va - vb).abs().reduce_add();
        i += 8;
    }

    while i < left.len() {
        acc += (left[i] - right[i]).abs();
        i += 1;
    }

    acc
}

/// Chebyshev/L-infinity distance.
fn chebyshev(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0f32, f32::max)
}

/// Hamming distance over truthy/non-truthy float coordinates.
fn hamming(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .filter(|(a, b)| (**a != 0.0) != (**b != 0.0))
        .count() as f32
}

/// Jaccard distance over truthy/non-truthy float coordinates.
fn jaccard(left: &[f32], right: &[f32]) -> f32 {
    let mut intersection = 0usize;
    let mut union = 0usize;

    for (a, b) in left.iter().zip(right) {
        let left_truthy = *a != 0.0;
        let right_truthy = *b != 0.0;
        if left_truthy || right_truthy {
            union += 1;
        }
        if left_truthy && right_truthy {
            intersection += 1;
        }
    }

    if union == 0 {
        0.0
    } else {
        1.0 - intersection as f32 / union as f32
    }
}

/// L2-normalizes a vector. Zero vectors stay zero.
pub fn normalize_l2(vector: Vec<f32>) -> Result<Vec<f32>, String> {
    validate_finite_vector(&vector)?;
    let norm = f64_dot(&vector, &vector).sqrt();
    if norm == 0.0 {
        Ok(vec![0.0; vector.len()])
    } else {
        Ok(vector
            .into_iter()
            .map(|value| (f64::from(value) / norm) as f32)
            .collect())
    }
}

/// Z-score normalizes a vector using population variance.
pub fn normalize_zscore(vector: Vec<f32>) -> Result<Vec<f32>, String> {
    validate_finite_vector(&vector)?;
    if vector.is_empty() {
        return Ok(vector);
    }

    let mean = vector.iter().map(|value| f64::from(*value)).sum::<f64>() / vector.len() as f64;
    let variance = vector
        .iter()
        .map(|value| {
            let diff = f64::from(*value) - mean;
            diff * diff
        })
        .sum::<f64>()
        / vector.len() as f64;
    let stddev = variance.sqrt();

    if stddev == 0.0 {
        Ok(vec![0.0; vector.len()])
    } else {
        Ok(vector
            .into_iter()
            .map(|value| ((f64::from(value) - mean) / stddev) as f32)
            .collect())
    }
}

/// Min-max normalizes a vector into `[0.0, 1.0]`. Constant vectors become zero.
pub fn normalize_minmax(vector: Vec<f32>) -> Result<Vec<f32>, String> {
    validate_finite_vector(&vector)?;
    if vector.is_empty() {
        return Ok(vector);
    }

    let min = vector.iter().copied().fold(f32::INFINITY, f32::min);
    let max = vector.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if min == max {
        Ok(vec![0.0; vector.len()])
    } else {
        Ok(vector
            .into_iter()
            .map(|value| {
                ((f64::from(value) - f64::from(min)) / (f64::from(max) - f64::from(min))) as f32
            })
            .collect())
    }
}

/// Encodes vector signs into packed 64-bit words.
pub fn compress_sign_bits(vector: &[f32]) -> Vec<u64> {
    let mut words = vec![0u64; vector.len().div_ceil(64)];

    for (index, value) in vector.iter().enumerate() {
        if *value >= 0.0 {
            words[index / 64] |= 1u64 << (index % 64);
        }
    }

    words
}

/// Hamming distance over packed bit words.
pub fn packed_hamming(left: &[u64], right: &[u64], dimensions: usize) -> Result<f32, String> {
    validate_packed_pair(left, right, dimensions)?;

    let distance = left
        .iter()
        .zip(right)
        .enumerate()
        .map(|(index, (a, b))| u64::from(((*a ^ *b) & word_mask(index, dimensions)).count_ones()))
        .sum::<u64>();

    Ok(distance as f32)
}

/// Jaccard distance over packed bit words.
pub fn packed_jaccard(left: &[u64], right: &[u64], dimensions: usize) -> Result<f32, String> {
    validate_packed_pair(left, right, dimensions)?;

    let mut intersection = 0u64;
    let mut union = 0u64;

    for (index, (a, b)) in left.iter().zip(right).enumerate() {
        let mask = word_mask(index, dimensions);
        intersection += u64::from(((*a & *b) & mask).count_ones());
        union += u64::from(((*a | *b) & mask).count_ones());
    }

    if union == 0 {
        Ok(0.0)
    } else {
        Ok(1.0 - intersection as f32 / union as f32)
    }
}

fn validate_packed_pair(left: &[u64], right: &[u64], dimensions: usize) -> Result<(), String> {
    let words = dimensions.div_ceil(64);

    if dimensions == 0 {
        return Err("dimensions must be positive".to_string());
    }
    if left.len() != words || right.len() != words {
        return Err("dimension mismatch".to_string());
    }

    Ok(())
}

fn word_mask(index: usize, dimensions: usize) -> u64 {
    let words = dimensions.div_ceil(64);
    let remainder = dimensions % 64;

    if index + 1 == words && remainder != 0 {
        (1u64 << remainder) - 1
    } else {
        u64::MAX
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32, tolerance: f32) {
        let scale = 1.0f32.max(actual.abs()).max(expected.abs());
        assert!(
            (actual - expected).abs() <= tolerance * scale,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn computes_every_metric_and_rank_semantics() {
        let left = [1.0, 0.0, 1.0];
        let right = [0.0, 1.0, 1.0];

        assert_eq!(compute(Metric::L2Squared, &left, &right), Ok(2.0));
        assert!((compute(Metric::L2, &left, &right).unwrap() - 2.0_f32.sqrt()).abs() < 1e-6);
        assert_eq!(compute(Metric::Cosine, &left, &right), Ok(1.0));
        assert_eq!(compute(Metric::InnerProduct, &left, &right), Ok(1.0));
        assert_eq!(
            compute(Metric::NegativeInnerProduct, &left, &right),
            Ok(-1.0)
        );
        assert_eq!(compute(Metric::Manhattan, &left, &right), Ok(2.0));
        assert_eq!(compute(Metric::Chebyshev, &left, &right), Ok(1.0));
        assert_eq!(compute(Metric::Hamming, &left, &right), Ok(2.0));
        assert!((compute(Metric::Jaccard, &left, &right).unwrap() - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(rank_value(Metric::InnerProduct, 2.0), -2.0);
        assert_eq!(rank_value(Metric::Cosine, 0.25), 0.75);
        assert_eq!(similarity_value(Metric::NegativeInnerProduct, -3.0), 3.0);
    }

    #[test]
    fn validates_dimensions_normalization_and_finite_values() {
        assert_eq!(
            compute(Metric::L2, &[1.0], &[1.0, 2.0]),
            Err("dimension mismatch".into())
        );
        assert_eq!(normalize_l2(vec![3.0, 4.0]).unwrap(), vec![0.6, 0.8]);
        assert_eq!(normalize_l2(vec![0.0, 0.0]).unwrap(), vec![0.0, 0.0]);
        assert_eq!(normalize_zscore(vec![4.0, 4.0]).unwrap(), vec![0.0, 0.0]);
        assert_eq!(normalize_minmax(vec![7.0, 7.0]).unwrap(), vec![0.0, 0.0]);
        assert!(validate_finite_vector(&[1.0, f32::NAN]).is_err());
        assert_eq!(cosine(&[2.0, 0.0], &[4.0, 0.0]), Ok(1.0));
        assert_eq!(cosine(&[0.0, 0.0], &[4.0, 0.0]), Ok(0.0));

        let max = f32::MAX;
        let normalized = normalize_l2(vec![max, max]).unwrap();
        assert!((normalized[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert_eq!(normalize_minmax(vec![-max, max]).unwrap(), vec![0.0, 1.0]);
        assert!(compute_checked(Metric::InnerProduct, &[max], &[max]).is_err());
        assert!(compute_checked(Metric::Hamming, &[f32::NAN], &[0.0]).is_err());
    }

    #[test]
    fn packs_bits_and_masks_unused_coordinates() {
        let left = compress_sign_bits(&[1.0, -1.0, 0.0]);
        let right = compress_sign_bits(&[-1.0, -1.0, 0.0]);
        assert_eq!(left, vec![5]);
        assert_eq!(packed_hamming(&left, &right, 3), Ok(1.0));
        assert_eq!(packed_jaccard(&left, &right, 3), Ok(0.5));
        assert!(packed_hamming(&left, &right, 0).is_err());
        assert!(packed_hamming(&left, &[], 3).is_err());
    }

    #[test]
    fn decodes_metric_codes() {
        let metrics = [
            Metric::L2,
            Metric::L2Squared,
            Metric::Cosine,
            Metric::InnerProduct,
            Metric::NegativeInnerProduct,
            Metric::Manhattan,
            Metric::Chebyshev,
            Metric::Hamming,
            Metric::Jaccard,
        ];
        for (code, expected) in metrics.into_iter().enumerate() {
            assert_eq!(Metric::from_code(code as u8), Ok(expected));
        }
        assert!(Metric::from_code(9).is_err());
        assert!(Metric::from_code(u8::MAX).is_err());
    }

    #[test]
    fn simd_and_tail_kernels_match_scalar_oracles() {
        for len in 0..=40 {
            let left: Vec<f32> = (0..len)
                .map(|index| ((index * 37 % 23) as f32 - 11.0) / 3.0)
                .collect();
            let right: Vec<f32> = (0..len)
                .map(|index| ((index * 19 % 29) as f32 - 14.0) / 5.0)
                .collect();

            let expected_dot = left
                .iter()
                .zip(&right)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum::<f64>() as f32;
            let expected_l2_squared = left
                .iter()
                .zip(&right)
                .map(|(a, b)| {
                    let difference = f64::from(*a) - f64::from(*b);
                    difference * difference
                })
                .sum::<f64>() as f32;
            let expected_manhattan = left
                .iter()
                .zip(&right)
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
                .sum::<f64>() as f32;
            let expected_chebyshev = left
                .iter()
                .zip(&right)
                .map(|(a, b)| (*a - *b).abs())
                .fold(0.0f32, f32::max);

            assert_close(dot(&left, &right), expected_dot, 2.0e-6);
            assert_close(l2_squared(&left, &right), expected_l2_squared, 2.0e-6);
            assert_close(manhattan(&left, &right), expected_manhattan, 2.0e-6);
            assert_eq!(chebyshev(&left, &right), expected_chebyshev);
        }
    }

    #[test]
    fn recovers_representable_results_after_f32_intermediate_overflow() {
        let large = 1.0e20f32;
        assert_close(
            compute(Metric::L2, &[large], &[0.0]).unwrap(),
            large,
            1.0e-6,
        );

        let max = f32::MAX;
        assert_eq!(
            compute(Metric::InnerProduct, &[max, max], &[2.0, -2.0]),
            Ok(0.0)
        );
        assert_eq!(
            compute(Metric::NegativeInnerProduct, &[max, max], &[2.0, -2.0]),
            Ok(-0.0)
        );

        assert!(compute(Metric::L2Squared, &[large], &[0.0]).is_err());
        assert!(compute(Metric::L2, &[max, max], &[0.0, 0.0]).is_err());
        assert!(compute(Metric::Manhattan, &[max, max], &[0.0, 0.0]).is_err());
        assert!(compute(Metric::Chebyshev, &[max], &[-max]).is_err());
        assert_eq!(compute(Metric::Jaccard, &[0.0, 0.0], &[0.0, 0.0]), Ok(0.0));
    }

    #[test]
    fn cosine_and_normalization_obey_numerical_invariants() {
        assert_eq!(cosine(&[], &[]), Ok(0.0));
        assert_eq!(
            cosine(&[1.0], &[1.0, 2.0]),
            Err("dimension mismatch".into())
        );
        assert_close(cosine(&[2.0, 0.0], &[-5.0, 0.0]).unwrap(), -1.0, 1.0e-6);
        assert_close(cosine(&[3.0, 4.0], &[6.0, 8.0]).unwrap(), 1.0, 1.0e-6);

        assert_eq!(normalize_l2(vec![]), Ok(vec![]));
        assert_eq!(normalize_zscore(vec![]), Ok(vec![]));
        assert_eq!(normalize_minmax(vec![]), Ok(vec![]));

        let l2_normalized = normalize_l2(vec![3.0, -4.0, 12.0]).unwrap();
        assert_close(f64_dot(&l2_normalized, &l2_normalized) as f32, 1.0, 1.0e-6);

        let zscore = normalize_zscore(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mean = zscore.iter().sum::<f32>() / zscore.len() as f32;
        let variance = zscore
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / zscore.len() as f32;
        assert_close(mean, 0.0, 1.0e-6);
        assert_close(variance, 1.0, 1.0e-6);

        let minmax = normalize_minmax(vec![-7.0, 0.0, 21.0]).unwrap();
        assert_eq!(minmax, vec![0.0, 0.25, 1.0]);

        for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(normalize_l2(vec![non_finite]).is_err());
            assert!(normalize_zscore(vec![non_finite]).is_err());
            assert!(normalize_minmax(vec![non_finite]).is_err());
            assert!(cosine(&[non_finite], &[1.0]).is_err());
        }
    }

    #[test]
    fn packed_distances_cover_word_boundaries_and_ignore_padding() {
        for dimensions in [1usize, 63, 64, 65, 127, 128, 129] {
            let words = dimensions.div_ceil(64);
            let left = vec![u64::MAX; words];
            let mut right = left.clone();
            let mut flipped = vec![0usize];
            if dimensions > 1 {
                flipped.push(dimensions - 1);
            }
            for coordinate in &flipped {
                right[coordinate / 64] ^= 1u64 << (coordinate % 64);
            }

            if !dimensions.is_multiple_of(64) {
                let used_mask = (1u64 << (dimensions % 64)) - 1;
                right[words - 1] ^= !used_mask;
            }

            assert_eq!(
                packed_hamming(&left, &right, dimensions),
                Ok(flipped.len() as f32)
            );
            assert_close(
                packed_jaccard(&left, &right, dimensions).unwrap(),
                flipped.len() as f32 / dimensions as f32,
                1.0e-6,
            );
        }

        assert_eq!(packed_jaccard(&[0], &[0], 64), Ok(0.0));
        assert!(packed_jaccard(&[], &[], 1).is_err());
    }
}
