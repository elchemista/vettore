//! Native distance, similarity, normalization, and compression kernels.
//!
//! This module is intentionally algorithm-only. It owns no collection records
//! and no database state; Elixir/ETS remains the canonical store. Hot dense
//! vector kernels use portable SIMD through `wide` where it is useful and fall
//! back to scalar tails for arbitrary dimensions.

use wide::f32x8;

#[derive(Clone, Copy)]
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

/// Dispatches a named metric to its native kernel after checking dimensions.
pub fn compute(metric: Metric, left: &[f32], right: &[f32]) -> Result<f32, String> {
    if left.len() != right.len() {
        return Err("dimension mismatch".to_string());
    }

    match metric {
        Metric::L2 => Ok(l2(left, right)),
        Metric::L2Squared => Ok(l2_squared(left, right)),
        Metric::Cosine => Ok(dot(left, right)),
        Metric::InnerProduct => Ok(dot(left, right)),
        Metric::NegativeInnerProduct => Ok(-dot(left, right)),
        Metric::Manhattan => Ok(manhattan(left, right)),
        Metric::Chebyshev => Ok(chebyshev(left, right)),
        Metric::Hamming => Ok(hamming(left, right)),
        Metric::Jaccard => Ok(jaccard(left, right)),
    }
}

/// Converts similarity metrics into ascending rank distances for indexes.
pub fn rank_distance(metric: Metric, left: &[f32], right: &[f32]) -> Result<f32, String> {
    match metric {
        Metric::Cosine => Ok(1.0 - compute(metric, left, right)?),
        Metric::InnerProduct => Ok(-compute(metric, left, right)?),
        _ => compute(metric, left, right),
    }
}

/// Euclidean distance derived from the SIMD squared L2 kernel.
pub fn l2(left: &[f32], right: &[f32]) -> f32 {
    l2_squared(left, right).sqrt()
}

/// Squared L2 distance using portable SIMD for dense f32 chunks.
pub fn l2_squared(left: &[f32], right: &[f32]) -> f32 {
    simd_l2_squared(left, right)
}

/// Inner product using portable SIMD for dense f32 chunks.
pub fn dot(left: &[f32], right: &[f32]) -> f32 {
    simd_dot(left, right)
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
    let norm = dot(&vector, &vector).sqrt();
    if norm == 0.0 {
        Ok(vec![0.0; vector.len()])
    } else {
        Ok(vector.into_iter().map(|value| value / norm).collect())
    }
}

/// Z-score normalizes a vector using population variance.
pub fn normalize_zscore(vector: Vec<f32>) -> Result<Vec<f32>, String> {
    if vector.is_empty() {
        return Ok(vector);
    }

    let mean = vector.iter().sum::<f32>() / vector.len() as f32;
    let variance = vector
        .iter()
        .map(|value| {
            let diff = *value - mean;
            diff * diff
        })
        .sum::<f32>()
        / vector.len() as f32;
    let stddev = variance.sqrt();

    if stddev == 0.0 {
        Ok(vec![0.0; vector.len()])
    } else {
        Ok(vector
            .into_iter()
            .map(|value| (value - mean) / stddev)
            .collect())
    }
}

/// Min-max normalizes a vector into `[0.0, 1.0]`. Constant vectors become zero.
pub fn normalize_minmax(vector: Vec<f32>) -> Result<Vec<f32>, String> {
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
            .map(|value| (value - min) / (max - min))
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
        .map(|(index, (a, b))| ((*a ^ *b) & word_mask(index, dimensions)).count_ones())
        .sum::<u32>();

    Ok(distance as f32)
}

/// Jaccard distance over packed bit words.
pub fn packed_jaccard(left: &[u64], right: &[u64], dimensions: usize) -> Result<f32, String> {
    validate_packed_pair(left, right, dimensions)?;

    let mut intersection = 0u32;
    let mut union = 0u32;

    for (index, (a, b)) in left.iter().zip(right).enumerate() {
        let mask = word_mask(index, dimensions);
        intersection += ((*a & *b) & mask).count_ones();
        union += ((*a | *b) & mask).count_ones();
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
