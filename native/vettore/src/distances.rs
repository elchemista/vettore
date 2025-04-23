use crate::simd_utils::load_f32x4;
use crate::types::{Distance, Embedding};

/// Clamp a float into the inclusive range `[0, 1]`.
pub fn clamp_0_1(val: f32) -> f32 {
    if val < 0.0 {
        0.0
    } else if val > 1.0 {
        1.0
    } else {
        val
    }
}

/// SIMD Euclidean distance (L2).
pub fn simd_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut i = 0;
    let mut sum_squares = 0.0;
    while i + 4 <= len {
        let va = load_f32x4(a, i);
        let vb = load_f32x4(b, i);
        sum_squares += ((va - vb) * (va - vb)).reduce_add();
        i += 4;
    }
    while i < len {
        let d = a[i] - b[i];
        sum_squares += d * d;
        i += 1;
    }
    sum_squares.sqrt()
}

/// SIMD dot product.
pub fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut i = 0;
    let mut accum = 0.0;
    while i + 4 <= len {
        let va = load_f32x4(a, i);
        let vb = load_f32x4(b, i);
        accum += (va * vb).reduce_add();
        i += 4;
    }
    while i < len {
        accum += a[i] * b[i];
        i += 1;
    }
    accum
}

/// Compress a vector into sign bits (1 bit per component → `u64`s).
pub fn compress_vector(vector: &[f32]) -> Vec<u64> {
    let mut out = Vec::new();
    let mut current: u64 = 0;
    let mut filled = 0;
    for &v in vector {
        current <<= 1;
        if v >= 0.0 {
            current |= 1;
        }
        filled += 1;
        if filled == 64 {
            out.push(current);
            current = 0;
            filled = 0;
        }
    }
    if filled > 0 {
        current <<= 64 - filled;
        out.push(current);
    }
    out
}

pub fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// Convert any raw distance / similarity metric to a `[0 … 1]` score.
pub fn compute_0_1_score(query: &[f32], emb: &Embedding, dist: Distance) -> f32 {
    match dist {
        Distance::Euclidean | Distance::Hnsw => {
            let d = simd_euclidean_distance(query, &emb.vector);
            clamp_0_1(1.0 / (1.0 + d))
        }
        Distance::Cosine => {
            let cos = simd_dot_product(query, &emb.vector);
            clamp_0_1((cos + 1.0) / 2.0)
        }
        Distance::DotProduct => {
            let dp = simd_dot_product(query, &emb.vector);
            clamp_0_1(1.0 / (1.0 + f32::exp(-dp)))
        }
        Distance::Binary => {
            let qbits = compress_vector(query);
            let ebits_buf;
            let ebits: &[u64] = match &emb.binary {
                Some(b) => b,
                None => {
                    ebits_buf = compress_vector(&emb.vector);
                    &ebits_buf
                }
            };
            let d_bits = hamming_distance(&qbits, ebits) as f32;
            let frac = clamp_0_1(d_bits / query.len() as f32);
            1.0 - frac
        }
    }
}
