use crate::simd_utils::load_f32x4;
use crate::types::Distance;

pub fn clamp_0_1(v: f32) -> f32 {
    v.max(0.0).min(1.0)
}

pub fn simd_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;
    let mut acc = 0.0;
    while i + 4 <= len {
        let va = load_f32x4(a, i);
        let vb = load_f32x4(b, i);
        acc += ((va - vb) * (va - vb)).reduce_add();
        i += 4;
    }
    while i < len {
        let d = a[i] - b[i];
        acc += d * d;
        i += 1;
    }
    acc.sqrt()
}

pub fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut i = 0;
    let mut acc = 0.0;
    while i + 4 <= len {
        let va = load_f32x4(a, i);
        let vb = load_f32x4(b, i);
        acc += (va * vb).reduce_add();
        i += 4;
    }
    while i < len {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

pub fn compress_vector(v: &[f32]) -> Vec<u64> {
    let mut out = Vec::new();
    let mut cur = 0u64;
    let mut fill = 0;
    for &x in v {
        cur <<= 1;
        if x >= 0.0 {
            cur |= 1;
        }
        fill += 1;
        if fill == 64 {
            out.push(cur);
            cur = 0;
            fill = 0;
        }
    }
    if fill > 0 {
        cur <<= 64 - fill;
        out.push(cur);
    }
    out
}

pub fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

pub fn score(query: &[f32], vector: &[f32], bin: Option<&Vec<u64>>, dist: Distance) -> f32 {
    match dist {
        Distance::Euclidean | Distance::Hnsw => {
            clamp_0_1(1.0 / (1.0 + simd_euclidean_distance(query, vector)))
        }
        Distance::Cosine => clamp_0_1((simd_dot_product(query, vector) + 1.0) / 2.0),
        Distance::DotProduct => clamp_0_1(1.0 / (1.0 + f32::exp(-simd_dot_product(query, vector)))),
        Distance::Binary => {
            let qbits = compress_vector(query);
            let ebits = bin.expect("binary column missing");
            let frac = hamming_distance(&qbits, ebits) as f32 / query.len() as f32;
            1.0 - clamp_0_1(frac)
        }
    }
}
