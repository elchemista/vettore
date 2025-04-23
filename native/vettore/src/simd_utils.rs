use wide::f32x4;

/// Load four `f32`s starting at index *i* into a SIMD register.
#[inline]
pub fn load_f32x4(slice: &[f32], i: usize) -> f32x4 {
    f32x4::from([slice[i], slice[i + 1], slice[i + 2], slice[i + 3]])
}

/// Return a new `Vec<f32>` whose L2‑norm is ~1. Used for cosine similarity.
pub fn normalize_vec(v: &[f32]) -> Vec<f32> {
    let mut sum_sq = 0.0;
    let mut i = 0;
    while i + 4 <= v.len() {
        let vv = load_f32x4(v, i);
        sum_sq += (vv * vv).reduce_add();
        i += 4;
    }
    while i < v.len() {
        sum_sq += v[i] * v[i];
        i += 1;
    }
    let norm = sum_sq.sqrt();
    if norm > std::f32::EPSILON {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}
