use wide::f32x4;
#[cfg(any(target_feature = "avx2", target_feature = "avx512f"))]
// import only when AVX is available
use wide::f32x8;

#[inline]
#[allow(clippy::needless_range_loop)]
pub fn load_f32x4(slice: &[f32], i: usize) -> f32x4 {
    f32x4::from([slice[i], slice[i + 1], slice[i + 2], slice[i + 3]])
}

#[cfg(any(target_feature = "avx2", target_feature = "avx512f"))]
#[inline]
pub fn load_f32x8(slice: &[f32], i: usize) -> f32x8 {
    f32x8::from([
        slice[i],
        slice[i + 1],
        slice[i + 2],
        slice[i + 3],
        slice[i + 4],
        slice[i + 5],
        slice[i + 6],
        slice[i + 7],
    ])
}

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
