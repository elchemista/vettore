//! similarity.rs – fast k-NN (top-k) scan without Rayon
//! ====================================================
//! * Uses the collection’s HNSW index when present.
//! * Otherwise performs a SIMD-accelerated brute-force scan
//!   and keeps the best *k* hits in a min-heap.
//
//! No external dependencies.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::db::Collection;
use crate::simd_utils::load_f32x4;
#[cfg(any(target_feature = "avx2", target_feature = "avx512f"))]
use crate::simd_utils::load_f32x8;
use crate::types::Distance;

#[derive(Clone, Copy, Debug)]
struct F32Ord(f32);

impl PartialEq for F32Ord {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for F32Ord {}
impl PartialOrd for F32Ord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Ord for F32Ord {
    fn cmp(&self, other: &Self) -> Ordering {
        // unwrap safe: scores are finite, no NaN
        self.partial_cmp(other).unwrap()
    }
}

#[inline(always)]
fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0;
    let mut i = 0;

    #[cfg(any(target_feature = "avx2", target_feature = "avx512f"))]
    {
        while i + 8 <= a.len() {
            acc += (load_f32x8(a, i) * load_f32x8(b, i)).reduce_add();
            i += 8;
        }
    }
    while i + 4 <= a.len() {
        acc += (load_f32x4(a, i) * load_f32x4(b, i)).reduce_add();
        i += 4;
    }
    while i < a.len() {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

/// Return the `k` most similar `(value, score)` pairs.
///
/// * Falls back to brute-force scan when HNSW is not available.
/// * SIMD kernels keep it fast up to a few million vectors.
pub fn similarity_search(
    col: &Collection,
    query: &[f32],
    k: usize,
) -> Result<Vec<(String, f32)>, String> {
    if query.len() != col.dimension {
        return Err("query dim mismatch".into());
    }

    /* ---- HNSW fast-path ----------------------------------------- */
    if let Some(h) = col.hnsw() {
        return h.search(query, k, col.distance);
    }

    /* ---- brute-force scan with min-heap ------------------------- */
    let mut heap: BinaryHeap<(Reverse<F32Ord>, String)> = BinaryHeap::with_capacity(k + 1);

    let q_norm2 = dot_simd(query, query);
    let rows = col.row_count();

    for row in 0..rows {
        let value = match col.value_by_row(row) {
            Some(v) => v,
            None => continue, // deleted row
        };

        let vec_slice = col.vector_slice(row);
        let mut sim = dot_simd(query, vec_slice);

        match col.distance {
            Distance::Cosine => {
                sim = (sim + 1.0) * 0.5; // cosine ∈ [-1,1]  →  [0,1]
            }
            Distance::DotProduct => { /* sim already ∝ similarity */ }
            Distance::Euclidean | Distance::Binary | Distance::Hnsw => {
                let v_norm2 = dot_simd(vec_slice, vec_slice);
                let dist = (q_norm2 + v_norm2 - 2.0 * sim).sqrt();
                sim = 1.0 / (1.0 + dist); // convert to similarity
            }
        }

        // push & keep only the best k (min-heap)
        heap.push((Reverse(F32Ord(sim)), value.clone()));
        if heap.len() > k {
            heap.pop();
        }
    }

    /* ---- heap → sorted Vec -------------------------------------- */
    let mut out: Vec<(String, f32)> = heap
        .into_iter()
        .map(|(Reverse(F32Ord(s)), v)| (v, s))
        .collect();

    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    Ok(out)
}
