//! fast k-NN (top-k) scan without Rayon
//! * Uses the collection’s HNSW index when present.
//! * Otherwise performs a SIMD-accelerated brute-force scan
//!   and keeps the best *k* hits in a min-heap.
use crate::distances::{
    clamp_0_1, compress_vector, hamming_distance, simd_dot_product, simd_euclidean_distance,
};
use rayon::prelude::*;

use crate::db::Collection;
#[cfg(any(target_feature = "avx2", target_feature = "avx512f"))]
use crate::simd_utils::load_f32x8;
use crate::simd_utils::normalize_vec;
use crate::types::Distance;

const PAR_THRESHOLD: usize = 2_000; // tune for your machine

/// Return the `k` most similar `(value, score)` pairs.
///
/// * Falls back to brute-force scan when HNSW is not available.
/// * SIMD kernels keep it fast up to a few million vectors.
pub fn similarity_search(
    coll: &Collection,
    query: &[f32],
    k: usize,
) -> Result<Vec<(String, f32)>, String> {
    if let Some(h) = coll.hnsw() {
        return h.search(query, k, coll.distance).map(|mut v| {
            v.truncate(k);
            v
        });
    }

    /*  brute-force */
    match coll.distance {
        Distance::Binary => brute_binary(coll, query, k),
        Distance::Euclidean | Distance::Cosine | Distance::DotProduct => {
            brute_l2_dot_cos(coll, query, k)
        }
        _ => Err("unsupported distance".into()),
    }
}

fn brute_binary(c: &Collection, q: &[f32], k: usize) -> Result<Vec<(String, f32)>, String> {
    let q_bits = compress_vector(q); // cached once
    let rows = c.row_count();

    let mut pairs: Vec<(usize, u32)> = if rows >= PAR_THRESHOLD {
        (0..rows)
            .into_par_iter() // Rayon
            .filter_map(|r| {
                c.compressed_by_row(r)
                    .map(|bits| (r, hamming_distance(&q_bits, bits)))
            })
            .collect()
    } else {
        let mut v = Vec::with_capacity(rows);
        for r in 0..rows {
            if let Some(bits) = c.compressed_by_row(r) {
                v.push((r, hamming_distance(&q_bits, bits)));
            }
        }
        v
    };

    // keep top-k by smallest Hamming distance, then convert to similarity in [0,1]
    pairs.sort_by_key(|&(_, d)| d);
    Ok(pairs
        .into_iter()
        .take(k)
        .filter_map(|(r, d)| {
            c.value_by_row(r).map(|v| {
                // fraction of differing bits
                let frac = d as f32 / q.len() as f32;
                // similarity = 1 - fraction, clamped to [0,1]
                let score = clamp_0_1(1.0 - frac);
                (v.clone(), score)
            })
        })
        .collect())
}

/*  SIMD L2 / cosine / dot */
fn brute_l2_dot_cos(c: &Collection, q: &[f32], k: usize) -> Result<Vec<(String, f32)>, String> {
    let rows = c.row_count();
    /*  normalize query for cosine once */
    let q_normed = normalize_vec(q);

    let mut pairs: Vec<(usize, f32)> = if rows >= PAR_THRESHOLD {
        (0..rows)
            .into_par_iter()
            .map(|r| {
                let vec = c.vector_slice(r);
                let score = match c.distance {
                    Distance::Euclidean => 1.0 / (1.0 + simd_euclidean_distance(vec, q)),
                    Distance::Cosine => {
                        let dp = simd_dot_product(vec, &q_normed);
                        (dp + 1.0) * 0.5
                    }
                    Distance::DotProduct => {
                        let dp = simd_dot_product(vec, q);
                        clamp_0_1(1.0 / (1.0 + (-dp).exp()))
                    }
                    _ => unreachable!(),
                };
                (r, score)
            })
            .collect()
    } else {
        let mut v = Vec::with_capacity(rows);
        for r in 0..rows {
            let vec = c.vector_slice(r);
            let s = match c.distance {
                Distance::Euclidean => 1.0 / (1.0 + simd_euclidean_distance(vec, q)),
                Distance::Cosine => {
                    let dp = simd_dot_product(vec, &q_normed);
                    (dp + 1.0) * 0.5
                }
                Distance::DotProduct => {
                    let dp = simd_dot_product(vec, q);
                    clamp_0_1(1.0 / (1.0 + (-dp).exp()))
                }
                _ => unreachable!(),
            };
            v.push((r, s));
        }
        v
    };

    /*  keep largest scores */
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    Ok(pairs
        .into_iter()
        .take(k)
        .filter_map(|(r, s)| c.value_by_row(r).map(|v| (v.clone(), s)))
        .collect())
}
