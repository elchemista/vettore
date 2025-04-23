use std::collections::HashMap;

use crate::distances::{simd_dot_product, simd_euclidean_distance};
use crate::types::Distance;

pub fn mmr_rerank_internal(
    initial: &[(String, f32)],
    vectors: &HashMap<String, Vec<f32>>, // id -> vec
    dist: Distance,
    alpha: f32,
    final_k: usize,
) -> Vec<(String, f32)> {
    let mut cand = initial.to_owned();
    let mut sel = Vec::<(String, f32)>::new();
    let mut sel_ids = Vec::<String>::new();
    while sel.len() < final_k && !cand.is_empty() {
        let mut best_idx = None;
        let mut best = f32::MIN;
        for (i, (cid, cscore_q)) in cand.iter().enumerate() {
            let mut max_sim = 0.0;
            if !sel_ids.is_empty() {
                let cvec = &vectors[cid];
                for sid in &sel_ids {
                    let svec = &vectors[sid];
                    let sim = match dist {
                        Distance::Euclidean | Distance::Hnsw => {
                            1.0 / (1.0 + simd_euclidean_distance(cvec, svec))
                        }
                        Distance::Cosine | Distance::DotProduct => simd_dot_product(cvec, svec),
                        Distance::Binary => 1.0 / (1.0 + simd_euclidean_distance(cvec, svec)),
                    };
                    if sim > max_sim {
                        max_sim = sim;
                    }
                }
            }
            let mmr = alpha * cscore_q - (1.0 - alpha) * max_sim;
            if mmr > best {
                best = mmr;
                best_idx = Some(i);
            }
        }
        if let Some(idx) = best_idx {
            let ch = cand.remove(idx);
            sel_ids.push(ch.0.clone());
            sel.push(ch);
        } else {
            break;
        }
    }
    sel
}
