//! mmr.rs – algorithm-only module, no DB dependencies
//! ==================================================
//! Implements Maximal-Marginal-Relevance re-ranking over a set of
//! candidate (value, score) pairs given their embedding vectors.

use std::collections::HashMap;

use crate::distances::{simd_dot_product, simd_euclidean_distance};
use crate::types::Distance;

/// Re-rank `initial` according to MMR.
///
/// * `initial` – list of (value, similarity) pairs returned by a first-pass search.
/// * `vectors` – map value → embedding vector (same dimension for all).
/// * `dist` – distance metric used for the redundancy penalty.
/// * `alpha` – trade-off between query relevance (α) and diversity (1 − α).
/// * `final_k` – number of results you want back.
///
/// Returns a new list of (value, score) pairs, length ≤ `final_k`, ordered by MMR.
pub fn mmr_rerank(
    initial: &[(String, f32)],
    vectors: &HashMap<String, Vec<f32>>, // value → vec
    dist: Distance,
    alpha: f32,
    final_k: usize,
) -> Vec<(String, f32)> {
    let mut cand = initial.to_owned();
    let mut selected = Vec::<(String, f32)>::new();
    let mut sel_ids = Vec::<String>::new();

    while selected.len() < final_k && !cand.is_empty() {
        let mut best_idx = None;
        let mut best_mmr = f32::MIN;

        for (i, (cid, cscore_q)) in cand.iter().enumerate() {
            let mut max_sim = 0.0;
            if !sel_ids.is_empty() {
                let cvec = &vectors[cid];
                for sid in &sel_ids {
                    let svec = &vectors[sid];
                    let sim = match dist {
                        Distance::Euclidean | Distance::Hnsw | Distance::Binary => {
                            1.0 / (1.0 + simd_euclidean_distance(cvec, svec))
                        }
                        Distance::Cosine | Distance::DotProduct => simd_dot_product(cvec, svec),
                    };
                    if sim > max_sim {
                        max_sim = sim;
                    }
                }
            }
            let mmr = alpha * cscore_q - (1.0 - alpha) * max_sim;
            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            let chosen = cand.remove(idx);
            sel_ids.push(chosen.0.clone());
            selected.push(chosen);
        } else {
            break; //  shouldn’t happen
        }
    }
    selected
}
