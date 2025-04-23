use std::collections::HashMap;

use crate::distances::{simd_dot_product, simd_euclidean_distance};
use crate::types::Distance;

pub fn mmr_rerank_internal(
    initial_results: &[(String, f32)],
    all_embeddings: &HashMap<String, Vec<f32>>, // id → vector
    distance: Distance,
    alpha: f32,
    final_k: usize,
) -> Vec<(String, f32)> {
    let mut candidates = initial_results.to_vec();
    let mut selected: Vec<(String, f32)> = Vec::new();
    let mut selected_ids: Vec<String> = Vec::new();

    while selected.len() < final_k && !candidates.is_empty() {
        let mut best_idx = None;
        let mut best_score = f32::MIN;
        for (idx, (cand_id, cand_sim_q)) in candidates.iter().enumerate() {
            // similarity to the closest already‑selected document
            let mut max_sim_cand_sel = 0.0;
            if !selected_ids.is_empty() {
                let cand_vec = &all_embeddings[cand_id];
                for sel_id in &selected_ids {
                    let sel_vec = &all_embeddings[sel_id];
                    let s = match distance {
                        Distance::Euclidean | Distance::Hnsw => {
                            let d = simd_euclidean_distance(cand_vec, sel_vec);
                            1.0 / (1.0 + d)
                        }
                        Distance::Cosine | Distance::DotProduct => {
                            simd_dot_product(cand_vec, sel_vec)
                        }
                        Distance::Binary => {
                            let d = simd_euclidean_distance(cand_vec, sel_vec);
                            1.0 / (1.0 + d)
                        }
                    };
                    if s > max_sim_cand_sel {
                        max_sim_cand_sel = s;
                    }
                }
            }
            let mmr_score = alpha * cand_sim_q - (1.0 - alpha) * max_sim_cand_sel;
            if mmr_score > best_score {
                best_score = mmr_score;
                best_idx = Some(idx);
            }
        }
        if let Some(bi) = best_idx {
            let chosen = candidates.remove(bi);
            selected_ids.push(chosen.0.clone());
            selected.push(chosen);
        } else {
            break;
        }
    }
    selected
}
