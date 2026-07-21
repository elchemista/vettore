//! Batched top-k helpers used by adaptive Elixir search paths.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::distances::Metric;

#[derive(Debug)]
struct Hit {
    id: String,
    raw: f32,
    rank: f32,
}

impl Eq for Hit {}

impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.rank.total_cmp(&other.rank) == Ordering::Equal && self.id == other.id
    }
}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank
            .total_cmp(&other.rank)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Scores a vector batch once and retains only its best `limit` results.
pub fn vector_top_k(
    vectors: Vec<(String, Vec<f32>)>,
    query: &[f32],
    metric: Metric,
    dimensions: usize,
    limit: usize,
) -> Result<Vec<(String, f32)>, String> {
    if dimensions == 0 || dimensions > query.len() {
        return Err("invalid prefix dimensions".to_string());
    }
    crate::distances::validate_finite_vector(&query[..dimensions])?;

    let mut heap = BinaryHeap::with_capacity(usize::min(limit, vectors.len()));
    for (id, vector) in vectors {
        if dimensions > vector.len() {
            return Err("dimension mismatch".to_string());
        }
        crate::distances::validate_finite_vector(&vector[..dimensions])?;
        let raw = if metric == Metric::Cosine {
            crate::distances::cosine(&query[..dimensions], &vector[..dimensions])?
        } else {
            crate::distances::compute(metric, &query[..dimensions], &vector[..dimensions])?
        };
        push_top_k(
            &mut heap,
            Hit {
                id,
                raw,
                rank: crate::distances::rank_value(metric, raw),
            },
            limit,
        );
    }

    sorted_hits(heap)
}

/// Scores packed sign vectors by Hamming distance in one native call.
pub fn binary_top_k(
    vectors: Vec<(String, Vec<u64>)>,
    query: &[u64],
    dimensions: usize,
    limit: usize,
) -> Result<Vec<(String, f32)>, String> {
    // Validate the query even for an empty candidate batch. Otherwise malformed
    // dimensions could appear valid merely because there was nothing to score.
    crate::distances::packed_hamming(query, query, dimensions)?;

    let mut heap = BinaryHeap::with_capacity(usize::min(limit, vectors.len()));
    for (id, vector) in vectors {
        let raw = crate::distances::packed_hamming(query, &vector, dimensions)?;
        push_top_k(&mut heap, Hit { id, raw, rank: raw }, limit);
    }
    sorted_hits(heap)
}

fn push_top_k(heap: &mut BinaryHeap<Hit>, hit: Hit, limit: usize) {
    if limit == 0 {
        return;
    }
    if heap.len() < limit {
        heap.push(hit);
    } else if heap.peek().is_some_and(|worst| hit < *worst) {
        heap.pop();
        heap.push(hit);
    }
}

fn sorted_hits(heap: BinaryHeap<Hit>) -> Result<Vec<(String, f32)>, String> {
    let mut hits = heap.into_vec();
    hits.sort();
    Ok(hits.into_iter().map(|hit| (hit.id, hit.raw)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_metrics() -> [Metric; 9] {
        [
            Metric::L2,
            Metric::L2Squared,
            Metric::Cosine,
            Metric::InnerProduct,
            Metric::NegativeInnerProduct,
            Metric::Manhattan,
            Metric::Chebyshev,
            Metric::Hamming,
            Metric::Jaccard,
        ]
    }

    fn exact_oracle(
        vectors: &[(String, Vec<f32>)],
        query: &[f32],
        metric: Metric,
        dimensions: usize,
        limit: usize,
    ) -> Vec<(String, f32)> {
        let mut hits: Vec<_> = vectors
            .iter()
            .map(|(id, vector)| {
                let raw = if metric == Metric::Cosine {
                    crate::distances::cosine(&query[..dimensions], &vector[..dimensions]).unwrap()
                } else {
                    crate::distances::compute(metric, &query[..dimensions], &vector[..dimensions])
                        .unwrap()
                };
                (id.clone(), raw)
            })
            .collect();
        hits.sort_by(|left, right| {
            crate::distances::rank_value(metric, left.1)
                .total_cmp(&crate::distances::rank_value(metric, right.1))
                .then_with(|| left.0.cmp(&right.0))
        });
        hits.truncate(limit);
        hits
    }

    #[test]
    fn vector_top_k_handles_prefixes_similarity_and_ties() {
        let vectors = vec![
            ("b".into(), vec![1.0, 10.0]),
            ("a".into(), vec![1.0, -10.0]),
            ("c".into(), vec![-1.0, 0.0]),
        ];
        assert_eq!(
            vector_top_k(vectors.clone(), &[1.0, 0.0], Metric::L2, 1, 2).unwrap(),
            vec![("a".into(), 0.0), ("b".into(), 0.0)]
        );
        assert_eq!(
            vector_top_k(vectors, &[1.0, 1.0], Metric::InnerProduct, 2, 1).unwrap()[0].0,
            "b"
        );
    }

    #[test]
    fn vector_top_k_rejects_bad_dimensions_and_values() {
        assert!(vector_top_k(vec![], &[1.0], Metric::L2, 0, 1).is_err());
        assert!(
            vector_top_k(vec![("a".into(), vec![1.0])], &[1.0, 2.0], Metric::L2, 2, 1).is_err()
        );
        assert!(
            vector_top_k(vec![("a".into(), vec![f32::NAN])], &[1.0], Metric::L2, 1, 1).is_err()
        );
    }

    #[test]
    fn binary_top_k_masks_padding_and_orders_ids() {
        let query = crate::distances::compress_sign_bits(&[1.0, -1.0, 1.0]);
        let vectors = vec![
            (
                "b".into(),
                crate::distances::compress_sign_bits(&[1.0, 1.0, 1.0]),
            ),
            (
                "a".into(),
                crate::distances::compress_sign_bits(&[1.0, -1.0, 1.0]),
            ),
        ];
        assert_eq!(
            binary_top_k(vectors, &query, 3, 2).unwrap(),
            vec![("a".into(), 0.0), ("b".into(), 1.0)]
        );
    }

    #[test]
    fn vector_top_k_matches_full_sort_for_every_metric_and_limit() {
        let vectors: Vec<_> = (0..37)
            .map(|index| {
                (
                    format!("id-{index:02}"),
                    vec![
                        (index as f32 - 18.0) / 7.0,
                        ((index * 11 % 17) as f32 - 8.0) / 5.0,
                        ((index * 7 % 13) as f32 - 6.0) / 3.0,
                        if index % 3 == 0 { 0.0 } else { 1.0 },
                    ],
                )
            })
            .collect();
        let query = [0.25, -0.75, 1.5, 0.0];

        for metric in all_metrics() {
            for dimensions in [1usize, 3, 4] {
                for limit in [0usize, 1, 5, 37, 100] {
                    assert_eq!(
                        vector_top_k(vectors.clone(), &query, metric, dimensions, limit).unwrap(),
                        exact_oracle(&vectors, &query, metric, dimensions, limit)
                    );
                }
            }
        }
    }

    #[test]
    fn vector_top_k_validates_queries_and_only_reads_the_requested_prefix() {
        assert!(vector_top_k(vec![], &[f32::NAN], Metric::L2, 1, 1).is_err());
        assert!(vector_top_k(vec![], &[1.0], Metric::L2, 2, 1).is_err());

        let vectors = vec![("a".into(), vec![1.0, f32::NAN])];
        assert_eq!(
            vector_top_k(vectors, &[1.0, f32::NAN], Metric::L2, 1, 1),
            Ok(vec![("a".into(), 0.0)])
        );
    }

    #[test]
    fn binary_top_k_validates_empty_batches_limits_and_word_boundaries() {
        assert!(binary_top_k(vec![], &[], 0, 1).is_err());
        assert!(binary_top_k(vec![], &[], 1, 1).is_err());
        assert_eq!(binary_top_k(vec![], &[0], 1, 1), Ok(vec![]));

        let query = vec![u64::MAX, 1];
        let vectors = vec![("same".into(), query.clone()), ("far".into(), vec![0, 0])];
        assert_eq!(binary_top_k(vectors.clone(), &query, 65, 0), Ok(vec![]));
        assert_eq!(
            binary_top_k(vectors.clone(), &query, 65, 10).unwrap(),
            vec![("same".into(), 0.0), ("far".into(), 65.0)]
        );
        assert!(binary_top_k(vec![("bad".into(), vec![0])], &query, 65, 1).is_err());
    }

    #[test]
    fn stable_ties_do_not_depend_on_candidate_order() {
        let forward = vec![
            ("c".into(), vec![1.0]),
            ("a".into(), vec![1.0]),
            ("b".into(), vec![1.0]),
        ];
        let mut reverse = forward.clone();
        reverse.reverse();

        let expected = vec![("a".into(), 0.0), ("b".into(), 0.0)];
        assert_eq!(
            vector_top_k(forward, &[1.0], Metric::L2, 1, 2).unwrap(),
            expected
        );
        assert_eq!(
            vector_top_k(reverse, &[1.0], Metric::L2, 1, 2).unwrap(),
            expected
        );
    }

    #[test]
    fn heap_hit_equality_and_partial_order_include_the_external_id() {
        let first = Hit {
            id: "a".into(),
            raw: 1.0,
            rank: 1.0,
        };
        let equal = Hit {
            id: "a".into(),
            raw: 99.0,
            rank: 1.0,
        };
        let other_id = Hit {
            id: "b".into(),
            raw: 1.0,
            rank: 1.0,
        };
        assert_eq!(first, equal);
        assert_ne!(first, other_id);
        assert_eq!(first.partial_cmp(&other_id), Some(Ordering::Less));
    }
}
