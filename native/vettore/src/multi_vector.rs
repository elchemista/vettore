//! Native MaxSim/ColBERT scoring and batched top-k selection.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::distances::Metric;

#[derive(Debug)]
struct Hit {
    id: String,
    score: f32,
}

impl Eq for Hit {}

impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score) == Ordering::Equal && self.id == other.id
    }
}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse score ordering so BinaryHeap::peek exposes the worst retained
        // result. For equal scores, lexicographically larger ids are worse.
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Computes one MaxSim score: each query vector contributes its best document match.
pub fn score(
    query_vectors: &[Vec<f32>],
    document_vectors: &[Vec<f32>],
    metric: Metric,
) -> Result<f32, String> {
    if query_vectors.is_empty() {
        validate_standalone_vectors(document_vectors)?;
        return Ok(0.0);
    }

    let dimension = query_vectors[0].len();
    if dimension == 0 {
        return Err("vectors must not be empty".to_string());
    }
    validate_vectors(query_vectors, dimension)?;

    if document_vectors.is_empty() {
        return Ok(0.0);
    }

    validate_vectors(document_vectors, dimension)?;

    score_validated(query_vectors, document_vectors, metric)
}

fn score_validated(
    query_vectors: &[Vec<f32>],
    document_vectors: &[Vec<f32>],
    metric: Metric,
) -> Result<f32, String> {
    let mut total = 0.0;
    for query in query_vectors {
        let mut best = f32::NEG_INFINITY;
        for document in document_vectors {
            let raw = if metric == Metric::Cosine {
                crate::distances::cosine(query, document)?
            } else {
                crate::distances::compute(metric, query, document)?
            };
            best = best.max(crate::distances::similarity_value(metric, raw));
        }
        total += best;
        if !total.is_finite() {
            return Err("score overflow".to_string());
        }
    }
    Ok(total)
}

/// Scores a document batch and retains the highest-scoring records.
pub fn top_k(
    documents: Vec<(String, Vec<Vec<f32>>)>,
    query_vectors: &[Vec<f32>],
    metric: Metric,
    limit: usize,
) -> Result<Vec<(String, f32)>, String> {
    validate_standalone_vectors(query_vectors)?;
    let query_dimension = query_vectors.first().map(Vec::len);

    let mut heap = BinaryHeap::with_capacity(usize::min(limit, documents.len()));
    for (id, vectors) in documents {
        let score = match query_dimension {
            None => {
                validate_standalone_vectors(&vectors)?;
                0.0
            }
            Some(_dimension) if vectors.is_empty() => 0.0,
            Some(dimension) => {
                validate_vectors(&vectors, dimension)?;
                score_validated(query_vectors, &vectors, metric)?
            }
        };
        let hit = Hit { id, score };
        if limit == 0 {
            continue;
        }
        if heap.len() < limit {
            heap.push(hit);
        } else if heap.peek().is_some_and(|worst| hit < *worst) {
            heap.pop();
            heap.push(hit);
        }
    }

    let mut hits = heap.into_vec();
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.id.cmp(&right.id))
    });
    Ok(hits.into_iter().map(|hit| (hit.id, hit.score)).collect())
}

fn validate_standalone_vectors(vectors: &[Vec<f32>]) -> Result<(), String> {
    let Some(first) = vectors.first() else {
        return Ok(());
    };
    if first.is_empty() {
        return Err("vectors must not be empty".to_string());
    }
    validate_vectors(vectors, first.len())
}

fn validate_vectors(vectors: &[Vec<f32>], dimension: usize) -> Result<(), String> {
    for vector in vectors {
        if vector.len() != dimension {
            return Err("dimension mismatch".to_string());
        }
        crate::distances::validate_finite_vector(vector)?;
    }
    Ok(())
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

    fn score_oracle(query: &[Vec<f32>], document: &[Vec<f32>], metric: Metric) -> f32 {
        query
            .iter()
            .map(|query_vector| {
                document
                    .iter()
                    .map(|document_vector| {
                        let raw = if metric == Metric::Cosine {
                            crate::distances::cosine(query_vector, document_vector).unwrap()
                        } else {
                            crate::distances::compute(metric, query_vector, document_vector)
                                .unwrap()
                        };
                        crate::distances::similarity_value(metric, raw)
                    })
                    .max_by(f32::total_cmp)
                    .unwrap()
            })
            .sum()
    }

    #[test]
    fn scores_similarity_and_distance_metrics() {
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let document = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(score(&query, &document, Metric::InnerProduct), Ok(2.0));
        assert_eq!(
            score(&query, &document, Metric::NegativeInnerProduct),
            Ok(2.0)
        );
        assert_eq!(score(&query, &document, Metric::Cosine), Ok(2.0));
        assert_eq!(score(&query, &document, Metric::L2), Ok(2.0));
        assert_eq!(score(&[], &document, Metric::L2), Ok(0.0));
        assert_eq!(score(&query, &[], Metric::L2), Ok(0.0));
    }

    #[test]
    fn top_k_is_stable_and_rejects_bad_shapes() {
        let query = vec![vec![1.0, 0.0]];
        let documents = vec![
            ("b".into(), vec![vec![1.0, 0.0]]),
            ("a".into(), vec![vec![1.0, 0.0]]),
            ("c".into(), vec![vec![-1.0, 0.0]]),
        ];
        assert_eq!(
            top_k(documents, &query, Metric::InnerProduct, 2).unwrap(),
            vec![("a".into(), 1.0), ("b".into(), 1.0)]
        );
        assert!(score(&query, &[vec![1.0]], Metric::InnerProduct).is_err());
        assert!(score(&[vec![f32::NAN, 0.0]], &query, Metric::InnerProduct).is_err());
    }

    #[test]
    fn every_metric_matches_an_independent_maxsim_oracle() {
        let query = vec![vec![1.0, -0.5, 0.0], vec![0.0, 1.0, 1.0]];
        let document = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, -1.0],
            vec![-1.0, 0.5, 1.0],
        ];

        for metric in all_metrics() {
            let actual = score(&query, &document, metric).unwrap();
            let expected = score_oracle(&query, &document, metric);
            assert!((actual - expected).abs() <= 1.0e-6, "metric {metric:?}");
        }
    }

    #[test]
    fn validates_the_nonempty_side_even_when_the_other_side_is_empty() {
        assert!(score(&[], &[vec![]], Metric::L2).is_err());
        assert!(score(&[], &[vec![f32::NAN]], Metric::L2).is_err());
        assert!(score(&[vec![]], &[], Metric::L2).is_err());
        assert!(score(&[vec![f32::INFINITY]], &[], Metric::L2).is_err());
        assert!(top_k(vec![], &[vec![]], Metric::L2, 1).is_err());
        assert!(top_k(vec![], &[vec![f32::NAN]], Metric::L2, 1).is_err());
    }

    #[test]
    fn detects_total_score_overflow_after_finite_pair_scores() {
        let query = vec![vec![1.0e19]; 4];
        let document = vec![vec![1.0e19]];
        assert_eq!(
            score(&query, &document, Metric::InnerProduct),
            Err("score overflow".into())
        );
    }

    #[test]
    fn batched_top_k_matches_full_sort_for_all_metrics_and_limits() {
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let documents: Vec<_> = (0..25)
            .map(|index| {
                (
                    format!("doc-{index:02}"),
                    vec![
                        vec![(index as f32 - 12.0) / 5.0, 1.0],
                        vec![0.0, ((index * 7 % 11) as f32 - 5.0) / 3.0],
                    ],
                )
            })
            .collect();

        for metric in all_metrics() {
            let mut expected: Vec<_> = documents
                .iter()
                .map(|(id, vectors)| (id.clone(), score(&query, vectors, metric).unwrap()))
                .collect();
            expected.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });

            for limit in [0usize, 1, 7, 25, 100] {
                let mut limited = expected.clone();
                limited.truncate(limit);
                assert_eq!(
                    top_k(documents.clone(), &query, metric, limit).unwrap(),
                    limited
                );
            }
        }
    }

    #[test]
    fn empty_queries_still_validate_documents_and_order_zero_score_ties() {
        let documents = vec![("b".into(), vec![vec![1.0]]), ("a".into(), vec![vec![2.0]])];
        assert_eq!(
            top_k(documents, &[], Metric::L2, 10),
            Ok(vec![("a".into(), 0.0), ("b".into(), 0.0)])
        );
    }

    #[test]
    fn heap_hit_equality_and_partial_order_include_the_external_id() {
        let first = Hit {
            id: "a".into(),
            score: 1.0,
        };
        let equal = Hit {
            id: "a".into(),
            score: 1.0,
        };
        let other_id = Hit {
            id: "b".into(),
            score: 1.0,
        };
        assert_eq!(first, equal);
        assert_ne!(first, other_id);
        assert_eq!(first.partial_cmp(&other_id), Some(Ordering::Less));
    }
}
