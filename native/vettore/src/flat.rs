//! Native exact flat index resource.
//!
//! ETS remains the canonical record store. This resource mirrors only ids and
//! dense vectors so exact scans happen in one native call instead of one NIF
//! metric call per stored row.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::RwLock;

use crate::distances::Metric;

pub struct FlatIndex {
    metric: Metric,
    vectors: HashMap<String, Vec<f32>>,
    dimension: Option<usize>,
}

#[derive(Debug)]
struct FlatHit {
    id: String,
    raw: f32,
    rank: f32,
}

impl Eq for FlatHit {}

impl PartialEq for FlatHit {
    fn eq(&self, other: &Self) -> bool {
        self.rank.total_cmp(&other.rank) == Ordering::Equal && self.id == other.id
    }
}

impl Ord for FlatHit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank
            .total_cmp(&other.rank)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for FlatHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl FlatIndex {
    /// Creates an empty exact flat index for one metric.
    pub fn new(metric: Metric) -> Self {
        Self {
            metric,
            vectors: HashMap::new(),
            dimension: None,
        }
    }

    /// Inserts or replaces one vector by external id.
    pub fn insert(&mut self, id: String, vector: Vec<f32>) -> Result<(), String> {
        self.validate_vector(&vector)?;
        if self.dimension.is_none() {
            self.dimension = Some(vector.len());
        }
        self.vectors.insert(id, vector);
        Ok(())
    }

    /// Inserts or replaces a batch of vectors.
    pub fn insert_many(&mut self, vectors: Vec<(String, Vec<f32>)>) -> Result<(), String> {
        let expected = self
            .dimension
            .or_else(|| vectors.first().map(|(_, vector)| vector.len()));

        for (_, vector) in &vectors {
            validate_vector(vector, expected)?;
        }

        for (id, vector) in vectors {
            self.vectors.insert(id, vector);
        }
        if self.dimension.is_none() {
            self.dimension = expected;
        }
        Ok(())
    }

    /// Deletes one vector by external id.
    pub fn delete(&mut self, id: &str) {
        self.vectors.remove(id);
        if self.vectors.is_empty() {
            self.dimension = None;
        }
    }

    /// Searches every stored vector and returns ids with raw metric values.
    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<(String, f32)>, String> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        validate_vector(query, self.dimension)?;

        let mut hits = BinaryHeap::with_capacity(usize::min(limit, self.vectors.len()));
        for (id, vector) in &self.vectors {
            let raw = crate::distances::compute(self.metric, query, vector)?;
            let hit = FlatHit {
                id: id.clone(),
                raw,
                rank: crate::distances::rank_value(self.metric, raw),
            };

            if hits.len() < limit {
                hits.push(hit);
            } else if hits.peek().is_some_and(|worst| hit < *worst) {
                hits.pop();
                hits.push(hit);
            }
        }

        let mut hits = hits.into_vec();
        hits.sort();

        Ok(hits.into_iter().map(|hit| (hit.id, hit.raw)).collect())
    }

    fn validate_vector(&self, vector: &[f32]) -> Result<(), String> {
        validate_vector(vector, self.dimension)
    }
}

pub struct FlatResource(pub RwLock<FlatIndex>);

fn validate_vector(vector: &[f32], dimension: Option<usize>) -> Result<(), String> {
    if vector.is_empty() {
        return Err("vector must not be empty".to_string());
    }
    if dimension.is_some_and(|expected| vector.len() != expected) {
        return Err("dimension mismatch".to_string());
    }
    crate::distances::validate_finite_vector(vector)
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

    #[test]
    fn inserts_replaces_deletes_and_returns_stable_top_k() {
        let mut index = FlatIndex::new(Metric::L2);
        index.insert("b".into(), vec![2.0]).unwrap();
        index.insert("a".into(), vec![0.0]).unwrap();
        index.insert("c".into(), vec![2.0]).unwrap();

        assert_eq!(
            index.search(&[1.0], 2).unwrap(),
            vec![("a".into(), 1.0), ("b".into(), 1.0)]
        );

        index.insert("a".into(), vec![10.0]).unwrap();
        assert_eq!(index.search(&[2.0], 1).unwrap()[0].0, "b");
        index.delete("b");
        assert_eq!(index.search(&[2.0], 1).unwrap()[0].0, "c");
    }

    #[test]
    fn batch_validation_is_atomic() {
        let mut index = FlatIndex::new(Metric::InnerProduct);
        index.insert("existing".into(), vec![1.0, 0.0]).unwrap();

        assert!(index
            .insert_many(vec![
                ("valid".into(), vec![0.0, 1.0]),
                ("invalid".into(), vec![1.0]),
            ])
            .is_err());
        assert_eq!(index.vectors.len(), 1);
        assert!(!index.vectors.contains_key("valid"));
        assert!(index.insert("nan".into(), vec![f32::NAN, 0.0]).is_err());
    }

    #[test]
    fn rejects_invalid_queries_and_handles_empty_limits() {
        let mut index = FlatIndex::new(Metric::Cosine);
        assert!(index.insert("empty".into(), vec![]).is_err());
        index.insert("a".into(), vec![1.0, 0.0]).unwrap();
        assert!(index.search(&[1.0], 1).is_err());
        assert!(index.search(&[f32::INFINITY, 0.0], 1).is_err());
        assert_eq!(index.search(&[1.0, 0.0], 0).unwrap(), Vec::new());
    }

    #[test]
    fn exact_heap_matches_a_full_sort_for_all_metrics() {
        let vectors: Vec<_> = (0..51)
            .map(|index| {
                (
                    format!("v-{index:02}"),
                    vec![
                        (index as f32 - 25.0) / 9.0,
                        ((index * 13 % 31) as f32 - 15.0) / 7.0,
                        if index % 2 == 0 { 0.0 } else { 1.0 },
                    ],
                )
            })
            .collect();
        let query = [0.5, -1.25, 1.0];

        for metric in all_metrics() {
            let mut index = FlatIndex::new(metric);
            index.insert_many(vectors.clone()).unwrap();

            let mut expected: Vec<_> = vectors
                .iter()
                .map(|(id, vector)| {
                    (
                        id.clone(),
                        crate::distances::compute(metric, &query, vector).unwrap(),
                    )
                })
                .collect();
            expected.sort_by(|left, right| {
                crate::distances::rank_value(metric, left.1)
                    .total_cmp(&crate::distances::rank_value(metric, right.1))
                    .then_with(|| left.0.cmp(&right.0))
            });

            for limit in [1usize, 7, 51, 100] {
                let mut limited = expected.clone();
                limited.truncate(limit);
                assert_eq!(index.search(&query, limit).unwrap(), limited);
            }
        }
    }

    #[test]
    fn empty_batches_unknown_deletes_and_dimension_resets_are_total() {
        let mut index = FlatIndex::new(Metric::L2);
        assert_eq!(index.insert_many(vec![]), Ok(()));
        assert_eq!(index.search(&[1.0], 10), Ok(vec![]));
        index.delete("missing");

        index.insert("one".into(), vec![1.0]).unwrap();
        index.delete("missing");
        assert_eq!(index.dimension, Some(1));
        index.delete("one");
        assert_eq!(index.dimension, None);

        index.insert("two".into(), vec![1.0, 2.0]).unwrap();
        assert_eq!(index.dimension, Some(2));
        assert_eq!(index.search(&[1.0, 2.0], usize::MAX).unwrap().len(), 1);
    }

    #[test]
    fn duplicate_batch_ids_replace_deterministically_and_large_l2_stays_finite() {
        let mut index = FlatIndex::new(Metric::L2);
        index
            .insert_many(vec![
                ("same".into(), vec![0.0]),
                ("same".into(), vec![1.0e20]),
            ])
            .unwrap();
        assert_eq!(index.vectors.len(), 1);
        assert_eq!(index.search(&[0.0], 1).unwrap()[0].0, "same");
        assert!(index.search(&[0.0], 1).unwrap()[0].1.is_finite());
    }

    #[test]
    fn heap_hit_equality_and_partial_order_include_the_external_id() {
        let first = FlatHit {
            id: "a".into(),
            raw: 1.0,
            rank: 1.0,
        };
        let equal = FlatHit {
            id: "a".into(),
            raw: 99.0,
            rank: 1.0,
        };
        let other_id = FlatHit {
            id: "b".into(),
            raw: 1.0,
            rank: 1.0,
        };
        assert_eq!(first, equal);
        assert_ne!(first, other_id);
        assert_eq!(first.partial_cmp(&other_id), Some(Ordering::Less));
    }
}
