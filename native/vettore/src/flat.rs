//! Native exact flat index resource.
//!
//! ETS remains the canonical record store. This resource mirrors only ids and
//! dense vectors so exact scans happen in one native call instead of one NIF
//! metric call per stored row.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::RwLock;

use crate::distances::Metric;

pub struct FlatIndex {
    metric: Metric,
    vectors: HashMap<String, Vec<f32>>,
}

impl FlatIndex {
    /// Creates an empty exact flat index for one metric.
    pub fn new(metric: Metric) -> Self {
        Self {
            metric,
            vectors: HashMap::new(),
        }
    }

    /// Inserts or replaces one vector by external id.
    pub fn insert(&mut self, id: String, vector: Vec<f32>) {
        self.vectors.insert(id, vector);
    }

    /// Inserts or replaces a batch of vectors.
    pub fn insert_many(&mut self, vectors: Vec<(String, Vec<f32>)>) {
        for (id, vector) in vectors {
            self.insert(id, vector);
        }
    }

    /// Deletes one vector by external id.
    pub fn delete(&mut self, id: &str) {
        self.vectors.remove(id);
    }

    /// Searches every stored vector and returns ids with raw metric values.
    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<(String, f32)>, String> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let mut hits = Vec::with_capacity(self.vectors.len());
        for (id, vector) in &self.vectors {
            let rank = crate::distances::rank_distance(self.metric, query, vector)?;
            let raw = crate::distances::compute(self.metric, query, vector)?;
            hits.push((id.clone(), raw, rank));
        }

        hits.sort_by(|left, right| {
            left.2
                .partial_cmp(&right.2)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });

        Ok(hits
            .into_iter()
            .take(limit)
            .map(|(id, raw, _rank)| (id, raw))
            .collect())
    }
}

pub struct FlatResource(pub RwLock<FlatIndex>);
