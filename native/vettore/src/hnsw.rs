//! Native HNSW index resource.
//!
//! This index stores only the ANN graph, external ids, and normalized vectors
//! needed for search. It deliberately does not own Vettore records or metadata;
//! ETS remains the canonical store. The graph uses Vettore's native distance
//! kernels for rank comparisons.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::RwLock;

use crate::distances::Metric;

#[derive(Clone, Copy)]
pub struct HnswParams {
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub max_level: usize,
}

impl HnswParams {
    /// Validates HNSW graph/search parameters.
    pub fn validate(self) -> Result<Self, String> {
        if self.m == 0 {
            return Err("m must be positive".to_string());
        }
        if self.m0 == 0 {
            return Err("m0 must be positive".to_string());
        }
        if self.ef_construction < self.m {
            return Err("ef_construction must be >= m".to_string());
        }
        if self.ef_search == 0 {
            return Err("ef_search must be positive".to_string());
        }
        if self.max_level == 0 {
            return Err("max_level must be positive".to_string());
        }

        Ok(self)
    }
}

#[derive(Clone)]
struct ScoredNode {
    id: usize,
    dist: f32,
}

#[derive(Clone)]
struct ClosestFirst(ScoredNode);

#[derive(Clone)]
struct WorstFirst(ScoredNode);

impl Eq for ClosestFirst {}

impl PartialEq for ClosestFirst {
    /// Compares candidate heap entries by distance equality.
    fn eq(&self, other: &Self) -> bool {
        self.0.dist == other.0.dist
    }
}

impl Ord for ClosestFirst {
    /// Reverses ordering so the binary heap pops the closest candidate first.
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.dist.total_cmp(&self.0.dist)
    }
}

impl PartialOrd for ClosestFirst {
    /// Delegates partial ordering to the total distance order.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for WorstFirst {}

impl PartialEq for WorstFirst {
    /// Compares bounded-result heap entries by distance equality.
    fn eq(&self, other: &Self) -> bool {
        self.0.dist == other.0.dist
    }
}

impl Ord for WorstFirst {
    /// Orders so the binary heap exposes the worst retained result first.
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.dist.total_cmp(&other.0.dist)
    }
}

impl PartialOrd for WorstFirst {
    /// Delegates partial ordering to the total distance order.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone)]
struct Node {
    external_id: String,
    vector: Vec<f32>,
    connections: Vec<Vec<usize>>,
    layer: usize,
}

pub struct HnswIndex {
    metric: Metric,
    params: HnswParams,
    nodes: HashMap<usize, Node>,
    external_to_internal: HashMap<String, usize>,
    entry: Option<usize>,
    next: usize,
}

impl HnswIndex {
    /// Creates an empty HNSW graph for one ranking metric.
    pub fn new(metric: Metric, params: HnswParams) -> Result<Self, String> {
        let params = params.validate()?;

        Ok(Self {
            metric,
            params,
            nodes: HashMap::new(),
            external_to_internal: HashMap::new(),
            entry: None,
            next: 0,
        })
    }

    /// Inserts or replaces one external id in the graph.
    pub fn insert(&mut self, external_id: String, vector: Vec<f32>) -> Result<(), String> {
        if self.external_to_internal.contains_key(&external_id) {
            self.delete(&external_id);
        }

        let internal_id = self.next;
        self.next += 1;
        let node_level = self.level_for(&external_id);

        if self.nodes.is_empty() {
            self.nodes.insert(
                internal_id,
                Node {
                    external_id: external_id.clone(),
                    vector,
                    connections: vec![Vec::new(); node_level + 1],
                    layer: node_level,
                },
            );
            self.external_to_internal.insert(external_id, internal_id);
            self.entry = Some(internal_id);
            return Ok(());
        }

        let mut entry = self.entry.ok_or_else(|| "missing hnsw entry".to_string())?;
        let mut entry_dist = self.rank_distance(&self.nodes[&entry].vector, &vector)?;
        let top_layer = self.nodes[&entry].layer;

        for layer in (node_level + 1..=top_layer).rev() {
            let (best_id, best_dist) = self.greedy_closest(entry, &vector, layer)?;
            entry = best_id;
            entry_dist = best_dist;
        }

        let mut new_connections = vec![Vec::new(); node_level + 1];

        for layer in (0..=usize::min(node_level, top_layer)).rev() {
            let mut candidates =
                self.search_layer(entry, &vector, layer, self.params.ef_construction)?;
            candidates.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
            candidates.dedup_by_key(|neighbor| neighbor.id);
            candidates.truncate(if layer == 0 {
                self.params.m0
            } else {
                self.params.m
            });

            for candidate in &candidates {
                new_connections[layer].push(candidate.id);
            }

            for candidate in candidates {
                if let Some(node) = self.nodes.get_mut(&candidate.id) {
                    if layer < node.connections.len()
                        && !node.connections[layer].contains(&internal_id)
                    {
                        node.connections[layer].push(internal_id);
                    }
                }
                self.prune(candidate.id, layer)?;
            }
        }

        self.nodes.insert(
            internal_id,
            Node {
                external_id: external_id.clone(),
                vector,
                connections: new_connections,
                layer: node_level,
            },
        );
        self.external_to_internal.insert(external_id, internal_id);

        if let Some(current_entry) = self.entry {
            if node_level > self.nodes[&current_entry].layer {
                self.entry = Some(internal_id);
            }
        }

        let _ = entry_dist;
        Ok(())
    }

    /// Deletes an external id and removes incoming graph edges.
    pub fn delete(&mut self, external_id: &str) {
        let Some(internal_id) = self.external_to_internal.remove(external_id) else {
            return;
        };
        self.nodes.remove(&internal_id);

        for node in self.nodes.values_mut() {
            for layer in &mut node.connections {
                layer.retain(|id| *id != internal_id);
            }
        }

        if self.entry == Some(internal_id) {
            self.entry = self
                .nodes
                .iter()
                .max_by_key(|(_, node)| node.layer)
                .map(|(id, _)| *id);
        }
    }

    /// Searches the graph and returns external ids with raw metric values.
    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<(String, f32)>, String> {
        let Some(mut entry) = self.entry else {
            return Ok(Vec::new());
        };

        let top_layer = self.nodes[&entry].layer;
        for layer in (1..=top_layer).rev() {
            entry = self.greedy_closest(entry, query, layer)?.0;
        }

        let mut best =
            self.search_layer(entry, query, 0, usize::max(self.params.ef_search, limit))?;
        best.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));

        best.into_iter()
            .take(limit)
            .filter_map(|neighbor| self.nodes.get(&neighbor.id))
            .map(|node| {
                crate::distances::compute(self.metric, query, &node.vector)
                    .map(|raw| (node.external_id.clone(), raw))
            })
            .collect()
    }

    /// Descends one graph layer until no neighbor improves the rank distance.
    fn greedy_closest(
        &self,
        start: usize,
        query: &[f32],
        layer: usize,
    ) -> Result<(usize, f32), String> {
        let mut current = start;
        let mut current_dist = self.rank_distance(&self.nodes[&current].vector, query)?;

        loop {
            let mut moved = false;
            let Some(node) = self.nodes.get(&current) else {
                break;
            };
            if layer >= node.connections.len() {
                break;
            }

            for neighbor_id in &node.connections[layer] {
                let Some(neighbor) = self.nodes.get(neighbor_id) else {
                    continue;
                };
                let dist = self.rank_distance(&neighbor.vector, query)?;
                if dist < current_dist {
                    current = *neighbor_id;
                    current_dist = dist;
                    moved = true;
                }
            }

            if !moved {
                break;
            }
        }

        Ok((current, current_dist))
    }

    /// Explores one layer with separate candidate and bounded-result heaps.
    fn search_layer(
        &self,
        entry: usize,
        query: &[f32],
        layer: usize,
        ef: usize,
    ) -> Result<Vec<ScoredNode>, String> {
        if !self.nodes.contains_key(&entry) {
            return Ok(Vec::new());
        }

        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();
        let dist = self.rank_distance(&self.nodes[&entry].vector, query)?;

        candidates.push(ClosestFirst(ScoredNode { id: entry, dist }));
        results.push(WorstFirst(ScoredNode { id: entry, dist }));
        visited.insert(entry);

        while let Some(current) = candidates.pop() {
            let current = current.0;
            let worst = results
                .peek()
                .map_or(f32::INFINITY, |neighbor| neighbor.0.dist);
            if results.len() >= ef && current.dist > worst {
                break;
            }

            let Some(node) = self.nodes.get(&current.id) else {
                continue;
            };
            if layer >= node.connections.len() {
                continue;
            }

            for neighbor_id in &node.connections[layer] {
                if !visited.insert(*neighbor_id) {
                    continue;
                }
                let Some(neighbor) = self.nodes.get(neighbor_id) else {
                    continue;
                };
                let dist = self.rank_distance(&neighbor.vector, query)?;
                if results.len() < ef || dist < worst {
                    let candidate = ScoredNode {
                        id: *neighbor_id,
                        dist,
                    };
                    candidates.push(ClosestFirst(candidate.clone()));
                    results.push(WorstFirst(candidate));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        Ok(results.into_iter().map(|neighbor| neighbor.0).collect())
    }

    /// Keeps each node's neighbor list bounded by the configured HNSW degree.
    fn prune(&mut self, node_id: usize, layer: usize) -> Result<(), String> {
        let limit = if layer == 0 {
            self.params.m0
        } else {
            self.params.m
        };
        let Some(node) = self.nodes.get(&node_id) else {
            return Ok(());
        };
        if layer >= node.connections.len() {
            return Ok(());
        }

        let vector = node.vector.clone();
        let connections = node.connections[layer].clone();
        let mut scored = Vec::with_capacity(connections.len());
        for neighbor_id in connections {
            if let Some(neighbor) = self.nodes.get(&neighbor_id) {
                scored.push((neighbor_id, self.rank_distance(&vector, &neighbor.vector)?));
            }
        }
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        scored.truncate(limit);

        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.connections[layer] = scored.into_iter().map(|(id, _)| id).collect();
        }
        Ok(())
    }

    /// Computes the ascending distance used internally by HNSW.
    fn rank_distance(&self, left: &[f32], right: &[f32]) -> Result<f32, String> {
        crate::distances::rank_distance(self.metric, left, right)
    }

    /// Assigns a deterministic pseudo-random layer from the external id.
    fn level_for(&self, external_id: &str) -> usize {
        let mut hash = hash64(external_id.as_bytes());
        let mut level = 0usize;
        while level < self.params.max_level && hash & 0b11 == 0 {
            level += 1;
            hash >>= 2;
        }
        level
    }
}

pub struct HnswResource(pub RwLock<HnswIndex>);

/// FNV-1a style hash used for deterministic graph level assignment.
fn hash64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}
