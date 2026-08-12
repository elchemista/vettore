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
        if self.m > 1_024 || self.m0 > 2_048 || self.m0 < self.m {
            return Err("invalid hnsw degree".to_string());
        }
        if self.ef_construction < self.m {
            return Err("ef_construction must be >= m".to_string());
        }
        if self.ef_construction > 1_000_000 {
            return Err("ef_construction exceeds safety limit".to_string());
        }
        if self.ef_search == 0 || self.ef_search > 1_000_000 {
            return Err("ef_search must be positive".to_string());
        }
        if self.max_level == 0 || self.max_level > 64 {
            return Err("max_level must be positive".to_string());
        }

        Ok(self)
    }
}

#[derive(Clone, Debug)]
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
        self.0.dist.total_cmp(&other.0.dist) == Ordering::Equal && self.0.id == other.0.id
    }
}

impl Ord for ClosestFirst {
    /// Reverses ordering so the binary heap pops the closest candidate first.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .dist
            .total_cmp(&self.0.dist)
            .then_with(|| other.0.id.cmp(&self.0.id))
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
        self.0.dist.total_cmp(&other.0.dist) == Ordering::Equal && self.0.id == other.0.id
    }
}

impl Ord for WorstFirst {
    /// Orders so the binary heap exposes the worst retained result first.
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .dist
            .total_cmp(&other.0.dist)
            .then_with(|| self.0.id.cmp(&other.0.id))
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
    incoming: HashMap<(usize, usize), HashSet<usize>>,
    entry: Option<usize>,
    next: usize,
    dimension: Option<usize>,
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
            incoming: HashMap::new(),
            entry: None,
            next: 0,
            dimension: None,
        })
    }

    /// Inserts or replaces one external id in the graph.
    pub fn insert(&mut self, external_id: String, vector: Vec<f32>) -> Result<(), String> {
        validate_vector(&vector, self.dimension)?;

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
            self.dimension = Some(self.nodes[&internal_id].vector.len());
            return Ok(());
        }

        let mut entry = self.entry.ok_or_else(|| "missing hnsw entry".to_string())?;
        let top_layer = self.nodes[&entry].layer;

        for layer in (node_level + 1..=top_layer).rev() {
            let (best_id, _best_dist) = self.greedy_closest(entry, &vector, layer)?;
            entry = best_id;
        }

        let mut new_connections = vec![Vec::new(); node_level + 1];

        for layer in (0..=usize::min(node_level, top_layer)).rev() {
            let mut candidates =
                self.search_layer(entry, &vector, layer, self.params.ef_construction)?;
            candidates = self.select_neighbors(
                candidates,
                if layer == 0 {
                    self.params.m0
                } else {
                    self.params.m
                },
            )?;

            for candidate in &candidates {
                new_connections[layer].push(candidate.id);
            }

            if let Some(closest) = candidates.first() {
                entry = closest.id;
            }
        }

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
        self.dimension = Some(self.nodes[&internal_id].vector.len());

        for (layer, neighbors) in new_connections.into_iter().enumerate() {
            self.replace_connections(internal_id, layer, neighbors);
        }

        // The new node must exist before reciprocal neighbors are pruned. If it
        // is added afterwards, `prune` cannot score it and silently removes
        // every incoming edge, leaving later inserts unreachable from `entry`.
        let reciprocal_connections = self.nodes[&internal_id].connections.clone();
        for (layer, neighbors) in reciprocal_connections.into_iter().enumerate() {
            for (position, neighbor_id) in neighbors.into_iter().enumerate() {
                self.add_connection(neighbor_id, layer, internal_id);
                if layer == 0 && position == 0 {
                    self.prune_retaining(neighbor_id, layer, Some(internal_id))?;
                } else {
                    self.prune(neighbor_id, layer)?;
                }
            }
        }

        if let Some(current_entry) = self.entry {
            if node_level > self.nodes[&current_entry].layer {
                self.entry = Some(internal_id);
            }
        }

        Ok(())
    }

    /// Validates a whole batch before mutating the graph, then inserts it while
    /// holding a single resource lock at the NIF boundary.
    pub fn insert_many(&mut self, vectors: Vec<(String, Vec<f32>)>) -> Result<(), String> {
        let expected = self
            .dimension
            .or_else(|| vectors.first().map(|(_, vector)| vector.len()));
        for (_, vector) in &vectors {
            validate_vector(vector, expected)?;
        }
        for (id, vector) in vectors {
            self.insert(id, vector)?;
        }
        Ok(())
    }

    /// Deletes an external id, removes its edges, and reconnects local neighbors.
    pub fn delete(&mut self, external_id: &str) {
        let Some(internal_id) = self.external_to_internal.remove(external_id) else {
            return;
        };
        let Some(removed) = self.nodes.remove(&internal_id) else {
            return;
        };

        let mut incoming_by_layer = vec![HashSet::new(); removed.connections.len()];

        for (layer, targets) in removed.connections.iter().enumerate() {
            for target in targets {
                self.remove_incoming(*target, layer, internal_id);
            }
        }

        let incoming_keys: Vec<_> = self
            .incoming
            .keys()
            .filter(|(target, _layer)| *target == internal_id)
            .copied()
            .collect();

        for (_target, layer) in incoming_keys {
            let sources = self
                .incoming
                .remove(&(internal_id, layer))
                .unwrap_or_default();
            if layer >= incoming_by_layer.len() {
                incoming_by_layer.resize_with(layer + 1, HashSet::new);
            }
            for source in sources {
                incoming_by_layer[layer].insert(source);
                self.remove_connection(source, layer, internal_id);
            }
        }

        // Redirect each incoming edge toward the deleted node's bounded
        // outgoing neighborhood. This repairs the same traversal direction in
        // O(in_degree * M) additions rather than scanning the whole graph or
        // building an unbounded clique around a popular node.
        for (layer, sources) in incoming_by_layer.into_iter().enumerate() {
            let mut sources: Vec<_> = sources.into_iter().collect();
            sources.sort_unstable();

            let mut anchors: Vec<_> = removed
                .connections
                .get(layer)
                .into_iter()
                .flatten()
                .copied()
                .filter(|id| {
                    self.nodes
                        .get(id)
                        .is_some_and(|node| layer < node.connections.len())
                })
                .collect();

            if anchors.is_empty() {
                let limit = if layer == 0 {
                    self.params.m0
                } else {
                    self.params.m
                };
                let mut scored: Vec<_> = sources
                    .iter()
                    .filter_map(|id| {
                        self.nodes.get(id).and_then(|node| {
                            self.rank_distance(&removed.vector, &node.vector)
                                .ok()
                                .map(|dist| (*id, dist))
                        })
                    })
                    .collect();
                scored.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
                anchors.extend(scored.into_iter().take(limit).map(|(id, _dist)| id));
            }

            for source in sources {
                if let Some(node) = self.nodes.get(&source) {
                    let mut connections = node.connections[layer].clone();
                    connections.extend(anchors.iter().copied());
                    self.replace_connections(source, layer, connections);
                }
                let _ = self.prune(source, layer);
            }
        }

        if self.entry == Some(internal_id) {
            self.entry = self
                .nodes
                .iter()
                .max_by(|(_, left), (_, right)| {
                    left.layer
                        .cmp(&right.layer)
                        .then_with(|| right.external_id.cmp(&left.external_id))
                })
                .map(|(id, _)| *id);
        }
        if self.nodes.is_empty() {
            self.dimension = None;
        }
    }

    /// Releases all graph state while keeping the resource reusable.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.external_to_internal.clear();
        self.incoming.clear();
        self.entry = None;
        self.next = 0;
        self.dimension = None;
    }

    /// Searches the graph and returns external ids with raw metric values.
    pub fn search(&self, query: &[f32], limit: usize) -> Result<Vec<(String, f32)>, String> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        validate_vector(query, self.dimension)?;

        let Some(mut entry) = self.entry else {
            return Ok(Vec::new());
        };

        let top_layer = self.nodes[&entry].layer;
        for layer in (1..=top_layer).rev() {
            entry = self.greedy_closest(entry, query, layer)?.0;
        }

        let mut best =
            self.search_layer(entry, query, 0, usize::max(self.params.ef_search, limit))?;
        best.sort_by(|a, b| {
            let left_id = self
                .nodes
                .get(&a.id)
                .map(|node| node.external_id.as_str())
                .unwrap_or("");
            let right_id = self
                .nodes
                .get(&b.id)
                .map(|node| node.external_id.as_str())
                .unwrap_or("");
            a.dist
                .total_cmp(&b.dist)
                .then_with(|| left_id.cmp(right_id))
        });

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
                let worst = results
                    .peek()
                    .map_or(f32::INFINITY, |neighbor| neighbor.0.dist);
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
        self.prune_retaining(node_id, layer, None)
    }

    /// Prunes with the standard diversified HNSW neighbor heuristic.
    fn prune_retaining(
        &mut self,
        node_id: usize,
        layer: usize,
        required: Option<usize>,
    ) -> Result<(), String> {
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

        let connections = node.connections[layer].clone();
        let mut scored = Vec::with_capacity(connections.len());
        for neighbor_id in connections {
            if let Some(neighbor) = self.nodes.get(&neighbor_id) {
                scored.push(ScoredNode {
                    id: neighbor_id,
                    dist: self.rank_distance(&node.vector, &neighbor.vector)?,
                });
            }
        }

        let selected = self.select_neighbors_retaining(scored, limit, required)?;
        self.replace_connections(
            node_id,
            layer,
            selected.into_iter().map(|neighbor| neighbor.id).collect(),
        );
        Ok(())
    }

    /// Selects close but mutually diverse neighbors, filling unused capacity
    /// with the closest rejected nodes to keep sparse graphs connected.
    fn select_neighbors(
        &self,
        candidates: Vec<ScoredNode>,
        limit: usize,
    ) -> Result<Vec<ScoredNode>, String> {
        self.select_neighbors_retaining(candidates, limit, None)
    }

    fn select_neighbors_retaining(
        &self,
        mut candidates: Vec<ScoredNode>,
        limit: usize,
        required: Option<usize>,
    ) -> Result<Vec<ScoredNode>, String> {
        candidates.sort_by(|a, b| a.dist.total_cmp(&b.dist).then_with(|| a.id.cmp(&b.id)));
        candidates.dedup_by_key(|neighbor| neighbor.id);

        let mut selected = Vec::with_capacity(usize::min(limit, candidates.len()));
        if let Some(required_id) = required {
            if let Some(position) = candidates
                .iter()
                .position(|candidate| candidate.id == required_id)
            {
                selected.push(candidates.remove(position));
            }
        }

        let mut rejected = Vec::new();
        for candidate in candidates {
            if selected.len() == limit {
                rejected.push(candidate);
                continue;
            }

            let candidate_vector = &self.nodes[&candidate.id].vector;
            let mut diverse = true;
            for retained in &selected {
                let retained_vector = &self.nodes[&retained.id].vector;
                if self.rank_distance(candidate_vector, retained_vector)? <= candidate.dist {
                    diverse = false;
                    break;
                }
            }

            if diverse {
                selected.push(candidate);
            } else {
                rejected.push(candidate);
            }
        }

        if selected.len() < limit {
            rejected.sort_by(|a, b| a.dist.total_cmp(&b.dist).then_with(|| a.id.cmp(&b.id)));
            selected.extend(rejected.into_iter().take(limit - selected.len()));
        }

        selected.sort_by(|a, b| a.dist.total_cmp(&b.dist).then_with(|| a.id.cmp(&b.id)));
        Ok(selected)
    }

    /// Replaces an adjacency list and keeps the reverse-edge map in sync.
    fn replace_connections(&mut self, source: usize, layer: usize, mut targets: Vec<usize>) {
        if self
            .nodes
            .get(&source)
            .is_none_or(|node| layer >= node.connections.len())
        {
            return;
        }

        let mut seen = HashSet::new();
        targets.retain(|target| {
            *target != source
                && self
                    .nodes
                    .get(target)
                    .is_some_and(|node| layer < node.connections.len())
                && seen.insert(*target)
        });

        let old = self
            .nodes
            .get(&source)
            .and_then(|node| node.connections.get(layer))
            .cloned()
            .unwrap_or_default();
        let old_set: HashSet<_> = old.iter().copied().collect();
        let new_set: HashSet<_> = targets.iter().copied().collect();

        for target in old_set.difference(&new_set) {
            self.remove_incoming(*target, layer, source);
        }
        for target in new_set.difference(&old_set) {
            self.incoming
                .entry((*target, layer))
                .or_default()
                .insert(source);
        }

        if let Some(node) = self.nodes.get_mut(&source) {
            if layer < node.connections.len() {
                node.connections[layer] = targets;
            }
        }
    }

    fn add_connection(&mut self, source: usize, layer: usize, target: usize) {
        let Some(node) = self.nodes.get(&source) else {
            return;
        };
        if layer >= node.connections.len() || node.connections[layer].contains(&target) {
            return;
        }
        let mut connections = node.connections[layer].clone();
        connections.push(target);
        self.replace_connections(source, layer, connections);
    }

    fn remove_connection(&mut self, source: usize, layer: usize, target: usize) {
        let Some(node) = self.nodes.get(&source) else {
            return;
        };
        if layer >= node.connections.len() {
            return;
        }
        let mut connections = node.connections[layer].clone();
        connections.retain(|neighbor| *neighbor != target);
        self.replace_connections(source, layer, connections);
    }

    fn remove_incoming(&mut self, target: usize, layer: usize, source: usize) {
        let key = (target, layer);
        if let Some(sources) = self.incoming.get_mut(&key) {
            sources.remove(&source);
            if sources.is_empty() {
                self.incoming.remove(&key);
            }
        }
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

#[rustler::resource_impl]
impl rustler::Resource for HnswResource {}

/// FNV-1a style hash used for deterministic graph level assignment.
fn hash64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    hash
}

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

    fn params() -> HnswParams {
        HnswParams {
            m: 8,
            m0: 16,
            ef_construction: 200,
            ef_search: 200,
            max_level: 12,
        }
    }

    #[test]
    fn validates_parameters() {
        assert!(HnswIndex::new(Metric::L2, params()).is_ok());

        for invalid in [
            HnswParams { m: 0, ..params() },
            HnswParams { m0: 0, ..params() },
            HnswParams {
                m: 1_025,
                m0: 2_048,
                ..params()
            },
            HnswParams {
                m0: 2_049,
                ..params()
            },
            HnswParams { m0: 4, ..params() },
            HnswParams {
                ef_construction: 4,
                ..params()
            },
            HnswParams {
                ef_construction: 1_000_001,
                ..params()
            },
            HnswParams {
                ef_search: 0,
                ..params()
            },
            HnswParams {
                ef_search: 1_000_001,
                ..params()
            },
            HnswParams {
                max_level: 0,
                ..params()
            },
            HnswParams {
                max_level: 65,
                ..params()
            },
        ] {
            assert!(HnswIndex::new(Metric::L2, invalid).is_err());
        }
    }

    #[test]
    fn every_inserted_node_remains_reachable() {
        let mut index = HnswIndex::new(Metric::L2, params()).unwrap();
        index
            .insert_many(
                (0..100)
                    .map(|value| (format!("{value:03}"), vec![value as f32]))
                    .collect(),
            )
            .unwrap();

        let hits = index.search(&[99.0], 100).unwrap();
        assert_eq!(hits.len(), 100);
        let ids: HashSet<_> = hits.into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids.len(), 100);

        for value in 0..100 {
            let hit = index.search(&[value as f32], 1).unwrap();
            assert_eq!(hit[0].0, format!("{value:03}"));
        }
    }

    #[test]
    fn batch_validation_is_atomic_and_replace_delete_work() {
        let mut index = HnswIndex::new(Metric::InnerProduct, params()).unwrap();
        index.insert("a".into(), vec![1.0, 0.0]).unwrap();
        assert!(index
            .insert_many(vec![
                ("b".into(), vec![0.0, 1.0]),
                ("bad".into(), vec![1.0]),
            ])
            .is_err());
        assert_eq!(index.nodes.len(), 1);

        index.insert("a".into(), vec![0.0, 1.0]).unwrap();
        assert_eq!(index.search(&[0.0, 1.0], 1).unwrap()[0].0, "a");
        index.delete("a");
        assert!(index.search(&[0.0, 1.0], 1).unwrap().is_empty());
        assert_eq!(index.dimension, None);

        index.insert("clear-me".into(), vec![1.0, 0.0]).unwrap();
        index.clear();
        assert!(index.nodes.is_empty());
        assert!(index.external_to_internal.is_empty());
        assert!(index.incoming.is_empty());
        assert_eq!(index.entry, None);
        assert_eq!(index.next, 0);
        assert_eq!(index.dimension, None);
    }

    #[test]
    fn rejects_non_finite_and_mismatched_vectors() {
        let mut index = HnswIndex::new(Metric::Cosine, params()).unwrap();
        assert!(index.insert("empty".into(), vec![]).is_err());
        index.insert("a".into(), vec![1.0, 0.0]).unwrap();
        assert!(index.insert("short".into(), vec![1.0]).is_err());
        assert!(index.insert("nan".into(), vec![f32::NAN, 0.0]).is_err());
        assert!(index.search(&[1.0], 1).is_err());
        assert!(index.search(&[f32::INFINITY, 0.0], 1).is_err());
    }

    #[test]
    fn empty_and_corrupted_internal_graph_paths_fail_safely() {
        let empty = HnswIndex::new(Metric::L2, params()).unwrap();
        assert_eq!(empty.search(&[1.0], 10), Ok(vec![]));
        assert!(empty.search(&[], 10).is_err());
        assert!(empty.search_layer(999, &[1.0], 0, 10).unwrap().is_empty());

        let mut index = HnswIndex::new(Metric::L2, params()).unwrap();
        index.insert("a".into(), vec![1.0]).unwrap();
        let entry = index.entry.unwrap();
        assert_eq!(index.prune(999, 0), Ok(()));
        assert_eq!(index.prune(entry, 999), Ok(()));

        index.nodes.get_mut(&entry).unwrap().connections[0].push(999);
        assert_eq!(index.greedy_closest(entry, &[1.0], 0).unwrap().0, entry);
        assert_eq!(index.search_layer(entry, &[1.0], 0, 10).unwrap().len(), 1);
        assert_eq!(index.prune(entry, 0), Ok(()));
        assert!(!index.nodes[&entry].connections[0].contains(&999));
    }

    #[test]
    fn candidate_heap_equality_and_partial_order_use_distance_and_id() {
        let closest = ClosestFirst(ScoredNode { id: 1, dist: 2.0 });
        let closest_equal = ClosestFirst(ScoredNode { id: 1, dist: 2.0 });
        let closest_other = ClosestFirst(ScoredNode { id: 2, dist: 2.0 });
        assert!(closest == closest_equal);
        assert!(closest != closest_other);
        assert_eq!(closest.partial_cmp(&closest_other), Some(Ordering::Greater));

        let worst = WorstFirst(ScoredNode { id: 1, dist: 2.0 });
        let worst_equal = WorstFirst(ScoredNode { id: 1, dist: 2.0 });
        let worst_other = WorstFirst(ScoredNode { id: 2, dist: 2.0 });
        assert!(worst == worst_equal);
        assert!(worst != worst_other);
        assert_eq!(worst.partial_cmp(&worst_other), Some(Ordering::Less));
    }

    #[test]
    fn high_ef_search_matches_exact_l2_on_a_two_dimensional_grid() {
        let mut index = HnswIndex::new(Metric::L2, params()).unwrap();
        let vectors: Vec<_> = (0..15)
            .flat_map(|x| {
                (0..15).map(move |y| (format!("{x:02}-{y:02}"), vec![x as f32, y as f32]))
            })
            .collect();
        index.insert_many(vectors.clone()).unwrap();

        for query in [[0.25, 0.75], [7.2, 8.6], [14.0, 14.0], [-3.0, 20.0]] {
            let mut expected: Vec<_> = vectors
                .iter()
                .map(|(id, vector)| {
                    (
                        id.clone(),
                        crate::distances::compute(Metric::L2, &query, vector).unwrap(),
                    )
                })
                .collect();
            expected.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            expected.truncate(20);

            assert_eq!(index.search(&query, 20).unwrap(), expected);
        }
    }

    #[test]
    fn self_queries_recall_every_unit_vector_for_supported_similarity_metrics() {
        let vectors: Vec<_> = (0..64)
            .map(|index| {
                let angle = std::f32::consts::TAU * index as f32 / 64.0;
                (format!("unit-{index:02}"), vec![angle.cos(), angle.sin()])
            })
            .collect();

        for metric in [Metric::Cosine, Metric::InnerProduct] {
            let mut index = HnswIndex::new(metric, params()).unwrap();
            index.insert_many(vectors.clone()).unwrap();
            for (id, vector) in &vectors {
                assert_eq!(index.search(vector, 1).unwrap()[0].0, *id);
            }
        }
    }

    #[test]
    fn graph_degrees_references_and_search_results_remain_well_formed() {
        let mut index = HnswIndex::new(Metric::L2, params()).unwrap();
        index
            .insert_many(
                (0..300)
                    .map(|value| {
                        (
                            format!("node-{value:03}"),
                            vec![
                                (value as f32).sin(),
                                (value as f32).cos(),
                                value as f32 / 300.0,
                            ],
                        )
                    })
                    .collect(),
            )
            .unwrap();

        for (node_id, node) in &index.nodes {
            for (layer, connections) in node.connections.iter().enumerate() {
                let limit = if layer == 0 {
                    index.params.m0
                } else {
                    index.params.m
                };
                assert!(connections.len() <= limit);
                assert_eq!(
                    connections.iter().collect::<HashSet<_>>().len(),
                    connections.len()
                );
                assert!(!connections.contains(node_id));
                assert!(connections.iter().all(|id| index.nodes.contains_key(id)));
                for target in connections {
                    assert!(index
                        .incoming
                        .get(&(*target, layer))
                        .is_some_and(|sources| sources.contains(node_id)));
                }
            }
        }

        for ((target, layer), sources) in &index.incoming {
            assert!(index.nodes.contains_key(target));
            for source in sources {
                assert!(index.nodes[source].connections[*layer].contains(target));
            }
        }

        let hits = index.search(&[0.0, 1.0, 0.5], 1_000).unwrap();
        assert_eq!(hits.len(), index.nodes.len());
        assert_eq!(
            hits.iter().map(|(id, _)| id).collect::<HashSet<_>>().len(),
            hits.len()
        );
    }

    #[test]
    fn sparse_graph_stays_reachable_after_heavy_deletion() {
        let sparse_params = HnswParams {
            m: 4,
            m0: 8,
            ef_construction: 256,
            ef_search: 1_000,
            max_level: 12,
        };
        let vectors: Vec<_> = (0..200)
            .map(|value| {
                (
                    format!("id-{value:03}"),
                    vec![(value as f32 * 0.73).sin(), (value as f32 * 0.37).cos()],
                )
            })
            .collect();
        let mut index = HnswIndex::new(Metric::L2, sparse_params).unwrap();
        index.insert_many(vectors.clone()).unwrap();

        assert_eq!(
            index.search(&vectors[0].1, vectors.len()).unwrap().len(),
            200
        );
        for (id, vector) in &vectors {
            assert_eq!(index.search(vector, 1).unwrap()[0].0, *id);
        }

        for (id, _vector) in vectors.iter().take(160) {
            index.delete(id);
        }

        assert_eq!(index.search(&vectors[199].1, 40).unwrap().len(), 40);
        for (id, vector) in vectors.iter().skip(160) {
            assert_eq!(index.search(vector, 1).unwrap()[0].0, *id);
        }
    }

    #[test]
    fn delete_repairs_nodes_even_without_outgoing_anchors() {
        let mut index = HnswIndex::new(Metric::L2, params()).unwrap();
        index
            .insert_many(
                (0..20)
                    .map(|value| (format!("id-{value:02}"), vec![value as f32]))
                    .collect(),
            )
            .unwrap();

        let target = index
            .incoming
            .iter()
            .find(|((_, layer), sources)| *layer == 0 && !sources.is_empty())
            .map(|((target, _layer), _sources)| *target)
            .unwrap();
        let external_id = index.nodes[&target].external_id.clone();
        index.replace_connections(target, 0, vec![]);
        index.delete(&external_id);

        assert!(!index.nodes.contains_key(&target));
        assert!(index.nodes.values().all(|node| node
            .connections
            .iter()
            .flatten()
            .all(|id| *id != target)));

        let source = *index.nodes.keys().next().unwrap();
        index.replace_connections(usize::MAX, 0, vec![source]);
        index.add_connection(usize::MAX, 0, source);
        index.remove_connection(usize::MAX, 0, source);
        index.add_connection(source, usize::MAX, target);
        index.remove_connection(source, usize::MAX, target);
        index.replace_connections(source, 0, vec![source, usize::MAX]);
        assert!(index.nodes[&source].connections[0].is_empty());
    }

    #[test]
    fn deleting_an_entry_selects_a_deterministic_replacement() {
        let mut index = HnswIndex::new(Metric::L2, params()).unwrap();
        index
            .insert_many(
                (0..80)
                    .map(|value| (format!("id-{value:02}"), vec![value as f32]))
                    .collect(),
            )
            .unwrap();

        let old_entry = index.entry.unwrap();
        let old_entry_id = index.nodes[&old_entry].external_id.clone();
        index.delete("missing");
        assert_eq!(index.entry, Some(old_entry));
        index.delete(&old_entry_id);

        let expected = index
            .nodes
            .values()
            .max_by(|left, right| {
                left.layer
                    .cmp(&right.layer)
                    .then_with(|| right.external_id.cmp(&left.external_id))
            })
            .unwrap()
            .external_id
            .clone();
        let replacement = &index.nodes[&index.entry.unwrap()].external_id;
        assert_eq!(replacement, &expected);

        assert_eq!(index.search(&[0.0], 0), Ok(vec![]));
    }

    #[test]
    fn deterministic_level_assignment_is_bounded_and_seedless() {
        let first = HnswIndex::new(Metric::L2, params()).unwrap();
        let second = HnswIndex::new(Metric::L2, params()).unwrap();
        for id in ["a", "b", "stable-id", "another-id", "\0"] {
            assert_eq!(first.level_for(id), second.level_for(id));
            assert!(first.level_for(id) <= first.params.max_level);
            assert_eq!(hash64(id.as_bytes()), hash64(id.as_bytes()));
        }
    }
}
