use rustler::{Env, ResourceArc, Term};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::{BinaryHeap, HashSet};
use std::sync::Mutex;

use rand::rng;
use rand::Rng;
use wide::f32x4;

/// Distance metric enum.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Distance {
    Euclidean,
    Cosine,
    DotProduct,
    Hnsw,
    Binary,
}

/// In-memory representation of a single embedding.
#[derive(Clone)]
pub struct Embedding {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<HashMap<String, String>>,
    /// For "binary" mode: precomputed bit-signature for Hamming distance.
    pub binary: Option<Vec<u64>>,
}

/// A collection has:
///  - a dimension (vector length)
///  - a distance metric
///  - a list of stored embeddings
///  - whether to keep the original float vectors in memory
///  - an optional HNSW index for "hnsw" distance
pub struct Collection {
    pub dimension: usize,
    pub keep_embeddings: bool,
    pub distance: Distance,
    pub embeddings: Vec<Embedding>,
    pub hnsw_index: Option<HnswIndexWrapper>,
}

/// The main DB holds multiple named Collections.
pub struct CacheDB {
    pub collections: HashMap<String, Collection>,
}

impl CacheDB {
    pub fn new() -> Self {
        Self {
            collections: HashMap::new(),
        }
    }

    /// Retrieve a single embedding by ID from a named collection.
    /// Returns an error if the collection or embedding is not found.
    pub fn get_embedding_by_id(
        &self,
        collection_name: &str,
        id: &str,
    ) -> Result<Embedding, String> {
        let coll = self
            .collections
            .get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

        coll.embeddings
            .iter()
            .find(|emb| emb.id == id)
            .cloned()
            .ok_or_else(|| {
                format!(
                    "Embedding '{}' not found in collection '{}'",
                    id, collection_name
                )
            })
    }
}

/// A Rustler Resource to wrap our `CacheDB` with a Mutex for concurrency safety.
pub struct DBResource(pub Mutex<CacheDB>);

impl rustler::Resource for DBResource {}

//
// Helper functions for computing distances (SIMD for performance).
//

#[inline]
fn load_f32x4(slice: &[f32], i: usize) -> f32x4 {
    f32x4::from([slice[i], slice[i + 1], slice[i + 2], slice[i + 3]])
}

/// SIMD Euclidean distance between two vectors.
fn simd_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut i = 0;
    let mut sum_squares = 0.0;

    while i + 4 <= len {
        let va = load_f32x4(a, i);
        let vb = load_f32x4(b, i);
        let diff = va - vb;
        let sq = diff * diff;
        sum_squares += sq.reduce_add();
        i += 4;
    }

    // leftover
    while i < len {
        let d = a[i] - b[i];
        sum_squares += d * d;
        i += 1;
    }
    sum_squares.sqrt()
}

/// SIMD dot product of two vectors.
fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut i = 0;
    let mut accum = 0.0;

    while i + 4 <= len {
        let va = load_f32x4(a, i);
        let vb = load_f32x4(b, i);
        let prod = va * vb;
        accum += prod.reduce_add();
        i += 4;
    }

    // leftover
    while i < len {
        accum += a[i] * b[i];
        i += 1;
    }
    accum
}

/// Compress a vector of floats by encoding the sign bit of each value into one bit of a `u64`.
/// (Used for "binary" distance.)
fn compress_vector(vector: &[f32]) -> Vec<u64> {
    let mut result = Vec::new();
    let mut current: u64 = 0;
    let mut bits_filled = 0;
    for &val in vector {
        current <<= 1;
        // If val >= 0.0, we set bit to 1, otherwise 0
        if val >= 0.0 {
            current |= 1;
        }
        bits_filled += 1;
        if bits_filled == 64 {
            result.push(current);
            current = 0;
            bits_filled = 0;
        }
    }
    if bits_filled > 0 {
        // Shift the remaining bits
        current <<= 64 - bits_filled;
        result.push(current);
    }
    result
}

/// Hamming distance between two compressed (binary) vectors.
fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones()) // XOR and count bits
        .sum()
}

/// Normalize a vector for "cosine" distance, so the magnitude is ~1.0
fn normalize_vec(v: &[f32]) -> Vec<f32> {
    let len = v.len();
    let mut i = 0;
    let mut sum_sq = 0.0;

    while i + 4 <= len {
        let vv = load_f32x4(v, i);
        let sq = vv * vv;
        sum_sq += sq.reduce_add();
        i += 4;
    }
    while i < len {
        sum_sq += v[i] * v[i];
        i += 1;
    }

    let norm = sum_sq.sqrt();
    if norm > std::f32::EPSILON {
        v.iter().map(|x| x / norm).collect()
    } else {
        // If norm is extremely small, just return the original vector
        v.to_vec()
    }
}

//
// HNSW data structures
//

const M: usize = 16;
const M_MAX0: usize = 32;
const EF_CONSTRUCTION: usize = 100;
const EF_SEARCH: usize = 64;
const MAX_LEVEL: usize = 12;

/// A minimal struct to store a vector in the HNSW graph.
#[derive(Clone)]
pub struct VectorItem {
    pub id: usize,
    pub vector: Vec<f32>,
}

/// A neighbor in the graph search.
#[derive(Clone)]
struct Neighbor {
    pub id: usize,
    pub distance: f32,
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering: smaller distance => higher priority
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
            .reverse()
    }
}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}
impl Eq for Neighbor {}

pub struct Node {
    id: usize,
    connections: Vec<Vec<usize>>, // connections per level
    item: VectorItem,
    layer: usize,
}

/// The main HNSW index for approximate nearest neighbors.
pub struct HnswIndex {
    pub(crate) nodes: HashMap<usize, Node>,
    pub entry_point: Option<usize>,
    pub max_level: usize,
    level_lambda: f32,
}

impl HnswIndex {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            entry_point: None,
            max_level: MAX_LEVEL,
            level_lambda: 1.0 / (M as f32).ln(),
        }
    }

    pub fn add(&mut self, item: VectorItem) -> Result<(), String> {
        let node_level = self.random_level();

        if self.nodes.is_empty() {
            // first node
            let node = Node {
                id: item.id,
                connections: vec![Vec::new(); node_level + 1],
                item,
                layer: node_level,
            };
            let node_id = node.id;
            self.nodes.insert(node_id, node);
            self.entry_point = Some(node_id);
            return Ok(());
        }

        // search from top
        let mut curr_ep = self.entry_point.unwrap();
        let mut curr_dist =
            simd_euclidean_distance(&self.nodes[&curr_ep].item.vector, &item.vector);
        let top_l = self.nodes[&curr_ep].layer;

        for level in (0..=top_l).rev() {
            let improved = self.search_layer(curr_ep, &item.vector, level, 1)?;
            if !improved.is_empty() {
                let best = &improved[0];
                if best.distance < curr_dist {
                    curr_ep = best.id;
                    curr_dist = best.distance;
                }
            }
        }

        // Link
        let mut new_conn = vec![Vec::new(); node_level + 1];
        for level in 0..=node_level {
            let neighbors = self.search_layer(curr_ep, &item.vector, level, EF_CONSTRUCTION)?;
            let final_neighbors = self.select_neighbors(&neighbors, &item.vector, level);
            new_conn[level] = final_neighbors.iter().map(|n| n.id).collect();

            for neigh in final_neighbors {
                if let Some(nd) = self.nodes.get_mut(&neigh.id) {
                    if level < nd.connections.len() {
                        nd.connections[level].push(item.id);
                    }
                }
            }
        }

        let node = Node {
            id: item.id,
            connections: new_conn,
            item,
            layer: node_level,
        };
        let node_id = node.id;
        self.nodes.insert(node_id, node);

        let ep_layer = self.nodes[&self.entry_point.unwrap()].layer;
        if node_level > ep_layer {
            self.entry_point = Some(node_id);
        }

        Ok(())
    }

    fn search_layer(
        &self,
        entry_id: usize,
        query: &[f32],
        level: usize,
        ef: usize,
    ) -> Result<Vec<Neighbor>, String> {
        let start_node = self
            .nodes
            .get(&entry_id)
            .ok_or_else(|| format!("Node not found: {}", entry_id))?;

        if level >= start_node.connections.len() {
            return Ok(Vec::new());
        }

        let dist = simd_euclidean_distance(&start_node.item.vector, query);
        let init = Neighbor {
            id: entry_id,
            distance: dist,
        };
        let mut visited = HashSet::new();
        visited.insert(entry_id);

        let mut candidates = BinaryHeap::new();
        candidates.push(init.clone());

        let mut results = BinaryHeap::new();
        results.push(init);

        while let Some(cur) = candidates.pop() {
            let worst = results.peek().map_or(f32::INFINITY, |n| n.distance);
            if cur.distance > worst {
                break;
            }
            let node = self.nodes.get(&cur.id).unwrap();
            if level < node.connections.len() {
                for &nbr in &node.connections[level] {
                    if !visited.insert(nbr) {
                        continue;
                    }
                    let d = simd_euclidean_distance(&self.nodes[&nbr].item.vector, query);
                    let neigh = Neighbor {
                        id: nbr,
                        distance: d,
                    };
                    let worst2 = results.peek().map_or(f32::INFINITY, |n| n.distance);
                    if results.len() < ef || d < worst2 {
                        candidates.push(neigh.clone());
                        results.push(neigh);
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        Ok(results.into_sorted_vec())
    }

    fn select_neighbors(
        &self,
        neighbors: &[Neighbor],
        _query: &[f32],
        level: usize,
    ) -> Vec<Neighbor> {
        let max_conn = if level == 0 { M_MAX0 } else { M };
        let mut sorted = neighbors.to_vec();
        sorted.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        if sorted.len() > max_conn {
            sorted.truncate(max_conn);
        }
        sorted
    }

    fn random_level(&self) -> usize {
        let mut r = rng();
        let mut lvl = 0usize;
        while r.random::<f32>() < self.level_lambda && lvl < self.max_level {
            lvl += 1;
        }
        lvl
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, String> {
        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }
        let ep = self.entry_point.unwrap();
        let mut curr_ep = ep;
        let mut curr_dist = simd_euclidean_distance(&self.nodes[&ep].item.vector, query);
        let top_l = self.nodes[&ep].layer;

        for level in (1..=top_l).rev() {
            loop {
                let mut changed = false;
                if let Some(node) = self.nodes.get(&curr_ep) {
                    if level < node.connections.len() {
                        for &nbr in &node.connections[level] {
                            let d = simd_euclidean_distance(&self.nodes[&nbr].item.vector, query);
                            if d < curr_dist {
                                curr_dist = d;
                                curr_ep = nbr;
                                changed = true;
                            }
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        let results = self.search_layer(curr_ep, query, 0, EF_SEARCH)?;
        let mut sorted = results;
        sorted.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        let topk = sorted
            .into_iter()
            .take(k)
            .map(|n| (n.id, n.distance))
            .collect();
        Ok(topk)
    }
}

pub struct HnswIndexWrapper {
    pub index: HnswIndex,
    pub id_map: HashMap<usize, String>,
    pub next_id: usize,
}

impl HnswIndexWrapper {
    pub fn new() -> Self {
        Self {
            index: HnswIndex::new(),
            id_map: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn insert_embedding(&mut self, emb: &Embedding) -> Result<(), String> {
        let hnsw_id = self.next_id;
        self.next_id += 1;
        self.id_map.insert(hnsw_id, emb.id.clone());

        let item = VectorItem {
            id: hnsw_id,
            vector: emb.vector.clone(),
        };
        self.index.add(item)
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>, String> {
        let results = self.index.search(query, k)?;
        let mut out = Vec::new();
        for (nid, dist) in results {
            if let Some(str_id) = self.id_map.get(&nid) {
                out.push((str_id.clone(), dist));
            }
        }
        Ok(out)
    }
}

impl Collection {
    pub fn create_with_distance(dimension: usize, dist_str: &str) -> Result<Self, String> {
        let distance = match dist_str.to_lowercase().as_str() {
            "euclidean" => Distance::Euclidean,
            "cosine" => Distance::Cosine,
            "dot" => Distance::DotProduct,
            "hnsw" => Distance::Hnsw,
            "binary" => Distance::Binary,
            other => {
                return Err(format!(
                "Unknown distance '{}'. Must be one of: euclidean, cosine, dot, hnsw, or binary.",
                other
            ))
            }
        };

        let mut col = Self {
            dimension,
            distance,
            embeddings: Vec::new(),
            keep_embeddings: true, // default
            hnsw_index: None,
        };

        if distance == Distance::Hnsw {
            col.hnsw_index = Some(HnswIndexWrapper::new());
        }
        Ok(col)
    }

    pub fn insert_embedding(&mut self, emb: Embedding) -> Result<(), String> {
        if emb.vector.len() != self.dimension {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension,
                emb.vector.len()
            ));
        }

        if self.embeddings.iter().any(|e| e.id == emb.id) {
            return Err(format!(
                "Embedding ID '{}' already exists in this collection",
                emb.id
            ));
        }

        let mut new_emb = emb;

        if self.distance == Distance::Cosine {
            new_emb.vector = normalize_vec(&new_emb.vector);
        }
        if self.distance == Distance::Binary {
            new_emb.binary = Some(compress_vector(&new_emb.vector));

            // If we're NOT keeping raw embeddings, clear out the float vector:
            if !self.keep_embeddings {
                new_emb.vector.clear();
            }
        }

        self.embeddings.push(new_emb.clone());

        if self.distance == Distance::Hnsw {
            if let Some(ref mut w) = self.hnsw_index {
                w.insert_embedding(&new_emb)?;
            }
        }

        Ok(())
    }

    pub fn get_similarity(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>, String> {
        match self.distance {
            Distance::Hnsw => {
                if let Some(ref w) = self.hnsw_index {
                    w.search(query, k)
                } else {
                    Err("No HNSW index found in this collection".to_string())
                }
            }
            Distance::Euclidean => {
                let mut scored: Vec<_> = self
                    .embeddings
                    .iter()
                    .map(|emb| {
                        let dist = simd_euclidean_distance(query, &emb.vector);
                        (emb.id.clone(), dist)
                    })
                    .collect();
                scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                scored.truncate(k);
                Ok(scored)
            }
            Distance::Cosine => {
                let mut scored: Vec<_> = self
                    .embeddings
                    .iter()
                    .map(|emb| {
                        let dp = simd_dot_product(query, &emb.vector);
                        (emb.id.clone(), dp)
                    })
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                scored.truncate(k);
                Ok(scored)
            }
            Distance::DotProduct => {
                let mut scored: Vec<_> = self
                    .embeddings
                    .iter()
                    .map(|emb| {
                        let dp = simd_dot_product(query, &emb.vector);
                        (emb.id.clone(), dp)
                    })
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                scored.truncate(k);
                Ok(scored)
            }
            Distance::Binary => {
                let query_binary = compress_vector(query);
                let mut scored: Vec<_> = self
                    .embeddings
                    .iter()
                    .map(|emb| {
                        let d = if let Some(ref emb_bin) = emb.binary {
                            hamming_distance(&query_binary, emb_bin)
                        } else {
                            let computed = compress_vector(&emb.vector);
                            hamming_distance(&query_binary, &computed)
                        };
                        (emb.id.clone(), d as f32)
                    })
                    .collect();
                scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                scored.truncate(k);
                Ok(scored)
            }
        }
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
fn new_db() -> ResourceArc<DBResource> {
    let db = CacheDB::new();
    ResourceArc::new(DBResource(Mutex::new(db)))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn nif_create_collection(
    db_res: ResourceArc<DBResource>,
    name: String,
    dimension: usize,
    distance: String,
    keep_embeddings: bool,
) -> Result<String, String> {
    let mut db = db_res.0.lock().map_err(|_| "Mutex poisoned")?;

    if db.collections.contains_key(&name) {
        return Err(format!("Collection '{}' already exists", name));
    }

    // Create collection with the chosen distance
    let mut col = Collection::create_with_distance(dimension, &distance)?;

    // Set keep_embeddings
    col.keep_embeddings = keep_embeddings;

    db.collections.insert(name.clone(), col);
    Ok(name)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn delete_collection(db_res: ResourceArc<DBResource>, name: String) -> Result<String, String> {
    let mut db = db_res.0.lock().map_err(|_| "Mutex poisoned")?;
    if db.collections.remove(&name).is_none() {
        return Err(format!("Collection '{}' not found; cannot delete", name));
    }
    Ok(name)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn nif_insert_embedding(
    db_res: ResourceArc<DBResource>,
    collection_name: String,
    id: String,
    vector: Vec<f32>,
    metadata: Option<HashMap<String, String>>,
) -> Result<String, String> {
    let mut db = db_res.0.lock().map_err(|_| "Mutex poisoned")?;
    let coll = db
        .collections
        .get_mut(&collection_name)
        .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

    let emb = Embedding {
        id: id.clone(),
        vector,
        metadata,
        binary: None,
    };
    coll.insert_embedding(emb)?;

    Ok(id)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn nif_insert_embeddings(
    db_res: ResourceArc<DBResource>,
    collection_name: String,
    embeddings: Vec<(String, Vec<f32>, Option<HashMap<String, String>>)>,
) -> Result<Vec<String>, String> {
    let mut db = db_res.0.lock().map_err(|_| "Mutex poisoned")?;
    let coll = db
        .collections
        .get_mut(&collection_name)
        .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

    let mut inserted_ids = Vec::new();

    for (id, vector, metadata) in embeddings {
        let emb = Embedding {
            id: id.clone(),
            vector,
            metadata,
            binary: None,
        };
        // If any fails, we stop
        coll.insert_embedding(emb)?;
        inserted_ids.push(id);
    }

    // Return {:ok, ["id1", "id2", ...]}
    Ok(inserted_ids)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn nif_similarity_search_with_filter(
    db_res: ResourceArc<DBResource>,
    collection_name: String,
    query: Vec<f32>,
    k: usize,
    filter: HashMap<String, String>,
) -> Result<Vec<(String, f32)>, String> {
    let db = db_res.0.lock().map_err(|_| "Mutex poisoned")?;
    let coll = db
        .collections
        .get(&collection_name)
        .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

    // If the collection uses HNSW, we just return an error:
    if matches!(coll.distance, Distance::Hnsw) {
        return Err("Filtering by metadata is not supported with HNSW".into());
    }

    // Otherwise, filter the embeddings by metadata
    let filtered: Vec<&Embedding> = coll
        .embeddings
        .iter()
        .filter(|emb| {
            if let Some(md) = &emb.metadata {
                filter.iter().all(|(kf, vf)| md.get(kf) == Some(vf))
            } else {
                false
            }
        })
        .collect();

    if filtered.is_empty() {
        return Ok(vec![]);
    }

    match coll.distance {
        Distance::Euclidean => {
            let mut scored: Vec<(String, f32)> = filtered
                .iter()
                .map(|emb| {
                    let dist = simd_euclidean_distance(&query, &emb.vector);
                    (emb.id.clone(), dist)
                })
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            scored.truncate(k);
            Ok(scored)
        }
        Distance::Cosine => {
            let mut scored: Vec<(String, f32)> = filtered
                .iter()
                .map(|emb| {
                    let dp = simd_dot_product(&query, &emb.vector);
                    (emb.id.clone(), dp)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored.truncate(k);
            Ok(scored)
        }
        Distance::DotProduct => {
            let mut scored: Vec<(String, f32)> = filtered
                .iter()
                .map(|emb| {
                    let dp = simd_dot_product(&query, &emb.vector);
                    (emb.id.clone(), dp)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored.truncate(k);
            Ok(scored)
        }
        Distance::Binary => {
            let query_binary = compress_vector(&query);
            let mut scored: Vec<(String, f32)> = filtered
                .iter()
                .map(|emb| {
                    let d = if let Some(ref emb_bin) = emb.binary {
                        hamming_distance(&query_binary, emb_bin)
                    } else {
                        let computed = compress_vector(&emb.vector);
                        hamming_distance(&query_binary, &computed)
                    };
                    (emb.id.clone(), d as f32)
                })
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            scored.truncate(k);
            Ok(scored)
        }
        // We already returned an Err for HNSW above
        Distance::Hnsw => unreachable!(),
    }
}

#[rustler::nif(schedule = "DirtyCpu")]
fn nif_similarity_search(
    db_res: ResourceArc<DBResource>,
    collection_name: String,
    query: Vec<f32>,
    k: usize,
) -> Result<Vec<(String, f32)>, String> {
    let db = db_res.0.lock().map_err(|_| "Mutex poisoned")?;
    let coll = db
        .collections
        .get(&collection_name)
        .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
    coll.get_similarity(&query, k)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn get_embeddings(
    db_res: ResourceArc<DBResource>,
    collection_name: String,
) -> Result<Vec<(String, Vec<f32>, Option<HashMap<String, String>>)>, String> {
    let db = db_res.0.lock().map_err(|_| "Mutex poisoned")?;
    let coll = db
        .collections
        .get(&collection_name)
        .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
    let out: Vec<_> = coll
        .embeddings
        .iter()
        .map(|e| (e.id.clone(), e.vector.clone(), e.metadata.clone()))
        .collect();
    Ok(out)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn nif_get_embedding_by_id(
    db_res: ResourceArc<DBResource>,
    collection_name: String,
    id: String,
) -> Result<(String, Vec<f32>, Option<HashMap<String, String>>), String> {
    let db_lock = db_res.0.lock().map_err(|_| "Mutex poisoned")?;
    let embedding = db_lock.get_embedding_by_id(&collection_name, &id)?;
    Ok((embedding.id, embedding.vector, embedding.metadata))
}

fn on_load(env: Env, _info: Term) -> bool {
    env.register::<DBResource>().is_ok()
}

rustler::init!("Elixir.Vettore", load = on_load);
