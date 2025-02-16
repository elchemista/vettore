use rustler::{Env, Term, ResourceArc};
use std::collections::{HashMap};
use std::sync::Mutex;

// ==============================
// == Distance Enum & DB Setup ==
// ==============================

/// We add a "Hnsw" variant for the approximate nearest neighbor approach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Distance {
    Euclidean,
    Cosine,
    DotProduct,
    Hnsw,
}

/// A single embedding: ID, vector, optional metadata
#[derive(Debug, Clone)]
pub struct Embedding {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<HashMap<String, String>>,
}

/// A collection with dimension, a distance metric, stored embeddings,
/// and optionally an HNSW index (only if `distance == Hnsw`).
#[derive(Debug, Clone)]
pub struct Collection {
    pub dimension: usize,
    pub distance: Distance,
    pub embeddings: Vec<Embedding>,
    pub hnsw_index: Option<HnswIndexWrapper>,
}

/// Our main in-memory DB
#[derive(Debug, Default)]
pub struct CacheDB {
    pub collections: HashMap<String, Collection>,
}

impl CacheDB {
    pub fn new() -> Self {
        CacheDB {
            collections: HashMap::new(),
        }
    }
}

/// The resource type we share with Elixir
pub struct DBResource(pub Mutex<CacheDB>);

/// Implement Rustler Resource
impl rustler::Resource for DBResource {}

// ===============================================
// == SIMD-Optimized Distance Functions (faster)==
// ===============================================

use faster::*; // from the `faster` crate

/// A SIMD Euclidean distance for f32 slices.
fn simd_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    // We'll rely on `a.simd_iter().zip(b.simd_iter())`
    // The leftover elements (if length not multiple of vector width) are handled automatically by `faster`.
    let sum_sq = a
        .simd_iter()
        .zip(b.simd_iter())
        .simd_map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .scalar_sum();

    sum_sq.sqrt()
}

/// A SIMD dot product for f32 slices.
fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.simd_iter()
        .zip(b.simd_iter())
        .simd_map(|(x, y)| x * y)
        .scalar_sum()
}

/// For Cosine, we rely on storing normalized vectors, then we just do dot product.
fn simd_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    // Because vectors are normalized, the "similarity" is just dot product.
    // Typically you'd do '1.0 - dot' to interpret as a "distance," 
    // but let's keep it consistent with your original approach (dot is the "score").
    simd_dot_product(a, b)
}

/// A helper to do scalar-based normalization
fn normalize_f32(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.simd_iter().simd_map(|x| x * x).scalar_sum().sqrt();
    if norm > std::f32::EPSILON {
        vec.iter().map(|x| x / norm).collect()
    } else {
        vec.to_vec()
    }
}

// =========================================
// == HNSW Implementation (with f32 SIMD) ==
// =========================================

use rand::Rng;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

const M: usize = 16;
const M_MAX0: usize = 32;
const EF_CONSTRUCTION: usize = 100;
const EF_SEARCH: usize = 64;
const MAX_LEVEL: usize = 12;

#[derive(Clone, Debug)]
struct VectorItem {
    pub id: usize,
    pub vector: Vec<f32>,
}

#[derive(Clone, Debug)]
struct Neighbor {
    pub id: usize,
    pub distance: f32,
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse to keep a max-heap
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

#[derive(Clone, Debug)]
struct Node {
    pub id: usize,
    pub connections: Vec<Vec<usize>>,
    pub item: VectorItem,
    pub layer: usize,
}

/// A minimal HNSW index struct that uses our SIMD distances.
pub struct HnswIndex {
    pub nodes: HashMap<usize, Node>,
    pub entry_point: Option<usize>,
    level_lambda: f32,
    pub max_level: usize,
}

#[derive(Debug)]
pub struct HnswIndexWrapper {
    pub index: HnswIndex,
    pub id_map: HashMap<usize, String>, // hnsw_id -> real string ID
    pub next_id: usize,
}

impl HnswIndexWrapper {
    pub fn new() -> Self {
        HnswIndexWrapper {
            index: HnswIndex::new(),
            id_map: HashMap::new(),
            next_id: 0,
        }
    }

    /// Insert an embedding incrementally
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

    /// Perform approximate search. Returns (string_id, distance).
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>, String> {
        let results = self.index.search(query, k)?;
        let mut out = Vec::new();
        for (nid, dist) in results {
            if let Some(orig_id) = self.id_map.get(&nid) {
                out.push((orig_id.clone(), dist));
            }
        }
        Ok(out)
    }
}

impl HnswIndex {
    pub fn new() -> Self {
        HnswIndex {
            nodes: HashMap::new(),
            entry_point: None,
            level_lambda: 1.0 / (M as f32).ln(),
            max_level: MAX_LEVEL,
        }
    }

    /// Incremental add
    pub fn add(&mut self, item: VectorItem) -> Result<(), String> {
        let node_id = item.id;
        let node_level = self.random_level();

        // if empty => first node
        if self.nodes.is_empty() {
            let first_node = Node {
                id: node_id,
                connections: vec![Vec::with_capacity(M_MAX0); node_level + 1],
                item,
                layer: node_level,
            };
            self.nodes.insert(node_id, first_node);
            self.entry_point = Some(node_id);
            return Ok(());
        }

        // Otherwise
        let mut curr_ep = self.entry_point.unwrap();
        let mut curr_dist = simd_euclidean_distance(&self.nodes[&curr_ep].item.vector, &item.vector);
        let top_layer = self.nodes[&curr_ep].layer;

        for level in (0..=top_layer).rev() {
            let improved = self.search_layer(curr_ep, &item.vector, level, 1)?;
            if !improved.is_empty() {
                let best = &improved[0];
                if best.distance < curr_dist {
                    curr_dist = best.distance;
                    curr_ep = best.id;
                }
            }
        }

        // insert each layer
        let mut new_conn = vec![Vec::new(); node_level + 1];
        for level in 0..=node_level {
            let neighbors = self.search_layer(curr_ep, &item.vector, level, EF_CONSTRUCTION)?;
            let final_neighbors = self.select_neighbors(&neighbors, &item.vector, level);
            new_conn[level] = final_neighbors.iter().map(|n| n.id).collect();

            for neigh in &final_neighbors {
                if let Some(nod) = self.nodes.get_mut(&neigh.id) {
                    if level < nod.connections.len() {
                        nod.connections[level].push(node_id);
                    }
                }
            }
        }

        let new_node = Node {
            id: node_id,
            connections: new_conn,
            item,
            layer: node_level,
        };
        self.nodes.insert(node_id, new_node);

        // maybe update entry point
        let ep_level = self.nodes[&self.entry_point.unwrap()].layer;
        if node_level > ep_level {
            self.entry_point = Some(node_id);
        }
        Ok(())
    }

    /// Approximate search
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, String> {
        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }
        let ep = self.entry_point.unwrap();
        let mut curr_ep = ep;
        let mut curr_dist = simd_euclidean_distance(&self.nodes[&ep].item.vector, query);
        let top_layer = self.nodes[&ep].layer;

        for level in (1..=top_layer).rev() {
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
        let topk = sorted.into_iter().take(k)
            .map(|n| (n.id, n.distance))
            .collect();
        Ok(topk)
    }

    fn search_layer(
        &self,
        entry_id: usize,
        query: &[f32],
        level: usize,
        ef: usize
    ) -> Result<Vec<Neighbor>, String> {
        let start_node = self.nodes.get(&entry_id)
            .ok_or_else(|| format!("No node: {}", entry_id))?;

        if level >= start_node.connections.len() {
            return Ok(Vec::new());
        }

        let dist = simd_euclidean_distance(&start_node.item.vector, query);
        let initial = Neighbor{ id: entry_id, distance: dist };
        let mut visited = HashSet::new();
        visited.insert(entry_id);

        let mut candidates = BinaryHeap::new();
        candidates.push(initial.clone());

        let mut results = BinaryHeap::new();
        results.push(initial);

        while let Some(current) = candidates.pop() {
            let worst_dist = results.peek().map_or(f32::INFINITY, |n| n.distance);
            if current.distance > worst_dist {
                break;
            }
            if let Some(node) = self.nodes.get(&current.id) {
                if level < node.connections.len() {
                    for &nbr_id in &node.connections[level] {
                        if !visited.insert(nbr_id) { continue; }
                        let d = simd_euclidean_distance(&self.nodes[&nbr_id].item.vector, query);
                        let neighbor = Neighbor { id: nbr_id, distance: d };
                        let worst_dist2 = results.peek().map_or(f32::INFINITY, |n| n.distance);
                        if results.len() < ef || d < worst_dist2 {
                            candidates.push(neighbor.clone());
                            results.push(neighbor);
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }
        Ok(results.into_sorted_vec())
    }

    fn select_neighbors(&self, neighbors: &[Neighbor], query: &[f32], level: usize) -> Vec<Neighbor> {
        let max_conn = if level == 0 { M_MAX0 } else { M };
        let mut best = neighbors.to_vec();
        best.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        best.truncate(max_conn);
        best
    }

    fn random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let mut level = 0usize;
        while rng.gen::<f32>() < self.level_lambda && level < self.max_level {
            level += 1;
        }
        level
    }
}

// ===========================================
// == Insert & Search Logic in Collections ==
// ===========================================

impl Collection {
    /// Create a new collection with the specified distance (including "hnsw").
    pub fn create_with_distance(
        _name: &str,
        dimension: usize,
        distance: &str
    ) -> Result<Self, String> {
        let dist = match distance {
            "euclidean" => Distance::Euclidean,
            "cosine" => Distance::Cosine,
            "dot" => Distance::DotProduct,
            "hnsw" => Distance::Hnsw,
            other => return Err(format!("Unknown distance '{}'", other)),
        };
        let mut col = Collection {
            dimension,
            distance: dist,
            embeddings: Vec::new(),
            hnsw_index: None,
        };

        // If "hnsw" => create an empty HnswIndexWrapper
        if dist == Distance::Hnsw {
            col.hnsw_index = Some(HnswIndexWrapper::new());
        }
        Ok(col)
    }

    /// Insert an embedding. If "hnsw", also add to the HNSW index.
    pub fn insert_embedding(&mut self, embedding: Embedding) -> Result<(), String> {
        if embedding.vector.len() != self.dimension {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension, embedding.vector.len()
            ));
        }
        if self.embeddings.iter().any(|e| e.id == embedding.id) {
            return Err(format!("Embedding ID '{}' already exists", embedding.id));
        }

        // If Cosine, we store normalized vectors
        let mut emb_mod = embedding.clone();
        if self.distance == Distance::Cosine {
            emb_mod.vector = normalize_f32(&emb_mod.vector);
        }

        self.embeddings.push(emb_mod.clone());

        // If distance is Hnsw => insert into the index
        if self.distance == Distance::Hnsw {
            if let Some(ref mut wrapper) = self.hnsw_index {
                wrapper.insert_embedding(&emb_mod)?;
            }
        }
        Ok(())
    }

    /// get_similarity just calls HNSW if distance=Hnsw, otherwise does a linear approach with SIMD.
    pub fn get_similarity(
        &self,
        query: &[f32],
        k: usize
    ) -> Result<Vec<(String, f32)>, String> {
        match self.distance {
            Distance::Hnsw => {
                if let Some(ref hnsw) = self.hnsw_index {
                    hnsw.search(query, k)
                } else {
                    Err("No HNSW index found".to_string())
                }
            }
            Distance::Euclidean => {
                let mut scored: Vec<(String, f32)> = self.embeddings
                    .iter()
                    .map(|emb| {
                        let dist = simd_euclidean_distance(query, &emb.vector);
                        (emb.id.clone(), dist)
                    })
                    .collect();
                // smaller = more similar
                scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                scored.truncate(k);
                Ok(scored)
            }
            Distance::DotProduct => {
                let mut scored: Vec<(String, f32)> = self.embeddings
                    .iter()
                    .map(|emb| {
                        let dp = simd_dot_product(query, &emb.vector);
                        (emb.id.clone(), dp)
                    })
                    .collect();
                // bigger = more similar
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                scored.truncate(k);
                Ok(scored)
            }
            Distance::Cosine => {
                let mut scored: Vec<(String, f32)> = self.embeddings
                    .iter()
                    .map(|emb| {
                        // Because stored vectors are normalized,
                        // dot product = similarity
                        let dot = simd_cosine_distance(query, &emb.vector);
                        (emb.id.clone(), dot)
                    })
                    .collect();
                // bigger = more similar
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                scored.truncate(k);
                Ok(scored)
            }
        }
    }
}

// ===========================
// == NIF Exported Functions ==
// ===========================

#[rustler::nif]
fn new_db() -> ResourceArc<DBResource> {
    ResourceArc::new(DBResource(Mutex::new(CacheDB::new())))
}

#[rustler::nif]
fn create_collection(
    db_res: ResourceArc<DBResource>,
    name: String,
    dimension: usize,
    distance: String,
) -> Result<(), String> {
    let mut db = db_res.0.lock().map_err(|_| "Mutex poisoned")?;
    if db.collections.contains_key(&name) {
        return Err(format!("Collection '{}' already exists", name));
    }
    let col = Collection::create_with_distance(&name, dimension, &distance)?;
    db.collections.insert(name, col);
    Ok(())
}

#[rustler::nif]
fn insert_embedding(
    db_res: ResourceArc<DBResource>,
    collection_name: String,
    id: String,
    vector: Vec<f32>,
    metadata: Option<HashMap<String, String>>,
) -> Result<(), String> {
    let mut db = db_res.0.lock().map_err(|_| "Mutex poisoned")?;
    let collection = db.collections.get_mut(&collection_name)
        .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

    let emb = Embedding { id, vector, metadata };
    collection.insert_embedding(emb)
}

#[rustler::nif]
fn similarity_search(
    db_res: ResourceArc<DBResource>,
    collection_name: String,
    query: Vec<f32>,
    k: usize,
) -> Result<Vec<(String, f32)>, String> {
    let db = db_res.0.lock().map_err(|_| "Mutex poisoned")?;
    let coll = db.collections.get(&collection_name)
        .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
    coll.get_similarity(&query, k)
}

#[rustler::nif]
fn get_embeddings(
    db_res: ResourceArc<DBResource>,
    collection_name: String,
) -> Result<Vec<(String, Vec<f32>, Option<HashMap<String, String>>)>, String> {
    let db = db_res.0.lock().map_err(|_| "Mutex poisoned")?;
    let coll = db.collections.get(&collection_name)
        .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;

    let out = coll.embeddings
        .iter()
        .map(|e| (e.id.clone(), e.vector.clone(), e.metadata.clone()))
        .collect();
    Ok(out)
}

fn on_load(_env: Env, _info: Term) -> bool {
    true
}

/// Register the module
rustler::init!(
    "Elixir.Vettore",
    load = on_load
);
