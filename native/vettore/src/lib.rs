use rustler::{Env, ResourceArc, Term};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::{BinaryHeap, HashSet};
use std::sync::Mutex;

use rand::rng;
use rand::Rng;
use wide::f32x4;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Distance {
    Euclidean,
    Cosine,
    DotProduct,
    Hnsw,
    Binary,
}

#[derive(Clone)]
pub struct Embedding {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<HashMap<String, String>>,
    pub binary: Option<Vec<u64>>,
}

pub struct Collection {
    pub dimension: usize,
    pub distance: Distance,
    pub embeddings: Vec<Embedding>,
    pub hnsw_index: Option<HnswIndexWrapper>,
}

pub struct CacheDB {
    pub collections: HashMap<String, Collection>,
}

impl CacheDB {
    pub fn new() -> Self {
        Self {
            collections: HashMap::new(),
        }
    }

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

pub struct DBResource(pub Mutex<CacheDB>);

impl rustler::Resource for DBResource {}

#[inline]
fn load_f32x4(slice: &[f32], i: usize) -> f32x4 {
    f32x4::from([slice[i], slice[i + 1], slice[i + 2], slice[i + 3]])
}

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
    while i < len {
        let d = a[i] - b[i];
        sum_squares += d * d;
        i += 1;
    }
    sum_squares.sqrt()
}

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
    while i < len {
        accum += a[i] * b[i];
        i += 1;
    }
    accum
}

fn compress_vector(vector: &[f32]) -> Vec<u64> {
    let mut result = Vec::new();
    let mut current: u64 = 0;
    let mut bits_filled = 0;
    for &val in vector {
        current <<= 1;
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
        current <<= 64 - bits_filled;
        result.push(current);
    }
    result
}

fn hamming_distance(a: &[u64], b: &[u64]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

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
        v.to_vec()
    }
}

const M: usize = 16;
const M_MAX0: usize = 32;
const EF_CONSTRUCTION: usize = 100;
const EF_SEARCH: usize = 64;
const MAX_LEVEL: usize = 12;

#[derive(Clone)]
pub struct VectorItem {
    pub id: usize,
    pub vector: Vec<f32>,
}

#[derive(Clone)]
struct Neighbor {
    pub id: usize,
    pub distance: f32,
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
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
    connections: Vec<Vec<usize>>,
    item: VectorItem,
    layer: usize,
}

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

    // NEW: approximate search with metadata filter. We do a standard BFS but
    // then only keep the matches that pass the filter. We expand with EF_SEARCH
    // so hopefully we get enough matching items. Then from those matching items
    // we return up to k.
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: &HashMap<String, String>,
        all_embeddings: &[Embedding],
        id_map: &HashMap<usize, String>,
    ) -> Result<Vec<(String, f32)>, String> {
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

        // Do a BFS at level 0 with EF_SEARCH
        let ef = EF_SEARCH;
        let results = self.search_layer(curr_ep, query, 0, ef)?;
        let mut sorted = results;
        sorted.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        // Now filter out those that don't match
        let mut final_hits = Vec::new();
        for neighbor in sorted {
            let node_id = neighbor.id;
            if let Some(real_id) = id_map.get(&node_id) {
                // We'll find the actual embedding in all_embeddings
                // (We do a simple .iter().find() for demonstration, but you could
                //  store a separate map from node_id -> index if you want faster lookups.)
                if let Some(emb) = all_embeddings.iter().find(|e| e.id == *real_id) {
                    if matches_filter(emb, filter) {
                        final_hits.push((emb.id.clone(), neighbor.distance));
                        if final_hits.len() == k {
                            break;
                        }
                    }
                }
            }
        }
        Ok(final_hits)
    }
}

// Simple helper to check if an embedding's metadata satisfies all filter pairs
fn matches_filter(emb: &Embedding, filter: &HashMap<String, String>) -> bool {
    if let Some(md) = &emb.metadata {
        for (fk, fv) in filter {
            if md.get(fk) != Some(fv) {
                return false;
            }
        }
        true
    } else {
        false
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

    // NEW: used for HNSW + filter
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: &HashMap<String, String>,
        all_embeddings: &[Embedding],
    ) -> Result<Vec<(String, f32)>, String> {
        self.index
            .search_with_filter(query, k, filter, all_embeddings, &self.id_map)
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

    // NEW: same idea but with metadata filtering
    pub fn get_similarity_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: &HashMap<String, String>,
    ) -> Result<Vec<(String, f32)>, String> {
        match self.distance {
            Distance::Hnsw => {
                if let Some(ref w) = self.hnsw_index {
                    w.search_with_filter(query, k, filter, &self.embeddings)
                } else {
                    Err("No HNSW index found in this collection".to_string())
                }
            }
            Distance::Euclidean => {
                let mut scored: Vec<_> = self
                    .embeddings
                    .iter()
                    .filter(|emb| matches_filter(emb, filter))
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
                    .filter(|emb| matches_filter(emb, filter))
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
                    .filter(|emb| matches_filter(emb, filter))
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
                    .filter(|emb| matches_filter(emb, filter))
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

#[rustler::nif]
fn new_db() -> ResourceArc<DBResource> {
    let db = CacheDB::new();
    ResourceArc::new(DBResource(Mutex::new(db)))
}

#[rustler::nif]
fn create_collection(
    db_res: ResourceArc<DBResource>,
    name: String,
    dimension: usize,
    distance: String,
) -> Result<String, String> {
    let mut db = db_res.0.lock().map_err(|_| "Mutex poisoned")?;
    if db.collections.contains_key(&name) {
        return Err(format!("Collection '{}' already exists", name));
    }
    let c = Collection::create_with_distance(dimension, &distance)?;
    db.collections.insert(name.clone(), c);

    Ok(name)
}

#[rustler::nif]
fn delete_collection(db_res: ResourceArc<DBResource>, name: String) -> Result<String, String> {
    let mut db = db_res.0.lock().map_err(|_| "Mutex poisoned")?;
    if db.collections.remove(&name).is_none() {
        return Err(format!("Collection '{}' not found; cannot delete", name));
    }
    Ok(name)
}

#[rustler::nif]
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
        coll.insert_embedding(emb)?;
        inserted_ids.push(id);
    }
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

    coll.get_similarity_with_filter(&query, k, &filter)
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

#[rustler::nif]
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

#[rustler::nif]
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
