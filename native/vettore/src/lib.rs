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
    pub binary: Option<Vec<u64>>, // for "binary" distance only
}

pub struct Collection {
    pub dimension: usize,
    pub keep_embeddings: bool,
    pub distance: Distance,
    pub embeddings: Vec<Embedding>,
    pub hnsw_index: Option<HnswIndexWrapper>,
}

pub struct CacheDB {
    pub collections: HashMap<String, Collection>,
}

pub struct DBResource(pub Mutex<CacheDB>);
impl rustler::Resource for DBResource {}

// Ensure vector is not empty and matches the collection's dimension.
fn check_vector_dimension(vec: &[f32], expected_dim: usize) -> Result<(), String> {
    if expected_dim == 0 {
        return Err("Collection dimension cannot be 0.".to_string());
    }
    if vec.is_empty() {
        return Err("Cannot insert an empty embedding vector.".to_string());
    }
    if vec.len() != expected_dim {
        return Err(format!(
            "Dimension mismatch: expected {}, got {}",
            expected_dim,
            vec.len()
        ));
    }
    Ok(())
}

// Generic clamp helper, ensures val is in [0..1].
fn clamp_0_1(val: f32) -> f32 {
    if val < 0.0 {
        0.0
    } else if val > 1.0 {
        1.0
    } else {
        val
    }
}

#[inline]
fn load_f32x4(slice: &[f32], i: usize) -> f32x4 {
    f32x4::from([slice[i], slice[i + 1], slice[i + 2], slice[i + 3]])
}

// SIMD Euclidean distance
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

// SIMD dot product
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

// Normalizing for Cosine distance (so the vector has length ~1)
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
        // if extremely small norm, just return original
        v.to_vec()
    }
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

// Convert raw distance / dot product → a 0..1 "score".
fn compute_0_1_score(query: &[f32], emb: &Embedding, dist_type: Distance) -> f32 {
    match dist_type {
        Distance::Euclidean => {
            let d = simd_euclidean_distance(query, &emb.vector);
            let score = 1.0 / (1.0 + d);
            clamp_0_1(score)
        }
        Distance::Cosine => {
            let cos = simd_dot_product(query, &emb.vector);
            clamp_0_1((cos + 1.0) / 2.0)
        }
        Distance::DotProduct => {
            let dp = simd_dot_product(query, &emb.vector);
            let score = 1.0 / (1.0 + f32::exp(-dp));
            clamp_0_1(score)
        }
        Distance::Hnsw => {
            // internally Euclidean
            let d = simd_euclidean_distance(query, &emb.vector);
            let score = 1.0 / (1.0 + d);
            clamp_0_1(score)
        }
        Distance::Binary => {
            let qbits = compress_vector(query);
            if let Some(b) = &emb.binary {
                let d_bits = hamming_distance(&qbits, b) as f32;
                let fraction = clamp_0_1(d_bits / query.len() as f32);
                1.0 - fraction
            } else {
                let temp = compress_vector(&emb.vector);
                let d_bits = hamming_distance(&qbits, &temp) as f32;
                let fraction = clamp_0_1(d_bits / query.len() as f32);
                1.0 - fraction
            }
        }
    }
}

// Compare two embeddings → a 0..1 similarity, used in MMR.
fn compute_0_1_similarity_between(emb1: &Embedding, emb2: &Embedding, dist_type: Distance) -> f32 {
    let v1 = &emb1.vector;
    let v2 = &emb2.vector;

    match dist_type {
        Distance::Euclidean => {
            let d = simd_euclidean_distance(v1, v2);
            clamp_0_1(1.0 / (1.0 + d))
        }
        Distance::Cosine => {
            let cos = simd_dot_product(v1, v2);
            clamp_0_1((cos + 1.0) / 2.0)
        }
        Distance::DotProduct => {
            let dp = simd_dot_product(v1, v2);
            clamp_0_1(1.0 / (1.0 + f32::exp(-dp)))
        }
        Distance::Hnsw => {
            let d = simd_euclidean_distance(v1, v2);
            clamp_0_1(1.0 / (1.0 + d))
        }
        Distance::Binary => match (emb1.binary.as_ref(), emb2.binary.as_ref()) {
            (Some(b1), Some(b2)) => {
                let d_bits = hamming_distance(b1, b2) as f32;
                let frac = clamp_0_1(d_bits / emb1.vector.len() as f32);
                1.0 - frac
            }
            (Some(b1), None) => {
                let temp2 = compress_vector(&emb2.vector);
                let d_bits = hamming_distance(b1, &temp2) as f32;
                let frac = clamp_0_1(d_bits / emb1.vector.len() as f32);
                1.0 - frac
            }
            (None, Some(b2)) => {
                let temp1 = compress_vector(&emb1.vector);
                let d_bits = hamming_distance(&temp1, b2) as f32;
                let frac = clamp_0_1(d_bits / emb1.vector.len() as f32);
                1.0 - frac
            }
            (None, None) => {
                let temp1 = compress_vector(&emb1.vector);
                let temp2 = compress_vector(&emb2.vector);
                let d_bits = hamming_distance(&temp1, &temp2) as f32;
                let frac = clamp_0_1(d_bits / emb1.vector.len() as f32);
                1.0 - frac
            }
        },
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
        // reverse order: smaller distance => higher priority
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
        if self.nodes.is_empty() {
            // first node
            let node_level = self.random_level();
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

        let node_level = self.random_level();
        let mut curr_ep = self.entry_point.unwrap();
        let mut curr_dist =
            simd_euclidean_distance(&self.nodes[&curr_ep].item.vector, &item.vector);
        let top_l = self.nodes[&curr_ep].layer;

        // search from top
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

        // link
        let mut new_conn = vec![Vec::new(); node_level + 1];
        for level in 0..=node_level {
            let neighbors = self.search_layer(curr_ep, &item.vector, level, EF_CONSTRUCTION)?;

            let final_neighbors = self.select_neighbors(&neighbors, level);
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

    pub fn remove(&mut self, node_id: usize) -> Result<(), String> {
        let connections = {
            let node = self
                .nodes
                .get(&node_id)
                .ok_or_else(|| format!("Node {} not found in HNSW index", node_id))?;
            node.connections.clone()
        };

        for (level, neighbors) in connections.into_iter().enumerate() {
            for &nbr_id in &neighbors {
                if let Some(nbr_node) = self.nodes.get_mut(&nbr_id) {
                    if level < nbr_node.connections.len() {
                        nbr_node.connections[level].retain(|&x| x != node_id);
                    }
                }
            }
        }

        // Remove the node itself
        self.nodes.remove(&node_id);

        if self.entry_point == Some(node_id) {
            self.entry_point = None;
        }
        if self.entry_point.is_none() {
            if let Some((&some_id, _)) = self.nodes.iter().next() {
                self.entry_point = Some(some_id);
            }
        }

        Ok(())
    }

    fn random_level(&self) -> usize {
        let mut r = rng();
        let mut lvl = 0usize;
        while r.random::<f32>() < self.level_lambda && lvl < self.max_level {
            lvl += 1;
        }
        lvl
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

    fn select_neighbors(&self, neighbors: &[Neighbor], level: usize) -> Vec<Neighbor> {
        let max_conn = if level == 0 { M_MAX0 } else { M };
        let mut sorted = neighbors.to_vec();
        sorted.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        if sorted.len() > max_conn {
            sorted.truncate(max_conn);
        }
        sorted
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

    // Perform the HNSW search. This returns raw Euclidean distances,
    // which we must convert to 0..1 scores.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>, String> {
        let results = self.index.search(query, k)?;
        let mut out = Vec::new();
        for (nid, dist) in results {
            if let Some(str_id) = self.id_map.get(&nid) {
                let score = clamp_0_1(1.0 / (1.0 + dist));
                out.push((str_id.clone(), score));
            }
        }
        // sort by descending score
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(out)
    }

    pub fn remove_by_str_id(&mut self, emb_id: &str) -> Result<(), String> {
        // Reverse-lookup the numeric ID
        let mut found_nid = None;
        for (nid, id_string) in &self.id_map {
            if id_string == emb_id {
                found_nid = Some(*nid);
                break;
            }
        }
        let node_id =
            found_nid.ok_or_else(|| format!("ID '{}' not found in HNSW index", emb_id))?;

        // Remove from the core index
        self.index.remove(node_id)?;

        // Remove from id_map
        self.id_map.remove(&node_id);

        Ok(())
    }
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
            .ok_or_else(|| format!("Collection '{}' not found.", collection_name))?;

        coll.embeddings
            .iter()
            .find(|emb| emb.id == id)
            .cloned()
            .ok_or_else(|| {
                format!(
                    "Embedding '{}' not found in collection '{}'.",
                    id, collection_name
                )
            })
    }
}

impl Collection {
    pub fn create_with_distance(dimension: usize, dist_str: &str) -> Result<Self, String> {
        if dimension == 0 {
            return Err("Collection dimension cannot be 0.".to_string());
        }
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
            keep_embeddings: true,
            hnsw_index: None,
        };

        if distance == Distance::Hnsw {
            col.hnsw_index = Some(HnswIndexWrapper::new());
        }
        Ok(col)
    }

    pub fn insert_embedding(&mut self, emb: Embedding) -> Result<(), String> {
        check_vector_dimension(&emb.vector, self.dimension)?;

        // ID uniqueness check
        if self.embeddings.iter().any(|e| e.id == emb.id) {
            return Err(format!(
                "Embedding ID '{}' already exists in this collection.",
                emb.id
            ));
        }

        let mut new_emb = emb;
        if self.distance == Distance::Cosine {
            new_emb.vector = normalize_vec(&new_emb.vector);
        }
        if self.distance == Distance::Binary {
            new_emb.binary = Some(compress_vector(&new_emb.vector));
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
        check_vector_dimension(query, self.dimension)?;

        match self.distance {
            Distance::Hnsw => {
                if let Some(ref wrapper) = self.hnsw_index {
                    wrapper.search(query, k)
                } else {
                    Err("No HNSW index found in this collection.".to_string())
                }
            }
            _ => {
                let mut scored: Vec<_> = self
                    .embeddings
                    .iter()
                    .map(|emb| {
                        let s = compute_0_1_score(query, emb, self.distance);
                        (emb.id.clone(), s)
                    })
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                scored.truncate(k);
                Ok(scored)
            }
        }
    }

    pub fn remove_embedding_by_id(&mut self, emb_id: &str) -> Result<(), String> {
        let pos = self
            .embeddings
            .iter()
            .position(|emb| emb.id == emb_id)
            .ok_or_else(|| format!("Embedding '{}' not found in this collection.", emb_id))?;
        self.embeddings.remove(pos);

        if self.distance == Distance::Hnsw {
            if let Some(ref mut w) = self.hnsw_index {
                w.remove_by_str_id(emb_id)?;
            }
        }

        Ok(())
    }
}

fn mmr_rerank_internal(
    collection: &Collection,
    query: &[f32],
    initial_ids: &[String],
    alpha: f32,
    final_k: usize,
) -> Result<Vec<(String, f32)>, String> {
    // Build a map from ID -> embedding for quick lookup
    let mut emb_map = HashMap::new();
    for e in &collection.embeddings {
        emb_map.insert(e.id.clone(), e);
    }

    // gather candidates by re-computing sim to query
    let mut candidates = Vec::new();
    for id in initial_ids {
        if let Some(emb) = emb_map.get(id) {
            let sim_q = compute_0_1_similarity_between(
                emb,
                &Embedding {
                    id: "QUERY".to_string(),
                    vector: query.to_vec(),
                    metadata: None,
                    binary: None,
                },
                collection.distance,
            );
            candidates.push((id.clone(), sim_q));
        }
    }

    let mut selected = Vec::<(String, f32)>::new();
    let mut selected_ids = Vec::<String>::new();

    while selected.len() < final_k && !candidates.is_empty() {
        let mut best_idx: Option<usize> = None;
        let mut best_score = f32::MIN;

        for (idx, (cand_id, cand_sim_to_query)) in candidates.iter().enumerate() {
            // max sim to any selected so far
            let mut max_sim_cand_sel = 0.0;
            if !selected_ids.is_empty() {
                let cand_emb = emb_map.get(cand_id).unwrap();
                for sel_id in &selected_ids {
                    let sel_emb = emb_map.get(sel_id).unwrap();
                    let sim_cs =
                        compute_0_1_similarity_between(cand_emb, sel_emb, collection.distance);
                    if sim_cs > max_sim_cand_sel {
                        max_sim_cand_sel = sim_cs;
                    }
                }
            }
            let mmr_score = alpha * cand_sim_to_query - (1.0 - alpha) * max_sim_cand_sel;
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

    Ok(selected)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn new_db() -> ResourceArc<DBResource> {
    ResourceArc::new(DBResource(Mutex::new(CacheDB::new())))
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
        return Err(format!("Collection '{}' already exists.", name));
    }

    let mut col = Collection::create_with_distance(dimension, &distance)?;
    col.keep_embeddings = keep_embeddings;

    db.collections.insert(name.clone(), col);
    Ok(name)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn delete_collection(db_res: ResourceArc<DBResource>, name: String) -> Result<String, String> {
    let mut db = db_res.0.lock().map_err(|_| "Mutex poisoned")?;
    if db.collections.remove(&name).is_none() {
        return Err(format!("Collection '{}' not found; cannot delete.", name));
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
        .ok_or_else(|| format!("Collection '{}' not found.", collection_name))?;

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
        .ok_or_else(|| format!("Collection '{}' not found.", collection_name))?;

    let mut inserted_ids = Vec::new();
    for (id, vec, meta) in embeddings {
        let emb = Embedding {
            id: id.clone(),
            vector: vec,
            metadata: meta,
            binary: None,
        };
        coll.insert_embedding(emb)?;
        inserted_ids.push(id);
    }
    Ok(inserted_ids)
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
        .ok_or_else(|| format!("Collection '{}' not found.", collection_name))?;

    let results = coll.get_similarity(&query, k)?;
    Ok(results)
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
        .ok_or_else(|| format!("Collection '{}' not found.", collection_name))?;

    // not supported for HNSW
    if matches!(coll.distance, Distance::Hnsw) {
        return Err("Filtering by metadata is not supported with HNSW.".into());
    }

    check_vector_dimension(&query, coll.dimension)?;

    // filter first
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

    let mut scored: Vec<(String, f32)> = filtered
        .iter()
        .map(|emb| {
            let s = compute_0_1_score(&query, emb, coll.distance);
            (emb.id.clone(), s)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(k);
    Ok(scored)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn nif_get_embedding_by_id(
    db_res: ResourceArc<DBResource>,
    collection_name: String,
    id: String,
) -> Result<(String, Vec<f32>, Option<HashMap<String, String>>), String> {
    let db = db_res.0.lock().map_err(|_| "Mutex poisoned")?;
    let embedding = db.get_embedding_by_id(&collection_name, &id)?;
    Ok((embedding.id, embedding.vector, embedding.metadata))
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
        .ok_or_else(|| format!("Collection '{}' not found.", collection_name))?;

    let out = coll
        .embeddings
        .iter()
        .map(|e| (e.id.clone(), e.vector.clone(), e.metadata.clone()))
        .collect();
    Ok(out)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn nif_delete_embedding_by_id(
    db_res: ResourceArc<DBResource>,
    collection_name: String,
    embedding_id: String,
) -> Result<String, String> {
    let mut db = db_res.0.lock().map_err(|_| "Mutex poisoned")?;

    let coll = db
        .collections
        .get_mut(&collection_name)
        .ok_or_else(|| format!("Collection '{}' not found.", collection_name))?;

    coll.remove_embedding_by_id(&embedding_id)?;
    Ok(embedding_id)
}

// initial_results is a list of (ID, _score_or_distance)
// re-lookup the embeddings to do real MMR in [0..1]
#[rustler::nif(schedule = "DirtyCpu")]
fn nif_mmr_rerank(
    db_res: ResourceArc<DBResource>,
    collection_name: String,
    initial_results: Vec<(String, f32)>,
    alpha: f32,
    final_k: usize,
) -> Result<Vec<(String, f32)>, String> {
    let db = db_res.0.lock().map_err(|_| "Mutex poisoned")?;
    let coll = db
        .collections
        .get(&collection_name)
        .ok_or_else(|| format!("Collection '{}' not found.", collection_name))?;

    // reuse the IDs from initial_results
    let ids: Vec<String> = initial_results.iter().map(|(id, _)| id.clone()).collect();
    let reranked = mmr_rerank_internal(coll, &[], &ids, alpha, final_k)?;
    Ok(reranked)
}

fn on_load(env: Env, _info: Term) -> bool {
    env.register::<DBResource>().is_ok()
}

rustler::init!("Elixir.Vettore", load = on_load);
