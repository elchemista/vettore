use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use rand::Rng;

use crate::distances::{clamp_0_1, simd_euclidean_distance};
use crate::types::Embedding;

const M: usize = 16; // max connections above layer 0
const M_MAX0: usize = 32; // max connections at layer 0
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
    id: usize,
    distance: f32,
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

#[allow(dead_code)]
pub struct Node {
    id: usize,
    vector: Vec<f32>,
    connections: Vec<Vec<usize>>, // per‑level neighbours
    layer: usize,
}

pub struct HnswIndex {
    nodes: HashMap<usize, Node>,
    entry_point: Option<usize>,
    max_level: usize,
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

    // Generate a random level with an exponential distribution.
    fn random_level(&self) -> usize {
        let mut rng = rand::rng();
        let mut lvl = 0usize;
        while rng.random::<f32>() < self.level_lambda && lvl < self.max_level {
            lvl += 1;
        }
        lvl
    }

    // Insert a new item.
    pub fn add(&mut self, item: VectorItem) -> Result<(), String> {
        if self.nodes.is_empty() {
            let lvl = self.random_level();
            self.nodes.insert(
                item.id,
                Node {
                    id: item.id,
                    vector: item.vector,
                    connections: vec![Vec::new(); lvl + 1],
                    layer: lvl,
                },
            );
            self.entry_point = Some(item.id);
            return Ok(());
        }

        //  Greedy search from the current entry point to find the best epi‑center.
        let node_level = self.random_level();
        let mut curr_ep = self.entry_point.unwrap();
        let mut curr_dist = simd_euclidean_distance(&self.nodes[&curr_ep].vector, &item.vector);
        let top_l = self.nodes[&curr_ep].layer;
        for level in (0..=top_l).rev() {
            if let Some(best) = self.search_layer(curr_ep, &item.vector, level, 1)?.first() {
                if best.distance < curr_dist {
                    curr_ep = best.id;
                    curr_dist = best.distance;
                }
            }
        }

        //  Connect the new node.
        let mut new_conns = vec![Vec::new(); node_level + 1];
        for level in 0..=node_level {
            let neighs = self.search_layer(curr_ep, &item.vector, level, EF_CONSTRUCTION)?;
            let final_neighs = self.select_neighbors(&neighs, level);
            new_conns[level] = final_neighs.iter().map(|n| n.id).collect();
            for n in final_neighs {
                if let Some(node) = self.nodes.get_mut(&n.id) {
                    if level < node.connections.len() {
                        node.connections[level].push(item.id);
                    }
                }
            }
        }

        //  Store the node.
        self.nodes.insert(
            item.id,
            Node {
                id: item.id,
                vector: item.vector,
                connections: new_conns,
                layer: node_level,
            },
        );
        if node_level > self.nodes[&self.entry_point.unwrap()].layer {
            self.entry_point = Some(item.id);
        }
        Ok(())
    }

    // Remove a node and patch links.
    pub fn remove(&mut self, node_id: usize) -> Result<(), String> {
        let conns = self
            .nodes
            .get(&node_id)
            .ok_or_else(|| format!("Node {} not found", node_id))?
            .connections
            .clone();
        for (lvl, ns) in conns.into_iter().enumerate() {
            for n in ns {
                if let Some(node) = self.nodes.get_mut(&n) {
                    if lvl < node.connections.len() {
                        node.connections[lvl].retain(|&x| x != node_id);
                    }
                }
            }
        }
        self.nodes.remove(&node_id);
        if self.entry_point == Some(node_id) {
            self.entry_point = self.nodes.keys().next().copied();
        }
        Ok(())
    }

    // Low‑level search routine confined to a single layer.
    fn search_layer(
        &self,
        entry_id: usize,
        query: &[f32],
        level: usize,
        ef: usize,
    ) -> Result<Vec<Neighbor>, String> {
        let start = self
            .nodes
            .get(&entry_id)
            .ok_or_else(|| format!("Node {} not found", entry_id))?;
        if level >= start.connections.len() {
            return Ok(Vec::new());
        }

        let mut visited = HashSet::new();
        visited.insert(entry_id);

        let mut results = BinaryHeap::new();
        let mut candidates = BinaryHeap::new();

        let init_dist = simd_euclidean_distance(&start.vector, query);
        let init = Neighbor {
            id: entry_id,
            distance: init_dist,
        };
        results.push(init.clone());
        candidates.push(init);

        while let Some(cur) = candidates.pop() {
            let worst = results.peek().map_or(f32::INFINITY, |n| n.distance);
            if cur.distance > worst {
                break;
            }
            let node = &self.nodes[&cur.id];
            if level < node.connections.len() {
                for &nbr in &node.connections[level] {
                    if !visited.insert(nbr) {
                        continue;
                    }
                    let d = simd_euclidean_distance(&self.nodes[&nbr].vector, query);
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

    // Trim to at most `M` or `M_MAX0` neighbours.
    fn select_neighbors(&self, neighs: &[Neighbor], level: usize) -> Vec<Neighbor> {
        let max_conn = if level == 0 { M_MAX0 } else { M };
        let mut v = neighs.to_vec();
        v.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        v.truncate(max_conn);
        v
    }

    // User‑facing top‑k search.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, String> {
        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }
        let ep = self.entry_point.unwrap();
        let mut curr_ep = ep;
        let mut curr_dist = simd_euclidean_distance(&self.nodes[&ep].vector, query);
        let top_l = self.nodes[&ep].layer;
        for level in (1..=top_l).rev() {
            loop {
                let mut changed = false;
                let node = &self.nodes[&curr_ep];
                if level < node.connections.len() {
                    for &nbr in &node.connections[level] {
                        let d = simd_euclidean_distance(&self.nodes[&nbr].vector, query);
                        if d < curr_dist {
                            curr_dist = d;
                            curr_ep = nbr;
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        let mut res = self.search_layer(curr_ep, query, 0, EF_SEARCH)?;
        res.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        Ok(res
            .into_iter()
            .take(k)
            .map(|n| (n.id, n.distance))
            .collect())
    }
}

pub struct HnswIndexWrapper {
    pub index: HnswIndex,
    id_map: HashMap<usize, String>,
    next_id: usize,
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
        let nid = self.next_id;
        self.next_id += 1;
        self.id_map.insert(nid, emb.id.clone());
        self.index.add(VectorItem {
            id: nid,
            vector: emb.vector.clone(),
        })
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>, String> {
        let raw = self.index.search(query, k)?;
        let mut out = Vec::with_capacity(raw.len());
        for (nid, dist) in raw {
            if let Some(sid) = self.id_map.get(&nid) {
                out.push((sid.clone(), clamp_0_1(1.0 / (1.0 + dist))));
            }
        }
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(out)
    }

    pub fn remove_by_str_id(&mut self, sid: &str) -> Result<(), String> {
        let nid = self
            .id_map
            .iter()
            .find_map(|(k, v)| if v == sid { Some(*k) } else { None })
            .ok_or_else(|| format!("ID '{}' not found", sid))?;
        self.index.remove(nid)?;
        self.id_map.remove(&nid);
        Ok(())
    }
}
