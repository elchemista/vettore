//! HNSW – minimal wrapper used by Vettore
//! --------------------------------------
//! • identical public API
//! • extra “existence” checks when following neighbour ids so that look-ups
//!   like `self.nodes[&nb]` are never attempted for a removed node.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use rand::Rng;
use smallvec::SmallVec;

use crate::distances::{clamp_0_1, simd_euclidean_distance};
use crate::types::Distance;

/* ───────────────────────── constants ───────────────────────── */
const M: usize = 16;
const M0: usize = 32;
const EF_CONSTRUCTION: usize = 100;
const EF_SEARCH: usize = 64;
const MAX_LEVEL: usize = 12;

/* ───────────────────────── helper types ────────────────────── */
#[derive(Clone)]
struct Neighbor {
    id: usize,
    dist: f32,
}
impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap().reverse() // max-heap
    }
}
impl PartialOrd for Neighbor {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl PartialEq for Neighbor {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist
    }
}
impl Eq for Neighbor {}

struct Node {
    vector: Vec<f32>,
    connections: Vec<SmallVec<[usize; M0]>>,
    layer: usize,
}

/* ───────────────────────── HNSW core ───────────────────────── */
pub struct HnswIndex {
    nodes: HashMap<usize, Node>,
    entry: Option<usize>,
    lambda: f32,
    max_level: usize,
}

impl HnswIndex {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            entry: None,
            lambda: 1.0 / (M as f32).ln(),
            max_level: MAX_LEVEL,
        }
    }

    #[inline]
    fn rand_level(&self) -> usize {
        let mut rng = rand::rng();
        let mut lvl = 0usize;
        while rng.random::<f32>() < self.lambda && lvl < self.max_level {
            lvl += 1;
        }
        lvl
    }

    /* ───────────── insert ───────────── */
    pub fn add(&mut self, id: usize, vector: Vec<f32>) -> Result<(), String> {
        /* first ever node */
        if self.nodes.is_empty() {
            let lvl = self.rand_level();
            self.nodes.insert(
                id,
                Node {
                    vector,
                    connections: vec![SmallVec::new(); lvl + 1],
                    layer: lvl,
                },
            );
            self.entry = Some(id);
            return Ok(());
        }

        let node_lvl = self.rand_level();

        /* greedy descent from current entry */
        let mut ep = self.entry.unwrap();
        // let mut ep_dist = simd_euclidean_distance(&self.nodes[&ep].vector, &vector);
        let top = self.nodes[&ep].layer;
        for layer in (0..=top).rev() {
            if let Some(best) = self.search_layer(ep, &vector, layer, 1)?.first() {
                ep = best.id;
                best.dist;
            }
        }

        /* neighbour selection & linking */
        let mut new_conns = vec![SmallVec::<[usize; M0]>::new(); node_lvl + 1];
        for layer in 0..=node_lvl {
            let mut cand = self.search_layer(ep, &vector, layer, EF_CONSTRUCTION)?;
            cand.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
            cand.truncate(if layer == 0 { M0 } else { M });

            for nbr in &cand {
                new_conns[layer].push(nbr.id);
            }
            for nbr in cand {
                if let Some(n) = self.nodes.get_mut(&nbr.id) {
                    if layer < n.connections.len() {
                        n.connections[layer].push(id);
                    }
                }
            }
        }

        /* actually insert */
        self.nodes.insert(
            id,
            Node {
                vector,
                connections: new_conns,
                layer: node_lvl,
            },
        );
        if node_lvl > self.nodes[&self.entry.unwrap()].layer {
            self.entry = Some(id);
        }
        Ok(())
    }

    /* ───────────── remove ───────────── */
    pub fn remove(&mut self, id: usize) -> Result<(), String> {
        let node = self
            .nodes
            .remove(&id)
            .ok_or_else(|| "node not found".to_string())?;

        /* unlink from every neighbour */
        for (layer, neighs) in node.connections.into_iter().enumerate() {
            for nb in neighs {
                if let Some(n) = self.nodes.get_mut(&nb) {
                    if layer < n.connections.len() {
                        n.connections[layer].retain(|x| *x != id);
                    }
                }
            }
        }

        /* fix entry pointer */
        if self.entry == Some(id) {
            self.entry = self.nodes.keys().next().copied();
        }
        if self.nodes.is_empty() {
            self.entry = None;
        }
        Ok(())
    }

    /* ───────── internal search steps ───────── */
    fn search_layer(
        &self,
        entry: usize,
        query: &[f32],
        layer: usize,
        ef: usize,
    ) -> Result<Vec<Neighbor>, String> {
        if !self.nodes.contains_key(&entry) {
            return Ok(Vec::new());
        }

        let mut visited = HashSet::new();
        let mut cand = BinaryHeap::<Neighbor>::new();
        let mut res = BinaryHeap::<Neighbor>::new();

        let d0 = simd_euclidean_distance(&self.nodes[&entry].vector, query);
        cand.push(Neighbor {
            id: entry,
            dist: d0,
        });
        res.push(Neighbor {
            id: entry,
            dist: d0,
        });
        visited.insert(entry);

        while let Some(cur) = cand.pop() {
            let worst = res.peek().map_or(f32::INFINITY, |n| n.dist);
            if cur.dist > worst {
                break;
            }
            let Some(node) = self.nodes.get(&cur.id) else {
                continue;
            };
            if layer < node.connections.len() {
                for &nb in &node.connections[layer] {
                    /* skip stale neighbour ids */
                    let Some(nb_node) = self.nodes.get(&nb) else {
                        continue;
                    };
                    if !visited.insert(nb) {
                        continue;
                    }
                    let dist = simd_euclidean_distance(&nb_node.vector, query);
                    let cand_n = Neighbor { id: nb, dist };
                    if res.len() < ef || dist < worst {
                        cand.push(cand_n.clone());
                        res.push(cand_n);
                        if res.len() > ef {
                            res.pop();
                        }
                    }
                }
            }
        }

        Ok(res.into_sorted_vec())
    }

    /* ───────── public top-k search ───────── */
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, String> {
        /* early-outs */
        let Some(mut ep) = self.entry else {
            return Ok(Vec::new());
        };
        let Some(entry_node) = self.nodes.get(&ep) else {
            return Ok(Vec::new());
        };

        /* greedy descent on upper layers */
        let mut ep_dist = simd_euclidean_distance(&entry_node.vector, query);
        let top_layer = entry_node.layer;
        for layer in (1..=top_layer).rev() {
            loop {
                let mut moved = false;
                let Some(node) = self.nodes.get(&ep) else {
                    break;
                };
                if layer < node.connections.len() {
                    for &nb in &node.connections[layer] {
                        let Some(nb_node) = self.nodes.get(&nb) else {
                            continue;
                        };
                        let d = simd_euclidean_distance(&nb_node.vector, query);
                        if d < ep_dist {
                            ep = nb;
                            ep_dist = d;
                            moved = true;
                        }
                    }
                }
                if !moved {
                    break;
                }
            }
        }

        /* final, layer-0 search */
        let mut best = self.search_layer(ep, query, 0, EF_SEARCH)?;
        best.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        Ok(best.into_iter().take(k).map(|n| (n.id, n.dist)).collect())
    }
}

/* ───────────────────────── external wrapper ───────────────── */
pub struct HnswIndexWrapper {
    index: HnswIndex,
    id_map: HashMap<usize, String>,
    next: usize,
}

impl HnswIndexWrapper {
    pub fn new() -> Self {
        Self {
            index: HnswIndex::new(),
            id_map: HashMap::new(),
            next: 0,
        }
    }

    pub fn insert(&mut self, value: &str, vector: Vec<f32>) -> Result<(), String> {
        let nid = self.next;
        self.index.add(nid, vector)?;
        self.id_map.insert(nid, value.to_owned());
        self.next += 1;
        Ok(())
    }

    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        _dist: Distance,
    ) -> Result<Vec<(String, f32)>, String> {
        let raw = self.index.search(query, k)?;
        Ok(raw
            .into_iter()
            .filter_map(|(nid, d)| {
                self.id_map
                    .get(&nid)
                    .map(|s| (s.clone(), clamp_0_1(1.0 / (1.0 + d))))
            })
            .collect())
    }

    pub fn remove(&mut self, value: &str) -> Result<(), String> {
        let (&nid, _) = self
            .id_map
            .iter()
            .find(|(_, v)| *v == value)
            .ok_or_else(|| "id not found".to_string())?;
        self.index.remove(nid)?;
        self.id_map.remove(&nid);
        Ok(())
    }
}
