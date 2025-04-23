use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use bitvec::prelude::*;
use rand::Rng;
use smallvec::SmallVec;

use crate::distances::{clamp_0_1, simd_euclidean_distance};
use crate::types::Distance;

// params
const M: usize = 16;
const M0: usize = 32;
const EF_CONSTRUCTION: usize = 100;
const EF_SEARCH: usize = 64;
const MAX_LEVEL: usize = 12;

#[derive(Clone)]
struct Neighbor {
    id: usize,
    dist: f32,
}
impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap().reverse()
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
    connections: Vec<SmallVec<[usize; M0]>>, // per‑layer neighbor ids
    layer: usize,
}

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
    fn rand_level(&self) -> usize {
        let mut rng = rand::rng();
        let mut lvl = 0usize;
        while rng.random::<f32>() < self.lambda && lvl < self.max_level {
            lvl += 1;
        }
        lvl
    }

    pub fn add(&mut self, id: usize, vector: Vec<f32>) -> Result<(), String> {
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
        let mut ep = self.entry.unwrap();
        let mut ep_dist = simd_euclidean_distance(&self.nodes[&ep].vector, &vector);
        let top = self.nodes[&ep].layer;
        for l in (0..=top).rev() {
            if let Some(best) = self.search_layer(ep, &vector, l, 1)?.first() {
                if best.dist < ep_dist {
                    ep = best.id;
                    ep_dist = best.dist;
                }
            }
        }
        let mut new_conns = vec![SmallVec::<[usize; M0]>::new(); node_lvl + 1];
        for l in 0..=node_lvl {
            let neighs = self.search_layer(ep, &vector, l, EF_CONSTRUCTION)?;
            let sel = self.select_neighbors(&neighs, l);
            new_conns[l].extend(sel.iter().map(|n| n.id));
            for n in sel {
                if let Some(node) = self.nodes.get_mut(&n.id) {
                    if l < node.connections.len() {
                        node.connections[l].push(id);
                    }
                }
            }
        }
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

    pub fn remove(&mut self, id: usize) -> Result<(), String> {
        let node = self
            .nodes
            .remove(&id)
            .ok_or_else(|| "node not found".to_string())?;
        for (l, neighs) in node.connections.into_iter().enumerate() {
            for n in neighs {
                if let Some(node) = self.nodes.get_mut(&n) {
                    if l < node.connections.len() {
                        node.connections[l].retain(|x| *x != id);
                    }
                }
            }
        }
        if self.entry == Some(id) {
            self.entry = self.nodes.keys().next().copied();
        }
        Ok(())
    }

    fn search_layer(
        &self,
        entry: usize,
        query: &[f32],
        lvl: usize,
        ef: usize,
    ) -> Result<Vec<Neighbor>, String> {
        let mut visited: BitVec = bitvec![0; self.nodes.len()];
        let mut cand = BinaryHeap::<Neighbor>::new();
        let mut res = BinaryHeap::<Neighbor>::new();
        let dist0 = simd_euclidean_distance(&self.nodes[&entry].vector, query);
        cand.push(Neighbor {
            id: entry,
            dist: dist0,
        });
        res.push(Neighbor {
            id: entry,
            dist: dist0,
        });
        visited.set(entry, true);
        while let Some(cur) = cand.pop() {
            let worst = res.peek().map_or(f32::INFINITY, |n| n.dist);
            if cur.dist > worst {
                break;
            }
            let node = &self.nodes[&cur.id];
            if lvl < node.connections.len() {
                for &nb in &node.connections[lvl] {
                    if visited[nb] {
                        continue;
                    }
                    visited.set(nb, true);
                    let d = simd_euclidean_distance(&self.nodes[&nb].vector, query);
                    let n = Neighbor { id: nb, dist: d };
                    if res.len() < ef || d < worst {
                        cand.push(n.clone());
                        res.push(n);
                        if res.len() > ef {
                            res.pop();
                        }
                    }
                }
            }
        }
        Ok(res.into_sorted_vec())
    }

    fn select_neighbors(&self, neighs: &[Neighbor], lvl: usize) -> Vec<Neighbor> {
        let mut v = neighs.to_vec();
        v.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        let m = if lvl == 0 { M0 } else { M };
        v.truncate(m);
        v
    }

    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, String> {
        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }
        let mut ep = self.entry.unwrap();
        let mut ep_dist = simd_euclidean_distance(&self.nodes[&ep].vector, query);
        let top = self.nodes[&ep].layer;
        for l in (1..=top).rev() {
            loop {
                let mut changed = false;
                let node = &self.nodes[&ep];
                if l < node.connections.len() {
                    for &nb in &node.connections[l] {
                        let d = simd_euclidean_distance(&self.nodes[&nb].vector, query);
                        if d < ep_dist {
                            ep = nb;
                            ep_dist = d;
                            changed = true;
                        }
                    }
                }
                if !changed {
                    break;
                }
            }
        }
        let mut res = self.search_layer(ep, query, 0, EF_SEARCH)?;
        res.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        Ok(res.into_iter().take(k).map(|n| (n.id, n.dist)).collect())
    }
}

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
    pub fn insert(&mut self, id_str: &str, vector: Vec<f32>) -> Result<(), String> {
        let nid = self.next;
        self.next += 1;
        self.id_map.insert(nid, id_str.to_owned());
        self.index.add(nid, vector)
    }
    pub fn search(
        &self,
        q: &[f32],
        k: usize,
        _dist: Distance,
    ) -> Result<Vec<(String, f32)>, String> {
        let raw = self.index.search(q, k)?;
        Ok(raw
            .into_iter()
            .filter_map(|(nid, d)| {
                self.id_map
                    .get(&nid)
                    .map(|s| (s.clone(), clamp_0_1(1.0 / (1.0 + d))))
            })
            .collect())
    }
    pub fn remove(&mut self, id_str: &str) -> Result<(), String> {
        let nid = *self
            .id_map
            .iter()
            .find(|(_, v)| *v == id_str)
            .map(|(k, _)| k)
            .ok_or_else(|| "id not found".to_string())?;
        self.index.remove(nid)?;
        self.id_map.remove(&nid);
        Ok(())
    }
}
