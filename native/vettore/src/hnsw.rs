//! hnsw.rs – minimal, safe HNSW used by Vettore
//! Public surface is unchanged; internals fixed & hardened.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use rand::{thread_rng, Rng};
use smallvec::SmallVec;

use crate::distances::{clamp_0_1, simd_euclidean_distance};
use crate::types::Distance;

pub const M: usize = 16;
pub const M0: usize = 32; // layer-0 width
pub const EF_CONSTRUCTION: usize = 100;
pub const EF_SEARCH: usize = 64;
pub const MAX_LEVEL: usize = 12;

#[derive(Clone)]
struct Neighbor {
    id: usize,
    dist: f32,
}
impl Eq for Neighbor {}
impl PartialEq for Neighbor {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist
    }
}
impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // max-heap; NaN treated as worst possible
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Neighbor {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        // reverse so that smaller distance == "greater priority"
        match (self.dist.is_nan(), o.dist.is_nan()) {
            (true, true) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Less), // NaN goes to bottom
            (false, true) => Some(Ordering::Greater),
            (false, false) => Some(o.dist.partial_cmp(&self.dist).unwrap()),
        }
    }
}

#[derive(Clone)]
struct Node {
    vector: Vec<f32>,
    connections: Vec<SmallVec<[usize; M0]>>,
    layer: usize,
}

/* core index */
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
        let mut rng = thread_rng(); // ‹thread_rng› → ‹rng›
        let mut lvl = 0;
        while rng.gen::<f32>() < self.lambda && lvl < self.max_level {
            lvl += 1;
        }
        lvl
    }

    /* keep the closest {M|M0} connections for `node_id` at `layer` */
    fn prune_node_layer(&mut self, node_id: usize, layer: usize) {
        let limit = if layer == 0 { M0 } else { M };

        let (conn_snapshot, reference_vec) = match self.nodes.get(&node_id) {
            Some(n) => (n.connections[layer].clone(), n.vector.clone()),
            None => return,
        };

        let mut scored: Vec<(usize, f32)> = conn_snapshot
            .into_iter()
            .filter_map(|nid| {
                self.nodes
                    .get(&nid)
                    .map(|nbr| (nid, simd_euclidean_distance(&nbr.vector, &reference_vec)))
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(limit);

        let mut new_conn: SmallVec<[usize; M0]> = SmallVec::new();
        new_conn.extend(scored.into_iter().map(|(nid, _)| nid));

        if let Some(n) = self.nodes.get_mut(&node_id) {
            n.connections[layer] = new_conn;
        }
    }

    /* ── insert─ */
    pub fn add(&mut self, id: usize, vector: Vec<f32>) -> Result<(), String> {
        if self.nodes.contains_key(&id) {
            return Err("duplicate id".into());
        }

        /* first node shortcut */
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
        let mut ep = self.entry.expect("entry must exist");
        let mut ep_dist = simd_euclidean_distance(&self.nodes[&ep].vector, &vector);
        let top_layer = self.nodes[&ep].layer;
        for layer in (0..=top_layer).rev() {
            if let Some(best) = self.search_layer(ep, &vector, layer, 1)?.into_iter().next() {
                if best.dist < ep_dist {
                    ep = best.id;
                    ep_dist = best.dist;
                }
            }
        }

        /* neighbour selection & two-way linking */
        let mut new_conns = vec![SmallVec::<[usize; M0]>::new(); node_lvl + 1];

        for layer in 0..=node_lvl {
            // EF-construction search at this layer
            let mut cand = self.search_layer(ep, &vector, layer, EF_CONSTRUCTION)?;
            cand.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
            cand.dedup_by_key(|n| n.id);
            cand.truncate(if layer == 0 { M0 } else { M });

            /* neighbours for the *new* node */
            for nb in &cand {
                new_conns[layer].push(nb.id);
            }

            /* stage nodes that need pruning */
            let mut to_prune = Vec::<usize>::new();

            /* symmetric link */
            for nb in cand {
                if let Some(n) = self.nodes.get_mut(&nb.id) {
                    if layer < n.connections.len() {
                        let conn = &mut n.connections[layer];
                        if !conn.contains(&id) {
                            conn.push(id);
                        }
                        to_prune.push(nb.id); // <- remember for later
                    }
                }
            }

            for pid in to_prune {
                self.prune_node_layer(pid, layer);
            }
        }

        /* finally insert the node */
        self.nodes.insert(
            id,
            Node {
                vector,
                connections: new_conns,
                layer: node_lvl,
            },
        );

        /* update entry if the new node reaches a higher layer */
        if node_lvl > self.nodes[&self.entry.unwrap()].layer {
            self.entry = Some(id);
        }
        Ok(())
    }

    /* ── delete ───────────────────────────────────────────────────── */
    pub fn remove(&mut self, id: usize) -> Result<(), String> {
        let node = self
            .nodes
            .remove(&id)
            .ok_or_else(|| "node not found".to_string())?;

        /* unlink from neighbours */
        for (layer, neighs) in node.connections.into_iter().enumerate() {
            for nb in neighs {
                if let Some(n) = self.nodes.get_mut(&nb) {
                    if layer < n.connections.len() {
                        n.connections[layer].retain(|x| *x != id);
                    }
                }
            }
        }

        /* repair entry pointer */
        if self.entry == Some(id) {
            self.entry = self
                .nodes
                .iter()
                .max_by_key(|(_, n)| n.layer)
                .map(|(&id, _)| id);
        }
        Ok(())
    }

    /* ── internal layer search */
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

            if layer >= node.connections.len() {
                continue;
            }

            for &nb in &node.connections[layer] {
                if !visited.insert(nb) {
                    continue;
                }
                let Some(nb_node) = self.nodes.get(&nb) else {
                    continue;
                };
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

        Ok(res.into_sorted_vec())
    }

    /* ─public k-NN search  */
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>, String> {
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

                let neigh_slice: &[usize] = if layer < node.connections.len() {
                    &node.connections[layer]
                } else {
                    &[]
                };
                for &nb in neigh_slice {
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
                if !moved {
                    break;
                }
            }
        }

        /* final search on layer-0 */
        let mut best = self.search_layer(ep, query, 0, EF_SEARCH)?;
        best.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        Ok(best.into_iter().take(k).map(|n| (n.id, n.dist)).collect())
    }
}

/* ── thin wrapper to map ids ⇄ payload strings */
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
