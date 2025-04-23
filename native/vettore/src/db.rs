use std::collections::HashMap;

#[cfg(feature = "parallel")]
use crate::distances::{compress_vector, score};
use crate::hnsw::HnswIndexWrapper;
use crate::simd_utils::normalize_vec;
use crate::types::{Distance, Metadata};

/// One vector collection inside the DB.
pub struct Collection {
    pub dimension: usize,
    pub keep_embeddings: bool,
    pub distance: Distance,

    pub(crate) vectors: Vec<f32>,
    pub(crate) id2row: HashMap<String, usize>,
    pub(crate) row2id: Vec<Option<String>>,
    pub(crate) meta: Vec<Option<Metadata>>, // optional user metadata
    pub(crate) binary: Vec<Option<Vec<u64>>>, // cached sign bits (Binary dist)

    free: Vec<usize>,
    hnsw: Option<HnswIndexWrapper>,
}

impl Collection {
    pub fn create_with_distance(dim: usize, dist_str: &str) -> Result<Self, String> {
        let dist = match dist_str.to_lowercase().as_str() {
            "euclidean" => Distance::Euclidean,
            "cosine" => Distance::Cosine,
            "dot" => Distance::DotProduct,
            "hnsw" => Distance::Hnsw,
            "binary" => Distance::Binary,
            _ => return Err("unknown distance".into()),
        };
        Ok(Self {
            dimension: dim,
            keep_embeddings: true,
            distance: dist,
            vectors: Vec::new(),
            id2row: HashMap::new(),
            row2id: Vec::new(),
            meta: Vec::new(),
            binary: Vec::new(),
            free: Vec::new(),
            hnsw: if dist == Distance::Hnsw {
                Some(HnswIndexWrapper::new())
            } else {
                None
            },
        })
    }

    fn alloc_row(&mut self) -> usize {
        if let Some(row) = self.free.pop() {
            row
        } else {
            let row = self.row_count();
            self.meta.push(None);
            self.binary.push(None);
            self.row2id.push(None);
            self.vectors.resize((row + 1) * self.dimension, 0.0);
            row
        }
    }
    fn row_count(&self) -> usize {
        self.meta.len()
    }

    pub fn insert(
        &mut self,
        id: String,
        mut vec: Vec<f32>,
        md: Option<Metadata>,
    ) -> Result<(), String> {
        if vec.len() != self.dimension {
            return Err("dimension mismatch".into());
        }
        if self.id2row.contains_key(&id) {
            return Err("duplicate id".into());
        }
        if matches!(self.distance, Distance::Cosine) {
            vec = normalize_vec(&vec);
        }

        let row = self.alloc_row();
        let offset = row * self.dimension;

        if !(matches!(self.distance, Distance::Binary) && !self.keep_embeddings) {
            self.vectors[offset..offset + self.dimension].copy_from_slice(&vec);
        }
        self.meta[row] = md;

        if matches!(self.distance, Distance::Binary) {
            self.binary[row] = Some(compress_vector(&vec));
            if !self.keep_embeddings {
                vec.clear(); // vec is only used for HNSW insert below, so clearing is fine
            }
        }

        self.row2id[row] = Some(id.clone());
        self.id2row.insert(id.clone(), row);
        if let Some(h) = &mut self.hnsw {
            h.insert(&id, vec)?;
        }
        Ok(())
    }

    pub fn get_similarity(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>, String> {
        // dimension & HNSW fast-path
        if query.len() != self.dimension {
            return Err("query dim mismatch".into());
        }
        if let Some(h) = &self.hnsw {
            return h.search(query, k, self.distance);
        }

        let rows = self.row_count();
        let use_parallel = cfg!(feature = "parallel") && rows >= 20_000; // ← threshold

        #[allow(unused_mut)]
        let mut scores: Vec<(String, f32)> = if !use_parallel {
            self.row2id
                .iter()
                .enumerate()
                .filter_map(|(row, maybe_id)| {
                    let id = maybe_id.as_ref()?;
                    let slice = &self.vectors[row * self.dimension..(row + 1) * self.dimension];
                    Some((
                        id.clone(),
                        score(query, slice, self.binary[row].as_ref(), self.distance),
                    ))
                })
                .collect()
        } else {
            // parallel
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                self.row2id
                    .par_iter()
                    .enumerate()
                    .filter_map(|(row, maybe_id)| {
                        let id = maybe_id.as_ref()?;
                        let slice = &self.vectors[row * self.dimension..(row + 1) * self.dimension];
                        Some((
                            id.clone(),
                            score(query, slice, self.binary[row].as_ref(), self.distance),
                        ))
                    })
                    .collect()
            }
            #[cfg(not(feature = "parallel"))]
            unreachable!();
        };

        // rank & truncate
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(scores.into_iter().take(k).collect())
    }

    /// Returns (id, float_vec_or_empty, metadata)
    pub(crate) fn row_to_tuple(
        &self,
        id: &str,
        row: usize,
    ) -> (String, Vec<f32>, Option<Metadata>) {
        let floats = if matches!(self.distance, Distance::Binary) && !self.keep_embeddings {
            Vec::new() // ── CHANGED: return [] when floats were not stored ──
        } else {
            self.vectors[row * self.dimension..(row + 1) * self.dimension].to_vec()
        };
        (id.to_owned(), floats, self.meta[row].clone())
    }

    pub fn remove(&mut self, id: &str) -> Result<(), String> {
        let row = *self
            .id2row
            .get(id)
            .ok_or_else(|| "id not found".to_string())?;
        self.id2row.remove(id);
        self.free.push(row);
        if let Some(h) = &mut self.hnsw {
            h.remove(id)?;
        }
        Ok(())
    }
}
