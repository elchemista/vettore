use std::collections::HashMap;
use std::sync::Mutex;

use crate::distances::{compress_vector, compute_0_1_score};
use crate::hnsw::HnswIndexWrapper;
use crate::simd_utils::normalize_vec;
use crate::types::{Distance, Embedding};

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

impl CacheDB {
    pub fn new() -> Self {
        Self {
            collections: HashMap::new(),
        }
    }

    pub fn get_embedding_by_id(&self, col: &str, id: &str) -> Result<Embedding, String> {
        let c = self
            .collections
            .get(col)
            .ok_or_else(|| format!("Collection '{}' not found", col))?;
        c.embeddings
            .iter()
            .find(|e| e.id == id)
            .cloned()
            .ok_or_else(|| format!("Embedding '{}' not found in '{}'", id, col))
    }
}

// Shareable across NIF boundary.
pub struct DBResource(pub Mutex<CacheDB>);
impl rustler::Resource for DBResource {}

impl Collection {
    pub fn create_with_distance(dim: usize, dist_str: &str) -> Result<Self, String> {
        if dim == 0 {
            return Err("Dimension cannot be 0".into());
        }
        let distance = match dist_str.to_lowercase().as_str() {
            "euclidean" => Distance::Euclidean,
            "cosine" => Distance::Cosine,
            "dot" => Distance::DotProduct,
            "hnsw" => Distance::Hnsw,
            "binary" => Distance::Binary,
            other => {
                return Err(format!(
                    "Unknown distance '{}'. Expected euclidean | cosine | dot | hnsw | binary",
                    other
                ))
            }
        };
        let mut c = Self {
            dimension: dim,
            keep_embeddings: true,
            distance,
            embeddings: Vec::new(),
            hnsw_index: None,
        };
        if distance == Distance::Hnsw {
            c.hnsw_index = Some(HnswIndexWrapper::new());
        }
        Ok(c)
    }

    pub fn insert_embedding(&mut self, mut emb: Embedding) -> Result<(), String> {
        if emb.vector.len() != self.dimension {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimension,
                emb.vector.len()
            ));
        }
        if self.embeddings.iter().any(|e| e.id == emb.id) {
            return Err(format!("ID '{}' already exists", emb.id));
        }
        if self.distance == Distance::Cosine {
            emb.vector = normalize_vec(&emb.vector);
        }
        if self.distance == Distance::Binary {
            emb.binary = Some(compress_vector(&emb.vector));
            if !self.keep_embeddings {
                emb.vector.clear();
            }
        }
        if let Some(idx) = &mut self.hnsw_index {
            idx.insert_embedding(&emb)?;
        }
        self.embeddings.push(emb);
        Ok(())
    }

    pub fn get_similarity(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>, String> {
        if query.len() != self.dimension {
            return Err("Query dimension mismatch".into());
        }
        match self.distance {
            Distance::Hnsw => {
                if let Some(idx) = &self.hnsw_index {
                    idx.search(query, k)
                } else {
                    Err("No HNSW index present".into())
                }
            }
            _ => {
                let mut scored: Vec<_> = self
                    .embeddings
                    .iter()
                    .map(|e| (e.id.clone(), compute_0_1_score(query, e, self.distance)))
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                scored.truncate(k);
                Ok(scored)
            }
        }
    }

    pub fn remove_embedding_by_id(&mut self, id: &str) -> Result<(), String> {
        let idx = self
            .embeddings
            .iter()
            .position(|e| e.id == id)
            .ok_or_else(|| format!("Embedding '{}' not found", id))?;
        self.embeddings.remove(idx);
        if let Some(h) = &mut self.hnsw_index {
            h.remove_by_str_id(id)?;
        }
        Ok(())
    }
}
