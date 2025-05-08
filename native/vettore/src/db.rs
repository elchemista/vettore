//! db.rs – storage only (no algorithms)
//! ======================================
//! * Provides fast CRUD on embeddings.
//! * Exposes **public getters** so search / rerank algorithms can live elsewhere.
//! * No similarity/MRR code remains inside this file.

use std::collections::HashMap;

use crate::distances::compress_vector;
use crate::hnsw::HnswIndexWrapper;
use crate::simd_utils::normalize_vec;
use crate::types::{Distance, Metadata};

type CompKey = Vec<u64>; // sign‑bit compression

#[derive(Debug, Clone)]
pub struct Record {
    pub vector: Vec<f32>,
    pub value: String,
    pub metadata: Option<Metadata>,
}

pub struct Collection {
    /* config */
    pub dimension: usize,
    pub keep_embeddings: bool,
    pub distance: Distance,

    /* tables */
    vectors: Vec<f32>,
    row2value: Vec<Option<String>>, // row → value
    meta: Vec<Option<Metadata>>,
    binary: Vec<Option<CompKey>>,

    /* indexes */
    comp2row: HashMap<CompKey, usize>,
    value2row: HashMap<String, usize>,

    free: Vec<usize>,
    hnsw: Option<HnswIndexWrapper>,
}

impl Collection {
    pub fn create_with_distance(dim: usize, dist: &str) -> Result<Self, String> {
        let distance = match dist.to_lowercase().as_str() {
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
            distance,
            vectors: Vec::new(),
            row2value: Vec::new(),
            meta: Vec::new(),
            binary: Vec::new(),
            comp2row: HashMap::new(),
            value2row: HashMap::new(),
            free: Vec::new(),
            hnsw: if distance == Distance::Hnsw {
                Some(HnswIndexWrapper::new())
            } else {
                None
            },
        })
    }

    /* ---------- public lightweight accessors for algorithms ---------- */
    pub fn row_count(&self) -> usize {
        self.row2value.len()
    }
    pub fn value_by_row(&self, r: usize) -> Option<&String> {
        self.row2value[r].as_ref()
    }
    pub fn vector_slice(&self, r: usize) -> &[f32] {
        &self.vectors[r * self.dimension..(r + 1) * self.dimension]
    }
    pub fn compressed_by_row(&self, r: usize) -> Option<&CompKey> {
        self.binary[r].as_ref()
    }
    pub fn hnsw(&self) -> Option<&HnswIndexWrapper> {
        self.hnsw.as_ref()
    }

    /* ---------- internal helpers ---------- */
    fn alloc_row(&mut self) -> usize {
        if let Some(r) = self.free.pop() {
            r
        } else {
            let r = self.row2value.len();
            self.row2value.push(None);
            self.meta.push(None);
            self.binary.push(None);
            self.vectors.resize((r + 1) * self.dimension, 0.0);
            r
        }
    }
    fn row_to_record(&self, row: usize) -> Record {
        let value = self.row2value[row].as_ref().unwrap().clone();
        let vector = if !self.keep_embeddings && matches!(self.distance, Distance::Binary) {
            Vec::new()
        } else {
            self.vector_slice(row).to_vec()
        };
        Record {
            vector,
            value,
            metadata: self.meta[row].clone(),
        }
    }

    /* -------------------- INSERT / DELETE / GET -------------------- */
    pub fn insert(
        &mut self,
        value: String,
        mut vec: Vec<f32>,
        md: Option<Metadata>,
    ) -> Result<(), String> {
        if vec.len() != self.dimension {
            return Err("dimension mismatch".into());
        }
        if self.value2row.contains_key(&value) {
            return Err("duplicate value".into());
        }

        // normalize if cosine
        if matches!(self.distance, Distance::Cosine) {
            vec = normalize_vec(&vec);
        }

        let comp = compress_vector(&vec);
        if self.comp2row.contains_key(&comp) {
            return Err("duplicate vector".into());
        }

        // now safe to allocate a new row
        let row = self.alloc_row();
        let offset = row * self.dimension;

        // store floats (unless binary+drop)
        if self.keep_embeddings || !matches!(self.distance, Distance::Binary) {
            self.vectors[offset..offset + self.dimension].copy_from_slice(&vec);
        }

        self.binary[row] = Some(comp.clone());
        self.meta[row] = md;
        self.row2value[row] = Some(value.clone());
        self.comp2row.insert(comp, row);
        self.value2row.insert(value.clone(), row);

        if let Some(h) = &mut self.hnsw {
            h.insert(&value, vec)?;
        }
        Ok(())
    }

    pub fn get_by_value(&self, value: &str) -> Option<Record> {
        let &row = self.value2row.get(value)?;
        Some(self.row_to_record(row))
    }

    pub fn get_by_vector(&self, vec: &[f32]) -> Option<Record> {
        if vec.len() != self.dimension {
            return None;
        }
        let &row = self.comp2row.get(&compress_vector(vec))?;
        Some(self.row_to_record(row))
    }

    pub fn remove(&mut self, value: &str) -> Result<(), String> {
        let row = *self
            .value2row
            .get(value)
            .ok_or_else(|| "value not found".to_string())?;
        self.value2row.remove(value);
        if let Some(comp) = &self.binary[row] {
            self.comp2row.remove(comp);
        }
        self.row2value[row] = None;
        self.free.push(row);
        if let Some(h) = &mut self.hnsw {
            h.remove(value)?;
        }
        Ok(())
    }
}

pub struct VettoreDB {
    cols: HashMap<String, Collection>,
}
impl Default for VettoreDB {
    fn default() -> Self {
        Self {
            cols: HashMap::new(),
        }
    }
}

impl VettoreDB {
    /* public collection accessors for external algorithms */
    pub fn collection(&self, name: &str) -> Result<&Collection, String> {
        self.cols
            .get(name)
            .ok_or_else(|| "collection not found".into())
    }
    pub fn collection_mut(&mut self, name: &str) -> Result<&mut Collection, String> {
        self.cols
            .get_mut(name)
            .ok_or_else(|| "collection not found".into())
    }

    /* management */
    pub fn create_collection(
        &mut self,
        name: String,
        dim: usize,
        dist: &str,
        keep: bool,
    ) -> Result<(), String> {
        if self.cols.contains_key(&name) {
            return Err("duplicate collection".into());
        }
        let mut c = Collection::create_with_distance(dim, dist)?;
        c.keep_embeddings = keep;
        self.cols.insert(name, c);
        Ok(())
    }
    pub fn delete_collection(&mut self, name: &str) -> Result<(), String> {
        self.cols
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| "collection not found".into())
    }

    /* thin data ops wrappers for NIFs */
    pub fn insert(
        &mut self,
        col: &str,
        value: String,
        vec: Vec<f32>,
        md: Option<Metadata>,
    ) -> Result<(), String> {
        self.collection_mut(col)?.insert(value, vec, md)
    }
    pub fn get_by_value(&self, col: &str, val: &str) -> Result<Record, String> {
        self.collection(col)?
            .get_by_value(val)
            .ok_or_else(|| "value not found".into())
    }
    pub fn get_by_vector(&self, col: &str, v: &[f32]) -> Result<Record, String> {
        self.collection(col)?
            .get_by_vector(v)
            .ok_or_else(|| "vector not found".into())
    }
    pub fn get_all(&self, col: &str) -> Result<Vec<Record>, String> {
        let c = self.collection(col)?;
        let mut out = Vec::with_capacity(c.row_count());
        for row in 0..c.row_count() {
            // only rows with a value are “live”
            if c.value_by_row(row).is_some() {
                out.push(c.row_to_record(row));
            }
        }
        Ok(out)
    }

    pub fn delete_by_value(&mut self, col: &str, val: &str) -> Result<(), String> {
        self.collection_mut(col)?.remove(val)
    }
}
