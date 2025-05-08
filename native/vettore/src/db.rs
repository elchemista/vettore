//! db.rs  –  storage layer (no algorithms)
//! ======================================

use dashmap::DashMap;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::distances::compress_vector;
use crate::hnsw::HnswIndexWrapper;
use crate::simd_utils::normalize_vec;
use crate::types::{Distance, Metadata};

/* ───────────── helper aliases ───────────── */
type CompKey = Vec<u64>; // sign-bit compression

/* ───────────── public record ───────────── */
#[derive(Debug, Clone)]
pub struct Record {
    pub vector: Vec<f32>,
    pub value: String,
    pub metadata: Option<Metadata>,
}

/* ───────────── one collection ───────────── */
pub struct Collection {
    /* static config */
    pub dimension: usize,
    pub keep_embeddings: bool,
    pub distance: Distance,

    /* storage tables */
    vectors: Vec<f32>,
    row2value: Vec<Option<String>>,
    meta: Vec<Option<Metadata>>,
    binary: Vec<Option<CompKey>>,

    /* indexes */
    comp2row: HashMap<CompKey, usize>,
    value2row: HashMap<String, usize>,

    /* housekeeping */
    free: Vec<usize>,
    hnsw: Option<HnswIndexWrapper>,
}

/* ---------- ctor ------------------------------------------------ */
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

    /* ---------- lightweight getters ------------------------------ */
    #[inline]
    pub fn row_count(&self) -> usize {
        self.row2value.len()
    }
    #[inline]
    pub fn value_by_row(&self, r: usize) -> Option<&String> {
        self.row2value[r].as_ref()
    }
    #[inline]
    pub fn vector_slice(&self, r: usize) -> &[f32] {
        &self.vectors[r * self.dimension..(r + 1) * self.dimension]
    }
    #[inline]
    pub fn compressed_by_row(&self, r: usize) -> Option<&CompKey> {
        self.binary[r].as_ref()
    }
    #[inline]
    pub fn hnsw(&self) -> Option<&HnswIndexWrapper> {
        self.hnsw.as_ref()
    }

    /* ---------- row allocator ------------------------------------ */
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

    /* ---------- CRUD --------------------------------------------- */
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

        if matches!(self.distance, Distance::Cosine) {
            vec = normalize_vec(&vec);
        }
        let comp = compress_vector(&vec);
        if self.comp2row.contains_key(&comp) {
            return Err("duplicate vector".into());
        }

        /* allocate row & copy -------------------------------------- */
        let row = self.alloc_row();
        let offset = row * self.dimension;
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

    /* read helpers */
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

    /* delete */
    pub fn remove(&mut self, value: &str) -> Result<(), String> {
        let row = *self
            .value2row
            .get(value)
            .ok_or("value not found".to_string())?;
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

/* ───────────── global DB  (sharded) ───────────── */
pub struct VettoreDB {
    cols: DashMap<String, Arc<RwLock<Collection>>>,
}

impl Default for VettoreDB {
    fn default() -> Self {
        Self {
            cols: DashMap::new(),
        }
    }
}
use std::panic::{RefUnwindSafe, UnwindSafe};

impl RefUnwindSafe for VettoreDB {}
impl UnwindSafe for VettoreDB {}

impl VettoreDB {
    #[inline]
    pub fn collection(&self, name: &str) -> Result<Arc<RwLock<Collection>>, String> {
        self.cols
            .get(name)
            .map(|e| e.clone())
            .ok_or_else(|| "collection not found".into())
    }
    #[inline]
    fn collection_mut(&self, name: &str) -> Result<Arc<RwLock<Collection>>, String> {
        self.collection(name)
    }

    /* management --------------------------------------------------- */
    pub fn create_collection(
        &self,
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
        self.cols.insert(name, Arc::new(RwLock::new(c)));
        Ok(())
    }

    pub fn delete_collection(&self, name: &str) -> Result<(), String> {
        self.cols
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| "collection not found".into())
    }

    /* CRUD wrappers ------------------------------------------------ */
    pub fn insert(
        &self,
        col: &str,
        v: String,
        vec: Vec<f32>,
        md: Option<Metadata>,
    ) -> Result<(), String> {
        let arc = self.collection_mut(col)?;
        let mut guard = arc
            .write()
            .map_err(|_| "collection lock poisoned".to_string())?;
        guard.insert(v, vec, md)
    }

    pub fn get_by_value(&self, col: &str, v: &str) -> Result<Record, String> {
        let arc = self.collection(col)?;
        let guard = arc
            .read()
            .map_err(|_| "collection lock poisoned".to_string())?;
        guard
            .get_by_value(v)
            .ok_or_else(|| "value not found".to_string())
    }

    pub fn get_by_vector(&self, col: &str, vec: &[f32]) -> Result<Record, String> {
        let arc = self.collection(col)?;
        let guard = arc
            .read()
            .map_err(|_| "collection lock poisoned".to_string())?;
        guard
            .get_by_vector(vec)
            .ok_or_else(|| "vector not found".to_string())
    }

    pub fn get_all(&self, col: &str) -> Result<Vec<Record>, String> {
        let arc = self.collection(col)?;
        let out = {
            let guard = arc
                .read()
                .map_err(|_| "collection lock poisoned".to_string())?;
            let mut tmp = Vec::with_capacity(guard.row_count());
            for row in 0..guard.row_count() {
                if guard.value_by_row(row).is_some() {
                    tmp.push(guard.row_to_record(row));
                }
            }
            tmp
        };
        Ok(out)
    }

    pub fn delete_by_value(&self, col: &str, v: &str) -> Result<(), String> {
        let arc = self.collection_mut(col)?;
        let mut guard = arc
            .write()
            .map_err(|_| "collection lock poisoned".to_string())?;
        guard.remove(v)
    }
}
