use std::collections::HashMap;

use rustler::{Env, ResourceArc, Term};
use std::sync::RwLock;

use crate::distances::{
    clamp_0_1, compress_vector, hamming_distance, simd_dot_product, simd_euclidean_distance,
};
use crate::simd_utils::normalize_vec;

use crate::db::Collection;

use crate::mmr::mmr_rerank_internal;
use crate::types::{Distance, Metadata};

#[derive(Default)]
pub struct CacheDB {
    cols: HashMap<String, Collection>,
}

pub struct DBResource(pub std::sync::RwLock<CacheDB>);
impl rustler::Resource for DBResource {}

macro_rules! db_read {
    ($res:expr) => {
        $res.0.read().expect("DB RwLock poisoned")
    };
}
macro_rules! db_write {
    ($res:expr) => {
        $res.0.write().expect("DB RwLock poisoned")
    };
}

macro_rules! badarg {
    ($msg:expr) => {
        Err(format!("[vettore] {}", $msg))
    };
}

#[rustler::nif(schedule = "DirtyCpu")]
fn new_db() -> ResourceArc<DBResource> {
    ResourceArc::new(DBResource(RwLock::new(CacheDB::default())))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn create_collection(
    db: ResourceArc<DBResource>,
    name: String,
    dimension: usize,
    distance: String,
    keep_embeddings: bool,
) -> Result<String, String> {
    let mut guard = db_write!(db);
    if guard.cols.contains_key(&name) {
        return badarg!(format!("collection '{}' already exists", name));
    }
    let mut col = Collection::create_with_distance(dimension, &distance)?;
    col.keep_embeddings = keep_embeddings;
    guard.cols.insert(name.clone(), col);
    Ok(name)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn delete_collection(db: ResourceArc<DBResource>, name: String) -> Result<String, String> {
    let mut guard = db_write!(db);
    if guard.cols.remove(&name).is_none() {
        return badarg!(format!("collection '{}' not found", name));
    }
    Ok(name)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn insert_embedding(
    db: ResourceArc<DBResource>,
    col_name: String,
    id: String,
    vector: Vec<f32>,
    metadata: Option<Metadata>,
) -> Result<String, String> {
    let mut guard = db_write!(db);
    let col = guard
        .cols
        .get_mut(&col_name)
        .ok_or_else(|| format!("[vettore] collection '{}' not found", col_name))?;
    col.insert(id.clone(), vector, metadata)?;
    Ok(id)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn insert_embeddings(
    db: ResourceArc<DBResource>,
    col_name: String,
    embeddings: Vec<(String, Vec<f32>, Option<Metadata>)>,
) -> Result<Vec<String>, String> {
    let mut guard = db_write!(db);
    let col = guard
        .cols
        .get_mut(&col_name)
        .ok_or_else(|| format!("[vettore] collection '{}' not found", col_name))?;
    let mut inserted = Vec::with_capacity(embeddings.len());
    for (id, vec, md) in embeddings {
        col.insert(id.clone(), vec, md)?;
        inserted.push(id);
    }
    Ok(inserted)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn similarity_search(
    db: ResourceArc<DBResource>,
    col_name: String,
    query: Vec<f32>,
    k: usize,
) -> Result<Vec<(String, f32)>, String> {
    let guard = db_read!(db);
    let col = guard
        .cols
        .get(&col_name)
        .ok_or_else(|| format!("[vettore] collection '{}' not found", col_name))?;
    col.get_similarity(&query, k)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn similarity_search_with_filter(
    db: ResourceArc<DBResource>,
    col_name: String,
    query: Vec<f32>,
    k: usize,
    filter: Metadata,
) -> Result<Vec<(String, f32)>, String> {
    let guard = db_read!(db);
    let col = guard
        .cols
        .get(&col_name)
        .ok_or_else(|| format!("[vettore] collection '{}' not found", col_name))?;

    if matches!(col.distance, Distance::Hnsw) {
        return badarg!("metadata filtering is not supported for HNSW collections");
    }

    // linear scan with filter first
    let mut prelim: Vec<(String, f32)> = Vec::new();
    for (id, &row) in &col.id2row {
        if let Some(Some(md)) = col.meta.get(row) {
            if filter.iter().all(|(k, v)| md.get(k) == Some(v)) {
                let vec_slice = &col.vectors[row * col.dimension..(row + 1) * col.dimension];
                let score = crate::distances::score(
                    &query,
                    vec_slice,
                    col.binary[row].as_ref(),
                    col.distance,
                );
                prelim.push((id.clone(), score));
            }
        }
    }
    prelim.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    prelim.truncate(k);
    Ok(prelim)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn get_embedding_by_id(
    db: ResourceArc<DBResource>,
    col_name: String,
    id: String,
) -> Result<(String, Vec<f32>, Option<Metadata>), String> {
    let guard = db_read!(db);
    let col = guard
        .cols
        .get(&col_name)
        .ok_or_else(|| format!("[vettore] collection '{}' not found", col_name))?;
    let row = *col
        .id2row
        .get(&id)
        .ok_or_else(|| format!("[vettore] id '{}' not found", id))?;
    let vec_slice = &col.vectors[row * col.dimension..(row + 1) * col.dimension];
    Ok((id, vec_slice.to_vec(), col.meta[row].clone()))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn get_all_embeddings(
    db: ResourceArc<DBResource>,
    col_name: String,
) -> Result<Vec<(String, Vec<f32>, Option<Metadata>)>, String> {
    let guard = db_read!(db);
    let col = guard
        .cols
        .get(&col_name)
        .ok_or_else(|| format!("[vettore] collection '{}' not found", col_name))?;
    let mut out = Vec::with_capacity(col.id2row.len());
    for (id, &row) in &col.id2row {
        out.push(col.row_to_tuple(id, row));
    }
    Ok(out)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn delete_embedding_by_id(
    db: ResourceArc<DBResource>,
    col_name: String,
    id: String,
) -> Result<String, String> {
    let mut guard = db_write!(db);
    let col = guard
        .cols
        .get_mut(&col_name)
        .ok_or_else(|| format!("[vettore] collection '{}' not found", col_name))?;
    col.remove(&id)?;
    Ok(id)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn mmr_rerank(
    db: ResourceArc<DBResource>,
    col_name: String,
    initial: Vec<(String, f32)>,
    alpha: f32,
    final_k: usize,
) -> Result<Vec<(String, f32)>, String> {
    let guard = db_read!(db);
    let col = guard
        .cols
        .get(&col_name)
        .ok_or_else(|| format!("[vettore] collection '{}' not found", col_name))?;
    let embed_map = col
        .id2row
        .iter()
        .map(|(id, &row)| {
            let vec_slice = &col.vectors[row * col.dimension..(row + 1) * col.dimension];
            (id.clone(), vec_slice.to_vec())
        })
        .collect::<HashMap<_, _>>();
    Ok(mmr_rerank_internal(
        &initial,
        &embed_map,
        col.distance,
        alpha,
        final_k,
    ))
}

// Core standalone distance‑algorithm NIFs

#[rustler::nif(schedule = "DirtyCpu")]
fn euclidean_distance(vec_a: Vec<f32>, vec_b: Vec<f32>) -> Result<f32, String> {
    if vec_a.len() != vec_b.len() {
        return badarg!("dimension mismatch");
    }
    let d = simd_euclidean_distance(&vec_a, &vec_b);
    Ok(clamp_0_1(1.0 / (1.0 + d))) // convert to similarity ∈ [0,1]
}

#[rustler::nif(schedule = "DirtyCpu")]
fn cosine_similarity(vec_a: Vec<f32>, vec_b: Vec<f32>) -> Result<f32, String> {
    if vec_a.len() != vec_b.len() {
        return badarg!("dimension mismatch");
    }
    let na = normalize_vec(&vec_a);
    let nb = normalize_vec(&vec_b);
    let sim = (simd_dot_product(&na, &nb) + 1.0) / 2.0;
    Ok(clamp_0_1(sim))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn dot_product(vec_a: Vec<f32>, vec_b: Vec<f32>) -> Result<f32, String> {
    if vec_a.len() != vec_b.len() {
        return badarg!("dimension mismatch");
    }
    Ok(simd_dot_product(&vec_a, &vec_b))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn hamming_distance_bits(bits_a: Vec<u64>, bits_b: Vec<u64>) -> Result<u32, String> {
    if bits_a.len() != bits_b.len() {
        return badarg!("length mismatch");
    }
    Ok(hamming_distance(&bits_a, &bits_b))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn compress_f32_vector(vec: Vec<f32>) -> Vec<u64> {
    compress_vector(&vec)
}

/// Convert a distance string sent from Elixir into the internal `Distance` enum.
fn distance_from_str(dist: &str) -> Result<Distance, String> {
    match dist.to_lowercase().as_str() {
        "euclidean" | "l2" => Ok(Distance::Euclidean),
        "cosine" => Ok(Distance::Cosine),
        "dot" | "dotproduct" => Ok(Distance::DotProduct),
        "binary" | "hamming" => Ok(Distance::Binary),
        "hnsw" => Ok(Distance::Hnsw),
        _ => badarg!("unknown distance"),
    }
}

/// MMR reranking directly on the initial and embedding vectors.
#[rustler::nif(schedule = "DirtyCpu")]
fn mmr_rerank_embeddings(
    initial: Vec<(String, f32)>,
    embeddings: Vec<(String, Vec<f32>)>,
    distance: String,
    alpha: f32,
    final_k: usize,
) -> Result<Vec<(String, f32)>, String> {
    let dist = distance_from_str(&distance)?;
    let embed_map: HashMap<_, _> = embeddings.into_iter().collect();
    Ok(mmr_rerank_internal(
        &initial, &embed_map, dist, alpha, final_k,
    ))
}

fn on_load(env: Env, _info: Term) -> bool {
    env.register::<DBResource>().is_ok()
}

rustler::init!("Elixir.Vettore.Nifs", load = on_load);
