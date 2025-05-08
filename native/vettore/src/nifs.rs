use std::collections::HashMap;

use rustler::{Env, ResourceArc, Term};
use std::sync::RwLock;

use crate::distances::{
    clamp_0_1, compress_vector, hamming_distance, simd_dot_product, simd_euclidean_distance,
};
use crate::simd_utils::normalize_vec;

use crate::db::VettoreDB;

use crate::mmr::mmr_rerank as algo_mmr_rerank;

use crate::similarity::similarity_search as algo_sim_search;

use crate::types::{Distance, Metadata};

pub struct DBResource(pub RwLock<VettoreDB>);
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
    ResourceArc::new(DBResource(RwLock::new(VettoreDB::default())))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn create_collection(
    db: ResourceArc<DBResource>,
    name: String,
    dimension: usize,
    distance: String,
    keep_embeddings: bool,
) -> Result<String, String> {
    db_write!(db).create_collection(name.clone(), dimension, &distance, keep_embeddings)?;
    Ok(name)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn delete_collection(db: ResourceArc<DBResource>, name: String) -> Result<String, String> {
    db_write!(db).delete_collection(&name)?;
    Ok(name)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn insert_embedding(
    db: ResourceArc<DBResource>,
    col_name: String,
    value: String,
    vector: Vec<f32>,
    metadata: Option<Metadata>,
) -> Result<String, String> {
    db_write!(db).insert(&col_name, value.clone(), vector, metadata)?;
    Ok(value)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn insert_embeddings(
    db: ResourceArc<DBResource>,
    col_name: String,
    embeddings: Vec<(String, Vec<f32>, Option<Metadata>)>,
) -> Result<Vec<String>, String> {
    let mut out = Vec::with_capacity(embeddings.len());
    let mut guard = db_write!(db);
    for (value, vec, md) in embeddings {
        guard.insert(&col_name, value.clone(), vec, md)?;
        out.push(value);
    }
    Ok(out)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn get_embedding_by_value(
    db: ResourceArc<DBResource>,
    col_name: String,
    value: String,
) -> Result<(String, Vec<f32>, Option<Metadata>), String> {
    let rec = db_read!(db).get_by_value(&col_name, &value)?;
    Ok((rec.value, rec.vector, rec.metadata))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn get_embedding_by_vector(
    db: ResourceArc<DBResource>,
    col_name: String,
    vector: Vec<f32>,
) -> Result<(String, Vec<f32>, Option<Metadata>), String> {
    let rec = db_read!(db).get_by_vector(&col_name, &vector)?;
    Ok((rec.value, rec.vector, rec.metadata))
}

/// Fetch *all* embeddings from a collection.
#[rustler::nif(schedule = "DirtyCpu")]
fn get_all_embeddings(
    db: ResourceArc<DBResource>,
    col_name: String,
) -> Result<Vec<(String, Vec<f32>, Option<Metadata>)>, String> {
    let recs = db_read!(db).get_all(&col_name)?;
    Ok(recs
        .into_iter()
        .map(|r| (r.value, r.vector, r.metadata))
        .collect())
}

#[rustler::nif(schedule = "DirtyCpu")]
fn similarity_search(
    db: ResourceArc<DBResource>,
    col_name: String,
    query: Vec<f32>,
    k: usize,
) -> Result<Vec<(String, f32)>, String> {
    // alias the pure algorithm to avoid a name clash with this NIF

    // 1. lock the DB and fetch the collection (read-only)
    let guard = db_read!(db);
    let collection = guard.collection(&col_name)?;

    // 2. delegate to the pure SIMD search
    algo_sim_search(collection, &query, k)
}

/* Delete by value (string) */
#[rustler::nif(schedule = "DirtyCpu")]
fn delete_embedding_by_value(
    db: ResourceArc<DBResource>,
    col_name: String,
    value: String,
) -> Result<String, String> {
    db_write!(db).delete_by_value(&col_name, &value)?;
    Ok(value)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn mmr_rerank(
    db: ResourceArc<DBResource>,
    col_name: String,
    initial: Vec<(String, f32)>,
    alpha: f32,
    final_k: usize,
) -> Result<Vec<(String, f32)>, String> {
    // 1. grab the collection (read-only)
    let guard = db_read!(db);
    let collection = guard.collection(&col_name)?;

    // 2. build value → vector map only for the candidates in `initial`
    let mut embed_map: HashMap<String, Vec<f32>> = HashMap::with_capacity(initial.len());

    for (val, _) in &initial {
        if let Some(rec) = collection.get_by_value(val) {
            embed_map.insert(val.clone(), rec.vector);
        }
    }

    // 3. call the pure algorithm and return
    Ok(algo_mmr_rerank(
        &initial,
        &embed_map,
        collection.distance,
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
    Ok(algo_mmr_rerank(&initial, &embed_map, dist, alpha, final_k))
}

fn on_load(env: Env, _info: Term) -> bool {
    env.register::<DBResource>().is_ok()
}

rustler::init!("Elixir.Vettore.Nifs", load = on_load);
