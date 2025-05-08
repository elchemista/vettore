//! nifs.rs – Rustler bridge (adapted for DashMap + Arc<RwLock<…>>)

use std::collections::HashMap;

use rustler::{Env, ResourceArc, Term};

use crate::distances::{
    clamp_0_1, compress_vector, hamming_distance, simd_dot_product, simd_euclidean_distance,
};
use crate::simd_utils::normalize_vec;

use crate::db::VettoreDB;
use crate::mmr::mmr_rerank as algo_mmr_rerank;
use crate::similarity::similarity_search as algo_sim_search;
use crate::types::{Distance, Metadata};

/* ───────────────────────── resource wrapper ───────────────────────── */
pub struct DBResource(pub VettoreDB);
impl rustler::Resource for DBResource {}

/* handy macro – just give me &VettoreDB */
macro_rules! db_ref {
    ($arc:expr) => {
        &$arc.0
    };
}

/* tiny helper for uniform error */
macro_rules! badarg {
    ($msg:expr) => {
        Err(format!("[vettore] {}", $msg))
    };
}

/* ───────────────────────── NIFs ───────────────────────── */
#[rustler::nif(schedule = "DirtyCpu")]
fn new_db() -> ResourceArc<DBResource> {
    ResourceArc::new(DBResource(VettoreDB::default()))
}

/* collection management */
#[rustler::nif(schedule = "DirtyCpu")]
fn create_collection(
    db: ResourceArc<DBResource>,
    name: String,
    dim: usize,
    distance: String,
    keep_embeddings: bool,
) -> Result<String, String> {
    db_ref!(db).create_collection(name.clone(), dim, &distance, keep_embeddings)?;
    Ok(name)
}

#[rustler::nif(schedule = "DirtyCpu")]
fn delete_collection(db: ResourceArc<DBResource>, name: String) -> Result<String, String> {
    db_ref!(db).delete_collection(&name)?;
    Ok(name)
}

/* single insert */
#[rustler::nif(schedule = "DirtyCpu")]
fn insert_embedding(
    db: ResourceArc<DBResource>,
    col_name: String,
    value: String,
    vector: Vec<f32>,
    metadata: Option<Metadata>,
) -> Result<String, String> {
    db_ref!(db).insert(&col_name, value.clone(), vector, metadata)?;
    Ok(value)
}

/* batch insert (each insert locks only its own collection) */
#[rustler::nif(schedule = "DirtyCpu")]
fn insert_embeddings(
    db: ResourceArc<DBResource>,
    col_name: String,
    embeddings: Vec<(String, Vec<f32>, Option<Metadata>)>,
) -> Result<Vec<String>, String> {
    let db_ref = db_ref!(db);
    let mut out = Vec::with_capacity(embeddings.len());
    for (value, vec, md) in embeddings {
        db_ref.insert(&col_name, value.clone(), vec, md)?;
        out.push(value);
    }
    Ok(out)
}

/* look-ups */
#[rustler::nif(schedule = "DirtyCpu")]
fn get_embedding_by_value(
    db: ResourceArc<DBResource>,
    col_name: String,
    value: String,
) -> Result<(String, Vec<f32>, Option<Metadata>), String> {
    let rec = db_ref!(db).get_by_value(&col_name, &value)?;
    Ok((rec.value, rec.vector, rec.metadata))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn get_embedding_by_vector(
    db: ResourceArc<DBResource>,
    col_name: String,
    vector: Vec<f32>,
) -> Result<(String, Vec<f32>, Option<Metadata>), String> {
    let rec = db_ref!(db).get_by_vector(&col_name, &vector)?;
    Ok((rec.value, rec.vector, rec.metadata))
}

/* dump */
#[rustler::nif(schedule = "DirtyCpu")]
fn get_all_embeddings(
    db: ResourceArc<DBResource>,
    col_name: String,
) -> Result<Vec<(String, Vec<f32>, Option<Metadata>)>, String> {
    let recs = db_ref!(db).get_all(&col_name)?;
    Ok(recs
        .into_iter()
        .map(|r| (r.value, r.vector, r.metadata))
        .collect())
}

/* similarity search – uses read-guard */
#[rustler::nif(schedule = "DirtyCpu")]
fn similarity_search(
    db: ResourceArc<DBResource>,
    col_name: String,
    query: Vec<f32>,
    k: usize,
) -> Result<Vec<(String, f32)>, String> {
    let arc = db_ref!(db).collection(&col_name)?;
    let guard = arc
        .read()
        .map_err(|_| "collection lock poisoned".to_string())?;
    algo_sim_search(&*guard, &query, k)
}

/* delete one */
#[rustler::nif(schedule = "DirtyCpu")]
fn delete_embedding_by_value(
    db: ResourceArc<DBResource>,
    col_name: String,
    value: String,
) -> Result<String, String> {
    db_ref!(db).delete_by_value(&col_name, &value)?;
    Ok(value)
}

/* MMR rerank (needs vectors of the initial hits) */
#[rustler::nif(schedule = "DirtyCpu")]
fn mmr_rerank(
    db: ResourceArc<DBResource>,
    col_name: String,
    initial: Vec<(String, f32)>,
    alpha: f32,
    final_k: usize,
) -> Result<Vec<(String, f32)>, String> {
    let arc = db_ref!(db).collection(&col_name)?;
    let guard = arc
        .read()
        .map_err(|_| "collection lock poisoned".to_string())?;

    let mut embed_map: HashMap<String, Vec<f32>> = HashMap::with_capacity(initial.len());
    for (val, _) in &initial {
        if let Some(rec) = guard.get_by_value(val) {
            embed_map.insert(val.clone(), rec.vector);
        }
    }

    Ok(algo_mmr_rerank(
        &initial,
        &embed_map,
        guard.distance,
        alpha,
        final_k,
    ))
}

/* ───────────── pure helpers exposed as NIFs ───────────── */
#[rustler::nif(schedule = "DirtyCpu")]
fn euclidean_distance(a: Vec<f32>, b: Vec<f32>) -> Result<f32, String> {
    if a.len() != b.len() {
        return badarg!("dimension mismatch");
    }
    Ok(clamp_0_1(1.0 / (1.0 + simd_euclidean_distance(&a, &b))))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn cosine_similarity(a: Vec<f32>, b: Vec<f32>) -> Result<f32, String> {
    if a.len() != b.len() {
        return badarg!("dimension mismatch");
    }
    let sim = (simd_dot_product(&normalize_vec(&a), &normalize_vec(&b)) + 1.0) / 2.0;
    Ok(clamp_0_1(sim))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn dot_product(a: Vec<f32>, b: Vec<f32>) -> Result<f32, String> {
    if a.len() != b.len() {
        return badarg!("dimension mismatch");
    }
    Ok(simd_dot_product(&a, &b))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn hamming_distance_bits(a: Vec<u64>, b: Vec<u64>) -> Result<u32, String> {
    if a.len() != b.len() {
        return badarg!("length mismatch");
    }
    Ok(hamming_distance(&a, &b))
}

#[rustler::nif(schedule = "DirtyCpu")]
fn compress_f32_vector(v: Vec<f32>) -> Vec<u64> {
    compress_vector(&v)
}

/* distance string → enum */
fn distance_from_str(s: &str) -> Result<Distance, String> {
    match s.to_lowercase().as_str() {
        "euclidean" | "l2" => Ok(Distance::Euclidean),
        "cosine" => Ok(Distance::Cosine),
        "dot" | "dotproduct" => Ok(Distance::DotProduct),
        "binary" | "hamming" => Ok(Distance::Binary),
        "hnsw" => Ok(Distance::Hnsw),
        _ => badarg!("unknown distance"),
    }
}

/* NIF form of standalone MMR (no DB involved) */
#[rustler::nif(schedule = "DirtyCpu")]
fn mmr_rerank_embeddings(
    initial: Vec<(String, f32)>,
    embeddings: Vec<(String, Vec<f32>)>,
    distance: String,
    alpha: f32,
    final_k: usize,
) -> Result<Vec<(String, f32)>, String> {
    let dist = distance_from_str(&distance)?;
    Ok(algo_mmr_rerank(
        &initial,
        &embeddings.into_iter().collect(),
        dist,
        alpha,
        final_k,
    ))
}

/* on-load */
fn on_load(env: Env, _info: Term) -> bool {
    env.register::<DBResource>().is_ok()
}

rustler::init!("Elixir.Vettore.Nifs", load = on_load);
