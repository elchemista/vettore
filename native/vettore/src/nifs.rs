use std::collections::HashMap;

use rustler::{Env, ResourceArc, Term};

use crate::db::{CacheDB, Collection, DBResource};
use crate::distances::compute_0_1_score;
use crate::mmr::mmr_rerank_internal;
use crate::types::{Distance, Embedding};

#[rustler::nif(schedule = "DirtyCpu")]
pub fn new_db() -> ResourceArc<DBResource> {
    ResourceArc::new(DBResource(std::sync::Mutex::new(CacheDB::new())))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn nif_create_collection(
    db: ResourceArc<DBResource>,
    name: String,
    dim: usize,
    distance: String,
    keep_embeddings: bool,
) -> Result<String, String> {
    let mut db_guard = db.0.lock().map_err(|_| "Mutex poisoned")?;
    if db_guard.collections.contains_key(&name) {
        return Err(format!("Collection '{}' already exists", name));
    }
    let mut col = Collection::create_with_distance(dim, &distance)?;
    col.keep_embeddings = keep_embeddings;
    db_guard.collections.insert(name.clone(), col);
    Ok(name)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn delete_collection(db: ResourceArc<DBResource>, name: String) -> Result<String, String> {
    let mut db_guard = db.0.lock().map_err(|_| "Mutex poisoned")?;
    db_guard
        .collections
        .remove(&name)
        .ok_or_else(|| format!("Collection '{}' not found", name))?;
    Ok(name)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn nif_insert_embedding(
    db: ResourceArc<DBResource>,
    collection: String,
    id: String,
    vector: Vec<f32>,
    metadata: Option<HashMap<String, String>>,
) -> Result<String, String> {
    let mut db_guard = db.0.lock().map_err(|_| "Mutex poisoned")?;
    let coll = db_guard
        .collections
        .get_mut(&collection)
        .ok_or_else(|| format!("Collection '{}' not found", collection))?;
    coll.insert_embedding(Embedding {
        id: id.clone(),
        vector,
        metadata,
        binary: None,
    })?;
    Ok(id)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn nif_insert_embeddings(
    db: ResourceArc<DBResource>,
    collection: String,
    embeddings: Vec<(String, Vec<f32>, Option<HashMap<String, String>>)>,
) -> Result<Vec<String>, String> {
    let mut db_guard = db.0.lock().map_err(|_| "Mutex poisoned")?;
    let coll = db_guard
        .collections
        .get_mut(&collection)
        .ok_or_else(|| format!("Collection '{}' not found", collection))?;
    let mut ids = Vec::new();
    for (id, vec, meta) in embeddings {
        coll.insert_embedding(Embedding {
            id: id.clone(),
            vector: vec,
            metadata: meta,
            binary: None,
        })?;
        ids.push(id);
    }
    Ok(ids)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn nif_similarity_search(
    db: ResourceArc<DBResource>,
    collection: String,
    query: Vec<f32>,
    k: usize,
) -> Result<Vec<(String, f32)>, String> {
    let db_guard = db.0.lock().map_err(|_| "Mutex poisoned")?;
    let coll = db_guard
        .collections
        .get(&collection)
        .ok_or_else(|| format!("Collection '{}' not found", collection))?;
    coll.get_similarity(&query, k)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn nif_similarity_search_with_filter(
    db: ResourceArc<DBResource>,
    collection: String,
    query: Vec<f32>,
    k: usize,
    filter: HashMap<String, String>,
) -> Result<Vec<(String, f32)>, String> {
    let db_guard = db.0.lock().map_err(|_| "Mutex poisoned")?;
    let coll = db_guard
        .collections
        .get(&collection)
        .ok_or_else(|| format!("Collection '{}' not found", collection))?;
    if matches!(coll.distance, Distance::Hnsw) {
        return Err("Filtering not supported with HNSW.".into());
    }
    let candidates: Vec<_> = coll
        .embeddings
        .iter()
        .filter(|e| match &e.metadata {
            Some(md) => filter.iter().all(|(k, v)| md.get(k) == Some(v)),
            None => false,
        })
        .collect();
    let mut scored: Vec<_> = candidates
        .iter()
        .map(|e| (e.id.clone(), compute_0_1_score(&query, e, coll.distance)))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.truncate(k);
    Ok(scored)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn nif_get_embedding_by_id(
    db: ResourceArc<DBResource>,
    collection: String,
    id: String,
) -> Result<(String, Vec<f32>, Option<HashMap<String, String>>), String> {
    let db_guard = db.0.lock().map_err(|_| "Mutex poisoned")?;
    let emb = db_guard.get_embedding_by_id(&collection, &id)?;
    Ok((emb.id, emb.vector, emb.metadata))
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn get_embeddings(
    db: ResourceArc<DBResource>,
    collection: String,
) -> Result<Vec<(String, Vec<f32>, Option<HashMap<String, String>>)>, String> {
    let db_guard = db.0.lock().map_err(|_| "Mutex poisoned")?;
    let coll = db_guard
        .collections
        .get(&collection)
        .ok_or_else(|| format!("Collection '{}' not found", collection))?;
    Ok(coll
        .embeddings
        .iter()
        .map(|e| (e.id.clone(), e.vector.clone(), e.metadata.clone()))
        .collect())
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn nif_delete_embedding_by_id(
    db: ResourceArc<DBResource>,
    collection: String,
    id: String,
) -> Result<String, String> {
    let mut db_guard = db.0.lock().map_err(|_| "Mutex poisoned")?;
    let coll = db_guard
        .collections
        .get_mut(&collection)
        .ok_or_else(|| format!("Collection '{}' not found", collection))?;
    coll.remove_embedding_by_id(&id)?;
    Ok(id)
}

#[rustler::nif(schedule = "DirtyCpu")]
pub fn nif_mmr_rerank(
    db: ResourceArc<DBResource>,
    collection: String,
    initial: Vec<(String, f32)>,
    alpha: f32,
    final_k: usize,
) -> Result<Vec<(String, f32)>, String> {
    let db_guard = db.0.lock().map_err(|_| "Mutex poisoned")?;
    let coll = db_guard
        .collections
        .get(&collection)
        .ok_or_else(|| format!("Collection '{}' not found", collection))?;
    let emb_map = coll
        .embeddings
        .iter()
        .map(|e| (e.id.clone(), e.vector.clone()))
        .collect::<HashMap<_, _>>();
    Ok(mmr_rerank_internal(
        &initial,
        &emb_map,
        coll.distance,
        alpha,
        final_k,
    ))
}

// Register resource on load.
fn on_load(env: Env, _info: Term) -> bool {
    env.register::<DBResource>().is_ok()
}

rustler::init!("Elixir.Vettore", load = on_load);
