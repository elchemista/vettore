use rustler::{Env, Term, ResourceArc};
use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy)]
pub enum Distance {
    Euclidean,
    Cosine,
    DotProduct,
}

#[derive(Debug, Clone)]
pub struct Embedding {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone)]
pub struct Collection {
    pub dimension: usize,
    pub distance: Distance,
    pub embeddings: Vec<Embedding>,
}

#[derive(Debug, Default)]
pub struct CacheDB {
    pub collections: HashMap<String, Collection>,
}

impl CacheDB {
    pub fn new() -> Self {
        CacheDB {
            collections: HashMap::new(),
        }
    }
}

pub struct DBResource(pub Mutex<CacheDB>);

impl rustler::Resource for DBResource {}

impl CacheDB {
    pub fn create_collection(
        &mut self,
        name: &str,
        dimension: usize,
        distance: &str,
    ) -> Result<(), String> {
        let dist = match distance {
            "euclidean" => Distance::Euclidean,
            "cosine" => Distance::Cosine,
            "dot" => Distance::DotProduct,
            _ => return Err(format!("Unknown distance '{}'", distance)),
        };

        if self.collections.contains_key(name) {
            return Err(format!("Collection '{}' already exists", name));
        }

        let c = Collection {
            dimension,
            distance: dist,
            embeddings: Vec::new(),
        };
        self.collections.insert(name.to_string(), c);
        Ok(())
    }

    pub fn delete_collection(&mut self, name: &str) -> Result<(), String> {
        if self.collections.remove(name).is_none() {
            return Err(format!("Collection '{}' not found", name));
        }
        Ok(())
    }

    pub fn insert_embedding(
        &mut self,
        collection: &str,
        embedding: Embedding,
    ) -> Result<(), String> {
        let coll = self.collections.get_mut(collection)
            .ok_or_else(|| format!("Collection '{}' not found", collection))?;
        if embedding.vector.len() != coll.dimension {
            return Err(format!(
                "Dimension mismatch: expected {}, got {}",
                coll.dimension,
                embedding.vector.len()
            ));
        }
        if coll.embeddings.iter().any(|e| e.id == embedding.id) {
            return Err(format!("Embedding ID '{}' already exists", embedding.id));
        }
        let mut emb = embedding.clone();
        if matches!(coll.distance, Distance::Cosine) {
            emb.vector = normalize(&embedding.vector);
        }
        coll.embeddings.push(emb);
        Ok(())
    }

    pub fn get_embedding_by_id(
        &self,
        collection_name: &str,
        id: &str,
    ) -> Result<Embedding, String> {
        let coll = self.collections.get(collection_name)
            .ok_or_else(|| format!("Collection '{}' not found", collection_name))?;
        coll.embeddings
            .iter()
            .find(|emb| emb.id == id)
            .cloned()
            .ok_or_else(|| format!("Embedding '{}' not found in collection '{}'", id, collection_name))
    }

    pub fn get_embeddings(&self, collection: &str) -> Result<Vec<Embedding>, String> {
        let coll = self.collections.get(collection)
            .ok_or_else(|| format!("Collection '{}' not found", collection))?;
        Ok(coll.embeddings.clone())
    }

    pub fn similarity_search(
        &self,
        collection: &str,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<(String, f32)>, String> {
        let coll = self.collections.get(collection)
            .ok_or_else(|| format!("Collection '{}' not found", collection))?;
        if query.len() != coll.dimension {
            return Err(format!(
                "Query dimension mismatch: expected {}, got {}",
                coll.dimension,
                query.len()
            ));
        }
        let memo = get_cache_attr(coll.distance, query);
        let dist_fn = get_distance_fn(coll.distance);
        let mut scored: Vec<(Embedding, f32)> = coll.embeddings.iter()
            .map(|emb| {
                let score = dist_fn(&emb.vector, query, memo);
                (emb.clone(), score)
            })
            .collect();
        match coll.distance {
            Distance::Euclidean => {
                scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            },
            Distance::Cosine | Distance::DotProduct => {
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            }
        }
        let top_k = scored.into_iter().take(k)
            .map(|(emb, score)| (emb.id, score))
            .collect();
        Ok(top_k)
    }
}

fn get_cache_attr(metric: Distance, vec: &[f32]) -> f32 {
    match metric {
        Distance::Euclidean | Distance::DotProduct => 0.0,
        Distance::Cosine => {
            let sum_sq: f32 = vec.iter().map(|x| x * x).sum();
            sum_sq.sqrt()
        }
    }
}

fn get_distance_fn(metric: Distance) -> impl Fn(&[f32], &[f32], f32) -> f32 {
    match metric {
        Distance::Euclidean => euclidean_distance,
        Distance::Cosine | Distance::DotProduct => dot_product,
    }
}

fn euclidean_distance(a: &[f32], b: &[f32], _memo: f32) -> f32 {
    let sum_sq = a.iter().zip(b.iter())
                  .map(|(x, y)| (x - y).powi(2))
                  .sum::<f32>();
    sum_sq.sqrt()
}

fn dot_product(a: &[f32], b: &[f32], _memo: f32) -> f32 {
    a.iter().zip(b.iter())
     .map(|(x, y)| x * y)
     .sum()
}

fn normalize(vec: &[f32]) -> Vec<f32> {
    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > std::f32::EPSILON {
        vec.iter().map(|x| x / norm).collect()
    } else {
        vec.to_vec()
    }
}

#[rustler::nif]
fn new_db() -> ResourceArc<DBResource> {
    let db = CacheDB::new();
    ResourceArc::new(DBResource(Mutex::new(db)))
}

#[rustler::nif]
fn create_collection(
    db_res: ResourceArc<DBResource>,
    name: String,
    dimension: usize,
    distance: String,
) -> Result<(), String> {
    let db_resource: &DBResource = &*db_res;
    let mut db = db_resource.0.lock().map_err(|_| "Mutex poisoned")?;
    db.create_collection(&name, dimension, &distance)
}

#[rustler::nif]
fn delete_collection(
    db_res: ResourceArc<DBResource>,
    name: String,
) -> Result<(), String> {
    let db_resource: &DBResource = &*db_res;
    let mut db = db_resource.0.lock().map_err(|_| "Mutex poisoned")?;
    db.delete_collection(&name)
}

#[rustler::nif]
fn insert_embedding(
    db_res: ResourceArc<DBResource>,
    collection_name: String,
    id: String,
    vector: Vec<f32>,
    metadata: Option<HashMap<String, String>>,
) -> Result<(), String> {
    let emb = Embedding { id, vector, metadata };
    let db_resource: &DBResource = &*db_res;
    let mut db = db_resource.0.lock().map_err(|_| "Mutex poisoned")?;
    db.insert_embedding(&collection_name, emb)
}

#[rustler::nif]
fn get_embedding_by_id(
    db_res: ResourceArc<DBResource>,
    collection_name: String,
    id: String,
) -> Result<(String, Vec<f32>, Option<HashMap<String, String>>), String> {
    let db_resource: &DBResource = &*db_res;
    let db = db_resource.0.lock().map_err(|_| "Mutex poisoned")?;
    let embedding = db.get_embedding_by_id(&collection_name, &id)?;
    Ok((embedding.id, embedding.vector, embedding.metadata))
}

#[rustler::nif]
fn get_embeddings(
    db_res: ResourceArc<DBResource>,
    collection_name: String,
) -> Result<Vec<(String, Vec<f32>, Option<HashMap<String, String>>)>, String> {
    let db_resource: &DBResource = &*db_res;
    let db = db_resource.0.lock().map_err(|_| "Mutex poisoned")?;
    let embeddings = db.get_embeddings(&collection_name)?;
    Ok(embeddings.into_iter().map(|e| (e.id, e.vector, e.metadata)).collect())
}

#[rustler::nif(schedule = "DirtyCpu")]
fn similarity_search(
    db_res: ResourceArc<DBResource>,
    collection_name: String,
    query: Vec<f32>,
    k: usize,
) -> Result<Vec<(String, f32)>, String> {
    let db_resource: &DBResource = &*db_res;
    let db = db_resource.0.lock().map_err(|_| "Mutex poisoned")?;
    db.similarity_search(&collection_name, &query, k)
}

fn on_load(env: Env, _info: Term) -> bool {
    env.register::<DBResource>().is_ok()
}

rustler::init!("Elixir.Vettore", load = on_load);
