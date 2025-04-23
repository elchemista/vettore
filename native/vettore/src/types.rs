use std::collections::HashMap;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Distance {
    Euclidean,
    Cosine,
    DotProduct,
    Hnsw,
    Binary,
}

#[derive(Clone)]
pub struct Embedding {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<HashMap<String, String>>,
    pub binary: Option<Vec<u64>>, // sign‑bit compression cache for Binary distance
}
