use std::collections::HashMap;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Distance {
    Euclidean,
    Cosine,
    DotProduct,
    Hnsw,
    Binary,
}

pub type Metadata = HashMap<String, String>;
