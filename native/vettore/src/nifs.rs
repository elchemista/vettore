//! Rustler NIF boundary for Vettore's native algorithms.
//!
//! Keep this file thin: each NIF should expose a named algorithm operation and
//! delegate the work to focused modules such as `distances`, `hnsw`, or
//! `muvera`.

use rustler::{NifResult, ResourceArc};

use crate::distances::Metric;
use crate::hnsw::{HnswIndex, HnswResource};

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes L2/Euclidean distance between two f32 vectors.
fn l2_distance(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::compute(Metric::L2, &left, &right))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes squared L2 distance without the final square root.
fn l2_squared_distance(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::compute(Metric::L2Squared, &left, &right))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes raw cosine similarity; callers normalize vectors when required.
fn cosine_similarity(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::compute(Metric::Cosine, &left, &right))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes raw inner product.
fn inner_product(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::compute(
        Metric::InnerProduct,
        &left,
        &right,
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes negative inner product for distance-style ordering.
fn negative_inner_product(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::compute(
        Metric::NegativeInnerProduct,
        &left,
        &right,
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes Manhattan/L1 distance.
fn manhattan_distance(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::compute(Metric::Manhattan, &left, &right))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes Chebyshev/L-infinity distance.
fn chebyshev_distance(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::compute(Metric::Chebyshev, &left, &right))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes Hamming distance over truthy/non-truthy coordinates.
fn hamming_distance(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::compute(Metric::Hamming, &left, &right))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes Jaccard distance over truthy/non-truthy coordinates.
fn jaccard_distance(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::compute(Metric::Jaccard, &left, &right))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// L2-normalizes a vector in native code.
fn normalize_l2(vector: Vec<f32>) -> NifResult<Result<Vec<f32>, String>> {
    Ok(crate::distances::normalize_l2(vector))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Z-score normalizes a vector in native code.
fn normalize_zscore(vector: Vec<f32>) -> NifResult<Result<Vec<f32>, String>> {
    Ok(crate::distances::normalize_zscore(vector))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Min-max normalizes a vector in native code.
fn normalize_minmax(vector: Vec<f32>) -> NifResult<Result<Vec<f32>, String>> {
    Ok(crate::distances::normalize_minmax(vector))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Compresses signs into integer bits for compatibility with old helpers.
fn compress_sign_bits(vector: Vec<f32>) -> Vec<u64> {
    crate::distances::compress_sign_bits(&vector)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Creates a native HNSW graph ordered by L2 distance.
fn hnsw_new_l2() -> ResourceArc<HnswResource> {
    hnsw_new(Metric::L2)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Creates a native HNSW graph ordered by cosine rank distance.
fn hnsw_new_cosine() -> ResourceArc<HnswResource> {
    hnsw_new(Metric::Cosine)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Creates a native HNSW graph ordered by inner-product rank distance.
fn hnsw_new_inner_product() -> ResourceArc<HnswResource> {
    hnsw_new(Metric::InnerProduct)
}

/// Allocates the Rust resource that owns only ANN graph state.
fn hnsw_new(metric: Metric) -> ResourceArc<HnswResource> {
    ResourceArc::new(HnswResource(std::sync::RwLock::new(HnswIndex::new(metric))))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Inserts or replaces one vector in the native HNSW graph.
fn hnsw_insert(
    index: ResourceArc<HnswResource>,
    id: String,
    vector: Vec<f32>,
) -> Result<(), String> {
    let mut guard = index
        .0
        .write()
        .map_err(|_| "hnsw lock poisoned".to_string())?;
    guard.insert(id, vector)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Removes one vector from the native HNSW graph.
fn hnsw_delete(index: ResourceArc<HnswResource>, id: String) -> Result<(), String> {
    let mut guard = index
        .0
        .write()
        .map_err(|_| "hnsw lock poisoned".to_string())?;
    guard.delete(&id);
    Ok(())
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Searches the native HNSW graph and returns external ids plus raw metric values.
fn hnsw_search(
    index: ResourceArc<HnswResource>,
    query: Vec<f32>,
    limit: usize,
) -> Result<Vec<(String, f32)>, String> {
    let guard = index
        .0
        .read()
        .map_err(|_| "hnsw lock poisoned".to_string())?;
    guard.search(&query, limit)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Encodes query-side MUVERA/FDE vectors by summing projected partition vectors.
fn muvera_encode_query(
    vectors: Vec<Vec<f32>>,
    dimension: usize,
    num_repetitions: usize,
    num_simhash_projections: usize,
    seed: u64,
    projection_dimension: usize,
    final_projection_dimension: Option<usize>,
) -> NifResult<Result<Vec<f32>, String>> {
    Ok(crate::muvera::encode(
        vectors,
        crate::muvera::Config {
            dimension,
            num_repetitions,
            num_simhash_projections,
            seed,
            projection_dimension,
            final_projection_dimension,
        },
        crate::muvera::Mode::Query,
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Encodes document-side MUVERA/FDE vectors by averaging projected partition vectors.
fn muvera_encode_document(
    vectors: Vec<Vec<f32>>,
    dimension: usize,
    num_repetitions: usize,
    num_simhash_projections: usize,
    seed: u64,
    projection_dimension: usize,
    final_projection_dimension: Option<usize>,
) -> NifResult<Result<Vec<f32>, String>> {
    Ok(crate::muvera::encode(
        vectors,
        crate::muvera::Config {
            dimension,
            num_repetitions,
            num_simhash_projections,
            seed,
            projection_dimension,
            final_projection_dimension,
        },
        crate::muvera::Mode::Document,
    ))
}

/// Registers Rust resources with the BEAM VM when the NIF library loads.
fn load(env: rustler::Env, _term: rustler::Term) -> bool {
    rustler::resource!(HnswResource, env)
}

rustler::init!("Elixir.Vettore.Nifs", load = load);
