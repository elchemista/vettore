//! Rustler NIF boundary for Vettore's native algorithms.
//!
//! Keep this file thin: each NIF should expose a named algorithm operation and
//! delegate the work to focused modules such as `distances`, `hnsw`, or
//! `muvera`.

use rustler::{Binary, NifResult, ResourceArc};

use crate::distances::Metric;
use crate::flat::{FlatIndex, FlatResource};
use crate::hnsw::{HnswIndex, HnswParams, HnswResource};

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes L2/Euclidean distance between two f32 vectors.
fn l2_distance(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::compute_checked(Metric::L2, &left, &right))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes squared L2 distance without the final square root.
fn l2_squared_distance(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::compute_checked(
        Metric::L2Squared,
        &left,
        &right,
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes the raw dot-product compatibility helper used by `normalize: :none`.
fn cosine_similarity(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::compute_checked(
        Metric::InnerProduct,
        &left,
        &right,
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes true cosine similarity without allocating normalized vectors.
fn normalized_cosine_similarity(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::validate_finite_vector(&left)
        .and_then(|()| crate::distances::validate_finite_vector(&right))
        .and_then(|()| crate::distances::cosine(&left, &right)))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes raw inner product.
fn inner_product(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::compute_checked(
        Metric::InnerProduct,
        &left,
        &right,
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes negative inner product for distance-style ordering.
fn negative_inner_product(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::compute_checked(
        Metric::NegativeInnerProduct,
        &left,
        &right,
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes Manhattan/L1 distance.
fn manhattan_distance(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::compute_checked(
        Metric::Manhattan,
        &left,
        &right,
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes Chebyshev/L-infinity distance.
fn chebyshev_distance(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::compute_checked(
        Metric::Chebyshev,
        &left,
        &right,
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes Hamming distance over truthy/non-truthy coordinates.
fn hamming_distance(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::compute_checked(
        Metric::Hamming,
        &left,
        &right,
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes Jaccard distance over truthy/non-truthy coordinates.
fn jaccard_distance(left: Vec<f32>, right: Vec<f32>) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::compute_checked(
        Metric::Jaccard,
        &left,
        &right,
    ))
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
/// Decodes a little-endian f32 binary without allocating BEAM float terms first.
fn decode_f32_binary(binary: Binary<'_>) -> NifResult<Result<Vec<f32>, String>> {
    Ok(crate::dense::decode_f32_le(binary.as_slice()))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes a named metric directly over little-endian f32 binaries.
fn metric_f32_binary(
    left: Binary<'_>,
    right: Binary<'_>,
    metric_code: u8,
) -> NifResult<Result<f32, String>> {
    Ok(Metric::from_code(metric_code)
        .and_then(|metric| crate::dense::metric_f32_le(left.as_slice(), right.as_slice(), metric)))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Normalizes one little-endian f32 binary into native f32 coordinates.
fn normalize_f32_binary(
    binary: Binary<'_>,
    normalization_code: u8,
) -> NifResult<Result<Vec<f32>, String>> {
    Ok(crate::dense::normalize_f32_le(
        binary.as_slice(),
        normalization_code,
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Mean-pools selected rows from a row-major little-endian f32 matrix.
fn mean_pool_f32(
    matrix: Binary<'_>,
    dimensions: usize,
    row_indices: Vec<usize>,
) -> NifResult<Result<Vec<f32>, String>> {
    Ok(crate::dense::mean_pool_f32_le(
        matrix.as_slice(),
        dimensions,
        &row_indices,
    ))
}

#[rustler::nif(schedule = "DirtyIo")]
/// Reports whether a non-software GPU adapter can be initialized through wgpu.
fn gpu_detected() -> bool {
    crate::gpu::detected()
}

#[rustler::nif(schedule = "DirtyIo")]
/// Returns stable diagnostic information for the selected GPU adapter.
fn gpu_info() -> Result<(String, String, String), String> {
    crate::gpu::info().map(|info| (info.name, info.backend, info.device_type))
}

#[rustler::nif(schedule = "DirtyIo")]
/// Computes a named dense-vector metric with the native wgpu pipeline.
fn gpu_metric(left: Vec<f32>, right: Vec<f32>, metric_code: u8) -> NifResult<Result<f32, String>> {
    Ok(Metric::from_code(metric_code).and_then(|metric| crate::gpu::metric(&left, &right, metric)))
}

#[rustler::nif(schedule = "DirtyIo")]
/// Computes a GPU metric over little-endian f32 binaries without BEAM float terms.
fn gpu_metric_f32_binary(
    left: Binary<'_>,
    right: Binary<'_>,
    metric_code: u8,
) -> NifResult<Result<f32, String>> {
    Ok(Metric::from_code(metric_code).and_then(|metric| {
        let left = crate::dense::decode_f32_le(left.as_slice())?;
        let right = crate::dense::decode_f32_le(right.as_slice())?;
        crate::gpu::metric(&left, &right, metric)
    }))
}

#[rustler::nif(schedule = "DirtyIo")]
/// Normalizes one dense vector with the native wgpu pipeline.
fn gpu_normalize(vector: Vec<f32>, normalization_code: u8) -> NifResult<Result<Vec<f32>, String>> {
    Ok(crate::gpu::normalize(vector, normalization_code))
}

#[rustler::nif(schedule = "DirtyIo")]
/// GPU-normalizes a little-endian f32 binary without BEAM float terms.
fn gpu_normalize_f32_binary(
    binary: Binary<'_>,
    normalization_code: u8,
) -> NifResult<Result<Vec<f32>, String>> {
    Ok(crate::dense::decode_f32_le(binary.as_slice())
        .and_then(|vector| crate::gpu::normalize(vector, normalization_code)))
}

#[rustler::nif(schedule = "DirtyIo")]
/// Mean-pools selected matrix rows with the native wgpu pipeline.
fn gpu_mean_pool_f32(
    matrix: Binary<'_>,
    dimensions: usize,
    row_indices: Vec<usize>,
) -> NifResult<Result<Vec<f32>, String>> {
    Ok(crate::gpu::mean_pool_f32_le(
        matrix.as_slice(),
        dimensions,
        &row_indices,
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Compresses signs into integer bits for compatibility with old helpers.
fn compress_sign_bits(vector: Vec<f32>) -> Vec<u64> {
    crate::distances::compress_sign_bits(&vector)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes Hamming distance between packed bit vectors.
fn packed_hamming_distance(
    left: Vec<u64>,
    right: Vec<u64>,
    dimensions: usize,
) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::packed_hamming(&left, &right, dimensions))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes Jaccard distance between packed bit vectors.
fn packed_jaccard_distance(
    left: Vec<u64>,
    right: Vec<u64>,
    dimensions: usize,
) -> NifResult<Result<f32, String>> {
    Ok(crate::distances::packed_jaccard(&left, &right, dimensions))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Scores a vector batch and returns only the best prefix-aware results.
fn vector_top_k(
    vectors: Vec<(String, Vec<f32>)>,
    query: Vec<f32>,
    metric_code: u8,
    dimensions: usize,
    limit: usize,
) -> NifResult<Result<Vec<(String, f32)>, String>> {
    Ok(Metric::from_code(metric_code)
        .and_then(|metric| crate::search::vector_top_k(vectors, &query, metric, dimensions, limit)))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Runs a packed-Hamming top-k pass over a whole candidate batch.
fn binary_top_k(
    vectors: Vec<(String, Vec<u64>)>,
    query: Vec<u64>,
    dimensions: usize,
    limit: usize,
) -> NifResult<Result<Vec<(String, f32)>, String>> {
    Ok(crate::search::binary_top_k(
        vectors, &query, dimensions, limit,
    ))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Runs a complete MMR pass without repeated BEAM/native transitions.
fn mmr_rerank(
    initial: Vec<(String, f64)>,
    embeddings: Vec<(String, Vec<f32>)>,
    metric_code: u8,
    alpha: f64,
    final_k: usize,
) -> NifResult<Result<Vec<String>, String>> {
    Ok(Metric::from_code(metric_code)
        .and_then(|metric| crate::search::mmr_rerank(initial, embeddings, metric, alpha, final_k)))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Computes one native MaxSim/ColBERT score.
fn multi_vector_score(
    query_vectors: Vec<Vec<f32>>,
    document_vectors: Vec<Vec<f32>>,
    metric_code: u8,
) -> NifResult<Result<f32, String>> {
    Ok(Metric::from_code(metric_code)
        .and_then(|metric| crate::multi_vector::score(&query_vectors, &document_vectors, metric)))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Scores and selects a whole multi-vector document batch in one native call.
fn multi_vector_top_k(
    documents: Vec<(String, Vec<Vec<f32>>)>,
    query_vectors: Vec<Vec<f32>>,
    metric_code: u8,
    limit: usize,
) -> NifResult<Result<Vec<(String, f32)>, String>> {
    Ok(Metric::from_code(metric_code)
        .and_then(|metric| crate::multi_vector::top_k(documents, &query_vectors, metric, limit)))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Creates a native exact flat index ordered by L2 distance.
fn flat_new_l2() -> ResourceArc<FlatResource> {
    flat_new(Metric::L2)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Creates a native exact flat index ordered by squared L2 distance.
fn flat_new_l2_squared() -> ResourceArc<FlatResource> {
    flat_new(Metric::L2Squared)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Creates a native exact flat index ordered by cosine rank distance.
fn flat_new_cosine() -> ResourceArc<FlatResource> {
    flat_new(Metric::Cosine)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Creates a native exact flat index ordered by inner product.
fn flat_new_inner_product() -> ResourceArc<FlatResource> {
    flat_new(Metric::InnerProduct)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Creates a native exact flat index ordered by negative inner product.
fn flat_new_negative_inner_product() -> ResourceArc<FlatResource> {
    flat_new(Metric::NegativeInnerProduct)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Creates a native exact flat index ordered by Manhattan distance.
fn flat_new_manhattan() -> ResourceArc<FlatResource> {
    flat_new(Metric::Manhattan)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Creates a native exact flat index ordered by Chebyshev distance.
fn flat_new_chebyshev() -> ResourceArc<FlatResource> {
    flat_new(Metric::Chebyshev)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Creates a native exact flat index ordered by Hamming distance.
fn flat_new_hamming() -> ResourceArc<FlatResource> {
    flat_new(Metric::Hamming)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Creates a native exact flat index ordered by Jaccard distance.
fn flat_new_jaccard() -> ResourceArc<FlatResource> {
    flat_new(Metric::Jaccard)
}

/// Allocates the Rust resource that owns exact flat vector state.
fn flat_new(metric: Metric) -> ResourceArc<FlatResource> {
    ResourceArc::new(FlatResource(std::sync::RwLock::new(FlatIndex::new(metric))))
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Inserts or replaces one vector in the native flat index.
fn flat_insert(
    index: ResourceArc<FlatResource>,
    id: String,
    vector: Vec<f32>,
) -> Result<(), String> {
    let mut guard = index
        .0
        .write()
        .map_err(|_| "flat lock poisoned".to_string())?;
    guard.insert(id, vector)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Inserts or replaces a batch of vectors in the native flat index.
fn flat_insert_many(
    index: ResourceArc<FlatResource>,
    vectors: Vec<(String, Vec<f32>)>,
) -> Result<(), String> {
    let mut guard = index
        .0
        .write()
        .map_err(|_| "flat lock poisoned".to_string())?;
    guard.insert_many(vectors)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Removes one vector from the native flat index.
fn flat_delete(index: ResourceArc<FlatResource>, id: String) -> Result<(), String> {
    let mut guard = index
        .0
        .write()
        .map_err(|_| "flat lock poisoned".to_string())?;
    guard.delete(&id);
    Ok(())
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Releases all vectors retained by a native flat index.
fn flat_clear(index: ResourceArc<FlatResource>) -> Result<(), String> {
    let mut guard = index
        .0
        .write()
        .map_err(|_| "flat lock poisoned".to_string())?;
    guard.clear();
    Ok(())
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Searches the native flat index and returns external ids plus raw metric values.
fn flat_search(
    index: ResourceArc<FlatResource>,
    query: Vec<f32>,
    limit: usize,
) -> Result<Vec<(String, f32)>, String> {
    let guard = index
        .0
        .read()
        .map_err(|_| "flat lock poisoned".to_string())?;
    guard.search(&query, limit)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Creates a native HNSW graph ordered by L2 distance.
fn hnsw_new_l2(
    m: usize,
    m0: usize,
    ef_construction: usize,
    ef_search: usize,
    max_level: usize,
) -> Result<ResourceArc<HnswResource>, String> {
    hnsw_new(Metric::L2, m, m0, ef_construction, ef_search, max_level)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Creates a native HNSW graph ordered by cosine rank distance.
fn hnsw_new_cosine(
    m: usize,
    m0: usize,
    ef_construction: usize,
    ef_search: usize,
    max_level: usize,
) -> Result<ResourceArc<HnswResource>, String> {
    hnsw_new(Metric::Cosine, m, m0, ef_construction, ef_search, max_level)
}

#[rustler::nif(schedule = "DirtyCpu")]
/// Creates a native HNSW graph ordered by inner-product rank distance.
fn hnsw_new_inner_product(
    m: usize,
    m0: usize,
    ef_construction: usize,
    ef_search: usize,
    max_level: usize,
) -> Result<ResourceArc<HnswResource>, String> {
    hnsw_new(
        Metric::InnerProduct,
        m,
        m0,
        ef_construction,
        ef_search,
        max_level,
    )
}

/// Allocates the Rust resource that owns only ANN graph state.
fn hnsw_new(
    metric: Metric,
    m: usize,
    m0: usize,
    ef_construction: usize,
    ef_search: usize,
    max_level: usize,
) -> Result<ResourceArc<HnswResource>, String> {
    let params = HnswParams {
        m,
        m0,
        ef_construction,
        ef_search,
        max_level,
    };

    Ok(ResourceArc::new(HnswResource(std::sync::RwLock::new(
        HnswIndex::new(metric, params)?,
    ))))
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
/// Inserts a validated batch while acquiring the HNSW write lock once.
fn hnsw_insert_many(
    index: ResourceArc<HnswResource>,
    vectors: Vec<(String, Vec<f32>)>,
) -> Result<(), String> {
    let mut guard = index
        .0
        .write()
        .map_err(|_| "hnsw lock poisoned".to_string())?;
    guard.insert_many(vectors)
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
/// Releases all nodes and edges retained by a native HNSW index.
fn hnsw_clear(index: ResourceArc<HnswResource>) -> Result<(), String> {
    let mut guard = index
        .0
        .write()
        .map_err(|_| "hnsw lock poisoned".to_string())?;
    guard.clear();
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

rustler::init!("Elixir.Vettore.Nifs");
