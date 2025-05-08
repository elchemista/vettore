# Vettore Rust Native Library

A high-performance, SIMD-accelerated vector store and similarity search engine in Rust. Exposed as a `cdylib` for use with Rustler (Elixir NIF), or as a native Rust crate.

## Crate Layout

- **src/lib.rs**  
  Re-exports all modules:
  - `db` – core in-memory database API
  - `distances` – distance and score computations
  - `hnsw` – hierarchical navigable small world graph index
  - `simd_utils` – vectorized load & normalize helpers
  - `similarity` – fast top-k search engine
  - `mmr` – Maximal Marginal Relevance reranking
  - `nifs` – Rustler bindings for Elixir
  - `types` – shared types (Distance, Metadata)

## Core Structs

### Record
```rust
pub struct Record {
    pub vector: Vec<f32>,
    pub value: String,
    pub metadata: Option<Metadata>,
}
```
Holds one embedding, its key and optional metadata.

### Collection
In-memory store of fixed-dimension vectors, supporting CRUD and search.
- **Config**  
  - `dimension: usize`  
  - `distance: Distance`  
  - `keep_embeddings: bool`
- **Storage**  
  - contiguous `Vec<f32>` for raw floats  
  - bit-compressed `Vec<u64>` for binary/fast dedupe  
  - maps: value→row, comp_key→row  
  - free list for reuse
- **Index**  
  - optional `HnswIndexWrapper` when `Distance::Hnsw`
- **API**  
  - `insert(value, vec, metadata)`  
  - `get_by_value`, `get_by_vector`, `get_all`  
  - `remove(value)`

### VettoreDB
Sharded global database:
```rust
pub struct VettoreDB {
    cols: DashMap<String, Arc<RwLock<Collection>>>,
}
```
Offers thread-safe collection management:
- `create_collection(name, dim, distance, keep_embeddings)`
- `delete_collection(name)`
- `insert`, `get_by_value`, `get_by_vector`, `get_all`, `delete_by_value`
- All wrapped in read/write locks per collection.

### HnswIndex & Wrapper
Graph index for sub-linear k-NN:
- **HnswIndex** – core nodes + multi-level neighbor links
- **HnswIndexWrapper** – assigns string IDs, clamps scores

## Modules

### types
Defines:
```rust
pub enum Distance { Euclidean, Cosine, DotProduct, Hnsw, Binary }
pub type Metadata = HashMap<String, String>;
```

### distances
- `simd_euclidean_distance(a, b) -> f32`
- `simd_dot_product(a, b) -> f32`
- `compress_vector(&[f32]) -> Vec<u64>` (sign-bit packing)
- `hamming_distance(&[u64], &[u64]) -> u32`
- `score(query, vector, bin, Distance) -> f32` (normalized 0‥1)

### simd_utils
Vectorized helpers using the `wide` crate:
- `load_f32x4(slice, i) -> f32x4`  
- `load_f32x8(slice, i) -> f32x8` (when AVX2/AVX512 enabled)
- `normalize_vec(&[f32]) -> Vec<f32>` (SIMD accumulate + fallback)

### similarity
`similarity_search(coll: &Collection, query: &[f32], k: usize)`  
- Uses HNSW if present, else brute-force scan with SIMD kernels  
- Binary distances use compressed bits + Hamming  
- L2, Cosine, DotProduct use `simd_euclidean_distance` / `simd_dot_product`  
- Keeps top-k via sorting or min-heap

### mmr
`mmr_rerank(initial, vectors_map, dist, alpha, final_k)`  
Performs Maximal Marginal Relevance to balance relevance vs. diversity.

### nifs
Elixir NIF bindings via Rustler:
- `new_db()`, `create_collection()`, `insert_embedding()`, `get_embedding_by_value()`, `similarity_search()`, `mmr_rerank()`, and standalone distance helpers (`euclidean_distance`, `cosine_similarity`, etc.)

## SIMD Acceleration

We leverage the `wide` crate for 128-bit and optional 256-bit vector operations:

- **load_f32x4** (always available)  
  Packs 4 `f32` into a single `f32x4` lane for parallel subtraction, multiplication, dot products.

- **load_f32x8** (with `target_feature=avx2` or `avx512f`)  
  Packs 8 `f32` into `f32x8` for double-width throughput.

- **normalize_vec**  
  Accumulates squared sums via SIMD, divides each lane by the norm.

- **Distance kernels**  
  `simd_euclidean_distance` and `simd_dot_product` loop in 4- or 8-lane chunks, then fallback scalar tail.

- **Brute-force search**  
  Uses those kernels inside parallel iterators (`rayon` when `parallel` feature is enabled) for millions of vectors.

## Cargo Features

- **default = ["parallel"]**  
  Enables `rayon`-powered parallel scan above `PAR_THRESHOLD`.
- **parallel**  
  Activates Rayon dependency for multi-threaded brute-force search.

## Building

```bash
cd native/vettore
cargo build --release
```

Configure `RUSTLER_NIF_VERSION` for Elixir integration, and use `mix compile` in your Phoenix/Elixir project.

## Usage Example (Rust)

```rust
use vettore::db::{VettoreDB, Collection};
use vettore::types::Distance;

let mut db = VettoreDB::default();
db.create_collection("mycol".into(), 128, "cosine", true).unwrap();
let vec = vec![0.1f32; 128];
db.insert("mycol", "id1".into(), vec.clone(), None).unwrap();
let hits = db.similarity_search("mycol", vec, 10).unwrap();
```

## Usage Example (Elixir)

```elixir
{:ok, pid} = Vettore.Nifs.new_db()
{:ok, _} = Vettore.Nifs.create_collection(pid, "col", 64, "euclidean", true)
{:ok, "val1"} = Vettore.Nifs.insert_embedding(pid, "col", "val1", embedding, nil)
results = Vettore.Nifs.similarity_search(pid, "col", query_embedding, 5)
```
