# Vettore

Vettore is an ETS-native vector toolkit for Elixir.

It provides:

- ETS-backed vector collections
- exact flat search
- optional native HNSW approximate search
- Matryoshka-style funnel search and binary quantized candidate search
- named distance and similarity functions
- vector normalization
- explicit scoring semantics
- multi-vector Chamfer/MaxSim scoring
- MUVERA-style fixed-dimensional encodings through Rust NIFs
- a small compatibility layer for the old `Vettore.new/0` API

The vNext architecture is intentionally simple:

- Elixir and ETS own canonical collection state.
- Rust is used for acceleration only.
- Indexes are separate from storage.
- Metrics are separate from indexes.
- Search results expose both score and distance semantics explicitly.

## Installation

```elixir
def deps do
  [
    {:vettore, "~> 0.3.0"}
  ]
end
```

Vettore uses Rust NIFs through `rustler`, so local compilation requires a Rust
toolchain.

For local development:

```bash
mix deps.get
mix test
```

## Architecture

Vettore has three main layers.

### Store

The store owns records and collection metadata.

The default store is:

```elixir
Vettore.Store.ETS
```

ETS is the canonical source of truth. Records, ids, metadata, and normalized
vectors live there. Native resources are acceleration structures, not the
database.

Keeping the full store in ETS is slower than moving the whole database into
Rust, especially for exact scans over many records. This is an intentional
tradeoff: Vettore favors simple ownership, observability, snapshotting, and
natural integration with the Elixir ecosystem and BEAM VM. Rust is still used
where it pays off most, but the canonical data stays in Elixir.

ETS compression can be enabled per collection:

```elixir
{:ok, collection} =
  Vettore.Collection.new(
    name: :compressed_documents,
    dimensions: 384,
    metric: :cosine,
    normalize: :l2,
    compressed: true
  )
```

This passes `:compressed` to the underlying ETS table. It may reduce memory for
large records at the cost of extra CPU on reads and writes.

ETS collections can be snapshotted to disk and loaded later:

```elixir
:ok = Vettore.Collection.snapshot(collection, "priv/snapshots/documents.ets")

{:ok, loaded} =
  Vettore.Collection.load_snapshot("priv/snapshots/documents.ets")
```

Snapshots store the canonical ETS table: records, metadata, normalized vectors,
and collection config. Native index state is not stored. When a snapshot is
loaded, Vettore rebuilds the configured index from ETS records.

You can override the loaded index:

```elixir
{:ok, loaded} =
  Vettore.Collection.load_snapshot(
    "priv/snapshots/documents.ets",
    index: :hnsw
  )
```

### Index

Indexes search over records owned by the store.

Supported indexes:

- `:flat` - exact scan over ETS records
- `:hnsw` - native Rust HNSW graph over stored ids and vectors

Index choice is independent from metric choice:

```elixir
index: :flat,
metric: :cosine
```

or:

```elixir
index: :hnsw,
metric: :l2
```

### Native Acceleration

Rust is used for:

- distance kernels
- normalization kernels
- sign-bit compression
- HNSW graph search
- MUVERA/FDE encoding

Rust does not own the canonical collection database.

## Collections

Create a collection with dimensions, metric, normalization mode, and index:

```elixir
{:ok, collection} =
  Vettore.Collection.new(
    name: :documents,
    dimensions: 3,
    store: :ets,
    index: :flat,
    metric: :cosine,
    normalize: :l2,
    score: :raw,
    compressed: false
  )
```

Insert records:

```elixir
:ok =
  Vettore.Collection.put_many(collection, [
    %Vettore.Embedding{
      id: "a",
      vector: [1.0, 0.0, 0.0],
      metadata: %{"kind" => "axis"}
    },
    %Vettore.Embedding{
      id: "b",
      vector: [0.0, 1.0, 0.0]
    }
  ])
```

Search:

```elixir
{:ok, results} =
  Vettore.Collection.search(collection, [1.0, 0.0, 0.0], limit: 2)
```

Results are `%Vettore.Result{}` structs:

```elixir
%Vettore.Result{
  id: "a",
  value: "a",
  score: 1.0,
  distance: 0.0,
  metric: :cosine,
  metadata: %{"kind" => "axis"}
}
```

## Embeddings

Collection records use `%Vettore.Embedding{}`:

```elixir
%Vettore.Embedding{
  id: "doc-1",
  value: "optional external value",
  vector: [0.1, 0.2, 0.3],
  binary_vector: [1, 1, 1],
  metadata: %{source: "local"}
}
```

Rules:

- `id` is the preferred unique identifier.
- If `id` is missing, `value` can be used as the id when it is a non-empty string.
- Duplicate ids are rejected.
- Duplicate vectors are allowed.
- Vectors are normalized at insertion according to collection config.
- Binary sign vectors are generated at insertion for quantized candidate search.

Maps are accepted too:

```elixir
Vettore.Collection.put(collection, %{
  id: "doc-2",
  vector: [0.2, 0.1, 0.0],
  metadata: %{source: "map"}
})
```

## Metrics

Collection metrics:

- `:l2`
- `:l2_squared`
- `:cosine`
- `:inner_product`
- `:negative_inner_product`
- `:manhattan`
- `:chebyshev`
- `:hamming`
- `:jaccard`

Aliases accepted by `Vettore.Collection.new/1`:

- `:euclidean` -> `:l2`
- `:dot` -> `:inner_product`
- `:dot_product` -> `:inner_product`

`Vettore.Distance` itself is intentionally explicit. Use named functions:

```elixir
Vettore.Distance.l2([0.0, 0.0], [3.0, 4.0])
# {:ok, 5.0}

Vettore.Distance.l2_squared([0.0, 0.0], [3.0, 4.0])
# {:ok, 25.0}

Vettore.Distance.cosine([1.0, 0.0], [0.0, 1.0])
# {:ok, 0.0}

Vettore.Distance.inner_product([1.0, 2.0], [3.0, 4.0])
# {:ok, 11.0}

Vettore.Distance.negative_inner_product([1.0, 2.0], [3.0, 4.0])
# {:ok, -11.0}

Vettore.Distance.manhattan([1.0, 2.0], [4.0, 6.0])
# {:ok, 7.0}

Vettore.Distance.chebyshev([1.0, 2.0], [4.0, 6.0])
# {:ok, 4.0}

Vettore.Distance.hamming([1, 0, 1], [0, 0, 0])
# {:ok, 2.0}

Vettore.Distance.jaccard([1, 0, 1], [0, 1, 1])
# {:ok, 0.6666667}
```

There is no generic public metric dispatcher. Call the metric function you want
directly.

## Normalization

Supported normalization modes:

- `:none`
- `:l2`
- `:zscore`
- `:minmax`

Examples:

```elixir
Vettore.Distance.normalize([3.0, 4.0], :l2)
# {:ok, [0.6, 0.8]}

Vettore.Distance.normalize([1.0, 2.0, 3.0], :zscore)
# {:ok, [-1.2247449, 0.0, 1.2247449]}

Vettore.Distance.normalize([2.0, 4.0, 6.0], :minmax)
# {:ok, [0.0, 0.5, 1.0]}
```

Collection defaults:

- `metric: :cosine` defaults to `normalize: :l2`
- all other metrics default to `normalize: :none`

For collection search, inserted vectors and query vectors are prepared with the
same collection normalization mode.

## Scoring Semantics

Vettore keeps scoring explicit.

Distance metrics are naturally lower-is-better. Similarity metrics are
higher-is-better. Search results always expose a score and, when possible, a
distance.

For `score: :raw`:

- cosine score is raw cosine similarity
- inner product score is raw inner product
- distance metric score is `-distance`

Examples:

```elixir
Vettore.Distance.result_values(:l2, 5.0, :raw)
# {-5.0, 5.0}

Vettore.Distance.result_values(:cosine, 0.25, :raw)
# {0.25, 0.75}

Vettore.Distance.result_values(:inner_product, 3.0, :raw)
# {3.0, -3.0}
```

For `score: :similarity`, distance metrics are converted to:

```elixir
1.0 / (1.0 + distance)
```

Cosine is converted to:

```elixir
(raw + 1.0) / 2.0
```

## Exact Flat Search

Flat search is the default:

```elixir
{:ok, collection} =
  Vettore.Collection.new(
    name: :exact_vectors,
    dimensions: 384,
    index: :flat,
    metric: :cosine,
    normalize: :l2
  )
```

The flat index scans ETS records exactly. It is useful for:

- small collections
- deterministic tests
- classifier centroids
- local caches
- correctness baselines

The current exact path keeps ETS as the store and uses native distance kernels
for scoring. This keeps the implementation simple and idiomatic for Elixir, but
it is not as fast as a fully Rust-owned exact index would be.

## Adaptive Search Helpers

Vettore includes two first-pass candidate helpers that keep ETS as the canonical
store and rerank final results with the full stored vectors.

Matryoshka-style funnel search scores progressively larger vector prefixes,
then reranks the surviving candidates exactly:

```elixir
{:ok, results} =
  Vettore.Collection.funnel_search(collection, query,
    stages: [128, 256],
    candidates: 200,
    limit: 10
  )
```

This works best with embedding models trained for Matryoshka/nested dimensions,
where earlier dimensions preserve coarse semantic signal.

Binary quantized search uses stored sign bits for a cheap Hamming first pass,
then reranks candidates with the collection metric:

```elixir
{:ok, results} =
  Vettore.Collection.quantized_search(collection, query,
    candidates: 200,
    limit: 10
  )
```

Both helpers are intended as simple candidate generators. They trade first-pass
precision for less work before exact reranking.

## HNSW Search

HNSW is optional approximate nearest-neighbor search:

```elixir
{:ok, collection} =
  Vettore.Collection.new(
    name: :ann_vectors,
    dimensions: 768,
    index: :hnsw,
    metric: :cosine,
    normalize: :l2
  )
```

Supported HNSW metrics:

- `:l2`
- `:cosine`
- `:inner_product`

HNSW stores only native graph state, external ids, and vectors needed for ANN
search. ETS remains canonical.

Insertions update both ETS and the HNSW graph:

```elixir
:ok =
  Vettore.Collection.put(collection, %Vettore.Embedding{
    id: "doc-1",
    vector: embedding
  })
```

Deletion removes the record from ETS and the graph:

```elixir
:ok = Vettore.Collection.delete(collection, "doc-1")
```

## Multi-Vector Search

`Vettore.MultiVector.chamfer/3` computes Chamfer/MaxSim-style similarity.

For every query vector, it finds the best matching document vector and sums the
best scores:

```elixir
query_vectors = [
  [1.0, 0.0],
  [0.0, 1.0]
]

document_vectors = [
  [1.0, 0.0],
  [1.0, 1.0]
]

Vettore.MultiVector.chamfer(query_vectors, document_vectors, metric: :inner_product)
# {:ok, 2.0}
```

This is useful for late interaction retrieval, reranking, and multi-embedding
documents.

Supported metrics are delegated to named `Vettore.Distance` functions.

## MUVERA-Style Encodings

`Vettore.Encoding.Muvera` provides Rust-backed fixed-dimensional encodings for
multi-vector retrieval.

The intended retrieval shape:

1. Encode query multi-vectors into a fixed-dimensional vector.
2. Encode document multi-vectors into fixed-dimensional vectors.
3. Search FDE vectors with inner product.
4. Rerank candidates with exact Chamfer/MaxSim over stored multi-vectors.

Example:

```elixir
vectors = [
  [1.0, 0.0],
  [0.0, 1.0]
]

config = [
  num_repetitions: 1,
  num_simhash_projections: 4,
  seed: 42,
  projection_dimension: 2
]

{:ok, query_fde} = Vettore.Encoding.Muvera.encode_query(vectors, config)
{:ok, document_fde} = Vettore.Encoding.Muvera.encode_document(vectors, config)
```

Query encodings sum projected vectors in each partition. Document encodings
average projected vectors in each partition. Both use the same deterministic
seed/config so their FDEs are comparable.

Config options:

- `:dimension` - inferred from vectors by default
- `:num_repetitions` - defaults to `1`
- `:num_simhash_projections` - defaults to `0`
- `:seed` - defaults to `1`
- `:projection_dimension` - defaults to input dimension
- `:final_projection_dimension` - optional count-sketch compression size

## MMR Reranking

`Vettore.Distance.mmr_rerank/5` reranks initial candidates with maximal marginal
relevance:

```elixir
initial = [
  {"a", 0.9},
  {"b", 0.8},
  {"c", 0.1}
]

embeddings = [
  {"a", [1.0, 0.0]},
  {"b", [1.0, 0.0]},
  {"c", [0.0, 1.0]}
]

Vettore.Distance.mmr_rerank(initial, embeddings, :cosine, 0.5, 2)
# {:ok, [{"a", 0.9}, {"c", 0.1}]}
```

The metric must be a canonical atom such as `:cosine`, `:l2`, or
`:inner_product`.

## Sign Compression

Vettore can convert float vector signs into integer bits:

```elixir
Vettore.Distance.compress_f32_vector([1.0, -2.0, 0.0])
# [1, 0, 1]
```

This is kept as a compatibility/helper utility. It is not used as a duplicate
vector index.

## Compatibility API

The old top-level API still exists as a small compatibility layer:

```elixir
db = Vettore.new()

{:ok, "legacy"} =
  Vettore.create_collection(db, "legacy", 2, :cosine)

{:ok, "a"} =
  Vettore.insert(db, "legacy", %Vettore.Embedding{
    value: "a",
    vector: [1.0, 0.0]
  })

{:ok, results} =
  Vettore.similarity_search(db, "legacy", [1.0, 0.0], limit: 1)
```

New code should prefer `Vettore.Collection`.

Compatibility functions:

- `Vettore.new/0`
- `Vettore.create_collection/5`
- `Vettore.delete_collection/2`
- `Vettore.insert/3`
- `Vettore.batch/3`
- `Vettore.get_by_value/3`
- `Vettore.get_by_vector/3`
- `Vettore.delete/3`
- `Vettore.get_all/2`
- `Vettore.similarity_search/4`
- `Vettore.rerank/4`

## Error Handling

Most public operations return tagged tuples:

```elixir
{:ok, value}
{:error, reason}
```

Common errors:

- `:invalid_dimensions`
- `:invalid_metric`
- `:invalid_vector`
- `:dimension_mismatch`
- `:duplicate_id`
- `:missing_id`
- `:not_found`
- `:invalid_limit`
- `{:unknown_metric, metric}`
- `{:unknown_normalization, mode}`

## Design Notes

Vettore avoids these vNext pitfalls:

- no Rust-owned canonical DB
- no duplicate-vector rejection by sign-bit hash
- no public generic metric dispatcher
- no hidden cosine score conversions
- no PostgreSQL dependency in the core library

The core contract is:

- choose a store
- choose an index
- choose a metric
- choose normalization
- get explicit result semantics back

## Development

Run tests:

```bash
mix test
```

Run style checks:

```bash
mix credo --all
```

Run Dialyzer:

```bash
mix dialyzer
```

Run Rust tests:

```bash
cd native/vettore
cargo test
```

## Current Verification

The current refactor is expected to pass:

```bash
mix test
mix credo --all
mix dialyzer
```

The Spectre integration profile is also expected to pass when using this local
Vettore checkout.

## Roadmap

Likely next improvements:

- Rust-backed exact flat index resource to avoid one NIF call per ETS row
- benchmarks for 384D and 768D at 1k, 10k, and 100k records
- packed bitset representation for Hamming/Jaccard
- configurable HNSW parameters
- stronger MUVERA recall tests against deterministic fixtures
- persistence/rebuild helpers for native indexes from ETS records
