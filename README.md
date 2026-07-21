# Vettore

Vettore is a small vector toolkit for Elixir that keeps your data in ETS and
uses Rust only where it helps: distance kernels, normalization, HNSW search, and
MUVERA-style encodings.

Earlier versions leaned toward a Rust-owned in-memory database. That was fast,
but it made the library feel less like an Elixir tool and more like an external
engine with Elixir bindings. Vettore now chooses ETS as the canonical store on
purpose:

- records are visible and easy to inspect from Elixir
- supervision, snapshots, and ownership stay simple
- metadata and application values live beside vectors naturally
- native indexes can be rebuilt from canonical ETS state
- the public API stays small, predictable, and BEAM-friendly

The important idea is simple:

- Elixir owns the records.
- ETS is the source of truth.
- Rust accelerates the expensive parts.
- Search results say clearly what is a score and what is a distance.

That choice is not the absolute fastest possible architecture. A fully
Rust-owned vector database can beat ETS for large exact scans, but Vettore
optimizes for a different kind of usefulness: simple integration with ordinary
Elixir systems, with Rust kept as acceleration rather than ownership.

## What You Get

- ETS-backed collections
- exact flat search
- native HNSW approximate search
- Matryoshka-style funnel search
- binary quantized candidate search
- hybrid candidate pipelines with exact or multi-vector reranking
- ColBERT-style late interaction over multi-vector records
- MUVERA-style fixed-dimensional encodings
- named distance, similarity, normalization, and MMR helpers
- a top-level `Vettore.*` API, plus compatibility wrappers for the older
  `Vettore.new/0` database-style API

## Installation

```elixir
def deps do
  [
    {:vettore, "~> 0.3.2"}
  ]
end
```

## Quick Start

Create a collection, insert a few records, and search:

```elixir
{:ok, collection} =
  Vettore.new(
    name: :documents,
    dimensions: 3,
    index: :flat,
    metric: :cosine,
    normalize: :l2
  )

:ok =
  Vettore.put_many(collection, [
    %{id: "east", vector: [1.0, 0.0, 0.0], metadata: %{kind: :axis}},
    %{id: "north", vector: [0.0, 1.0, 0.0]},
    %{id: "west", vector: [-1.0, 0.0, 0.0]}
  ])

{:ok, results} =
  Vettore.search(collection, [1.0, 0.0, 0.0], limit: 2)
```

Results are `%Vettore.Result{}` structs:

```elixir
%Vettore.Result{
  id: "east",
  value: "east",
  score: 1.0,
  distance: 0.0,
  metric: :cosine,
  metadata: %{kind: :axis}
}
```

## Public API

New code can stay under the top-level `Vettore` module:

```elixir
Vettore.new(opts)
Vettore.put(collection, embedding)
Vettore.put_many(collection, embeddings)
Vettore.get(collection, id)
Vettore.delete(collection, id)
Vettore.all(collection)
Vettore.search(collection, query, opts)
Vettore.funnel_search(collection, query, opts)
Vettore.quantized_search(collection, query, opts)
Vettore.multi_vector_search(collection, query_vectors, opts)
Vettore.hybrid_search(collection, query, opts)
Vettore.snapshot(collection, path)
Vettore.load_snapshot(path, opts)
Vettore.close(collection)
```

`Vettore.new/1` creates a collection. `Vettore.new/0` still creates the older
compatibility database.

## Choosing A Search Path

Start with the simplest thing that matches your job.

| Use this | When |
| --- | --- |
| `search/3` with `index: :flat` | Small data, tests, correctness baselines, exact results |
| `search/3` with `index: :hnsw` | Fast approximate search over larger collections |
| `funnel_search/3` | Matryoshka embeddings where early dimensions are meaningful |
| `quantized_search/3` | Cheap sign-bit candidate search before exact reranking |
| `multi_vector_search/3` | ColBERT-style late interaction over token/page vectors |
| `hybrid_search/3` | Combine candidate generators, then rerank once |

The standalone helpers are nice while exploring. For production-style retrieval,
`hybrid_search/3` is usually the most ergonomic surface.

## Exact Search

Flat search keeps ids and vectors in a Rust resource and scores the whole exact
scan in one native call. ETS remains the canonical store for values, metadata,
snapshots, and usability.

```elixir
{:ok, collection} =
  Vettore.new(
    name: :exact_vectors,
    dimensions: 384,
    index: :flat,
    metric: :cosine,
    normalize: :l2
  )

{:ok, results} =
  Vettore.search(collection, query_vector, limit: 10)
```

This path is intentionally boring. It is great for small collections, local
caches, classifier centroids, deterministic tests, and recall baselines.

## HNSW Search

HNSW keeps a native graph beside the ETS store. ETS remains canonical; the graph
is an acceleration structure.

```elixir
{:ok, collection} =
  Vettore.new(
    name: :ann_vectors,
    dimensions: 768,
    index: :hnsw,
    index_options: [
      m: 16,
      m0: 32,
      ef_construction: 100,
      ef_search: 64,
      max_level: 12
    ],
    metric: :cosine,
    normalize: :l2
  )

:ok = Vettore.put(collection, %{id: "doc-1", vector: embedding})

{:ok, results} =
  Vettore.search(collection, query_vector, limit: 10)
```

Supported HNSW metrics:

- `:l2`
- `:cosine`
- `:inner_product`

HNSW results are hydrated from ETS, so they contain the same `value`,
`metadata`, score, and distance fields as exact flat results.

## Adaptive Candidate Search

These helpers first find a candidate set, then rerank with full stored vectors.
They are useful when you want to make the first pass cheaper without changing
the canonical store.

### Matryoshka Funnel

Funnel search scores progressively larger vector prefixes. It works best with
models trained for Matryoshka or nested embeddings.

```elixir
{:ok, results} =
  Vettore.funnel_search(collection, query_vector,
    stages: [128, 256, 384],
    candidates: 200,
    limit: 10
  )
```

### Binary Quantized Candidates

Quantized search uses stored sign bits for a cheap Hamming-distance first pass,
then reranks with the collection metric.

```elixir
{:ok, results} =
  Vettore.quantized_search(collection, query_vector,
    candidates: 200,
    limit: 10
  )
```

Vettore generates `binary_vector` at insert time:

```elixir
{:ok, embedding} = Vettore.get(collection, "doc-1")
embedding.binary_vector
# [7]
```

Sign bits are packed into unsigned 64-bit words; they are not stored as one
integer per vector dimension.

## Hybrid Search

`hybrid_search/3` lets you combine candidate generators, union their ids, fetch
the canonical records from ETS, and rerank once.

```elixir
{:ok, results} =
  Vettore.hybrid_search(collection, query_vector,
    generators: [
      funnel: [stages: [128, 384], candidates: 200],
      quantized: [candidates: 200]
    ],
    rerank: :exact,
    limit: 10
  )
```

For HNSW collections, add `:hnsw` as a generator:

```elixir
{:ok, results} =
  Vettore.hybrid_search(collection, query_vector,
    generators: [
      hnsw: [candidates: 100],
      quantized: [candidates: 200]
    ],
    rerank: :exact,
    limit: 10
  )
```

The same pipeline can rerank with late interaction:

```elixir
{:ok, results} =
  Vettore.hybrid_search(collection, query_vector,
    generators: [quantized: [candidates: 200]],
    rerank: {:multi_vector, query_vectors},
    limit: 10
  )
```

That is the general pattern:

1. Generate cheap candidates.
2. Merge them by id.
3. Rerank with the expensive scorer you actually care about.

## Multi-Vector Search

Multi-vector search is for ColBERT-style retrieval: each record can hold many
vectors, usually token vectors or page-patch vectors. A query also has many
vectors. For each query vector, Vettore finds the best matching document vector
and sums those best scores.

```elixir
:ok =
  Vettore.put(collection, %Vettore.Embedding{
    id: "page-1",
    vectors: [
      [1.0, 0.0],
      [0.0, 1.0]
    ],
    metadata: %{source: "manual"}
  })

{:ok, results} =
  Vettore.multi_vector_search(
    collection,
    [[1.0, 0.0], [0.0, 1.0]],
    metric: :inner_product,
    limit: 10
  )
```

The lower-level scoring helper is available too:

```elixir
Vettore.MultiVector.colbert_score(
  [[1.0, 0.0], [0.0, 1.0]],
  [[1.0, 0.0], [1.0, 1.0]],
  metric: :inner_product
)
# {:ok, 2.0}
```

`Vettore.MultiVector.chamfer/3` is the same MaxSim-style operation under a more
general name.

## MUVERA-Style Encodings

MUVERA reduces multi-vector retrieval to fixed-dimensional vectors. The intended
flow is:

1. Encode query multi-vectors into a fixed-dimensional query vector.
2. Encode document multi-vectors into fixed-dimensional document vectors.
3. Search those vectors with inner product.
4. Rerank candidates with exact MaxSim/Chamfer.

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
{:ok, doc_fde} = Vettore.Encoding.Muvera.encode_document(vectors, config)
```

Config options:

- `:dimension` - inferred from vectors by default
- `:num_repetitions` - defaults to `1`
- `:num_simhash_projections` - defaults to `0`
- `:seed` - defaults to `1`
- `:projection_dimension` - defaults to input dimension
- `:final_projection_dimension` - optional count-sketch compression size

## Records And Storage

Records are `%Vettore.Embedding{}` structs or maps with equivalent keys.

```elixir
%Vettore.Embedding{
  id: "doc-1",
  value: "optional external value",
  vector: [0.1, 0.2, 0.3],
  vectors: [[0.1, 0.2, 0.3], [0.0, 0.5, 0.5]],
  binary_vector: [7],
  metadata: %{source: "local"}
}
```

Useful details:

- `id` is the preferred unique identifier.
- If `id` is missing, a non-empty string `value` can be used as the id.
- Duplicate ids are rejected.
- Duplicate vectors are allowed.
- Vectors are normalized at insertion according to the collection config.
- If `vectors` is present but `vector` is omitted, Vettore stores an averaged
  representative vector for ordinary search/indexing.
- `binary_vector` is generated automatically for quantized candidate search.

Each collection table is owned by a supervised Vettore worker, so it remains
alive if the process that created the collection exits. Tables are `:protected`:
all caller processes can read them directly and concurrently, with ETS
`read_concurrency` enabled. Reads do not pass through the owner process. Perform
writes through `Vettore.put/2`, `Vettore.put_many/2`, and `Vettore.delete/2` so
ETS and the native index stay in sync. Release resources deterministically when
they are no longer needed:

```elixir
:ok = Vettore.close(collection)
```

Closing is idempotent. Later operations return `{:error, :closed}`.

ETS collections can be snapshotted:

```elixir
:ok = Vettore.snapshot(collection, "priv/snapshots/docs.ets")

{:ok, loaded} =
  Vettore.load_snapshot("priv/snapshots/docs.ets")
```

Snapshots store the ETS table: records, metadata, normalized vectors, binary
vectors, multi-vectors, and collection config. Native indexes are rebuilt from
ETS when loaded. Snapshot writes use a same-directory temporary file followed
by a rename and include ETS integrity metadata. Loads validate the table type,
schema, and every stored record before rebuilding the index; legacy public
tables are restored as protected tables.

You can load the same data with a different index:

```elixir
{:ok, loaded} =
  Vettore.load_snapshot("priv/snapshots/docs.ets", index: :hnsw)
```

Supported load overrides are `:name`, `:index`, `:index_options`, `:score`, and
`:store`. Structural fields such as dimensions, metric, normalization, and
compression cannot be changed because stored vectors were already prepared
with those settings. Name, index, index options, and score overrides are written
back to the collection config, so they persist through later snapshots. The
`:store` option selects the loader for that call and must be supplied again for
a custom snapshot format.

ETS compression is available when you want to trade CPU for memory:

```elixir
{:ok, collection} =
  Vettore.new(
    name: :compressed_documents,
    dimensions: 384,
    metric: :cosine,
    normalize: :l2,
    compressed: true
  )
```

## Metrics And Scoring

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

Aliases accepted by `Vettore.new/1`:

- `:euclidean` -> `:l2`
- `:dot` -> `:inner_product`
- `:dot_product` -> `:inner_product`

with `Vettore.Distance` you can use directly all distance functions:

```elixir
Vettore.Distance.l2([0.0, 0.0], [3.0, 4.0])
# {:ok, 5.0}

Vettore.Distance.cosine([1.0, 0.0], [0.0, 1.0])
# {:ok, 0.0}

Vettore.Distance.inner_product([1.0, 2.0], [3.0, 4.0])
# {:ok, 11.0}
```

## Normalization

Supported normalization modes:

- `:none`
- `:l2`
- `:zscore`
- `:minmax`

```elixir
Vettore.Distance.normalize([3.0, 4.0], :l2)
# {:ok, [0.6, 0.8]}
```

Collection defaults:

- `metric: :cosine` defaults to `normalize: :l2`
- all other metrics default to `normalize: :none`

Inserted vectors and query vectors are prepared with the same collection
normalization mode.

## Other Helpers

MMR reranking:

```elixir
initial = [{"a", 0.9}, {"b", 0.8}, {"c", 0.1}]
embeddings = [{"a", [1.0, 0.0]}, {"b", [1.0, 0.0]}, {"c", [0.0, 1.0]}]

Vettore.Distance.mmr_rerank(initial, embeddings, :cosine, 0.5, 2)
# {:ok, [{"a", 0.9}, {"c", 0.1}]}
```

Sign compression:

```elixir
Vettore.Distance.compress_f32_vector([1.0, -2.0, 0.0])
# [5]

left = Vettore.Distance.compress_f32_vector([1.0, -2.0, 0.0])
right = Vettore.Distance.compress_f32_vector([-1.0, -2.0, 0.0])
Vettore.Distance.packed_hamming(left, right, 3)
# {:ok, 1.0}
```

## Compatibility API

The old top-level API still exists as a small compatibility layer backed by ETS
collections:

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

New code should prefer the collection-style top-level API: `Vettore.new/1`, `Vettore.put/2`, and `Vettore.search/3`.

## Development

CI includes a real `ex_fastembed` integration with `BAAI/bge-small-en-v1.5`
over a small phrase corpus. It compares exact search, HNSW, and hybrid
retrieval while checking canonical values and metadata. Enable that test
locally without adding its older Rustler constraint to the published package:

```bash
MIX_ENV=test VETTORE_TEST_EX_FASTEMBED=1 mix deps.get --locked
VETTORE_BUILD=1 VETTORE_TEST_EX_FASTEMBED=1 mix test --cover
```

Build the Rust crate locally with Rust 1.91 or newer by setting
`VETTORE_BUILD=1`:

```bash
VETTORE_BUILD=1 mix test --cover
cargo test --manifest-path native/vettore/Cargo.toml
```

Run the deterministic latency-and-overlap matrix for every search mode with:

```bash
VETTORE_BUILD=1 mix run bench/search_modes_bench.exs
```

See `bench/performance.md` for the full search, metric, MUVERA, MaxSim, and ETS
benchmark matrix.

Without that variable, Vettore uses the published precompiled NIF for the
current package version. See `RELEASE.md` for the full release verification and
checksum workflow.
