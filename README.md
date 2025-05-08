# Vettore

Vettore is a high-performance Elixir library for fast, in-memory operations on vector (embedding) data. It leverages a Rust backend via Rustler to store and manipulate vectors efficiently in a concurrent-safe `HashMap`.

## Features

* **Collections**: Create named sets of embeddings with a fixed dimension and a choice of similarity metric.
* **CRUD operations**: Insert, batch-insert, retrieve, and delete embeddings by ID or by vector.
* **Similarity search**: Nearest-neighbor search with customizable `:limit` and optional metadata filtering.
* **Reranking**: Maximal Marginal Relevance (MMR) reranker for diversity-aware result reordering.
* **Distance helpers**: Standalone Euclidean, Cosine, Dot, and Hamming metrics, plus binary compression for ultra-fast comparisons.

## Installation

Add `vettore` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:vettore, "~> 0.2.0"}
  ]
end
```

Then fetch and compile:

```bash
mix deps.get
mix compile
```

*Note*: The first compile will build the Rust crate; ensure you have a recent Rust toolchain installed.

## Quickstart

```elixir
# 1. Start a new in-memory database reference
db = Vettore.new()

# 2. Create a collection named "my_collection" with 3-dimensional vectors
:ok = Vettore.create_collection(db, "my_collection", 3, :euclidean)

# 3. Insert a single embedding
embedding = %Vettore.Embedding{
  value: "item_1",
  vector: [1.0, 2.0, 3.0],
  metadata: %{"note" => "first vector"}
}
:ok = Vettore.insert(db, "my_collection", embedding)

# 4. Retrieve by ID
{:ok, emb} = Vettore.get_by_value(db, "my_collection", "item_1")
IO.inspect(emb.vector, label: "Vector")

# 5. Similarity search (top 2 nearest neighbors)
{:ok, results} = Vettore.similarity_search(db, "my_collection", [1.5, 1.5, 1.5], limit: 2)
IO.inspect(results, label: "Top-2 Results")

# 6. Rerank with MMR for diversity (alpha = 0.7)
{:ok, reranked} = Vettore.rerank(db, "my_collection", results, limit: 2, alpha: 0.7)
IO.inspect(reranked, label: "MMR Reranked")
```

## API Reference

### `Vettore.new/0`

```elixir
def new() :: reference()
```

Allocates and returns an in-memory database handle backed by Rust.

---

### `Vettore.create_collection/5`

```elixir
@spec create_collection(
        db :: reference(),
        name :: String.t(),
        dim :: pos_integer(),
        metric :: :euclidean | :cosine | :dot | :hnsw | :binary,
        opts :: [keep_embeddings: boolean()]
      ) :: {:ok, String.t()} | {:error, String.t()}
```

* **name**: Collection identifier
* **dim**: Dimensionality of vectors
* **metric**: Similarity measure
* **opts**:

  * `:keep_embeddings` (default: `true`) — whether to retain embeddings on deletion

---

### `Vettore.insert/3`

```elixir
@spec insert(
        db :: reference(),
        collection :: String.t(),
        embedding :: Vettore.Embedding.t()
      ) :: {:ok, String.t()} | {:error, String.t()}
```

Insert a single `%Vettore.Embedding{}` struct into the named collection.

---

### `Vettore.batch/3`

```elixir
@spec batch(
        db :: reference(),
        collection :: String.t(),
        embeddings :: [Vettore.Embedding.t()]
      ) :: {:ok, [String.t()]} | {:error, String.t()}
```

Batch-insert multiple embeddings at once; non-embedding elements are ignored.

---

### Retrieval and Deletion

* `Vettore.get_by_value/3` — fetch by embedding ID
* `Vettore.get_by_vector/3` — fetch by exactly matching vector
* `Vettore.get_all/2` — returns all `{value, vector, metadata}`
* `Vettore.delete/3` — delete by ID

---

### `Vettore.similarity_search/4`

```elixir
@spec similarity_search(
        db :: reference(),
        collection :: String.t(),
        query :: [number()],
        opts :: [limit: pos_integer(), filter: map()]
      ) :: {:ok, [{String.t(), float()}]} | {:error, String.t()}
```

* **limit** (default: `10`)
* **filter**: metadata map to pre-filter embeddings

---

### `Vettore.rerank/4` (MMR)

```elixir
@spec rerank(
        db :: reference(),
        collection :: String.t(),
        initial :: [{String.t(), number()}],
        opts :: [limit: pos_integer(), alpha: float()]
      ) :: {:ok, [{String.t(), number()}]} | {:error, String.t()}
```

* **alpha**: `0.0..1.0` balance between relevance and diversity

---

## Distance Helpers (`Vettore.Distance`)

You can call these functions without creating a DB or collection:

```elixir
Vettore.Distance.euclidean([1.0,2.0], [2.0,3.0])      # => 1 / (1 + L2)
Vettore.Distance.cosine([1,0],[0,1])                  # => (dot + 1) / 2
Vettore.Distance.dot_product([1,2],[3,4])             # => raw dot
Vettore.Distance.hamming(bits1, bits2)                # => Hamming distance
bits = Vettore.Distance.compress_f32_vector([0.1,0.4])

# MMR re-ranker standalone (collection-agnostic)
initial = [{"id1", 0.9}, {"id2", 0.85}, ...]
embeds  = [{"id1", [v1...]}, {"id2", [v2...]}, ...]
Vettore.Distance.mmr_rerank(initial, embeds, "cosine", 0.5, 5)
```

## Similarity Search

The `similarity_search` function works as follows:

1. It retrieves the target collection and verifies that the query vector’s dimension matches.
2. Depending on the chosen distance metric, it selects an appropriate function:
   - **Euclidean:** Computes the standard Euclidean distance.
   - **Cosine / DotProduct:** Computes the dot product (with normalization applied for Cosine).
   - **HNSW:** Uses a graph-based approach for approximate nearest neighbor search.
   - **Binary:** Compresses the query vector into a binary signature and computes the Hamming distance between this signature and those of all stored embeddings.
3. For every embedding in the collection, it calculates a “score” between the stored vector (or its compressed representation) and the query.
4. The results are sorted:
   - For **Euclidean distance**, lower scores (closer to zero) are better.
   - For **Cosine/DotProduct**, higher scores are considered more similar.
   - For **Binary**, a lower Hamming distance means the vectors are more similar.
5. Finally, the top‑k results are returned as a list of tuples `(embedding_id, score)`.

| Technique/Algorithm    | Measures                                                | Magnitude Sensitive?¹ | Scale Invariant?¹   | Best Use Cases                                                             | Pros                                                                     | Cons                                                                            |
| ---------------------- | ------------------------------------------------------- | --------------------- | ------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| **Euclidean Distance** | Straight-line distance                                  | Yes                   | No                  | Dense data where both magnitude & direction are important                  | Intuitive, widely used, captures magnitude differences                   | Sensitive to scale differences, high dimensionality issues                      |
| **Cosine Similarity**  | Directional similarity (angle)                          | No                    | Yes                 | Text or high-dimensional data where scale invariance is desired            | Insensitive to magnitude, works well with normalized vectors             | Ignores magnitude differences                                                   |
| **Dot Product**        | Combination of direction & magnitude                    | Yes                   | No                  | Applications where both direction & magnitude matter                       | Computationally efficient, captures both aspects                         | Sensitive to vector magnitudes                                                  |
| **HNSW Indexing**      | Graph-based Approximate Nearest Neighbor Search         | Dependent on Metric   | Dependent on Metric | Large datasets, real-time search when approximate results are acceptable   | **Sublinear search time**, good speed-accuracy trade-off, scalable       | Approximate results, index build time and memory overhead                       |
| **Binary (Hamming)**   | Fast binary signature comparison using Hamming distance | No                    | Yes                 | Applications requiring ultra‑fast approximate searches on large-scale data | Extremely fast comparison via bit-level operations, low memory footprint | Loses precision due to compression, less suited when exact distances are needed |

---

## Performance Notes

- **HNSW** can speed up searches significantly for large datasets but comes with higher memory usage for the index.
- **Binary** distance uses bit-level compression and Hamming distance for extremely fast approximate similarity checks (especially beneficial for large or high-dimensional vectors, though it trades off some precision).
- **Cosine** normalizes vectors once on insertion, so queries and stored embeddings use a straightforward dot product.
- **Dot Product** directly multiplies corresponding elements.
- **Euclidean** uses a SIMD approach (`wide::f32x4`) for partial vectorization.

## Contributing

Contributions are welcome! Please open an issue or submit a PR.

1. Fork the repo
2. Create a feature branch
3. Add tests in `test/`
4. Submit a PR

## License

Apache 2.0 [LICENSE](LICENSE)
