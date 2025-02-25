# Vettore: In-Memory Vector Database with Elixir & Rustler

**Vettore** is an in-memory vector database implemented in Rust and exposed to Elixir via [Rustler](https://github.com/rusterlium/rustler). It allows you to create collections of vectors (embeddings), insert new embeddings (with optional metadata), retrieve all embeddings or a specific embedding by its ID, and perform similarity searches using common metrics with CPU-specific instructions for maximum performance.

The supported distance metrics are:

- **Euclidean**
- **Cosine**
- **Dot Product**
- **HNSW** (Hierarchical Navigable Small World graphs for approximate nearest neighbor search)
- **Binary** (Compressed vectors with fast Hamming distance)

---

## Installation

```elixir
def deps do
  [
    {:vettore, "~> 0.1.3", github: "elchemista/vettore"}
  ]
end
```

## Compile

```bash
RUSTFLAGS="-C target-cpu=native" mix compile
```

This sets the Rust compiler’s “target-cpu” to native, instructing it to generate code optimized for your machine’s CPU features (SSE, AVX, etc.).

---

## Overview

The Vettore library is designed for fast, in-memory operations on vector data. All vectors (embeddings) are stored in a Rust data structure (a HashMap) and accessed via a shared resource (using Rustler’s `ResourceArc` with a Mutex). The core operations include:

- **Creating a collection** – A collection is a set of embeddings with a fixed dimension and a chosen similarity metric: **hnsw**, **binary**, **euclidean**, **cosine**, or **dot**.
- **Inserting an embedding** – Add a new vector with an identifier and optional metadata to a specific collection.
- **Batch Inserting embeddings** – Insert a list of embeddings (batch) into a collection in one call.
- **Retrieving embeddings** – Fetch all embeddings from a collection or look up a single embedding by its unique ID.
- **Similarity search** – Given a query vector, calculate a “score” (distance or similarity) for every embedding in the collection and return the top‑k results.

---

## Under the Hood

### Data Structures

1. **CacheDB**  
   The main in‑memory database is defined as a `CacheDB` struct. Internally, it stores collections in a Rust [`HashMap<String, Collection>`](https://doc.rust-lang.org/std/collections/struct.HashMap.html), mapping collection names to a `Collection` struct.

2. **Collection**  
   Each collection is a struct with:

   - `dimension: usize` – The fixed length of every vector in this collection.
   - `distance: Distance` – The similarity metric used (e.g., Euclidean, Cosine, DotProduct, HNSW, Binary).
   - `embeddings: Vec<Embedding>` – A vector containing all embeddings.
   - `hnsw_index: Option<HnswIndexWrapper>` – An optional HNSW index for approximate nearest neighbor search.

3. **Embedding**  
   An embedding is represented by:

   - `id: String` – A unique identifier.
   - `vector: Vec<f32>` – The actual vector values.
   - `metadata: Option<HashMap<String, String>>` – Optional additional information.
   - `binary: Option<Vec<u64>>` – _For the **binary** distance metric_, a compressed signature.

4. **DBResource**  
   The `CacheDB` is wrapped in a `DBResource` (and stored as a Rustler `ResourceArc`). This allows Elixir to hold a reference to the database across NIF calls, and it is guarded by a `Mutex` to ensure thread safety.

---

## Public API / NIF Functions

All core functions are accessible in Elixir via `Vettore.*` calls. Their **return values** (on success) now include more information:

1. **`create_collection(db, name, dimension, distance)`**  
   Returns `{:ok, collection_name}` or `{:error, reason}`.

   - Creates a new collection in the database with a specified dimension and distance metric.

2. **`delete_collection(db, name)`**  
   Returns `{:ok, collection_name}` or `{:error, reason}`.

   - Deletes an existing collection (by name).

3. **`insert_embedding(db, collection_name, embedding_struct)`**  
   Returns `{:ok, embedding_id}` or `{:error, reason}`.

   - Inserts a single embedding (with an ID, vector, and optional metadata).

4. **`insert_embeddings(db, collection_name, [embedding_structs])`**  
   Returns `{:ok, list_of_inserted_ids}` or `{:error, reason}`.

   - **Batch insertion**: Insert a list of embeddings in one call.
   - If any embedding fails (dimension mismatch, duplicate ID, etc.), an error is returned immediately and the rest are not inserted.

5. **`get_embeddings(db, collection_name)`**  
   Returns `{:ok, list_of({id, vector, metadata})}` or `{:error, reason}`.

   - Retrieves all embeddings from the specified collection.

6. **`get_embedding_by_id(db, collection_name, id)`**  
   Returns `{:ok, %Vettore.Embedding{}}` or `{:error, reason}`.

   - Looks up a single embedding by its ID.

7. **`similarity_search(db, collection_name, query_vector, k)`**  
   Returns `{:ok, list_of({id, score})}` or `{:error, reason}`.

   - Performs a similarity or distance search with the given query vector, returning the top‑k results.

8. **`new_db()`**  
   Returns a **DB resource** (reference to the underlying Rust `CacheDB`).

---

### How It Works

#### Storage

- **In-Memory Storage:**  
  All data is stored in memory (within the `CacheDB` struct). No external databases or disk storage is used by default.
  - Collections are stored as key–value pairs in a HashMap.
  - Each collection’s embeddings are stored in a `Vec`.

#### Inserting Vectors

When you call the `insert_embedding`, `insert_embeddings` functions:

1. The collection is retrieved from the database.
2. The function checks that the vector’s dimension matches the collection’s defined dimension.
3. It verifies that there isn’t already an embedding with the same ID in the collection.
4. For **Cosine** distance collections, the vector is normalized before insertion.
5. For **Binary** distance collections, a binary signature is precomputed and stored (using the sign of each float to produce a bit-packed representation).
6. The new embeddings is then pushed into the collection’s vector of embeddings.

#### Similarity Search

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

## Batch Insert Example

Because `insert_embeddings/3` now accepts a **list** of `%Vettore.Embedding{}` structs and returns a list of IDs, you can do the following:

```elixir
db = Vettore.new_db()
{:ok, "my_coll"} = Vettore.create_collection(db, "my_coll", 3, "cosine")

# Define a list of embeddings
batch = [
  %Vettore.Embedding{id: "a", vector: [1.0, 2.0, 3.0], metadata: %{"tag" => "alpha"}},
  %Vettore.Embedding{id: "b", vector: [2.0, 2.5, 3.5], metadata: nil}
]

# Insert them all in one call
case Vettore.insert_embeddings(db, "my_coll", batch) do
  {:ok, inserted_ids} ->
    IO.inspect(inserted_ids, label: "Batch inserted IDs")

  {:error, reason} ->
    IO.puts("Failed to insert batch: #{reason}")
end
```

On success, you might see something like:

```
Batch inserted IDs: ["a", "b"]
```

---

## Example Usage for Single Insert & Retrieval

```elixir
defmodule VettoreExample do
  alias Vettore.Embedding

  def demo do
    # 1) Create a new in-memory DB resource
    db = Vettore.new_db()

    # 2) Create a new collection named "my_collection"
    {:ok, "my_collection"} = Vettore.create_collection(db, "my_collection", 3, "euclidean")

    # 3) Insert a single embedding
    embed = %Embedding{id: "emb1", vector: [1.0, 2.0, 3.0], metadata: %{"info" => "test"}}
    {:ok, "emb1"} = Vettore.insert_embedding(db, "my_collection", embed)

    # 4) Retrieve all embeddings
    {:ok, all_embs} = Vettore.get_embeddings(db, "my_collection")
    IO.inspect(all_embs, label: "All embeddings in my_collection")

    # 5) Retrieve specific embedding by ID
    {:ok, %Embedding{id: e_id, vector: e_vec, metadata: e_meta}} =
      Vettore.get_embedding_by_id(db, "my_collection", "emb1")

    IO.inspect(e_id, label: "Single embedding ID")
    IO.inspect(e_vec, label: "Single embedding vector")
    IO.inspect(e_meta, label: "Single embedding metadata")

    # 6) Similarity search
    {:ok, top_results} = Vettore.similarity_search(db, "my_collection", [1.0, 2.0, 3.0], 2)
    IO.inspect(top_results, label: "Similarity search results")
  end
end
```

---

## Performance Notes

- **HNSW** can speed up searches significantly for large datasets but comes with higher memory usage for the index.
- **Binary** distance uses bit-level compression and Hamming distance for extremely fast approximate similarity checks (especially beneficial for large or high-dimensional vectors, though it trades off some precision).
- **Cosine** normalizes vectors once on insertion, so queries and stored embeddings use a straightforward dot product.
- **Dot Product** directly multiplies corresponding elements.
- **Euclidean** uses a SIMD approach (`wide::f32x4`) for partial vectorization.

You can further optimize by compiling with:

```
RUSTFLAGS="-C target-cpu=native -C opt-level=3" mix compile
```

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests on GitHub.

---

## License

This project is licensed under the MIT License.

---
