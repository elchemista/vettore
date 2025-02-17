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
    {:vettore, "~> 0.1.2", github: "elchemista/vettore"}
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

- **Creating a collection** – A collection is a set of embeddings with a fixed dimension and a chosen similarity metric from **hnsw**, **binary** (Hamming distance), **euclidean**, **cosine**, or **dot**.
- **Inserting an embedding** – Add a new vector with an identifier and optional metadata to a specific collection.
- **Retrieving embeddings** – Fetch all embeddings from a collection or look up a single embedding by its unique ID.
- **Similarity search** – Given a query vector, calculate a “score” (distance or similarity) for every embedding in the collection and return the top‑k results.

---

## Under the Hood

### Data Structures

1. **CacheDB**  
   The main in‑memory database is defined as a `CacheDB` struct. Internally, it stores collections in a Rust [`HashMap<String, Collection>`](https://doc.rust-lang.org/std/collections/struct.HashMap.html), mapping collection names to a `Collection` struct.

2. **Collection**  
   Each collection is a struct with the following fields:
   - `dimension: usize` – The fixed length of every vector in this collection.
   - `distance: Distance` – The similarity metric used (e.g., Euclidean, Cosine, DotProduct, HNSW, Binary).
   - `embeddings: Vec<Embedding>` – A vector containing all embeddings in the collection.

3. **Embedding**  
   An embedding is represented by:
   - `id: String` – A unique identifier.
   - `vector: Vec<f32>` – The actual vector values.
   - `metadata: Option<HashMap<String, String>>` – Optional additional information.
   - `binary: Option<Vec<u64>>` – *New:* A compressed binary signature (used with the **binary** distance metric).

4. **DBResource**  
   The `CacheDB` is wrapped in a `DBResource` which is stored inside a Rustler `ResourceArc`. This allows the Elixir side to hold a reference to the in‑memory database across NIF calls. The database is guarded by a `Mutex` to ensure safe concurrent access.

---

### How It Works

#### Storage

- **In-Memory Storage:**  
  All data is stored in memory (within the `CacheDB` struct). No external databases or disk storage is used by default.
  - Collections are stored as key–value pairs in a HashMap.
  - Each collection’s embeddings are stored in a `Vec`.

#### Inserting Vectors

When you call the `insert_embedding` function:
1. The collection is retrieved from the database.
2. The function checks that the vector’s dimension matches the collection’s defined dimension.
3. It verifies that there isn’t already an embedding with the same ID in the collection.
4. For **Cosine** distance collections, the vector is normalized before insertion.
5. For **Binary** distance collections, a binary signature is precomputed and stored (using the sign of each float to produce a bit-packed representation).
6. The new embedding is then pushed into the collection’s vector of embeddings.

#### Retrieving Vectors

- **Get All Embeddings:**  
  The `get_embeddings` NIF returns all embeddings from a collection as a list of triples: `(id, vector, metadata)`.

- **Get Embedding by ID:**  
  The `get_embedding_by_id` function searches the collection’s embeddings (using an iterator) for a matching ID and returns the complete embedding (including metadata).

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

| Technique/Algorithm    | Measures                                   | Magnitude Sensitive?¹ | Scale Invariant?¹ | Best Use Cases                                                                       | Pros                                                                                                | Cons                                                                                                                 |
| ---------------------- | ------------------------------------------ | ---------------------- | ----------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Euclidean Distance** | Straight-line distance                     | Yes                    | No                | Dense data where both magnitude & direction are important                         | Intuitive, widely used, captures magnitude differences                                               | Sensitive to scale differences, high dimensionality issues                                                           |
| **Cosine Similarity**  | Directional similarity (angle)             | No                     | Yes               | Text or high-dimensional data where scale invariance is desired                     | Insensitive to magnitude, works well with normalized vectors                                         | Ignores magnitude differences                                                                                        |
| **Dot Product**        | Combination of direction & magnitude       | Yes                    | No                | Applications where both direction & magnitude matter                                | Computationally efficient, captures both aspects                                                     | Sensitive to vector magnitudes                                                                                       |
| **HNSW Indexing**      | Graph-based Approximate Nearest Neighbor Search | Dependent on Metric   | Dependent on Metric | Large datasets, real-time search when approximate results are acceptable             | **Sublinear search time**, good speed-accuracy trade-off, scalable                                    | Approximate results, index build time and memory overhead                                                             |
| **Binary (Hamming)**   | Fast binary signature comparison using Hamming distance | No                     | Yes               | Applications requiring ultra‑fast approximate searches on large-scale data           | Extremely fast comparison via bit-level operations, low memory footprint                             | Loses precision due to compression, less suited when exact distances are needed                                        |

---

## Exposed NIF Functions

All functions are exposed to Elixir using Rustler’s attribute-based NIFs. Here’s a summary:

- `new_db/0`  
  Returns a new DB resource (wrapped in `ResourceArc`).

- `create_collection/4`  
  Creates a new collection in the database with a specified name, vector dimension, and distance metric (e.g., `"euclidean"`, `"cosine"`, `"dot"`, `"hnsw"`, or `"binary"`).

- `delete_collection/2`  
  Deletes a collection by its name.

- `insert_embedding/5`  
  Inserts an embedding into a collection. Parameters include the collection name, embedding ID, vector (as a list of floats), and optional metadata (a map). For **binary** collections, the embedding’s binary signature is automatically computed.

- `get_embeddings/2`  
  Returns all embeddings from a given collection, each with their ID, vector, and metadata.

- `get_embedding_by_id/3`  
  Returns a single embedding (with full metadata) for a given collection and embedding ID.

- `similarity_search/4`  
  Given a collection name, a query vector, and a number `k`, it returns the top‑k embeddings as a list of `(id, score)` tuples.

---

## Example Usage

### Elixir Code

Below is an example of how you might use the Vettore library from Elixir:

```elixir
defmodule VettoreExample do
  def demo do
    # Create a new in-memory DB resource
    db = Vettore.new_db()

    # Create a new collection called "my_collection" with vectors of dimension 3,
    # using Euclidean distance.
    {:ok, {}} = Vettore.create_collection(db, "my_collection", 3, "euclidean")

    # Insert two embeddings into the collection.
    {:ok, {}} =
      Vettore.insert_embedding(db, "my_collection", "emb1", [1.0, 2.0, 3.0], %{"info" => "test"})

    {:ok, {}} = Vettore.insert_embedding(db, "my_collection", "emb2", [2.0, 3.0, 4.0], nil)

    # Retrieve all embeddings from the collection.
    {:ok, embeddings} = Vettore.get_embeddings(db, "my_collection")
    IO.inspect(embeddings, label: "All embeddings:")

    # Retrieve a specific embedding by ID.
    {:ok, {id, vector, metadata}} = Vettore.get_embedding_by_id(db, "my_collection", "emb1")
    IO.inspect(id, label: "Embedding ID")
    IO.inspect(vector, label: "Embedding vector")
    IO.inspect(metadata, label: "Embedding metadata")

    # Perform a similarity search with the query vector [1.0, 2.0, 3.0] and return top 2 matches.
    {:ok, top_results} = Vettore.similarity_search(db, "my_collection", [1.0, 2.0, 3.0], 2)
    IO.inspect(top_results, label: "Similarity results (id, score):")
  end
end
```

### Expected Output

Running the above code (for example, in an IEx session) might produce:

```
All embeddings: [{"emb1", [1.0, 2.0, 3.0], %{"info" => "test"}}, {"emb2", [2.0, 3.0, 4.0], nil}]
Embedding ID: "emb1"
Embedding vector: [1.0, 2.0, 3.0]
Embedding metadata: %{"info" => "test"}
Similarity results (id, score): [{"emb1", 0.0}, {"emb2", 1.7320507764816284}]
```

For collections using the **binary** distance metric, the similarity search compresses the vectors into binary signatures and computes the Hamming distance. This yields extremely fast similarity comparisons, especially useful for large-scale datasets, albeit with a trade-off in precision.

---

## Detailed Explanation: How It Works Under the Hood

1. **Resource Management**  
   The in-memory database is created using `CacheDB::new` and then wrapped in a `DBResource`, which is stored as a Rustler `ResourceArc`. This allows the Elixir process to hold a reference to the DB, and all operations (create, insert, search) lock the underlying Mutex to ensure thread safety.

2. **Collections & Embeddings**

   - **Collections:** Each collection is a separate entry in the `CacheDB.collections` HashMap, identified by its name. A collection stores the expected vector dimension, the similarity metric, and a vector of embeddings.
   - **Embeddings:** An embedding holds an ID, the vector data, optional metadata, and (if applicable) a precomputed binary signature for fast Hamming comparisons.

3. **Distance Metrics and Similarity**  
   The library supports several distance metrics:
   - **Euclidean:** Calculates the standard Euclidean distance between two vectors.
   - **Cosine:** Vectors are normalized before insertion; similarity is computed via the dot product.
   - **Dot Product:** Uses the dot product directly.
   - **HNSW:** Leverages a graph-based approach for approximate nearest neighbor search.
   - **Binary:** Compresses each vector into a binary signature (using the sign of each component) and computes similarity via Hamming distance. This provides very fast approximate search by using bit-level operations.

4. **Search Algorithm**

   - For each embedding in the target collection, the library calculates a score (distance or similarity) between the stored vector (or its binary signature) and the query vector.
   - The resulting list of scores is sorted:
     - **Euclidean:** Sorted in ascending order.
     - **Cosine/Dot Product:** Sorted in descending order.
     - **Binary:** Sorted in ascending order of Hamming distance.
   - The top‑k entries are returned to the caller.

5. **NIF Integration**  
   Each core operation is exposed as a NIF function (annotated with `#[rustler::nif]`), so that they can be called directly from Elixir. The Rustler macro `rustler::init!` (with the `on_load` callback) ensures that the NIF is properly loaded and that the resource type is registered.

---
