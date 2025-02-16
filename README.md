# Vettore: In-Memory Vector Database with Elixir & Rustler

**Vettore** is an in-memory vector database implemented in Rust and exposed to Elixir via [Rustler](https://github.com/rusterlium/rustler). It allows you to create collections of vectors (embeddings), insert new embeddings (with optional metadata), retrieve all embeddings or a specific embedding by its ID, and perform similarity searches using common metrics (Euclidean, Cosine, or Dot Product).

---

## Installation

```elixir
def deps do
  [
    {:vettore, "~> 0.1.1", github: "elchemista/vettore"}
  ]
end
```

---

## Overview

The Vettore library is designed for fast, in-memory operations on vector data. All vectors (embeddings) are stored in a Rust data structure (a HashMap) and accessed via a shared resource (using Rustler’s `ResourceArc` with a Mutex). The core operations include:

- **Creating a collection** – A collection is a set of embeddings with a fixed dimension and a chosen similarity metric from euclidean, cosine, dot.
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
   - `distance: Distance` – The similarity metric used (e.g., Euclidean, Cosine, DotProduct).
   - `embeddings: Vec<Embedding>` – A vector containing all embeddings in the collection.

3. **Embedding**  
   An embedding is represented by:

   - `id: String` – A unique identifier.
   - `vector: Vec<f32>` – The actual vector values.
   - `metadata: Option<HashMap<String, String>>` – Optional additional information.

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
4. For Cosine distance collections, the vector is normalized before insertion.
5. The new embedding is then pushed into the collection’s vector of embeddings.

#### Retrieving Vectors

- **Get All Embeddings:**  
  The `get_embeddings` NIF returns all embeddings from a collection. In the default implementation, it returns a list of triples: `(id, vector, metadata)`.

- **Get Embedding by ID:**  
  The `get_embedding_by_id` function searches the collection’s embeddings (using an iterator) for a matching ID and returns the complete embedding (including metadata).

#### Similarity Search

The `similarity_search` function works as follows:

1. It retrieves the target collection and verifies that the query vector’s dimension matches.
2. A helper function, `get_cache_attr`, is used to precompute any required attribute (for Cosine similarity, it computes the magnitude of the query vector).
3. Depending on the chosen distance metric, it selects an appropriate function:
   - **Euclidean:** Computes the standard Euclidean distance.
   - **Cosine / DotProduct:** Computes the dot product (if Cosine, vectors are normalized on insertion).
4. For every embedding in the collection, it calculates a “score” (distance or similarity value) between the stored vector and the query.
5. The results are sorted:
   - For **Euclidean distance**, lower scores (closer to zero) are better.
   - For **Cosine/DotProduct**, higher scores are considered more similar.
6. Finally, the top‑k results are returned as a list of tuples `(embedding_id, score)`.


| Distance Metric    | Measures                       | Magnitude Sensitive? | Scale Invariant? | Best Use Cases                                                                 | Pros                                                                       | Cons                                                                               |
| ------------------ | ----------------------------- | -------------------- | ---------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Euclidean**      | Straight-line distance        | Yes                  | No               | Magnitude & direction important, dense data, similar scales                  | Intuitive, widely used, magnitudes matter                                    | Magnitude sensitive, curse of dimensionality, not scale-invariant               |
| **Cosine Similarity** | Directional similarity (angle) | No                   | Yes              | Directional similarity, text, varying magnitudes, scale invariance needed  | Magnitude-insensitive, high-dimensional data, text similarity                | Ignores magnitude differences, less intuitive for some magnitude-dependent apps |
| **Dot Product**    | Direction & Magnitude combined | Yes                  | No               | Both direction & magnitude matter, ranking, pre-normalized vectors          | Computationally efficient, captures direction & magnitude                   | Magnitude dependent, less intuitive distance, problematic with unmeaningful magnitudes |



---

## Exposed NIF Functions

All functions are exposed to Elixir using Rustler’s attribute-based NIFs. Here’s a summary:

- `new_db/0`  
  Returns a new DB resource (wrapped in `ResourceArc`).

- `create_collection/4`  
  Creates a new collection in the database with a specified name, vector dimension, and distance metric (e.g., `"euclidean"`, `"cosine"`, `"dot"`).

- `delete_collection/2`  
  Deletes a collection by its name.

- `insert_embedding/5`  
  Inserts an embedding into a collection. Parameters include the collection name, embedding ID, vector (as a list of floats), and optional metadata (a map).

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

    # Create a new collection called "my_collection" with vectors of dimension 3, using Euclidean distance
    {:ok, {}} = Vettore.create_collection(db, "my_collection", 3, "euclidean")

    # Insert two embeddings into the collection
    {:ok, {}} =
      Vettore.insert_embedding(db, "my_collection", "emb1", [1.0, 2.0, 3.0], %{"info" => "test"})

    {:ok, {}} = Vettore.insert_embedding(db, "my_collection", "emb2", [2.0, 3.0, 4.0], nil)

    # Retrieve all embeddings from the collection
    {:ok, embeddings} = Vettore.get_embeddings(db, "my_collection")
    IO.inspect(embeddings, label: "All embeddings:")

    # Retrieve a specific embedding by ID
    {:ok, {id, vector, metadata}} = Vettore.get_embedding_by_id(db, "my_collection", "emb1")
    IO.inspect(id, label: "Embedding ID")
    IO.inspect(vector, label: "Embedding vector")
    IO.inspect(metadata, label: "Embedding metadata")

    # Perform a similarity search with the query vector [1.0, 2.0, 3.0] and return top 2 matches
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

In this example:

- The similarity search returns `emb1` with a score of `0.0` (an exact match) and `emb2` with a score of approximately `1.732` (the Euclidean distance between `[1,2,3]` and `[2,3,4]`).

---

## Detailed Explanation: How It Works Under the Hood

1. **Resource Management**  
   The in-memory database is created using `CacheDB::new` and then wrapped in a `DBResource`, which is stored as a Rustler `ResourceArc`. This allows the Elixir process to hold a reference to the DB, and all operations (create, insert, search) lock the underlying Mutex to ensure thread safety.

2. **Collections & Embeddings**

   - **Collections:** Each collection is a separate entry in the `CacheDB.collections` HashMap, identified by its name. A collection stores the expected vector dimension, the similarity metric, and a vector of embeddings.
   - **Embeddings:** An embedding holds an ID, the vector data, and optional metadata. When inserting an embedding, the function verifies that the provided vector has the correct dimension and that there isn’t a duplicate ID already in the collection.

3. **Distance Metrics and Similarity**  
   The library supports three distance metrics:

   - **Euclidean:** Calculates the standard Euclidean distance between two vectors. For similarity search, a lower distance means more similar.
   - **Cosine:** Vectors are normalized before insertion. The similarity is then calculated via the dot product, with higher values meaning more similar.
   - **DotProduct:** Similar to cosine, except vectors aren’t necessarily normalized.

   The helper function `get_distance_fn` selects the correct distance function based on the collection’s metric. The `similarity_search` function computes a “score” for each embedding by applying the distance function between the stored vector and the query vector.

4. **Search Algorithm**

   - For each embedding in the target collection, the library calculates a score (either the distance or dot product).
   - The resulting list of scores is sorted.
     - **Euclidean:** Sorted in ascending order (lower distances are better).
     - **Cosine/DotProduct:** Sorted in descending order (higher scores are better).
   - The top‑k entries are then returned to the caller.

5. **NIF Integration**  
   Each core operation is exposed as a NIF function (annotated with `#[rustler::nif]`), so that they can be called directly from Elixir. The Rustler macro `rustler::init!` (with the `on_load` callback) ensures that the NIF is properly loaded and that the resource type is registered.
