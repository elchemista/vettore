defmodule Vettore do
  use Rustler,
    # or your app name
    otp_app: :vettore,
    # matches the Cargo.toml "name"
    crate: "vettore"

  @moduledoc """
  The Vettore library is designed for fast, in-memory operations on vector data. All vectors (embeddings) are stored in a Rust data structure (a HashMap) and accessed via a shared resource (using Rustler’s `ResourceArc` with a Mutex). The core operations include:
    - **Creating a collection** – A collection is a set of embeddings with a fixed dimension and a chosen similarity metric from **hnsw**, **binary** (Hamming distance), **euclidean**, **cosine**, or **dot**.
    - **Inserting an embedding** – Add a new vector with an identifier and optional metadata to a specific collection.
    - **Retrieving embeddings** – Fetch all embeddings from a collection or look up a single embedding by its unique ID.
    - **Similarity search** – Given a query vector, calculate a “score” (distance or similarity) for every embedding in the collection and return the top‑k results.
  """

  # Fallbacks in case the NIF isn't loaded# Fallbacks in case the NIF isn't loaded
  @spec new_db() :: any()
  @doc """
  Returns a new DB resource (wrapped in `ResourceArc`).

  ## Examples

      Vettore.new_db()

  """
  def new_db(), do: :erlang.nif_error(:nif_not_loaded)

  @spec create_collection(any(), String.t(), integer(), String.t()) :: any()
  @doc """
  Creates a new collection in the database with a specified name, vector dimension, and distance metric (e.g., `"euclidean"`).

  ## Examples

      Vettore.create_collection(db, "my_collection", 3, "euclidean")

  """
  def create_collection(_db, _name, _dim, _distance),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec delete_collection(any(), String.t()) :: any()
  @doc """
  Deletes a collection by its name.

  ## Examples

      Vettore.delete_collection(db, "my_collection")

  """
  def delete_collection(_db, _name),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec insert_embedding(any(), String.t(), String.t(), list(), map()) :: any()
  @doc """
  Inserts an embedding into a collection. Parameters include the collection name, embedding ID, vector (as a list of floats), and optional metadata (a map).

  ## Examples

      Vettore.insert_embedding(db, "my_collection", "emb1", [1.0, 2.0, 3.0], %{"info" => "test"})

  """
  def insert_embedding(_db, _collection, _id, _vector, _metadata \\ nil),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec get_embedding_by_id(any(), String.t(), String.t()) :: any()
  @doc """
  Returns a single embedding (with full metadata) for a given collection and embedding ID.

  ## Examples

      Vettore.get_embedding_by_id(db, "my_collection", "emb1")

  """
  def get_embedding_by_id(_db, _collection_name, _id),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec get_embeddings(any(), String.t()) :: any()
  @doc """
  Returns all embeddings from a given collection, each with their ID, vector, and metadata.

  ## Examples

      Vettore.get_embeddings(db, "my_collection")

  """
  def get_embeddings(_db, _collection),
    do: :erlang.nif_error(:nif_not_loaded)

  @spec similarity_search(any(), String.t(), list(), integer()) :: any()
  @doc """
  Given a collection name, a query vector, and a number `k`, it returns the top② embeddings as a list of `(id, score)` tuples.

  ## Examples

      Vettore.similarity_search(db, "my_collection", [1.0, 2.0, 3.0], 2)

  """
  def similarity_search(_db, _collection, _query, _k),
    do: :erlang.nif_error(:nif_not_loaded)
end
