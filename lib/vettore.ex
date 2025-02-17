defmodule Vettore do
  use Rustler,
    # or your app name
    otp_app: :vettore,
    # matches the Cargo.toml "name"
    crate: "vettore"

  # Fallbacks in case the NIF isn't loaded# Fallbacks in case the NIF isn't loaded
  @doc """
  Returns a new DB resource (wrapped in `ResourceArc`).

  ## Examples

      Vettore.new_db()

  """
  def new_db(), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Creates a new collection in the database with a specified name, vector dimension, and distance metric (e.g., `"euclidean"`).

  ## Examples

      Vettore.create_collection(db, "my_collection", 3, "euclidean")

  """
  def create_collection(_db, _name, _dim, _distance),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Deletes a collection by its name.

  ## Examples

      Vettore.delete_collection(db, "my_collection")

  """
  def delete_collection(_db, _name),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Inserts an embedding into a collection. Parameters include the collection name, embedding ID, vector (as a list of floats), and optional metadata (a map).

  ## Examples

      Vettore.insert_embedding(db, "my_collection", "emb1", [1.0, 2.0, 3.0], %{"info" => "test"})

  """
  def insert_embedding(_db, _collection, _id, _vector, _metadata \\ nil),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Returns a single embedding (with full metadata) for a given collection and embedding ID.

  ## Examples

      Vettore.get_embedding_by_id(db, "my_collection", "emb1")

  """
  def get_embedding_by_id(_db, _collection_name, _id),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Returns all embeddings from a given collection, each with their ID, vector, and metadata.

  ## Examples

      Vettore.get_embeddings(db, "my_collection")

  """
  def get_embeddings(_db, _collection),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Given a collection name, a query vector, and a number `k`, it returns the top② embeddings as a list of `(id, score)` tuples.

  ## Examples

      Vettore.similarity_search(db, "my_collection", [1.0, 2.0, 3.0], 2)

  """
  def similarity_search(_db, _collection, _query, _k),
    do: :erlang.nif_error(:nif_not_loaded)
end
