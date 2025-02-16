defmodule Vettore do
  use Rustler,
    # or your app name
    otp_app: :vettore,
    # matches the Cargo.toml "name"
    crate: "vettore"

  # Fallbacks in case the NIF isn't loaded# Fallbacks in case the NIF isn't loaded
  def new_db(), do: :erlang.nif_error(:nif_not_loaded)

  def create_collection(_db, _name, _dim, _distance),
    do: :erlang.nif_error(:nif_not_loaded)

  def delete_collection(_db, _name),
    do: :erlang.nif_error(:nif_not_loaded)

  def insert_embedding(_db, _collection, _id, _vector, _metadata \\ nil),
    do: :erlang.nif_error(:nif_not_loaded)

  def get_embedding_by_id(_db, _collection_name, _id),
    do: :erlang.nif_error(:nif_not_loaded)

  def get_embeddings(_db, _collection),
    do: :erlang.nif_error(:nif_not_loaded)

  def similarity_search(_db, _collection, _query, _k),
    do: :erlang.nif_error(:nif_not_loaded)
end
