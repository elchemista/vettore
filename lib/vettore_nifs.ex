defmodule Vettore.Nifs do
  @moduledoc false

  # vNext keeps Rust optional. This module remains as a compatibility namespace
  # for callers that reached into the old NIF layer directly.

  def new_db, do: {:error, :native_db_removed}
  def create_collection(_db, _name, _dim, _dist, _keep?), do: {:error, :native_db_removed}
  def delete_collection(_db, _name), do: {:error, :native_db_removed}
  def insert_embedding(_db, _col, _id, _vec, _meta \\ nil), do: {:error, :native_db_removed}
  def insert_embeddings(_db, _col, _embeddings), do: {:error, :native_db_removed}
  def get_embedding_by_value(_db, _col, _id), do: {:error, :native_db_removed}
  def get_embedding_by_vector(_db, _col, _vec), do: {:error, :native_db_removed}
  def get_all_embeddings(_db, _col), do: {:error, :native_db_removed}
  def delete_embedding_by_value(_db, _col, _id), do: {:error, :native_db_removed}
  def similarity_search(_db, _col, _q, _k), do: {:error, :native_db_removed}
  def similarity_search_with_filter(_db, _col, _q, _k, _filter), do: {:error, :native_db_removed}
  def mmr_rerank(_db, _col, _init, _alpha, _k), do: {:error, :native_db_removed}

  def euclidean_distance(left, right), do: Vettore.Distance.euclidean(left, right)
  def cosine_similarity(left, right), do: Vettore.Distance.cosine(left, right)
  def dot_product(left, right), do: Vettore.Distance.dot_product(left, right)
  def hamming_distance_bits(left, right), do: Vettore.Distance.hamming(left, right)
  def compress_f32_vector(vector), do: Vettore.Distance.compress_f32_vector(vector)

  def mmr_rerank_embeddings(initial, embeddings, distance, alpha, k),
    do: Vettore.Distance.mmr_rerank(initial, embeddings, distance, alpha, k)
end
