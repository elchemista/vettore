defmodule Vettore.Distance do
  @moduledoc """
  Stand-alone distance/similarity helpers and the collection-agnostic **MMR**
  re-ranker exposed by the Rust NIF layer.
  """
  alias Vettore.Nifs, as: N

  @type embeddings :: [{String.t(), [number()]}]
  @type search_result :: [{String.t(), float()}]
  @type distance :: String.t()
  @type alpha :: float()
  @type final_k :: pos_integer()
  @type vector :: [float()]
  @type vector_bits :: [integer()]

  @doc """
  Similarity based on Euclidean (L2) distance.

  The result is in **`0.0..1.0`** via the mapping `1 / (1 + d)` so that
  identical vectors yield `1.0`.

  #Examples

      iex> Vettore.Distance.euclidean([1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0])
      0.0
  """
  @spec euclidean(vector(), vector()) :: float()
  def euclidean(vec_a, vec_b) when is_list(vec_a) and is_list(vec_b),
    do: N.euclidean_distance(vec_a, vec_b)

  @doc """
  Cosine similarity in **`0.0..1.0`** (`(dot + 1) / 2` after length-normalisation).

  #Examples

      iex> Vettore.Distance.cosine([1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0])
      0.0
  """
  @spec cosine(vector(), vector()) :: float()
  def cosine(vec_a, vec_b) when is_list(vec_a) and is_list(vec_b),
    do: N.cosine_similarity(vec_a, vec_b)

  @doc """
  Raw dot product (no post-processing).

  #Examples

      iex> Vettore.Distance.dot_product([1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0])
      1
  """
  @spec dot_product(vector(), vector()) :: float()
  def dot_product(vec_a, vec_b) when is_list(vec_a) and is_list(vec_b),
    do: N.dot_product(vec_a, vec_b)

  @doc """
  Bit-wise Hamming distance between two compressed vectors (see
  `compress_f32_vector/1`).

  #Examples

      iex> Vettore.Distance.hamming([1, 0, 0, 1, 0, 0], [1, 0, 0, 1, 0, 0])
      0
  """
  @spec hamming(vector_bits(), vector_bits()) :: float()
  def hamming(bits_a, bits_b) when is_list(bits_a) and is_list(bits_b),
    do: N.hamming_distance_bits(bits_a, bits_b)

  @doc """
  Compress a float vector into its sign-bit representation (64 floats → 64 bits →
  one `u64`).  Useful for ultra-fast binary similarity.

  #Examples

      iex> Vettore.Distance.compress_f32_vector([1.0, 2.0, 3.0])
      [1, 0, 0,..... 1, 0, 0]
  """
  @spec compress_f32_vector(vector()) :: vector_bits()
  def compress_f32_vector(vec) when is_list(vec), do: N.compress_f32_vector(vec)

  @doc """
  **MMR** (Maximal-Marginal-Relevance) re-ranker that trades off query relevance
  and result diversity.

  * `initial`   – list of `{id, similarity_to_query}` tuples (first-pass hits)
  * `embeddings` – `{id, vector}` pairs (dimension must be consistent)
  * `distance`  – `"euclidean" | "cosine" | "dot" | "binary"`
  * `alpha`     – 0 ⇢ only diversity, 1 ⇢ only query-relevance
  * `final_k`   – length of the wanted output list

  #Examples

      iex> Vettore.Distance.mmr_rerank([{"my_id", 0.0}], [{"my_id", [1.0, 2.0, 3.0]}], "euclidean", 0.5, 1)
      [{"my_id", 0.0}]
  """

  @spec mmr_rerank(
          search_result(),
          embeddings(),
          distance(),
          alpha(),
          final_k()
        ) :: embeddings()
  def mmr_rerank(initial, embeddings, distance, alpha, k)
      when is_list(initial) and is_list(embeddings) and is_bitstring(distance) and alpha >= 0 and
             alpha <= 1 and k > 0,
      do: N.mmr_rerank_embeddings(initial, embeddings, distance, alpha, k)
end
