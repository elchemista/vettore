defmodule Vettore.Nifs do
  @moduledoc false

  version = Mix.Project.config()[:version]

  force_build? = System.get_env("VETTORE_BUILD") in ["1", "true"]

  use RustlerPrecompiled,
    otp_app: :vettore,
    crate: "vettore",
    base_url: "https://github.com/elchemista/vettore/releases/download/v#{version}",
    force_build: force_build?,
    nif_versions: ["2.15", "2.16"],
    version: version

  @doc false
  @spec l2_distance([float()], [float()]) :: {:ok, float()} | {:error, String.t()}
  def l2_distance(_left, _right), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec l2_squared_distance([float()], [float()]) :: {:ok, float()} | {:error, String.t()}
  def l2_squared_distance(_left, _right), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec cosine_similarity([float()], [float()]) :: {:ok, float()} | {:error, String.t()}
  def cosine_similarity(_left, _right), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec normalized_cosine_similarity([float()], [float()]) ::
          {:ok, float()} | {:error, String.t()}
  def normalized_cosine_similarity(_left, _right), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec inner_product([float()], [float()]) :: {:ok, float()} | {:error, String.t()}
  def inner_product(_left, _right), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec negative_inner_product([float()], [float()]) :: {:ok, float()} | {:error, String.t()}
  def negative_inner_product(_left, _right), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec manhattan_distance([float()], [float()]) :: {:ok, float()} | {:error, String.t()}
  def manhattan_distance(_left, _right), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec chebyshev_distance([float()], [float()]) :: {:ok, float()} | {:error, String.t()}
  def chebyshev_distance(_left, _right), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec hamming_distance([float()], [float()]) :: {:ok, float()} | {:error, String.t()}
  def hamming_distance(_left, _right), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec jaccard_distance([float()], [float()]) :: {:ok, float()} | {:error, String.t()}
  def jaccard_distance(_left, _right), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec normalize_l2([float()]) :: {:ok, [float()]} | {:error, String.t()}
  def normalize_l2(_vector), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec normalize_zscore([float()]) :: {:ok, [float()]} | {:error, String.t()}
  def normalize_zscore(_vector), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec normalize_minmax([float()]) :: {:ok, [float()]} | {:error, String.t()}
  def normalize_minmax(_vector), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec compress_sign_bits([float()]) :: [non_neg_integer()]
  def compress_sign_bits(_vector), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec packed_hamming_distance([non_neg_integer()], [non_neg_integer()], pos_integer()) ::
          {:ok, float()} | {:error, String.t()}
  def packed_hamming_distance(_left, _right, _dimensions),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec packed_jaccard_distance([non_neg_integer()], [non_neg_integer()], pos_integer()) ::
          {:ok, float()} | {:error, String.t()}
  def packed_jaccard_distance(_left, _right, _dimensions),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec vector_top_k(
          [{String.t(), [float()]}],
          [float()],
          non_neg_integer(),
          pos_integer(),
          non_neg_integer()
        ) :: {:ok, [{String.t(), float()}]} | {:error, String.t()}
  def vector_top_k(_vectors, _query, _metric_code, _dimensions, _limit),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec binary_top_k(
          [{String.t(), [non_neg_integer()]}],
          [non_neg_integer()],
          pos_integer(),
          non_neg_integer()
        ) :: {:ok, [{String.t(), float()}]} | {:error, String.t()}
  def binary_top_k(_vectors, _query, _dimensions, _limit),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec multi_vector_score([[float()]], [[float()]], non_neg_integer()) ::
          {:ok, float()} | {:error, String.t()}
  def multi_vector_score(_query_vectors, _document_vectors, _metric_code),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec multi_vector_top_k(
          [{String.t(), [[float()]]}],
          [[float()]],
          non_neg_integer(),
          non_neg_integer()
        ) :: {:ok, [{String.t(), float()}]} | {:error, String.t()}
  def multi_vector_top_k(_documents, _query_vectors, _metric_code, _limit),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec flat_new_l2() :: reference()
  def flat_new_l2, do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec flat_new_l2_squared() :: reference()
  def flat_new_l2_squared, do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec flat_new_cosine() :: reference()
  def flat_new_cosine, do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec flat_new_inner_product() :: reference()
  def flat_new_inner_product, do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec flat_new_negative_inner_product() :: reference()
  def flat_new_negative_inner_product, do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec flat_new_manhattan() :: reference()
  def flat_new_manhattan, do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec flat_new_chebyshev() :: reference()
  def flat_new_chebyshev, do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec flat_new_hamming() :: reference()
  def flat_new_hamming, do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec flat_new_jaccard() :: reference()
  def flat_new_jaccard, do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec flat_insert(reference(), String.t(), [float()]) :: :ok | {:ok, {}} | {:error, String.t()}
  def flat_insert(_index, _id, _vector), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec flat_insert_many(reference(), [{String.t(), [float()]}]) ::
          :ok | {:ok, {}} | {:error, String.t()}
  def flat_insert_many(_index, _vectors), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec flat_delete(reference(), String.t()) :: :ok | {:ok, {}} | {:error, String.t()}
  def flat_delete(_index, _id), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec flat_search(reference(), [float()], pos_integer()) ::
          {:ok, [{String.t(), float()}]} | {:error, String.t()}
  def flat_search(_index, _query, _limit), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec hnsw_new_l2(pos_integer(), pos_integer(), pos_integer(), pos_integer(), pos_integer()) ::
          {:ok, reference()} | {:error, String.t()}
  def hnsw_new_l2(_m, _m0, _ef_construction, _ef_search, _max_level),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec hnsw_new_cosine(pos_integer(), pos_integer(), pos_integer(), pos_integer(), pos_integer()) ::
          {:ok, reference()} | {:error, String.t()}
  def hnsw_new_cosine(_m, _m0, _ef_construction, _ef_search, _max_level),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec hnsw_new_inner_product(
          pos_integer(),
          pos_integer(),
          pos_integer(),
          pos_integer(),
          pos_integer()
        ) ::
          {:ok, reference()} | {:error, String.t()}
  def hnsw_new_inner_product(_m, _m0, _ef_construction, _ef_search, _max_level),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec hnsw_insert(reference(), String.t(), [float()]) :: :ok | {:ok, {}} | {:error, String.t()}
  def hnsw_insert(_index, _id, _vector), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec hnsw_insert_many(reference(), [{String.t(), [float()]}]) ::
          :ok | {:ok, {}} | {:error, String.t()}
  def hnsw_insert_many(_index, _vectors), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec hnsw_delete(reference(), String.t()) :: :ok | {:ok, {}} | {:error, String.t()}
  def hnsw_delete(_index, _id), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec hnsw_search(reference(), [float()], pos_integer()) ::
          {:ok, [{String.t(), float()}]} | {:error, String.t()}
  def hnsw_search(_index, _query, _limit), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec muvera_encode_query(
          [[float()]],
          pos_integer(),
          pos_integer(),
          non_neg_integer(),
          non_neg_integer(),
          pos_integer(),
          pos_integer() | nil
        ) :: {:ok, [float()]} | {:error, String.t()}
  def muvera_encode_query(
        _vectors,
        _dimension,
        _num_repetitions,
        _num_simhash_projections,
        _seed,
        _projection_dimension,
        _final_projection_dimension
      ),
      do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec muvera_encode_document(
          [[float()]],
          pos_integer(),
          pos_integer(),
          non_neg_integer(),
          non_neg_integer(),
          pos_integer(),
          pos_integer() | nil
        ) :: {:ok, [float()]} | {:error, String.t()}
  def muvera_encode_document(
        _vectors,
        _dimension,
        _num_repetitions,
        _num_simhash_projections,
        _seed,
        _projection_dimension,
        _final_projection_dimension
      ),
      do: :erlang.nif_error(:nif_not_loaded)
end
