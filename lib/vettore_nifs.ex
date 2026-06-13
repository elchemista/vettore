defmodule Vettore.Nifs do
  @moduledoc false

  version = Mix.Project.config()[:version]

  force_build? =
    Mix.env() in [:dev, :test] or
      System.get_env("RUSTLER_PRECOMPILATION_EXAMPLE_BUILD") in ["1", "true"]

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
  @spec hnsw_new_l2() :: reference()
  def hnsw_new_l2, do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec hnsw_new_cosine() :: reference()
  def hnsw_new_cosine, do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec hnsw_new_inner_product() :: reference()
  def hnsw_new_inner_product, do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec hnsw_insert(reference(), String.t(), [float()]) :: :ok | {:ok, {}} | {:error, String.t()}
  def hnsw_insert(_index, _id, _vector), do: :erlang.nif_error(:nif_not_loaded)

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
