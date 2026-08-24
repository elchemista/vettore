defmodule Vettore.Distance do
  @moduledoc """
  Independent distance, similarity, normalization, and reranking helpers.

  Named distance functions return raw metric values:

    * distance metrics return a distance where lower is better
    * similarity metrics return a similarity where higher is better
  """

  alias Vettore.{Compute, Nifs}

  @type vector :: [number()]
  @type normalized_vector :: [float()]
  @type metric ::
          :l2
          | :l2_squared
          | :cosine
          | :inner_product
          | :negative_inner_product
          | :manhattan
          | :chebyshev
          | :hamming
          | :jaccard
  @type score_mode :: :raw | :similarity

  @similarity_metrics [:cosine, :inner_product]
  @distance_metrics [
    :l2,
    :l2_squared,
    :negative_inner_product,
    :manhattan,
    :chebyshev,
    :hamming,
    :jaccard
  ]
  @f32_max 3.402_823_466_385_288_6e38
  @u64_max 18_446_744_073_709_551_615
  @native_error_reasons %{
    "metric overflow" => :metric_overflow,
    "vector contains a non-finite value" => :invalid_vector
  }
  @mmr_error_reasons %{"invalid mmr arguments" => :invalid_mmr_args}

  @doc """
  Normalizes a vector.

  ## Examples

      iex> {:ok, normalized} = Vettore.Distance.normalize([3.0, 4.0], :l2)
      iex> Enum.map(normalized, &Float.round(&1, 1))
      [0.6, 0.8]

      iex> Vettore.Distance.normalize([2.0, 4.0, 6.0], :minmax)
      {:ok, [0.0, 0.5, 1.0]}

      iex> Vettore.Distance.normalize([1.0], :unknown)
      {:error, {:unknown_normalization, :unknown}}
  """
  @spec normalize(vector(), :none | :l2 | :zscore | :minmax, keyword()) ::
          {:ok, [float()]} | {:error, term()}
  def normalize(vector, method, opts \\ [])

  def normalize(vector, method, opts)
      when is_list(vector) and method in [:none, :l2, :zscore, :minmax] and is_list(opts) do
    with :ok <- validate_compute_options(opts),
         :ok <- validate_vector(vector) do
      vector = float_vector(vector)

      Compute.run(
        opts,
        length(vector),
        fn -> cpu_normalize(vector, method) end,
        fn -> gpu_normalize(vector, method) end
      )
    end
  end

  def normalize(_vector, method, opts)
      when method in [:none, :l2, :zscore, :minmax] and is_list(opts) do
    with :ok <- validate_compute_options(opts), do: {:error, :invalid_vector}
  end

  def normalize(_vector, _method, opts) when not is_list(opts), do: {:error, :invalid_options}
  def normalize(_vector, method, _opts), do: {:error, {:unknown_normalization, method}}

  @doc """
  Converts a raw metric value into the explicit result score and distance fields.

  ## Examples

      iex> Vettore.Distance.result_values(:l2, 5.0, :raw)
      {-5.0, 5.0}

      iex> Vettore.Distance.result_values(:cosine, 0.25, :raw)
      {0.25, 0.75}

      iex> Vettore.Distance.result_values(:l2, 5.0, :similarity)
      {0.16666666666666666, 5.0}
  """
  @spec result_values(metric() | atom(), number(), score_mode() | atom()) ::
          {float(), float() | nil}
  def result_values(metric, raw, score_mode \\ :raw) do
    do_result_values(metric, raw, score_mode)
  end

  @doc """
  L2 distance.

  ## Examples

      iex> Vettore.Distance.l2([0.0, 0.0], [3.0, 4.0])
      {:ok, 5.0}

      iex> Vettore.Distance.l2([1.0], [1.0, 2.0])
      {:error, :dimension_mismatch}
  """
  @spec l2(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def l2(left, right), do: native_metric(:l2, left, right)

  @doc "L2 distance with per-call CPU/GPU options."
  @spec l2(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def l2(left, right, opts), do: native_metric(:l2, left, right, opts)

  @doc """
  Squared L2 distance.

  ## Examples

      iex> Vettore.Distance.l2_squared([0.0, 0.0], [3.0, 4.0])
      {:ok, 25.0}

      iex> Vettore.Distance.l2_squared([1.0, :bad], [1.0, 2.0])
      {:error, :invalid_vector}
  """
  @spec l2_squared(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def l2_squared(left, right), do: native_metric(:l2_squared, left, right)

  @doc "Squared L2 distance with per-call CPU/GPU options."
  @spec l2_squared(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def l2_squared(left, right, opts), do: native_metric(:l2_squared, left, right, opts)

  @doc """
  Cosine similarity. Defaults to L2-normalizing inputs and returns `[-1.0, 1.0]`.

  ## Examples

      iex> Vettore.Distance.cosine([2.0, 0.0], [4.0, 0.0])
      {:ok, 1.0}

      iex> Vettore.Distance.cosine([2.0, 0.0], [4.0, 0.0], normalize: :none)
      {:ok, 8.0}
  """
  @spec cosine(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def cosine(left, right, opts \\ [])

  def cosine(left, right, opts) when is_list(opts) do
    with :ok <- validate_cosine_options(opts),
         normalize_method = Keyword.get(opts, :normalize, :l2),
         :ok <- validate_pair(left, right) do
      normalized_cosine(left, right, normalize_method, compute_options(opts))
    end
  end

  def cosine(_left, _right, _opts), do: {:error, :invalid_options}

  @doc """
  Inner product.

  ## Examples

      iex> Vettore.Distance.inner_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
      {:ok, 32.0}

      iex> Vettore.Distance.inner_product([1.0], [1.0, 2.0])
      {:error, :dimension_mismatch}
  """
  @spec inner_product(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def inner_product(left, right), do: native_metric(:inner_product, left, right)

  @doc "Inner product with per-call CPU/GPU options."
  @spec inner_product(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def inner_product(left, right, opts), do: native_metric(:inner_product, left, right, opts)

  @doc """
  Negative inner product.

  ## Examples

      iex> Vettore.Distance.negative_inner_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
      {:ok, -32.0}
  """
  @spec negative_inner_product(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def negative_inner_product(left, right), do: native_metric(:negative_inner_product, left, right)

  @doc "Negative inner product with per-call CPU/GPU options."
  @spec negative_inner_product(vector(), vector(), keyword()) ::
          {:ok, float()} | {:error, term()}
  def negative_inner_product(left, right, opts),
    do: native_metric(:negative_inner_product, left, right, opts)

  @doc """
  Manhattan/L1 distance.

  ## Examples

      iex> Vettore.Distance.manhattan([1.0, 2.0], [4.0, 6.0])
      {:ok, 7.0}
  """
  @spec manhattan(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def manhattan(left, right), do: native_metric(:manhattan, left, right)

  @doc "Manhattan distance with per-call CPU/GPU options."
  @spec manhattan(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def manhattan(left, right, opts), do: native_metric(:manhattan, left, right, opts)

  @doc """
  Chebyshev/L-infinity distance.

  ## Examples

      iex> Vettore.Distance.chebyshev([1.0, 2.0], [4.0, 6.0])
      {:ok, 4.0}
  """
  @spec chebyshev(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def chebyshev(left, right), do: native_metric(:chebyshev, left, right)

  @doc "Chebyshev distance with per-call CPU/GPU options."
  @spec chebyshev(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def chebyshev(left, right, opts), do: native_metric(:chebyshev, left, right, opts)

  @doc """
  Hamming distance for equal-length bit/integer vectors.

  ## Examples

      iex> Vettore.Distance.hamming([1, 0, 1], [0, 0, 0])
      {:ok, 2.0}
  """
  @spec hamming(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def hamming(left, right), do: native_metric(:hamming, left, right)

  @doc "Hamming distance with per-call CPU/GPU options."
  @spec hamming(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def hamming(left, right, opts), do: native_metric(:hamming, left, right, opts)

  @doc """
  Jaccard distance for truthy/non-truthy coordinates.

  ## Examples

      iex> {:ok, distance} = Vettore.Distance.jaccard([1, 0, 1], [0, 1, 1])
      iex> Float.round(distance, 6)
      0.666667
  """
  @spec jaccard(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def jaccard(left, right), do: native_metric(:jaccard, left, right)

  @doc "Jaccard distance with per-call CPU/GPU options."
  @spec jaccard(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def jaccard(left, right, opts), do: native_metric(:jaccard, left, right, opts)

  @doc """
  Compatibility alias for L2 distance.

  ## Examples

      iex> Vettore.Distance.euclidean([0.0, 0.0], [3.0, 4.0])
      {:ok, 5.0}
  """
  @spec euclidean(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def euclidean(left, right), do: l2(left, right)

  @doc "Compatibility alias for `l2/3`."
  @spec euclidean(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def euclidean(left, right, opts), do: l2(left, right, opts)

  @doc """
  Compatibility alias for inner product.

  ## Examples

      iex> Vettore.Distance.dot_product([1.0, 2.0], [3.0, 4.0])
      {:ok, 11.0}
  """
  @spec dot_product(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def dot_product(left, right), do: inner_product(left, right)

  @doc "Compatibility alias for `inner_product/3`."
  @spec dot_product(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def dot_product(left, right, opts), do: inner_product(left, right, opts)

  @doc """
  Compress a float vector into packed sign bits.

  ## Examples

      iex> Vettore.Distance.compress_f32_vector([1.0, -2.0, 0.0])
      [5]
  """
  @spec compress_f32_vector(vector()) ::
          [non_neg_integer()] | {:error, :invalid_vector}
  def compress_f32_vector(vector) when is_list(vector) do
    with :ok <- validate_vector(vector) do
      vector
      |> float_vector()
      |> Nifs.compress_sign_bits()
    end
  end

  def compress_f32_vector(_vector), do: {:error, :invalid_vector}

  @doc """
  Hamming distance over packed bit vectors.

  ## Examples

      iex> left = Vettore.Distance.compress_f32_vector([1.0, -2.0, 0.0])
      iex> right = Vettore.Distance.compress_f32_vector([-1.0, -2.0, 0.0])
      iex> Vettore.Distance.packed_hamming(left, right, 3)
      {:ok, 1.0}
  """
  @spec packed_hamming([non_neg_integer()], [non_neg_integer()], pos_integer()) ::
          {:ok, float()} | {:error, term()}
  def packed_hamming(left, right, dimensions)
      when is_list(left) and is_list(right) and is_integer(dimensions) do
    with :ok <- validate_packed_vectors(left, right, dimensions) do
      Nifs.packed_hamming_distance(left, right, dimensions)
      |> normalize_native_error()
    end
  end

  def packed_hamming(_left, _right, _dimensions), do: {:error, :invalid_vector}

  @doc """
  Jaccard distance over packed bit vectors.

  ## Examples

      iex> left = Vettore.Distance.compress_f32_vector([1.0, -2.0, 0.0])
      iex> right = Vettore.Distance.compress_f32_vector([1.0, 2.0, -1.0])
      iex> {:ok, distance} = Vettore.Distance.packed_jaccard(left, right, 3)
      iex> Float.round(distance, 6)
      0.666667
  """
  @spec packed_jaccard([non_neg_integer()], [non_neg_integer()], pos_integer()) ::
          {:ok, float()} | {:error, term()}
  def packed_jaccard(left, right, dimensions)
      when is_list(left) and is_list(right) and is_integer(dimensions) do
    with :ok <- validate_packed_vectors(left, right, dimensions) do
      Nifs.packed_jaccard_distance(left, right, dimensions)
      |> normalize_native_error()
    end
  end

  def packed_jaccard(_left, _right, _dimensions), do: {:error, :invalid_vector}

  @doc """
  Collection-agnostic MMR reranker.

  ## Examples

      iex> initial = [{"a", 0.9}, {"b", 0.8}, {"c", 0.1}]
      iex> embeddings = [{"a", [1.0, 0.0]}, {"b", [1.0, 0.0]}, {"c", [0.0, 1.0]}]
      iex> Vettore.Distance.mmr_rerank(initial, embeddings, :cosine, 0.5, 2)
      {:ok, [{"a", 0.9}, {"c", 0.1}]}
      iex> Vettore.Distance.mmr_rerank(initial, embeddings, :unknown, 0.5, 2)
      {:error, {:unknown_metric, :unknown}}
  """
  @spec mmr_rerank(
          [{String.t(), number()}],
          [{String.t(), vector()}],
          metric() | atom() | String.t(),
          number(),
          pos_integer()
        ) ::
          {:ok, [{String.t(), number()}]}
          | {:error, :invalid_mmr_args | {:unknown_metric, term()}}
  def mmr_rerank(initial, embeddings, metric, alpha, final_k)
      when is_list(initial) and is_list(embeddings) and is_number(alpha) and alpha >= 0 and
             alpha <= 1 and is_integer(final_k) and final_k > 0 do
    with :ok <- validate_metric(metric),
         {:ok, vectors} <- validate_mmr_embeddings(embeddings),
         :ok <- validate_mmr_initial(initial, vectors) do
      native_mmr(initial, vectors, metric, alpha, final_k)
    end
  end

  def mmr_rerank(_initial, _embeddings, _metric, _alpha, _final_k),
    do: {:error, :invalid_mmr_args}

  @spec validate_mmr_embeddings(term()) ::
          {:ok, %{String.t() => vector()}} | {:error, :invalid_mmr_args}
  defp validate_mmr_embeddings(embeddings) do
    result =
      Enum.reduce_while(embeddings, {:ok, %{}, nil}, fn embedding,
                                                        {:ok, vectors, expected_dimensions} ->
        case validate_mmr_embedding(embedding, vectors, expected_dimensions) do
          {:ok, vectors, dimensions} -> {:cont, {:ok, vectors, dimensions}}
          {:error, :invalid_mmr_args} = error -> {:halt, error}
        end
      end)

    case result do
      {:ok, vectors, _dimensions} -> {:ok, vectors}
      {:error, :invalid_mmr_args} = error -> error
    end
  end

  @spec validate_mmr_embedding(term(), map(), pos_integer() | nil) ::
          {:ok, map(), pos_integer()} | {:error, :invalid_mmr_args}
  defp validate_mmr_embedding({id, vector}, vectors, expected_dimensions)
       when is_binary(id) and is_list(vector) do
    if id != "" and vector != [] do
      validate_mmr_embedding_values(id, vector, vectors, expected_dimensions)
    else
      {:error, :invalid_mmr_args}
    end
  end

  defp validate_mmr_embedding(_embedding, _vectors, _expected_dimensions),
    do: {:error, :invalid_mmr_args}

  @spec validate_mmr_embedding_values(String.t(), vector(), map(), pos_integer() | nil) ::
          {:ok, map(), pos_integer()} | {:error, :invalid_mmr_args}
  defp validate_mmr_embedding_values(id, vector, vectors, expected_dimensions) do
    dimensions = length(vector)

    cond do
      Map.has_key?(vectors, id) ->
        {:error, :invalid_mmr_args}

      expected_dimensions not in [nil, dimensions] ->
        {:error, :invalid_mmr_args}

      not Enum.all?(vector, &finite_number?/1) ->
        {:error, :invalid_mmr_args}

      not String.valid?(id) ->
        {:error, :invalid_mmr_args}

      true ->
        {:ok, Map.put(vectors, id, vector), expected_dimensions || dimensions}
    end
  end

  @spec validate_mmr_initial(term(), map()) :: :ok | {:error, :invalid_mmr_args}
  defp validate_mmr_initial(initial, vectors) do
    {valid?, _ids} =
      Enum.reduce_while(initial, {true, MapSet.new()}, fn
        {id, score}, {true, ids} when is_binary(id) and id != "" ->
          if String.valid?(id) and finite_number?(score) and Map.has_key?(vectors, id) and
               not MapSet.member?(ids, id) do
            {:cont, {true, MapSet.put(ids, id)}}
          else
            {:halt, {false, ids}}
          end

        _entry, {_valid, ids} ->
          {:halt, {false, ids}}
      end)

    if valid?, do: :ok, else: {:error, :invalid_mmr_args}
  end

  @spec finite_number?(term()) :: boolean()
  defp finite_number?(value) when is_integer(value),
    do: value >= -@f32_max and value <= @f32_max

  defp finite_number?(value) when is_float(value),
    do: value >= -@f32_max and value <= @f32_max

  defp finite_number?(_value), do: false

  @spec native_mmr([{String.t(), number()}], map(), metric(), number(), pos_integer()) ::
          {:ok, [{String.t(), number()}]} | {:error, term()}
  defp native_mmr(initial, vectors, metric, alpha, final_k) do
    native_initial = Enum.map(initial, fn {id, score} -> {id, score / 1} end)
    native_vectors = Enum.map(vectors, fn {id, vector} -> {id, float_vector(vector)} end)
    native_k = min(final_k, max(length(initial), 1))
    scores = Map.new(initial)

    case Nifs.mmr_rerank(native_initial, native_vectors, metric_code(metric), alpha / 1, native_k)
         |> normalize_native_error() do
      {:ok, ids} -> {:ok, Enum.map(ids, &{&1, Map.fetch!(scores, &1)})}
      {:error, reason} -> {:error, Map.get(@mmr_error_reasons, reason, reason)}
    end
  end

  @spec similarity_distance(metric(), number()) :: float() | nil
  defp similarity_distance(:cosine, raw), do: 1.0 - raw
  defp similarity_distance(:inner_product, raw), do: -raw

  @spec do_result_values(metric() | atom(), number(), score_mode() | atom()) ::
          {float(), float() | nil}
  defp do_result_values(:negative_inner_product, raw, score_mode)
       when score_mode in [:raw, :similarity],
       do: {-raw / 1, raw / 1}

  defp do_result_values(metric, raw, :raw) when metric in @similarity_metrics,
    do: {raw / 1, similarity_distance(metric, raw)}

  defp do_result_values(metric, raw, :raw) when metric in @distance_metrics,
    do: {-raw / 1, raw / 1}

  defp do_result_values(metric, raw, :similarity) when metric in @similarity_metrics,
    do: {similarity_score(metric, raw), similarity_distance(metric, raw)}

  defp do_result_values(metric, raw, :similarity) when metric in @distance_metrics,
    do: {1.0 / (1.0 + raw), raw / 1}

  defp do_result_values(_metric, raw, _score_mode), do: {raw / 1, nil}

  @spec similarity_score(metric(), number()) :: float()
  defp similarity_score(:cosine, raw), do: (raw + 1.0) / 2.0
  defp similarity_score(:inner_product, raw), do: raw

  @spec validate_pair(vector(), vector()) :: :ok | {:error, :dimension_mismatch | :invalid_vector}
  defp validate_pair(left, right) when is_list(left) and is_list(right) do
    cond do
      length(left) != length(right) -> {:error, :dimension_mismatch}
      Enum.all?(left, &finite_number?/1) and Enum.all?(right, &finite_number?/1) -> :ok
      true -> {:error, :invalid_vector}
    end
  end

  defp validate_pair(_left, _right), do: {:error, :invalid_vector}

  @spec validate_vector(term()) :: :ok | {:error, :invalid_vector}
  defp validate_vector(vector) when is_list(vector) do
    if Enum.all?(vector, &finite_number?/1), do: :ok, else: {:error, :invalid_vector}
  end

  @spec validate_cosine_options(term()) :: :ok | {:error, :invalid_options}
  defp validate_cosine_options(opts) do
    validate_options(opts, [:normalize, :gpu, :gpu_fallback, :gpu_min_size])
  end

  @spec validate_compute_options(term()) :: :ok | {:error, :invalid_options}
  defp validate_compute_options(opts),
    do: validate_options(opts, [:gpu, :gpu_fallback, :gpu_min_size])

  @spec validate_options(term(), [atom()]) :: :ok | {:error, :invalid_options}
  defp validate_options(opts, allowed) do
    if Keyword.keyword?(opts) do
      keys = Keyword.keys(opts)

      if keys == Enum.uniq(keys) and Enum.all?(keys, &(&1 in allowed)),
        do: :ok,
        else: {:error, :invalid_options}
    else
      {:error, :invalid_options}
    end
  end

  @spec validate_packed_vectors(term(), term(), term()) :: :ok | {:error, :invalid_vector}
  defp validate_packed_vectors(left, right, dimensions) do
    words = if dimensions > 0, do: div(dimensions + 63, 64), else: 0

    if dimensions > 0 and length(left) == words and length(right) == words and
         Enum.all?(left, &valid_u64?/1) and Enum.all?(right, &valid_u64?/1),
       do: :ok,
       else: {:error, :invalid_vector}
  end

  @spec valid_u64?(term()) :: boolean()
  defp valid_u64?(value), do: is_integer(value) and value >= 0 and value <= @u64_max

  @spec validate_metric(term()) :: :ok | {:error, {:unknown_metric, term()}}
  defp validate_metric(metric) when metric in @similarity_metrics or metric in @distance_metrics,
    do: :ok

  defp validate_metric(metric), do: {:error, {:unknown_metric, metric}}

  @spec metric_code(metric()) :: non_neg_integer()
  defp metric_code(:l2), do: 0
  defp metric_code(:l2_squared), do: 1
  defp metric_code(:cosine), do: 2
  defp metric_code(:inner_product), do: 3
  defp metric_code(:negative_inner_product), do: 4
  defp metric_code(:manhattan), do: 5
  defp metric_code(:chebyshev), do: 6
  defp metric_code(:hamming), do: 7
  defp metric_code(:jaccard), do: 8

  @spec native_metric(metric(), vector(), vector(), keyword()) ::
          {:ok, float()} | {:error, term()}
  defp native_metric(metric, left, right, opts \\ []) do
    with :ok <- validate_compute_options(opts),
         :ok <- validate_metric(metric),
         :ok <- validate_pair(left, right) do
      Compute.run(
        opts,
        length(left),
        fn -> native_call(metric, left, right) end,
        fn -> gpu_metric(left, right, metric) end
      )
    end
  end

  @spec native_call(metric(), vector(), vector()) :: {:ok, float()} | {:error, term()}
  defp native_call(:l2, left, right), do: native_pair(left, right, &Nifs.l2_distance/2)

  defp native_call(:l2_squared, left, right),
    do: native_pair(left, right, &Nifs.l2_squared_distance/2)

  defp native_call(:cosine, left, right),
    do: native_pair(left, right, &Nifs.normalized_cosine_similarity/2)

  defp native_call(:inner_product, left, right),
    do: native_pair(left, right, &Nifs.inner_product/2)

  defp native_call(:negative_inner_product, left, right),
    do: native_pair(left, right, &Nifs.negative_inner_product/2)

  defp native_call(:manhattan, left, right),
    do: native_pair(left, right, &Nifs.manhattan_distance/2)

  defp native_call(:chebyshev, left, right),
    do: native_pair(left, right, &Nifs.chebyshev_distance/2)

  defp native_call(:hamming, left, right),
    do: native_pair(left, right, &Nifs.hamming_distance/2)

  defp native_call(:jaccard, left, right),
    do: native_pair(left, right, &Nifs.jaccard_distance/2)

  @spec normalized_cosine(vector(), vector(), term(), keyword()) ::
          {:ok, float()} | {:error, term()}
  defp normalized_cosine(left, right, :l2, opts) do
    Compute.run(
      opts,
      length(left),
      fn -> native_call(:cosine, left, right) end,
      fn -> gpu_metric(left, right, :cosine) end
    )
  end

  defp normalized_cosine(left, right, :none, opts),
    do: native_metric(:inner_product, left, right, opts)

  defp normalized_cosine(left, right, normalize_method, opts) do
    with {:ok, left} <- normalize(left, normalize_method, opts),
         {:ok, right} <- normalize(right, normalize_method, opts) do
      native_metric(:inner_product, left, right, opts)
    end
  end

  @spec gpu_metric(vector(), vector(), metric()) :: {:ok, float()} | {:error, term()}
  defp gpu_metric(left, right, metric) do
    left
    |> float_vector()
    |> Nifs.gpu_metric(float_vector(right), metric_code(metric))
    |> normalize_native_error()
  end

  @spec cpu_normalize(normalized_vector(), :none | :l2 | :zscore | :minmax) ::
          {:ok, normalized_vector()} | {:error, term()}
  defp cpu_normalize(vector, :none), do: {:ok, vector}

  defp cpu_normalize(vector, :l2),
    do: vector |> Nifs.normalize_l2() |> normalize_native_error()

  defp cpu_normalize(vector, :zscore),
    do: vector |> Nifs.normalize_zscore() |> normalize_native_error()

  defp cpu_normalize(vector, :minmax),
    do: vector |> Nifs.normalize_minmax() |> normalize_native_error()

  @spec gpu_normalize(normalized_vector(), :none | :l2 | :zscore | :minmax) ::
          {:ok, normalized_vector()} | {:error, term()}
  defp gpu_normalize(vector, method) do
    vector
    |> Nifs.gpu_normalize(normalization_code(method))
    |> normalize_native_error()
  end

  @spec normalization_code(:none | :l2 | :zscore | :minmax) :: 0..3
  defp normalization_code(:none), do: 0
  defp normalization_code(:l2), do: 1
  defp normalization_code(:zscore), do: 2
  defp normalization_code(:minmax), do: 3

  @spec compute_options(keyword()) :: keyword()
  defp compute_options(opts),
    do: Keyword.take(opts, [:gpu, :gpu_fallback, :gpu_min_size])

  @spec native_pair(vector(), vector(), (normalized_vector(), normalized_vector() -> term())) ::
          term()
  defp native_pair(left, right, fun) do
    left = float_vector(left)
    right = float_vector(right)

    fun.(left, right)
    |> normalize_native_error()
  end

  @spec normalize_native_error({:error, String.t()} | {:ok, term()} | term()) ::
          {:error, String.t()} | {:ok, term()} | term()
  defp normalize_native_error({:error, "gpu " <> _reason}), do: {:error, :gpu_failed}

  defp normalize_native_error({:error, reason}) when is_binary(reason),
    do: {:error, Map.get(@native_error_reasons, reason, reason)}

  defp normalize_native_error(other), do: other

  @spec float_vector(vector()) :: normalized_vector()
  defp float_vector(vector), do: Enum.map(vector, &(&1 / 1))
end
