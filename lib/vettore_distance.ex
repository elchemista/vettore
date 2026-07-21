defmodule Vettore.Distance do
  @moduledoc """
  Independent distance, similarity, normalization, and reranking helpers.

  Named distance functions return raw metric values:

    * distance metrics return a distance where lower is better
    * similarity metrics return a similarity where higher is better
  """

  alias Vettore.Nifs

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
  @spec normalize(vector(), :none | :l2 | :zscore | :minmax) ::
          {:ok, [float()]} | {:error, term()}
  def normalize(vector, :none) when is_list(vector) do
    with :ok <- validate_vector(vector) do
      {:ok, Enum.map(vector, &(&1 / 1))}
    end
  end

  def normalize(vector, :l2) when is_list(vector) do
    with :ok <- validate_vector(vector) do
      vector |> float_vector() |> Nifs.normalize_l2() |> normalize_native_error()
    end
  end

  def normalize(vector, :zscore) when is_list(vector) do
    with :ok <- validate_vector(vector) do
      vector |> float_vector() |> Nifs.normalize_zscore() |> normalize_native_error()
    end
  end

  def normalize(vector, :minmax) when is_list(vector) do
    with :ok <- validate_vector(vector) do
      vector |> float_vector() |> Nifs.normalize_minmax() |> normalize_native_error()
    end
  end

  def normalize(_vector, method) when method in [:none, :l2, :zscore, :minmax],
    do: {:error, :invalid_vector}

  def normalize(_vector, method), do: {:error, {:unknown_normalization, method}}

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
      normalized_cosine(left, right, normalize_method)
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

  @doc """
  Negative inner product.

  ## Examples

      iex> Vettore.Distance.negative_inner_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
      {:ok, -32.0}
  """
  @spec negative_inner_product(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def negative_inner_product(left, right), do: native_metric(:negative_inner_product, left, right)

  @doc """
  Manhattan/L1 distance.

  ## Examples

      iex> Vettore.Distance.manhattan([1.0, 2.0], [4.0, 6.0])
      {:ok, 7.0}
  """
  @spec manhattan(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def manhattan(left, right), do: native_metric(:manhattan, left, right)

  @doc """
  Chebyshev/L-infinity distance.

  ## Examples

      iex> Vettore.Distance.chebyshev([1.0, 2.0], [4.0, 6.0])
      {:ok, 4.0}
  """
  @spec chebyshev(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def chebyshev(left, right), do: native_metric(:chebyshev, left, right)

  @doc """
  Hamming distance for equal-length bit/integer vectors.

  ## Examples

      iex> Vettore.Distance.hamming([1, 0, 1], [0, 0, 0])
      {:ok, 2.0}
  """
  @spec hamming(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def hamming(left, right), do: native_metric(:hamming, left, right)

  @doc """
  Jaccard distance for truthy/non-truthy coordinates.

  ## Examples

      iex> {:ok, distance} = Vettore.Distance.jaccard([1, 0, 1], [0, 1, 1])
      iex> Float.round(distance, 6)
      0.666667
  """
  @spec jaccard(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def jaccard(left, right), do: native_metric(:jaccard, left, right)

  @doc """
  Compatibility alias for L2 distance.

  ## Examples

      iex> Vettore.Distance.euclidean([0.0, 0.0], [3.0, 4.0])
      {:ok, 5.0}
  """
  @spec euclidean(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def euclidean(left, right), do: l2(left, right)

  @doc """
  Compatibility alias for inner product.

  ## Examples

      iex> Vettore.Distance.dot_product([1.0, 2.0], [3.0, 4.0])
      {:ok, 11.0}
  """
  @spec dot_product(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def dot_product(left, right), do: inner_product(left, right)

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
      do_mmr(initial, vectors, metric, alpha, final_k, [])
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
       when is_binary(id) and id != "" and is_list(vector) and vector != [] do
    dimensions = length(vector)

    cond do
      Map.has_key?(vectors, id) ->
        {:error, :invalid_mmr_args}

      expected_dimensions not in [nil, dimensions] ->
        {:error, :invalid_mmr_args}

      not Enum.all?(vector, &finite_number?/1) ->
        {:error, :invalid_mmr_args}

      true ->
        {:ok, Map.put(vectors, id, vector), expected_dimensions || dimensions}
    end
  end

  defp validate_mmr_embedding(_embedding, _vectors, _expected_dimensions),
    do: {:error, :invalid_mmr_args}

  @spec validate_mmr_initial(term(), map()) :: :ok | {:error, :invalid_mmr_args}
  defp validate_mmr_initial(initial, vectors) do
    {valid?, _ids} =
      Enum.reduce_while(initial, {true, MapSet.new()}, fn
        {id, score}, {true, ids} when is_binary(id) and id != "" ->
          if finite_number?(score) and Map.has_key?(vectors, id) and not MapSet.member?(ids, id) do
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
    do: value >= -3.402_823_466_385_288_6e38 and value <= 3.402_823_466_385_288_6e38

  defp finite_number?(value) when is_float(value),
    do: value >= -3.402_823_466_385_288_6e38 and value <= 3.402_823_466_385_288_6e38

  defp finite_number?(_value), do: false

  @spec do_mmr(
          [{String.t(), number()}],
          %{String.t() => vector()},
          metric(),
          number(),
          non_neg_integer(),
          [{String.t(), number()}]
        ) :: {:ok, [{String.t(), number()}]} | {:error, term()}
  defp do_mmr(_remaining, _vectors, _metric, _alpha, 0, selected),
    do: {:ok, Enum.reverse(selected)}

  defp do_mmr([], _vectors, _metric, _alpha, _left, selected),
    do: {:ok, Enum.reverse(selected)}

  defp do_mmr(remaining, vectors, metric, alpha, left, selected) do
    with {:ok, scored} <- score_mmr_candidates(remaining, selected, vectors, metric, alpha) do
      {chosen, index, _mmr_score} = Enum.max_by(scored, &elem(&1, 2))
      rest = List.delete_at(remaining, index)
      do_mmr(rest, vectors, metric, alpha, left - 1, [chosen | selected])
    end
  end

  @spec score_mmr_candidates(
          [{String.t(), number()}],
          [{String.t(), number()}],
          %{String.t() => vector()},
          metric(),
          number()
        ) :: {:ok, [{{String.t(), number()}, non_neg_integer(), float()}]} | {:error, term()}
  defp score_mmr_candidates(remaining, selected, vectors, metric, alpha) do
    remaining
    |> Enum.with_index()
    |> Enum.reduce_while({:ok, []}, fn {{id, query_score} = candidate, index}, {:ok, acc} ->
      case maximum_redundancy(id, selected, vectors, metric) do
        {:ok, redundancy} ->
          score = alpha * query_score - (1.0 - alpha) * redundancy
          {:cont, {:ok, [{candidate, index, score / 1} | acc]}}

        {:error, reason} ->
          {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, scored} -> {:ok, Enum.reverse(scored)}
      {:error, reason} -> {:error, reason}
    end
  end

  @spec maximum_redundancy(
          String.t(),
          [{String.t(), number()}],
          %{String.t() => vector()},
          metric()
        ) :: {:ok, float()} | {:error, term()}
  defp maximum_redundancy(_id, [], _vectors, _metric), do: {:ok, 0.0}

  defp maximum_redundancy(id, selected, vectors, metric) do
    Enum.reduce_while(selected, {:ok, nil}, fn {selected_id, _score}, {:ok, maximum} ->
      case pair_similarity(metric, Map.fetch!(vectors, id), Map.fetch!(vectors, selected_id)) do
        {:ok, similarity} ->
          maximum = maximum_similarity(maximum, similarity)
          {:cont, {:ok, maximum}}

        {:error, reason} ->
          {:halt, {:error, reason}}
      end
    end)
  end

  @spec maximum_similarity(float() | nil, float()) :: float()
  defp maximum_similarity(nil, similarity), do: similarity
  defp maximum_similarity(maximum, similarity), do: max(maximum, similarity)

  @spec pair_similarity(metric(), vector(), vector()) :: {:ok, float()} | {:error, term()}
  defp pair_similarity(:cosine, left, right), do: cosine(left, right)

  defp pair_similarity(:inner_product, left, right),
    do: inner_product(left, right)

  defp pair_similarity(:l2, left, right), do: distance_similarity(l2(left, right))

  defp pair_similarity(:l2_squared, left, right),
    do: distance_similarity(l2_squared(left, right))

  defp pair_similarity(:negative_inner_product, left, right) do
    with {:ok, raw} <- negative_inner_product(left, right), do: {:ok, -raw / 1}
  end

  defp pair_similarity(:manhattan, left, right),
    do: distance_similarity(manhattan(left, right))

  defp pair_similarity(:chebyshev, left, right),
    do: distance_similarity(chebyshev(left, right))

  defp pair_similarity(:hamming, left, right),
    do: distance_similarity(hamming(left, right))

  defp pair_similarity(:jaccard, left, right),
    do: distance_similarity(jaccard(left, right))

  @spec distance_similarity({:ok, number()} | {:error, term()}) ::
          {:ok, float()} | {:error, term()}
  defp distance_similarity({:ok, raw}), do: {:ok, 1.0 / (1.0 + raw)}
  defp distance_similarity({:error, reason}), do: {:error, reason}

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
    if Keyword.keyword?(opts) and Keyword.keys(opts) in [[], [:normalize]],
      do: :ok,
      else: {:error, :invalid_options}
  end

  @spec validate_packed_vectors(term(), term(), term()) :: :ok | {:error, :invalid_vector}
  defp validate_packed_vectors(left, right, dimensions) do
    words = if dimensions > 0, do: div(dimensions + 63, 64), else: 0

    if dimensions > 0 and length(left) == words and length(right) == words and
         Enum.all?(left ++ right, &valid_u64?/1),
       do: :ok,
       else: {:error, :invalid_vector}
  end

  @spec valid_u64?(term()) :: boolean()
  defp valid_u64?(value),
    do: is_integer(value) and value >= 0 and value <= 18_446_744_073_709_551_615

  @spec validate_metric(term()) :: :ok | {:error, {:unknown_metric, term()}}
  defp validate_metric(metric) when metric in @similarity_metrics or metric in @distance_metrics,
    do: :ok

  defp validate_metric(metric), do: {:error, {:unknown_metric, metric}}

  @spec native_metric(metric(), vector(), vector()) :: {:ok, float()} | {:error, term()}
  defp native_metric(metric, left, right) do
    with :ok <- validate_metric(metric),
         :ok <- validate_pair(left, right) do
      native_call(metric, left, right)
    end
  end

  @spec native_call(metric(), vector(), vector()) :: {:ok, float()} | {:error, term()}
  defp native_call(:l2, left, right), do: native_pair(left, right, &Nifs.l2_distance/2)

  defp native_call(:l2_squared, left, right),
    do: native_pair(left, right, &Nifs.l2_squared_distance/2)

  defp native_call(:cosine, left, right),
    do: native_pair(left, right, &Nifs.cosine_similarity/2)

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

  @spec normalized_cosine(vector(), vector(), term()) :: {:ok, float()} | {:error, term()}
  defp normalized_cosine(left, right, :l2) do
    native_pair(left, right, &Nifs.normalized_cosine_similarity/2)
  end

  defp normalized_cosine(left, right, normalize_method) do
    with {:ok, left} <- normalize(left, normalize_method),
         {:ok, right} <- normalize(right, normalize_method) do
      native_metric(:cosine, left, right)
    end
  end

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
  defp normalize_native_error({:error, "metric overflow"}), do: {:error, :metric_overflow}

  defp normalize_native_error({:error, "vector contains a non-finite value"}),
    do: {:error, :invalid_vector}

  defp normalize_native_error({:error, reason}) when is_binary(reason), do: {:error, reason}
  defp normalize_native_error(other), do: other

  @spec float_vector(vector()) :: normalized_vector()
  defp float_vector(vector), do: Enum.map(vector, &(&1 / 1))
end
