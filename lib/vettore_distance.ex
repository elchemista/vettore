defmodule Vettore.Distance do
  @moduledoc """
  Independent distance, similarity, normalization, and reranking helpers.

  `compute/4` returns raw metric values:

    * distance metrics return a distance where lower is better
    * similarity metrics return a similarity where higher is better
  """

  @type vector :: [number()]
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
  Computes a raw metric value.
  """
  @spec compute(metric() | atom(), vector(), vector(), keyword()) ::
          {:ok, float()} | {:error, term()}
  def compute(metric, left, right, opts \\ []) do
    metric = normalize_metric(metric)

    with :ok <- validate_metric(metric),
         :ok <- validate_pair(left, right) do
      do_compute(metric, left, right, opts)
    end
  end

  @doc """
  Normalizes a vector.
  """
  @spec normalize(vector(), :none | :l2 | :zscore | :minmax) ::
          {:ok, [float()]} | {:error, term()}
  def normalize(vector, :none) when is_list(vector), do: {:ok, Enum.map(vector, &(&1 / 1))}

  def normalize(vector, :l2) when is_list(vector) do
    norm =
      vector
      |> Enum.reduce(0.0, fn value, acc -> acc + value * value end)
      |> :math.sqrt()

    if norm == 0.0 do
      {:ok, Enum.map(vector, fn _ -> 0.0 end)}
    else
      {:ok, Enum.map(vector, &(&1 / norm))}
    end
  end

  def normalize(vector, :zscore) when is_list(vector) do
    count = length(vector)

    if count == 0 do
      {:ok, []}
    else
      mean = Enum.sum(vector) / count

      variance =
        Enum.reduce(vector, 0.0, fn value, acc ->
          diff = value - mean
          acc + diff * diff
        end) / count

      stddev = :math.sqrt(variance)

      if stddev == 0.0 do
        {:ok, Enum.map(vector, fn _ -> 0.0 end)}
      else
        {:ok, Enum.map(vector, &((&1 - mean) / stddev))}
      end
    end
  end

  def normalize(vector, :minmax) when is_list(vector) do
    case Enum.min_max(vector, fn -> nil end) do
      nil ->
        {:ok, []}

      {min, max} when min == max ->
        {:ok, Enum.map(vector, fn _ -> 0.0 end)}

      {min, max} ->
        range = max - min
        {:ok, Enum.map(vector, &((&1 - min) / range))}
    end
  end

  def normalize(_vector, method), do: {:error, {:unknown_normalization, method}}

  @doc """
  Returns whether a metric is ordered by descending similarity or ascending distance.
  """
  def order(metric) do
    metric = normalize_metric(metric)

    cond do
      metric in @similarity_metrics -> :desc
      metric in @distance_metrics -> :asc
      true -> :unknown
    end
  end

  @doc """
  Converts a raw metric value into the explicit result score and distance fields.
  """
  def result_values(metric, raw, score_mode \\ :raw) do
    metric = normalize_metric(metric)

    case {order(metric), score_mode} do
      {:desc, :raw} -> {raw / 1, similarity_distance(metric, raw)}
      {:asc, :raw} -> {-raw / 1, raw / 1}
      {:desc, :similarity} -> {similarity_score(metric, raw), similarity_distance(metric, raw)}
      {:asc, :similarity} -> {1.0 / (1.0 + raw), raw / 1}
      _other -> {raw / 1, nil}
    end
  end

  @doc """
  L2 distance.
  """
  def euclidean(left, right), do: compute(:l2, left, right)

  @doc """
  Cosine similarity. Defaults to L2-normalizing inputs and returns `[-1.0, 1.0]`.
  """
  def cosine(left, right), do: compute(:cosine, left, right, normalize: :l2)

  @doc """
  Inner product.
  """
  def dot_product(left, right), do: compute(:inner_product, left, right)

  @doc """
  Hamming distance for equal-length bit/integer vectors.
  """
  def hamming(left, right), do: compute(:hamming, left, right)

  @doc """
  Compress a float vector into sign bits represented as integers.
  """
  def compress_f32_vector(vector) when is_list(vector) do
    Enum.map(vector, fn value -> if value >= 0, do: 1, else: 0 end)
  end

  @doc """
  Collection-agnostic MMR reranker.
  """
  def mmr_rerank(initial, embeddings, metric, alpha, final_k)
      when is_list(initial) and is_list(embeddings) and is_number(alpha) and alpha >= 0 and
             alpha <= 1 and is_integer(final_k) and final_k > 0 do
    metric = normalize_metric(metric)

    with :ok <- validate_metric(metric) do
      vectors = Map.new(embeddings)

      initial
      |> do_mmr(vectors, metric, alpha, final_k, [])
      |> then(&{:ok, &1})
    end
  end

  def mmr_rerank(_initial, _embeddings, _metric, _alpha, _final_k),
    do: {:error, :invalid_mmr_args}

  defp do_compute(:l2, left, right, _opts) do
    {:ok,
     left
     |> Enum.zip(right)
     |> Enum.reduce(0.0, fn {a, b}, acc ->
       diff = a - b
       acc + diff * diff
     end)
     |> :math.sqrt()}
  end

  defp do_compute(:l2_squared, left, right, _opts) do
    {:ok,
     left
     |> Enum.zip(right)
     |> Enum.reduce(0.0, fn {a, b}, acc ->
       diff = a - b
       acc + diff * diff
     end)}
  end

  defp do_compute(:cosine, left, right, opts) do
    normalize_method = Keyword.get(opts, :normalize, :none)

    with {:ok, left} <- normalize(left, normalize_method),
         {:ok, right} <- normalize(right, normalize_method) do
      do_compute(:inner_product, left, right, opts)
    end
  end

  defp do_compute(:inner_product, left, right, _opts) do
    {:ok, Enum.zip_reduce(left, right, 0.0, fn a, b, acc -> acc + a * b end)}
  end

  defp do_compute(:negative_inner_product, left, right, opts) do
    with {:ok, dot} <- do_compute(:inner_product, left, right, opts), do: {:ok, -dot}
  end

  defp do_compute(:manhattan, left, right, _opts) do
    {:ok, Enum.zip_reduce(left, right, 0.0, fn a, b, acc -> acc + abs(a - b) end)}
  end

  defp do_compute(:chebyshev, left, right, _opts) do
    {:ok, Enum.zip_reduce(left, right, 0.0, fn a, b, acc -> max(acc, abs(a - b)) end)}
  end

  defp do_compute(:hamming, left, right, _opts) do
    {:ok, Enum.zip_reduce(left, right, 0, fn a, b, acc -> if a == b, do: acc, else: acc + 1 end)}
  end

  defp do_compute(:jaccard, left, right, _opts) do
    left_set = truthy_index_set(left)
    right_set = truthy_index_set(right)
    union = MapSet.union(left_set, right_set) |> MapSet.size()

    if union == 0 do
      {:ok, 0.0}
    else
      intersection = MapSet.intersection(left_set, right_set) |> MapSet.size()
      {:ok, 1.0 - intersection / union}
    end
  end

  defp do_mmr(_remaining, _vectors, _metric, _alpha, 0, selected), do: Enum.reverse(selected)
  defp do_mmr([], _vectors, _metric, _alpha, _left, selected), do: Enum.reverse(selected)

  defp do_mmr(remaining, vectors, metric, alpha, left, selected) do
    {chosen, rest} =
      remaining
      |> Enum.with_index()
      |> Enum.max_by(fn {{id, query_score}, _index} ->
        redundancy =
          selected
          |> Enum.map(fn {selected_id, _score} ->
            pair_similarity(metric, Map.fetch!(vectors, id), Map.fetch!(vectors, selected_id))
          end)
          |> Enum.max(fn -> 0.0 end)

        alpha * query_score - (1.0 - alpha) * redundancy
      end)
      |> then(fn {chosen, index} -> {chosen, List.delete_at(remaining, index)} end)

    do_mmr(rest, vectors, metric, alpha, left - 1, [chosen | selected])
  end

  defp pair_similarity(metric, left, right) do
    case compute(metric, left, right, normalize: if(metric == :cosine, do: :l2, else: :none)) do
      {:ok, raw} when metric in @similarity_metrics -> raw
      {:ok, raw} -> 1.0 / (1.0 + raw)
      {:error, _reason} -> 0.0
    end
  end

  defp similarity_distance(:cosine, raw), do: 1.0 - raw
  defp similarity_distance(:inner_product, raw), do: -raw
  defp similarity_distance(_metric, _raw), do: nil

  defp similarity_score(:cosine, raw), do: (raw + 1.0) / 2.0
  defp similarity_score(:inner_product, raw), do: raw
  defp similarity_score(_metric, raw), do: raw

  defp truthy_index_set(vector) do
    vector
    |> Enum.with_index()
    |> Enum.reduce(MapSet.new(), fn {value, index}, acc ->
      if value in [0, 0.0, false, nil], do: acc, else: MapSet.put(acc, index)
    end)
  end

  defp validate_pair(left, right) when is_list(left) and is_list(right) do
    cond do
      length(left) != length(right) -> {:error, :dimension_mismatch}
      Enum.all?(left ++ right, &is_number/1) -> :ok
      true -> {:error, :invalid_vector}
    end
  end

  defp validate_pair(_left, _right), do: {:error, :invalid_vector}

  defp validate_metric(metric) when metric in @similarity_metrics or metric in @distance_metrics,
    do: :ok

  defp validate_metric(metric), do: {:error, {:unknown_metric, metric}}

  defp normalize_metric(metric) when is_binary(metric),
    do: metric |> String.to_atom() |> normalize_metric()

  defp normalize_metric(:euclidean), do: :l2
  defp normalize_metric(:dot), do: :inner_product
  defp normalize_metric(:dot_product), do: :inner_product
  defp normalize_metric(metric), do: metric
end
