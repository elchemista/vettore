defmodule Vettore.MultiVector do
  @moduledoc """
  Multi-vector scoring helpers.
  """

  alias Vettore.Distance

  @type vector :: [number()]
  @type metric :: Distance.metric() | atom()

  @doc """
  Computes Chamfer/MaxSim similarity.

  For each query vector, this finds the best matching document vector and sums
  those best scores.

  ## Examples

      iex> Vettore.MultiVector.chamfer(
      ...>   [[1.0, 0.0], [0.0, 1.0]],
      ...>   [[1.0, 0.0], [1.0, 1.0]],
      ...>   metric: :inner_product
      ...> )
      {:ok, 2.0}
  """
  @spec chamfer([vector()], [vector()], keyword()) ::
          {:ok, float()} | {:error, term()}
  def chamfer(query_vectors, document_vectors, opts \\ [])

  def chamfer([], document_vectors, _opts) when is_list(document_vectors), do: {:ok, 0.0}
  def chamfer(query_vectors, [], _opts) when is_list(query_vectors), do: {:ok, 0.0}

  def chamfer(query_vectors, document_vectors, opts)
      when is_list(query_vectors) and is_list(document_vectors) do
    metric = Keyword.get(opts, :metric, :inner_product)
    sum_best_similarities(query_vectors, document_vectors, metric)
  end

  def chamfer(_query_vectors, _document_vectors, _opts), do: {:error, :invalid_multi_vector}

  @doc """
  Computes a ColBERT-style late interaction score.

  This is an explicit alias for Chamfer/MaxSim scoring: each query vector takes
  its best match from the document vectors, then the best-match scores are
  summed.

  ## Examples

      iex> Vettore.MultiVector.colbert_score(
      ...>   [[1.0, 0.0], [0.0, 1.0]],
      ...>   [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
      ...>   metric: :inner_product
      ...> )
      {:ok, 2.0}
  """
  @spec colbert_score([vector()], [vector()], keyword()) ::
          {:ok, float()} | {:error, term()}
  def colbert_score(query_vectors, document_vectors, opts \\ []) do
    chamfer(query_vectors, document_vectors, opts)
  end

  @spec sum_best_similarities([vector()], [vector()], metric()) ::
          {:ok, float()} | {:error, term()}
  defp sum_best_similarities(query_vectors, document_vectors, metric) do
    query_vectors
    |> Enum.reduce_while(0.0, fn query_vector, acc ->
      continue_with_best_similarity(query_vector, document_vectors, metric, acc)
    end)
    |> wrap_score()
  end

  @spec continue_with_best_similarity(vector(), [vector()], metric(), float()) ::
          {:cont, float()} | {:halt, {:error, term()}}
  defp continue_with_best_similarity(query_vector, document_vectors, metric, acc) do
    case best_similarity(query_vector, document_vectors, metric) do
      {:ok, score} -> {:cont, acc + score}
      {:error, reason} -> {:halt, {:error, reason}}
    end
  end

  @spec best_similarity(vector(), [vector()], metric()) :: {:ok, float()} | {:error, term()}
  defp best_similarity(query_vector, document_vectors, metric) do
    document_vectors
    |> Enum.reduce_while(nil, fn document_vector, best ->
      case metric_value(metric, query_vector, document_vector) do
        {:ok, value} ->
          similarity = as_similarity(metric, value)
          {:cont, if(best == nil, do: similarity, else: max(best, similarity))}

        {:error, reason} ->
          {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:error, reason} -> {:error, reason}
      nil -> {:ok, 0.0}
      best -> {:ok, best}
    end
  end

  @spec metric_value(metric(), vector(), vector()) :: {:ok, float()} | {:error, term()}
  defp metric_value(:l2, left, right), do: Distance.l2(left, right)
  defp metric_value(:l2_squared, left, right), do: Distance.l2_squared(left, right)
  defp metric_value(:cosine, left, right), do: Distance.cosine(left, right, normalize: :l2)
  defp metric_value(:inner_product, left, right), do: Distance.inner_product(left, right)
  defp metric_value(:dot, left, right), do: Distance.inner_product(left, right)

  defp metric_value(:negative_inner_product, left, right),
    do: Distance.negative_inner_product(left, right)

  defp metric_value(:manhattan, left, right), do: Distance.manhattan(left, right)
  defp metric_value(:chebyshev, left, right), do: Distance.chebyshev(left, right)
  defp metric_value(:hamming, left, right), do: Distance.hamming(left, right)
  defp metric_value(:jaccard, left, right), do: Distance.jaccard(left, right)
  defp metric_value(metric, _left, _right), do: {:error, {:unknown_metric, metric}}

  @spec wrap_score(float() | {:error, term()}) :: {:ok, float()} | {:error, term()}
  defp wrap_score({:error, reason}), do: {:error, reason}
  defp wrap_score(score), do: {:ok, score}

  @spec as_similarity(metric(), number()) :: float()
  defp as_similarity(metric, value) when metric in [:cosine, :inner_product, :dot], do: value
  defp as_similarity(_metric, distance), do: 1.0 / (1.0 + distance)
end
