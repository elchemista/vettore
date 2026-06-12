defmodule Vettore.Index.Flat do
  @moduledoc """
  Exact flat-scan index over the canonical store.
  """

  @behaviour Vettore.Index

  alias Vettore.{Collection, Distance, Result}

  @spec new(atom()) :: {:ok, nil}
  @impl true
  def new(_metric), do: {:ok, nil}

  @spec put(Collection.t(), Vettore.Embedding.t()) :: :ok
  @impl true
  def put(_collection, _embedding), do: :ok

  @spec put_many(Collection.t(), [Vettore.Embedding.t()]) :: :ok
  @impl true
  def put_many(_collection, _embeddings), do: :ok

  @spec delete(Collection.t(), String.t()) :: :ok
  @impl true
  def delete(_collection, _id), do: :ok

  @spec search(Collection.t(), [number()], keyword()) :: {:ok, [Result.t()]} | {:error, term()}
  @impl true
  def search(%Collection{} = collection, query, opts) do
    limit = Keyword.get(opts, :limit, 10)

    with :ok <- validate_limit(limit),
         {:ok, query} <- Collection.prepare_query(collection, query) do
      fold_results(collection, query, limit)
    end
  end

  @spec fold_results(Collection.t(), [float()], pos_integer()) ::
          {:ok, [Result.t()]} | {:error, term()}
  defp fold_results(collection, query, limit) do
    collection.store_mod.fold(collection.store_state, [], fn embedding, acc ->
      score_and_insert(collection, query, embedding, acc, limit)
    end)
  end

  @spec score_and_insert(
          Collection.t(),
          [float()],
          Vettore.Embedding.t(),
          [Result.t()],
          pos_integer()
        ) ::
          [Result.t()]
  defp score_and_insert(collection, query, embedding, acc, limit) do
    case score_embedding(collection, query, embedding) do
      {:ok, result} -> insert_top(acc, result, limit)
      {:error, _reason} -> acc
    end
  end

  @spec score_embedding(Collection.t(), [float()], Vettore.Embedding.t()) ::
          {:ok, Result.t()} | {:error, term()}
  defp score_embedding(collection, query, embedding) do
    with {:ok, distance_or_similarity} <- metric_value(collection.metric, query, embedding.vector) do
      {score, distance} =
        Distance.result_values(collection.metric, distance_or_similarity, collection.score)

      {:ok,
       %Result{
         id: embedding.id,
         value: embedding.value,
         score: score,
         distance: distance,
         metric: collection.metric,
         metadata: embedding.metadata
       }}
    end
  end

  @spec metric_value(Distance.metric(), [float()], [float()]) :: {:ok, float()} | {:error, term()}
  defp metric_value(:l2, left, right), do: Distance.l2(left, right)
  defp metric_value(:l2_squared, left, right), do: Distance.l2_squared(left, right)
  defp metric_value(:cosine, left, right), do: Distance.cosine(left, right, normalize: :none)
  defp metric_value(:inner_product, left, right), do: Distance.inner_product(left, right)

  defp metric_value(:negative_inner_product, left, right),
    do: Distance.negative_inner_product(left, right)

  defp metric_value(:manhattan, left, right), do: Distance.manhattan(left, right)
  defp metric_value(:chebyshev, left, right), do: Distance.chebyshev(left, right)
  defp metric_value(:hamming, left, right), do: Distance.hamming(left, right)
  defp metric_value(:jaccard, left, right), do: Distance.jaccard(left, right)

  @spec insert_top([Result.t()], Result.t(), pos_integer()) :: [Result.t()]
  defp insert_top(results, result, limit) do
    {higher_or_equal, lower} = Enum.split_while(results, &(&1.score >= result.score))

    (higher_or_equal ++ [result | lower])
    |> Enum.take(limit)
  end

  @spec validate_limit(term()) :: :ok | {:error, :invalid_limit}
  defp validate_limit(limit) when is_integer(limit) and limit > 0, do: :ok
  defp validate_limit(_limit), do: {:error, :invalid_limit}
end
