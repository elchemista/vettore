defmodule Vettore.Index.Flat do
  @moduledoc """
  Exact flat-scan index over the canonical store.
  """

  @behaviour Vettore.Index

  alias Vettore.{Collection, Distance, Result}

  @impl true
  def search(%Collection{} = collection, query, opts) do
    limit = Keyword.get(opts, :limit, 10)

    with :ok <- validate_limit(limit),
         {:ok, query} <- Collection.prepare_query(collection, query),
         {:ok, embeddings} <- collection.store_mod.all(collection.store_state) do
      results =
        Enum.reduce(embeddings, [], fn embedding, acc ->
          case score_embedding(collection, query, embedding) do
            {:ok, result} -> insert_top(acc, result, limit)
            {:error, _reason} -> acc
          end
        end)

      {:ok, results}
    end
  end

  defp score_embedding(collection, query, embedding) do
    with {:ok, distance_or_similarity} <-
           Distance.compute(collection.metric, query, embedding.vector, score: :raw) do
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

  defp insert_top(results, result, limit) do
    [result | results]
    |> Enum.sort_by(& &1.score, :desc)
    |> Enum.take(limit)
  end

  defp validate_limit(limit) when is_integer(limit) and limit > 0, do: :ok
  defp validate_limit(_limit), do: {:error, :invalid_limit}
end
