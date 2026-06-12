defmodule Vettore.Index.HNSW do
  @moduledoc """
  Native HNSW index boundary.

  ETS remains the canonical store. This resource stores ids and normalized
  vectors only for ANN search.
  """

  @behaviour Vettore.Index

  alias Vettore.{Collection, Distance, Nifs, Result}

  @spec new(:l2 | :cosine | :inner_product | atom()) ::
          {:ok, reference()} | {:error, {:unsupported_hnsw_metric, atom()}}
  @impl true
  def new(:l2), do: {:ok, Nifs.hnsw_new_l2()}
  def new(:cosine), do: {:ok, Nifs.hnsw_new_cosine()}
  def new(:inner_product), do: {:ok, Nifs.hnsw_new_inner_product()}
  def new(metric), do: {:error, {:unsupported_hnsw_metric, metric}}

  @spec put(Collection.t(), Vettore.Embedding.t()) :: :ok | {:error, String.t()}
  @impl true
  def put(%Collection{} = collection, embedding) do
    normalize_ok(Nifs.hnsw_insert(collection.index_state, embedding.id, embedding.vector))
  end

  @spec put_many(Collection.t(), [Vettore.Embedding.t()]) :: :ok | {:error, String.t()}
  @impl true
  def put_many(%Collection{} = collection, embeddings) do
    Enum.reduce_while(embeddings, :ok, fn embedding, :ok ->
      case put(collection, embedding) do
        :ok -> {:cont, :ok}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
  end

  @spec delete(Collection.t(), String.t()) :: :ok | {:error, String.t()}
  @impl true
  def delete(%Collection{} = collection, id),
    do: normalize_ok(Nifs.hnsw_delete(collection.index_state, id))

  @spec search(Collection.t(), [number()], keyword()) :: {:ok, [Result.t()]} | {:error, term()}
  @impl true
  def search(%Collection{} = collection, query, opts) do
    limit = Keyword.get(opts, :limit, 10)

    with :ok <- validate_limit(limit),
         {:ok, query} <- Collection.prepare_query(collection, query),
         {:ok, hits} <- Nifs.hnsw_search(collection.index_state, query, limit) do
      {:ok, Enum.map(hits, &to_result(collection, &1))}
    end
  end

  @spec to_result(Collection.t(), {String.t(), float()}) :: Result.t()
  defp to_result(collection, {id, raw}) do
    {score, distance} = Distance.result_values(collection.metric, raw, collection.score)

    %Result{
      id: id,
      value: id,
      score: score,
      distance: distance,
      metric: collection.metric,
      metadata: nil
    }
  end

  @spec validate_limit(term()) :: :ok | {:error, :invalid_limit}
  defp validate_limit(limit) when is_integer(limit) and limit > 0, do: :ok
  defp validate_limit(_limit), do: {:error, :invalid_limit}

  @spec normalize_ok(:ok | {:ok, {}} | {:error, String.t()}) :: :ok | {:error, String.t()}
  defp normalize_ok({:ok, {}}), do: :ok
  defp normalize_ok(:ok), do: :ok
  defp normalize_ok(other), do: other
end
