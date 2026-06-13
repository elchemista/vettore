defmodule Vettore.Index.Flat do
  @moduledoc """
  Native exact flat-scan index over mirrored ids and vectors.

  ETS remains the canonical record store for values and metadata. The native
  resource keeps only ids and vectors so an exact scan is one native call.
  """

  @behaviour Vettore.Index

  alias Vettore.{Collection, Distance, Embedding, Nifs, Result}

  @spec new(Distance.metric(), keyword()) :: {:ok, reference()} | {:error, term()}
  @impl true
  def new(metric, _opts \\ []), do: new_metric(metric)

  @spec put(Collection.t(), Embedding.t()) :: :ok | {:error, term()}
  @impl true
  def put(%Collection{} = collection, %Embedding{} = embedding) do
    normalize_ok(Nifs.flat_insert(collection.index_state, embedding.id, embedding.vector))
  end

  @spec put_many(Collection.t(), [Embedding.t()]) :: :ok | {:error, term()}
  @impl true
  def put_many(%Collection{} = collection, embeddings) do
    vectors = Enum.map(embeddings, &{&1.id, &1.vector})

    normalize_ok(Nifs.flat_insert_many(collection.index_state, vectors))
  end

  @spec delete(Collection.t(), String.t()) :: :ok | {:error, term()}
  @impl true
  def delete(%Collection{} = collection, id) do
    normalize_ok(Nifs.flat_delete(collection.index_state, id))
  end

  @spec search(Collection.t(), [number()], keyword()) :: {:ok, [Result.t()]} | {:error, term()}
  @impl true
  def search(%Collection{} = collection, query, opts) do
    limit = Keyword.get(opts, :limit, 10)

    with :ok <- validate_limit(limit),
         {:ok, query} <- Collection.prepare_query(collection, query),
         {:ok, hits} <- Nifs.flat_search(collection.index_state, query, limit) do
      {:ok, Enum.map(hits, &to_result(collection, &1))}
    end
  end

  @spec new_metric(Distance.metric() | atom()) :: {:ok, reference()} | {:error, term()}
  defp new_metric(:l2), do: {:ok, Nifs.flat_new_l2()}
  defp new_metric(:l2_squared), do: {:ok, Nifs.flat_new_l2_squared()}
  defp new_metric(:cosine), do: {:ok, Nifs.flat_new_cosine()}
  defp new_metric(:inner_product), do: {:ok, Nifs.flat_new_inner_product()}
  defp new_metric(:negative_inner_product), do: {:ok, Nifs.flat_new_negative_inner_product()}
  defp new_metric(:manhattan), do: {:ok, Nifs.flat_new_manhattan()}
  defp new_metric(:chebyshev), do: {:ok, Nifs.flat_new_chebyshev()}
  defp new_metric(:hamming), do: {:ok, Nifs.flat_new_hamming()}
  defp new_metric(:jaccard), do: {:ok, Nifs.flat_new_jaccard()}
  defp new_metric(metric), do: {:error, {:unsupported_flat_metric, metric}}

  @spec to_result(Collection.t(), {String.t(), float()}) :: Result.t()
  defp to_result(collection, {id, raw}) do
    embedding =
      case Collection.get(collection, id) do
        {:ok, embedding} -> embedding
        {:error, _reason} -> %Embedding{id: id, value: id}
      end

    {score, distance} = Distance.result_values(collection.metric, raw, collection.score)

    %Result{
      id: id,
      value: embedding.value,
      score: score,
      distance: distance,
      metric: collection.metric,
      metadata: embedding.metadata
    }
  end

  @spec normalize_ok(:ok | {:ok, {}} | {:error, term()}) :: :ok | {:error, term()}
  defp normalize_ok({:ok, {}}), do: :ok
  defp normalize_ok(:ok), do: :ok
  defp normalize_ok(other), do: other

  @spec validate_limit(term()) :: :ok | {:error, :invalid_limit}
  defp validate_limit(limit) when is_integer(limit) and limit > 0, do: :ok
  defp validate_limit(_limit), do: {:error, :invalid_limit}
end
