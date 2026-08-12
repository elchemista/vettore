defmodule Vettore.Index.Flat do
  @moduledoc """
  Native exact flat-scan index over mirrored ids and vectors.

  ETS remains the canonical record store for values and metadata. The native
  resource keeps only ids and vectors so an exact scan is one native call.
  """

  @behaviour Vettore.Index

  alias Vettore.{Distance, Embedding, Identifier, Index, Nifs, Result}

  @spec new(Distance.metric(), keyword()) :: {:ok, reference()} | {:error, term()}
  @impl Vettore.Index
  def new(metric, opts \\ [])

  def new(metric, opts) when is_list(opts) do
    if Keyword.keyword?(opts) and opts == [],
      do: new_metric(metric),
      else: {:error, :invalid_flat_options}
  end

  def new(_metric, _opts), do: {:error, :invalid_flat_options}

  @spec put(Index.context(), Embedding.t()) :: :ok | {:error, term()}
  @impl Vettore.Index
  def put(%{index_state: index_state}, %Embedding{} = embedding) do
    with :ok <- Identifier.validate(embedding.id) do
      Index.normalize_write_result(Nifs.flat_insert(index_state, embedding.id, embedding.vector))
    end
  end

  @spec put_many(Index.context(), [Embedding.t()]) :: :ok | {:error, term()}
  @impl Vettore.Index
  def put_many(%{index_state: index_state}, embeddings) do
    with :ok <- Identifier.validate_embeddings(embeddings) do
      vectors = Enum.map(embeddings, &{&1.id, &1.vector})
      Index.normalize_write_result(Nifs.flat_insert_many(index_state, vectors))
    end
  end

  @spec delete(Index.context(), String.t()) :: :ok | {:error, term()}
  @impl Vettore.Index
  def delete(%{index_state: index_state}, id) do
    with :ok <- Identifier.validate_utf8(id) do
      Index.normalize_write_result(Nifs.flat_delete(index_state, id))
    end
  end

  @spec close(Index.context()) :: :ok | {:error, term()}
  @impl Vettore.Index
  def close(%{index_state: index_state}),
    do: Index.normalize_write_result(Nifs.flat_clear(index_state))

  @spec search(Index.context(), [number()], keyword()) ::
          {:ok, [Result.t()]} | {:error, term()}
  @impl Vettore.Index
  def search(%{index_state: index_state} = context, query, opts) do
    with :ok <- Index.validate_search_options(opts),
         limit = Keyword.get(opts, :limit, 10),
         :ok <- Index.validate_limit(limit),
         {:ok, query} <- Index.prepare_query(context, query),
         {:ok, hits} <- Nifs.flat_search(index_state, query, limit) do
      {:ok, Index.hydrate_results(context, hits)}
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
end
