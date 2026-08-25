defmodule Vettore.Index.Flat do
  @moduledoc """
  Native exact flat-scan index over mirrored ids and vectors.

  ETS remains the canonical record store for values and metadata. The native
  resource keeps ids plus a contiguous vector matrix so an exact CPU scan is one
  native SIMD call. `:gpu`, `:gpu_min_size`, and `:gpu_fallback` index options
  enable a generation-aware resident matrix: warm searches upload only the
  query, reduce top-k on the device, and read back compact hits.

  Effective mutations invalidate the device snapshot. The next eligible query
  rebuilds it once; concurrent warm queries share the immutable matrix and use
  independent pooled scratch buffers. GPU top-k supports up to 64 returned
  results and follows the configured fallback policy above that limit.
  """

  @behaviour Vettore.Index

  alias Vettore.{Compute, Distance, Embedding, Identifier, Index, Nifs, Result}

  @spec new(Distance.metric(), keyword()) :: {:ok, reference()} | {:error, term()}
  @impl Vettore.Index
  def new(metric, opts \\ [])

  def new(metric, opts) when is_list(opts) do
    with :ok <- validate_options(opts) do
      new_metric(metric)
    end
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
         {:ok, workload} <- search_workload(context.index_options, index_state),
         {:ok, hits} <- run_search(context.index_options, workload, index_state, query, limit) do
      {:ok, Index.hydrate_results(context, hits)}
    end
  end

  @spec search_workload(keyword(), reference()) :: {:ok, non_neg_integer()} | {:error, term()}
  defp search_workload(compute_options, index_state) do
    with {:ok, selection} <- Compute.selection(compute_options) do
      workload_for_selection(selection, index_state)
    end
  end

  @spec workload_for_selection(boolean() | :auto, reference()) ::
          {:ok, non_neg_integer()} | {:error, term()}
  defp workload_for_selection(:auto, index_state) do
    with {:ok, {rows, dimensions}} <- Nifs.flat_workload(index_state) do
      {:ok, rows * dimensions}
    end
  end

  defp workload_for_selection(_selection, _index_state), do: {:ok, 0}

  @spec run_search(keyword(), non_neg_integer(), reference(), [float()], pos_integer()) ::
          {:ok, [{String.t(), float()}]} | {:error, term()}
  defp run_search(compute_options, workload, index_state, query, limit) do
    Compute.run(
      compute_options,
      workload,
      fn -> Nifs.flat_search(index_state, query, limit) end,
      fn -> Nifs.flat_gpu_search(index_state, query, limit) end
    )
    |> Compute.normalize_search_result()
  end

  @spec validate_options(keyword()) :: :ok | {:error, term()}
  defp validate_options(opts) do
    case Compute.validate_options(opts) do
      {:error, :invalid_options} -> {:error, :invalid_flat_options}
      result -> result
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
