defmodule Vettore.Index.HNSW do
  @moduledoc """
  Native HNSW index boundary.

  ETS remains the canonical store. This resource stores ids and normalized
  vectors only for ANN search.
  """

  @behaviour Vettore.Index

  alias Vettore.{Collection, Distance, Nifs, Result}

  @default_options [
    m: 16,
    m0: 32,
    ef_construction: 100,
    ef_search: 64,
    max_level: 12
  ]

  @spec new(:l2 | :cosine | :inner_product | atom(), keyword()) ::
          {:ok, reference()} | {:error, {:unsupported_hnsw_metric, atom()}}
  @impl true
  def new(metric, opts \\ []) do
    with {:ok, options} <- normalize_options(opts) do
      new_metric(metric, options)
    end
  end

  @spec defaults() :: keyword(pos_integer())
  def defaults, do: @default_options

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

  @spec new_metric(atom(), keyword()) ::
          {:ok, reference()} | {:error, {:unsupported_hnsw_metric, atom()} | String.t()}
  defp new_metric(:l2, opts), do: apply_new(&Nifs.hnsw_new_l2/5, opts)
  defp new_metric(:cosine, opts), do: apply_new(&Nifs.hnsw_new_cosine/5, opts)
  defp new_metric(:inner_product, opts), do: apply_new(&Nifs.hnsw_new_inner_product/5, opts)
  defp new_metric(metric, _opts), do: {:error, {:unsupported_hnsw_metric, metric}}

  @spec apply_new(function(), keyword()) :: {:ok, reference()} | {:error, String.t()}
  defp apply_new(fun, opts) do
    fun.(
      Keyword.fetch!(opts, :m),
      Keyword.fetch!(opts, :m0),
      Keyword.fetch!(opts, :ef_construction),
      Keyword.fetch!(opts, :ef_search),
      Keyword.fetch!(opts, :max_level)
    )
  end

  @spec normalize_options(keyword()) :: {:ok, keyword()} | {:error, :invalid_hnsw_options}
  defp normalize_options(opts) when is_list(opts) do
    options = Keyword.merge(@default_options, opts)

    if Enum.all?(@default_options, fn {key, _default} -> positive_integer?(options[key]) end) do
      {:ok, options}
    else
      {:error, :invalid_hnsw_options}
    end
  end

  defp normalize_options(_opts), do: {:error, :invalid_hnsw_options}

  @spec positive_integer?(term()) :: boolean()
  defp positive_integer?(value), do: is_integer(value) and value > 0
end
