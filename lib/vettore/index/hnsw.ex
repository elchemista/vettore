defmodule Vettore.Index.HNSW do
  @moduledoc """
  Native HNSW index boundary.

  ETS remains the canonical store. This resource stores ids and normalized
  vectors only for ANN search.
  """

  @behaviour Vettore.Index

  alias Vettore.{Collection, Distance, Embedding, Nifs, Result}

  @default_options [
    m: 16,
    m0: 32,
    ef_construction: 100,
    ef_search: 64,
    max_level: 12
  ]

  @option_keys Keyword.keys(@default_options)
  @max_m 1_024
  @max_m0 2_048
  @max_ef 1_000_000
  @max_level 64
  @max_nif_usize 4_294_967_295

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
    vectors = Enum.map(embeddings, &{&1.id, &1.vector})
    normalize_ok(Nifs.hnsw_insert_many(collection.index_state, vectors))
  end

  @spec delete(Collection.t(), String.t()) :: :ok | {:error, String.t()}
  @impl true
  def delete(%Collection{} = collection, id),
    do: normalize_ok(Nifs.hnsw_delete(collection.index_state, id))

  @spec search(Collection.t(), [number()], keyword()) :: {:ok, [Result.t()]} | {:error, term()}
  @impl true
  def search(%Collection{} = collection, query, opts) do
    with :ok <- validate_search_options(opts),
         limit = Keyword.get(opts, :limit, 10),
         :ok <- validate_limit(limit),
         {:ok, query} <- Collection.prepare_query(collection, query),
         {:ok, hits} <- Nifs.hnsw_search(collection.index_state, query, limit) do
      {:ok, Enum.flat_map(hits, &to_result(collection, &1))}
    end
  end

  @spec to_result(Collection.t(), {String.t(), float()}) :: [Result.t()]
  defp to_result(collection, {id, raw}) do
    case Collection.get(collection, id) do
      {:ok, %Embedding{} = embedding} ->
        {score, distance} = Distance.result_values(collection.metric, raw, collection.score)

        [
          %Result{
            id: id,
            value: embedding.value,
            score: score,
            distance: distance,
            metric: collection.metric,
            metadata: embedding.metadata
          }
        ]

      {:error, _reason} ->
        []
    end
  end

  @spec validate_limit(term()) :: :ok | {:error, :invalid_limit}
  defp validate_limit(limit)
       when is_integer(limit) and limit > 0 and limit <= @max_nif_usize,
       do: :ok

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
    with true <- Keyword.keyword?(opts),
         true <- Enum.all?(Keyword.keys(opts), &(&1 in @option_keys)),
         true <- unique_keys?(opts) do
      options = Keyword.merge(@default_options, opts)

      if valid_options?(options),
        do: {:ok, options},
        else: {:error, :invalid_hnsw_options}
    else
      false -> {:error, :invalid_hnsw_options}
    end
  end

  defp normalize_options(_opts), do: {:error, :invalid_hnsw_options}

  @spec positive_integer?(term()) :: boolean()
  defp positive_integer?(value), do: is_integer(value) and value > 0

  @spec valid_options?(keyword()) :: boolean()
  defp valid_options?(options) do
    m = options[:m]
    m0 = options[:m0]
    ef_construction = options[:ef_construction]
    ef_search = options[:ef_search]
    max_level = options[:max_level]

    valid_degrees?(m, m0) and valid_ef?(m, ef_construction, ef_search) and
      valid_level?(max_level)
  end

  @spec valid_degrees?(term(), term()) :: boolean()
  defp valid_degrees?(m, m0) do
    positive_integer?(m) and m <= @max_m and positive_integer?(m0) and m0 >= m and
      m0 <= @max_m0
  end

  @spec valid_ef?(term(), term(), term()) :: boolean()
  defp valid_ef?(m, ef_construction, ef_search) do
    positive_integer?(ef_construction) and ef_construction >= m and
      ef_construction <= @max_ef and positive_integer?(ef_search) and ef_search <= @max_ef
  end

  @spec valid_level?(term()) :: boolean()
  defp valid_level?(max_level), do: positive_integer?(max_level) and max_level <= @max_level

  @spec unique_keys?(keyword()) :: boolean()
  defp unique_keys?(opts) do
    keys = Keyword.keys(opts)
    length(keys) == MapSet.size(MapSet.new(keys))
  end

  @spec validate_search_options(term()) :: :ok | {:error, :invalid_search_options}
  defp validate_search_options(opts) when is_list(opts) do
    if Keyword.keyword?(opts) and Enum.all?(Keyword.keys(opts), &(&1 == :limit)),
      do: :ok,
      else: {:error, :invalid_search_options}
  end

  defp validate_search_options(_opts), do: {:error, :invalid_search_options}
end
