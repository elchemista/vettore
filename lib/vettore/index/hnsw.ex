defmodule Vettore.Index.HNSW do
  @moduledoc """
  Native HNSW index boundary.

  ETS remains the canonical store. This resource stores ids and normalized
  vectors only for ANN search.
  """

  @behaviour Vettore.Index

  alias Vettore.{Embedding, Identifier, Index, Nifs, Result}

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
  @type new_error ::
          :invalid_hnsw_options | {:unsupported_hnsw_metric, atom()} | String.t()

  @spec new(atom(), term()) :: {:ok, reference()} | {:error, new_error()}
  @impl Vettore.Index
  def new(metric, opts \\ []) do
    with {:ok, options} <- normalize_options(opts) do
      new_metric(metric, options)
    end
  end

  @spec defaults() :: keyword(pos_integer())
  def defaults, do: @default_options

  @spec put(Index.context(), Embedding.t()) :: :ok | {:error, term()}
  @impl Vettore.Index
  def put(%{index_state: index_state}, %Embedding{} = embedding) do
    with :ok <- Identifier.validate(embedding.id) do
      Index.normalize_write_result(Nifs.hnsw_insert(index_state, embedding.id, embedding.vector))
    end
  end

  @spec put_many(Index.context(), [Embedding.t()]) :: :ok | {:error, term()}
  @impl Vettore.Index
  def put_many(%{index_state: index_state}, embeddings) do
    with :ok <- Identifier.validate_embeddings(embeddings) do
      vectors = Enum.map(embeddings, &{&1.id, &1.vector})
      Index.normalize_write_result(Nifs.hnsw_insert_many(index_state, vectors))
    end
  end

  @spec delete(Index.context(), String.t()) :: :ok | {:error, term()}
  @impl Vettore.Index
  def delete(%{index_state: index_state}, id) do
    with :ok <- Identifier.validate_utf8(id) do
      Index.normalize_write_result(Nifs.hnsw_delete(index_state, id))
    end
  end

  @spec close(Index.context()) :: :ok | {:error, term()}
  @impl Vettore.Index
  def close(%{index_state: index_state}),
    do: Index.normalize_write_result(Nifs.hnsw_clear(index_state))

  @spec search(Index.context(), [number()], keyword()) ::
          {:ok, [Result.t()]} | {:error, term()}
  @impl Vettore.Index
  def search(%{index_state: index_state} = context, query, opts) do
    with :ok <- Index.validate_search_options(opts),
         limit = Keyword.get(opts, :limit, 10),
         :ok <- Index.validate_limit(limit),
         {:ok, query} <- Index.prepare_query(context, query),
         {:ok, hits} <- Nifs.hnsw_search(index_state, query, limit) do
      {:ok, Index.hydrate_results(context, hits)}
    end
  end

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

  @spec normalize_options(term()) :: {:ok, keyword()} | {:error, :invalid_hnsw_options}
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
end
