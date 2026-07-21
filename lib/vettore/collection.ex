defmodule Vettore.Collection do
  @moduledoc """
  Collection module for managing vector embeddings.

  This module provides functions for creating, managing, and querying vector collections.
  """

  alias Vettore.{Distance, Embedding, Nifs, Result}

  @type t :: %__MODULE__{
          name: atom() | String.t(),
          dimensions: pos_integer(),
          metric: atom(),
          normalize: atom(),
          score: atom(),
          store_mod: module(),
          store_state: term(),
          index_mod: module(),
          index_state: term(),
          index: atom(),
          index_options: keyword(),
          compressed: boolean()
        }
  @type embedding_input :: Embedding.t() | map()
  @type hybrid_generator ::
          :funnel
          | :quantized
          | :search
          | :hnsw
          | {:funnel | :quantized | :search | :hnsw, keyword()}
  @type hybrid_rerank ::
          :exact | {:multi_vector, [[number()]]} | {:multi_vector, [[number()]], keyword()}

  defstruct [
    :name,
    :dimensions,
    :metric,
    :normalize,
    :score,
    :store_mod,
    :store_state,
    :index_mod,
    :index_state,
    :index,
    :index_options,
    :compressed
  ]

  @metrics ~w(l2 l2_squared cosine inner_product negative_inner_product manhattan chebyshev hamming jaccard)a
  @normalizations ~w(none l2 zscore minmax)a
  @score_modes ~w(raw similarity)a
  @snapshot_version 1
  @new_option_keys ~w(name dimensions metric normalize store index index_options score compressed)a
  @snapshot_override_keys ~w(name index index_options score store)a
  @search_option_keys ~w(limit)a
  @funnel_option_keys ~w(limit candidates stages dimensions)a
  @quantized_option_keys ~w(limit candidates)a
  @multi_vector_option_keys ~w(limit metric)a
  @hybrid_option_keys ~w(limit generators rerank)a
  @max_nif_usize 4_294_967_295
  @f32_max 3.402_823_466_385_288_6e38
  @store_callbacks [
    new: 1,
    put: 2,
    put_many: 2,
    get: 2,
    delete: 2,
    all: 1,
    snapshot: 2,
    load_snapshot: 1
  ]
  @index_callbacks [new: 2, put: 2, put_many: 2, delete: 2, search: 3]

  @doc false
  @spec new(keyword()) :: {:ok, t()} | {:error, term()}
  def new(opts) when is_list(opts) do
    with :ok <- validate_options(opts, @new_option_keys) do
      do_new(opts)
    end
  end

  def new(_opts), do: {:error, :invalid_options}

  @spec do_new(keyword()) :: {:ok, t()} | {:error, term()}
  defp do_new(opts) do
    metric = normalize_metric(Keyword.get(opts, :metric, :cosine))
    dimensions = Keyword.get(opts, :dimensions)
    normalize = Keyword.get(opts, :normalize, default_normalize(metric))
    store = Keyword.get(opts, :store, :ets)
    index = Keyword.get(opts, :index, :flat)
    index_options = Keyword.get(opts, :index_options, [])
    score = Keyword.get(opts, :score, :raw)
    compressed = Keyword.get(opts, :compressed, false)

    with :ok <- validate_dimensions(dimensions),
         :ok <- validate_metric(metric),
         :ok <- validate_normalization(normalize),
         :ok <- validate_score_mode(score),
         :ok <- validate_boolean(compressed, :invalid_compressed),
         :ok <- validate_keyword(index_options, :invalid_index_options),
         {:ok, store_mod} <- store_module(store),
         {:ok, index_mod} <- index_module(index),
         {:ok, index_state} <- index_mod.new(metric, index_options),
         {:ok, store_state} <-
           store_mod.new(%{
             snapshot_version: @snapshot_version,
             name: Keyword.get(opts, :name),
             dimensions: dimensions,
             metric: metric,
             normalize: normalize,
             score: score,
             index: index,
             index_options: index_options,
             compressed: compressed
           }) do
      {:ok,
       %__MODULE__{
         name: Keyword.get(opts, :name),
         dimensions: dimensions,
         metric: metric,
         normalize: normalize,
         score: score,
         store_mod: store_mod,
         store_state: store_state,
         index_mod: index_mod,
         index_state: index_state,
         index: index,
         index_options: index_options,
         compressed: compressed
       }}
    end
  end

  @doc false
  @spec snapshot(t(), Path.t()) :: :ok | {:error, term()}
  def snapshot(%__MODULE__{} = collection, path) when is_binary(path) do
    with :ok <- ensure_open(collection),
         :ok <- configure_store(collection) do
      collection.store_mod.snapshot(collection.store_state, path)
    end
  end

  def snapshot(_collection, _path), do: {:error, :invalid_snapshot}

  @doc false
  @spec load_snapshot(Path.t(), keyword()) :: {:ok, t()} | {:error, term()}
  def load_snapshot(path, opts \\ [])

  def load_snapshot(path, opts) when is_binary(path) and is_list(opts) do
    with :ok <- validate_snapshot_options(opts),
         {:ok, store_mod} <- store_module(Keyword.get(opts, :store, :ets)),
         {:ok, {store_state, config}} <- store_mod.load_snapshot(path) do
      case restore_and_rebuild(store_mod, store_state, config, opts) do
        {:ok, collection} ->
          {:ok, collection}

        {:error, reason} ->
          close_store(store_mod, store_state)
          {:error, reason}
      end
    end
  end

  def load_snapshot(_path, _opts), do: {:error, :invalid_snapshot}

  @doc false
  @spec put(t(), Embedding.t() | map()) :: :ok | {:error, term()}
  def put(%__MODULE__{} = collection, embedding) do
    with {:ok, embedding} <- prepare_embedding(collection, embedding),
         :ok <- collection.store_mod.put(collection.store_state, embedding) do
      case collection.index_mod.put(collection, embedding) do
        :ok ->
          :ok

        {:error, reason} ->
          rollback_insert(collection, [embedding])
          {:error, reason}
      end
    end
  end

  @doc false
  @spec put_many(t(), [Embedding.t() | map()]) :: :ok | {:error, term()}
  def put_many(%__MODULE__{} = collection, embeddings) when is_list(embeddings) do
    with {:ok, prepared} <- prepare_embeddings(collection, embeddings),
         :ok <- collection.store_mod.put_many(collection.store_state, prepared) do
      put_many_index(collection, prepared)
    end
  end

  def put_many(%__MODULE__{}, _embeddings), do: {:error, :invalid_embeddings}

  @doc false
  @spec get(t(), String.t()) :: {:ok, Embedding.t()} | {:error, term()}
  def get(%__MODULE__{} = collection, id) when is_binary(id) do
    collection.store_mod.get(collection.store_state, id)
  end

  @doc false
  @spec delete(t(), String.t()) :: :ok | {:error, term()}
  def delete(%__MODULE__{} = collection, id) when is_binary(id) do
    case collection.store_mod.get(collection.store_state, id) do
      {:ok, embedding} ->
        delete_existing(collection, id, embedding)

      {:error, :not_found} ->
        collection.index_mod.delete(collection, id)

      {:error, reason} ->
        {:error, reason}
    end
  end

  def delete(%__MODULE__{}, _id), do: {:error, :invalid_id}

  @doc false
  @spec all(t()) :: {:ok, [Embedding.t()]} | {:error, term()}
  def all(%__MODULE__{} = collection), do: collection.store_mod.all(collection.store_state)

  @doc false
  @spec search(t(), [number()], keyword()) :: {:ok, [Vettore.Result.t()]} | {:error, term()}
  def search(collection, query, opts \\ [])

  def search(%__MODULE__{} = collection, query, opts) when is_list(opts) do
    with :ok <- validate_options(opts, @search_option_keys) do
      collection.index_mod.search(collection, query, opts)
    end
  end

  def search(%__MODULE__{}, _query, _opts), do: {:error, :invalid_options}

  @doc false
  @spec funnel_search(t(), [number()], keyword()) :: {:ok, [Result.t()]} | {:error, term()}
  def funnel_search(collection, query, opts \\ [])

  def funnel_search(%__MODULE__{} = collection, query, opts) when is_list(opts) do
    with :ok <- validate_options(opts, @funnel_option_keys) do
      do_funnel_search(collection, query, opts)
    end
  end

  def funnel_search(%__MODULE__{}, _query, _opts), do: {:error, :invalid_options}

  @spec do_funnel_search(t(), [number()], keyword()) :: {:ok, [Result.t()]} | {:error, term()}
  defp do_funnel_search(collection, query, opts) do
    limit = Keyword.get(opts, :limit, 10)
    candidates = candidate_count(opts, limit)
    stages = funnel_stages(collection, opts)

    with :ok <- validate_limit(limit),
         :ok <- validate_candidates(candidates, limit),
         :ok <- validate_funnel_stages(stages, collection.dimensions),
         {:ok, query} <- prepare_query(collection, query),
         {:ok, embeddings} <- collection.store_mod.all(collection.store_state) do
      with {:ok, stage_embeddings} <-
             funnel_stage_embeddings(collection, embeddings, query, stages, candidates) do
        exact_rerank(collection, query, stage_embeddings, limit)
      end
    end
  end

  @doc false
  @spec quantized_search(t(), [number()], keyword()) :: {:ok, [Result.t()]} | {:error, term()}
  def quantized_search(collection, query, opts \\ [])

  def quantized_search(%__MODULE__{} = collection, query, opts) when is_list(opts) do
    with :ok <- validate_options(opts, @quantized_option_keys) do
      do_quantized_search(collection, query, opts)
    end
  end

  def quantized_search(%__MODULE__{}, _query, _opts), do: {:error, :invalid_options}

  @spec do_quantized_search(t(), [number()], keyword()) ::
          {:ok, [Result.t()]} | {:error, term()}
  defp do_quantized_search(collection, query, opts) do
    limit = Keyword.get(opts, :limit, 10)
    candidates = candidate_count(opts, limit)

    with :ok <- validate_limit(limit),
         :ok <- validate_candidates(candidates, limit),
         {:ok, query} <- prepare_query(collection, query),
         {:ok, query_bits} <- compress_vector(query),
         {:ok, embeddings} <- collection.store_mod.all(collection.store_state) do
      with {:ok, stage_embeddings} <-
             binary_candidate_embeddings(
               embeddings,
               query_bits,
               collection.dimensions,
               candidates
             ) do
        exact_rerank(collection, query, stage_embeddings, limit)
      end
    end
  end

  @doc false
  @spec multi_vector_search(t(), [[number()]], keyword()) ::
          {:ok, [Result.t()]} | {:error, term()}
  def multi_vector_search(collection, query_vectors, opts \\ [])

  def multi_vector_search(%__MODULE__{} = collection, query_vectors, opts) when is_list(opts) do
    with :ok <- validate_options(opts, @multi_vector_option_keys) do
      do_multi_vector_search(collection, query_vectors, opts)
    end
  end

  def multi_vector_search(%__MODULE__{}, _query_vectors, _opts),
    do: {:error, :invalid_options}

  @spec do_multi_vector_search(t(), [[number()]], keyword()) ::
          {:ok, [Result.t()]} | {:error, term()}
  defp do_multi_vector_search(collection, query_vectors, opts) do
    limit = Keyword.get(opts, :limit, 10)
    metric = normalize_metric(Keyword.get(opts, :metric, collection.metric))

    with :ok <- validate_limit(limit),
         :ok <- validate_metric(metric),
         {:ok, query_vectors} <- prepare_query_vectors(collection, query_vectors),
         {:ok, embeddings} <- collection.store_mod.all(collection.store_state) do
      multi_vector_results(collection, query_vectors, embeddings, metric, limit)
    end
  end

  @doc false
  @spec hybrid_search(t(), [number()], keyword()) :: {:ok, [Result.t()]} | {:error, term()}
  def hybrid_search(collection, query, opts \\ [])

  def hybrid_search(%__MODULE__{} = collection, query, opts) when is_list(opts) do
    with :ok <- validate_options(opts, @hybrid_option_keys) do
      do_hybrid_search(collection, query, opts)
    end
  end

  def hybrid_search(%__MODULE__{}, _query, _opts), do: {:error, :invalid_options}

  @spec do_hybrid_search(t(), [number()], keyword()) :: {:ok, [Result.t()]} | {:error, term()}
  defp do_hybrid_search(collection, query, opts) do
    limit = Keyword.get(opts, :limit, 10)
    generators = Keyword.get(opts, :generators, default_hybrid_generators(collection))
    rerank = Keyword.get(opts, :rerank, :exact)

    with :ok <- validate_limit(limit),
         {:ok, query} <- prepare_query(collection, query),
         {:ok, candidates} <- hybrid_candidates(collection, query, generators, limit) do
      hybrid_rerank(collection, query, candidates, rerank, limit)
    end
  end

  @doc false
  @spec prepare_query(t(), [number()]) :: {:ok, [float()]} | {:error, term()}
  def prepare_query(%__MODULE__{} = collection, query) do
    with :ok <- ensure_open(collection),
         :ok <- validate_vector(query, collection.dimensions) do
      Distance.normalize(query, collection.normalize)
    end
  end

  @doc false
  @spec close(t()) :: :ok | {:error, term()}
  def close(%__MODULE__{} = collection) do
    close_store(collection.store_mod, collection.store_state)
  end

  @doc false
  @spec ensure_open(t()) :: :ok | {:error, :closed}
  def ensure_open(%__MODULE__{} = collection) do
    if function_exported?(collection.store_mod, :alive?, 1) and
         not collection.store_mod.alive?(collection.store_state) do
      {:error, :closed}
    else
      :ok
    end
  end

  @spec restore_collection(module(), term(), map(), keyword()) :: {:ok, t()} | {:error, term()}
  defp restore_collection(store_mod, store_state, config, opts) when is_map(config) do
    metric = config_option(config, opts, :metric, :cosine) |> normalize_metric()
    dimensions = config_option(config, opts, :dimensions, nil)
    normalize = config_option(config, opts, :normalize, default_normalize(metric))
    index = config_option(config, opts, :index, :flat)
    index_options = config_option(config, opts, :index_options, [])
    score = config_option(config, opts, :score, :raw)
    compressed = Map.get(config, :compressed, false)

    with :ok <- validate_snapshot_version(config),
         :ok <- validate_dimensions(dimensions),
         :ok <- validate_metric(metric),
         :ok <- validate_normalization(normalize),
         :ok <- validate_score_mode(score),
         :ok <- validate_boolean(compressed, :invalid_compressed),
         :ok <- validate_keyword(index_options, :invalid_index_options),
         {:ok, index_mod} <- index_module(index),
         {:ok, index_state} <- index_mod.new(metric, index_options) do
      {:ok,
       %__MODULE__{
         name: config_option(config, opts, :name, nil),
         dimensions: dimensions,
         metric: metric,
         normalize: normalize,
         score: score,
         store_mod: store_mod,
         store_state: store_state,
         index_mod: index_mod,
         index_state: index_state,
         index: index,
         index_options: index_options,
         compressed: compressed
       }}
    end
  end

  defp restore_collection(_store_mod, _store_state, _config, _opts),
    do: {:error, :invalid_snapshot}

  @spec restore_and_rebuild(module(), term(), map(), keyword()) ::
          {:ok, t()} | {:error, term()}
  defp restore_and_rebuild(store_mod, store_state, config, opts) do
    with {:ok, collection} <- restore_collection(store_mod, store_state, config, opts),
         {:ok, collection} <- rebuild_index(collection),
         :ok <- configure_store(collection) do
      {:ok, collection}
    end
  end

  @spec rebuild_index(t()) :: {:ok, t()} | {:error, term()}
  defp rebuild_index(%__MODULE__{} = collection) do
    with {:ok, embeddings} <- collection.store_mod.all(collection.store_state),
         :ok <- validate_snapshot_embeddings(collection, embeddings),
         :ok <- collection.index_mod.put_many(collection, Enum.sort_by(embeddings, & &1.id)) do
      {:ok, collection}
    end
  end

  @spec configure_store(t()) :: :ok | {:error, term()}
  defp configure_store(%__MODULE__{} = collection) do
    if function_exported?(collection.store_mod, :configure, 2) do
      collection.store_mod.configure(collection.store_state, collection_config(collection))
    else
      :ok
    end
  end

  @spec collection_config(t()) :: map()
  defp collection_config(%__MODULE__{} = collection) do
    %{
      snapshot_version: @snapshot_version,
      name: collection.name,
      dimensions: collection.dimensions,
      metric: collection.metric,
      normalize: collection.normalize,
      score: collection.score,
      index: collection.index,
      index_options: collection.index_options,
      compressed: collection.compressed
    }
  end

  @spec rollback_insert(t(), [Embedding.t()]) :: :ok
  defp rollback_insert(%__MODULE__{} = collection, embeddings) do
    Enum.each(embeddings, fn embedding ->
      collection.index_mod.delete(collection, embedding.id)
      collection.store_mod.delete(collection.store_state, embedding.id)
    end)

    :ok
  end

  @spec put_many_index(t(), [Embedding.t()]) :: :ok | {:error, term()}
  defp put_many_index(%__MODULE__{} = collection, embeddings) do
    case collection.index_mod.put_many(collection, embeddings) do
      :ok ->
        :ok

      {:error, reason} ->
        rollback_insert(collection, embeddings)
        {:error, reason}
    end
  end

  @spec delete_existing(t(), String.t(), Embedding.t()) :: :ok | {:error, term()}
  defp delete_existing(%__MODULE__{} = collection, id, embedding) do
    with :ok <- collection.index_mod.delete(collection, id) do
      delete_stored_embedding(collection, id, embedding)
    end
  end

  @spec delete_stored_embedding(t(), String.t(), Embedding.t()) :: :ok | {:error, term()}
  defp delete_stored_embedding(%__MODULE__{} = collection, id, embedding) do
    case collection.store_mod.delete(collection.store_state, id) do
      :ok -> :ok
      {:error, reason} -> restore_deleted_index(collection, embedding, reason)
    end
  end

  @spec restore_deleted_index(t(), Embedding.t(), term()) :: {:error, term()}
  defp restore_deleted_index(%__MODULE__{} = collection, embedding, store_reason) do
    case collection.index_mod.put(collection, embedding) do
      :ok -> {:error, store_reason}
      {:error, index_reason} -> {:error, {:index_restore_failed, store_reason, index_reason}}
    end
  end

  @spec close_store(module(), term()) :: :ok | {:error, term()}
  defp close_store(store_mod, store_state) do
    if function_exported?(store_mod, :close, 1), do: store_mod.close(store_state), else: :ok
  end

  @spec candidate_count(keyword(), pos_integer()) :: pos_integer()
  defp candidate_count(opts, limit), do: Keyword.get(opts, :candidates, max(limit * 10, limit))

  @spec default_hybrid_generators(t()) :: [hybrid_generator()]
  defp default_hybrid_generators(%__MODULE__{index: :hnsw}), do: [:hnsw, :quantized]
  defp default_hybrid_generators(%__MODULE__{}), do: [:funnel, :quantized]

  @spec hybrid_candidates(t(), [float()], [hybrid_generator()], pos_integer()) ::
          {:ok, [Embedding.t()]} | {:error, term()}
  defp hybrid_candidates(%__MODULE__{} = collection, query, generators, limit)
       when is_list(generators) and generators != [] do
    Enum.reduce_while(generators, {:ok, []}, fn generator, {:ok, acc} ->
      case run_hybrid_generator(collection, query, generator, limit) do
        {:ok, embeddings} -> {:cont, {:ok, acc ++ embeddings}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, embeddings} -> {:ok, unique_embeddings(embeddings)}
      {:error, reason} -> {:error, reason}
    end
  end

  defp hybrid_candidates(_collection, _query, _generators, _limit),
    do: {:error, :invalid_generators}

  @spec run_hybrid_generator(t(), [float()], hybrid_generator(), pos_integer()) ::
          {:ok, [Embedding.t()]} | {:error, term()}
  defp run_hybrid_generator(collection, query, generator, limit) when is_atom(generator) do
    run_hybrid_generator(collection, query, {generator, []}, limit)
  end

  defp run_hybrid_generator(collection, query, {name, opts}, limit)
       when is_atom(name) and is_list(opts) do
    with :ok <- validate_generator_options(name, opts) do
      opts = Keyword.put_new(opts, :candidates, max(limit * 10, limit))

      case name do
        :funnel -> funnel_candidates(collection, query, opts, limit)
        :quantized -> quantized_candidates(collection, query, opts, limit)
        :search -> index_candidates(collection, query, opts, limit)
        :hnsw -> hnsw_candidates(collection, query, opts, limit)
      end
    end
  end

  defp run_hybrid_generator(_collection, _query, generator, _limit),
    do: {:error, {:invalid_generator, generator}}

  @spec funnel_candidates(t(), [float()], keyword(), pos_integer()) ::
          {:ok, [Embedding.t()]} | {:error, term()}
  defp funnel_candidates(collection, query, opts, limit) do
    candidates = candidate_count(opts, limit)
    stages = funnel_stages(collection, opts)

    with :ok <- validate_generator_candidates(candidates),
         :ok <- validate_funnel_stages(stages, collection.dimensions),
         {:ok, embeddings} <- collection.store_mod.all(collection.store_state) do
      funnel_stage_embeddings(collection, embeddings, query, stages, candidates)
    end
  end

  @spec quantized_candidates(t(), [float()], keyword(), pos_integer()) ::
          {:ok, [Embedding.t()]} | {:error, term()}
  defp quantized_candidates(collection, query, opts, limit) do
    candidates = candidate_count(opts, limit)

    with :ok <- validate_generator_candidates(candidates),
         {:ok, query_bits} <- compress_vector(query),
         {:ok, embeddings} <- collection.store_mod.all(collection.store_state) do
      binary_candidate_embeddings(embeddings, query_bits, collection.dimensions, candidates)
    end
  end

  @spec index_candidates(t(), [float()], keyword(), pos_integer()) ::
          {:ok, [Embedding.t()]} | {:error, term()}
  defp index_candidates(collection, query, opts, limit) do
    candidate_limit = candidate_count(opts, limit)

    with :ok <- validate_generator_candidates(candidate_limit),
         {:ok, results} <- collection.index_mod.search(collection, query, limit: candidate_limit) do
      fetch_result_embeddings(collection, results)
    end
  end

  @spec hnsw_candidates(t(), [float()], keyword(), pos_integer()) ::
          {:ok, [Embedding.t()]} | {:error, term()}
  defp hnsw_candidates(%__MODULE__{index: :hnsw} = collection, query, opts, limit) do
    index_candidates(collection, query, opts, limit)
  end

  defp hnsw_candidates(_collection, _query, _opts, _limit), do: {:error, :hnsw_index_required}

  @spec fetch_result_embeddings(t(), [Result.t()]) :: {:ok, [Embedding.t()]} | {:error, term()}
  defp fetch_result_embeddings(collection, results) do
    Enum.reduce_while(results, {:ok, []}, fn %Result{id: id}, {:ok, acc} ->
      case get(collection, id) do
        {:ok, embedding} -> {:cont, {:ok, [embedding | acc]}}
        {:error, :not_found} -> {:cont, {:ok, acc}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, embeddings} -> {:ok, Enum.reverse(embeddings)}
      {:error, reason} -> {:error, reason}
    end
  end

  @spec unique_embeddings([Embedding.t()]) :: [Embedding.t()]
  defp unique_embeddings(embeddings) do
    embeddings
    |> Enum.reduce({[], MapSet.new()}, fn %Embedding{id: id} = embedding, {acc, seen} ->
      if MapSet.member?(seen, id) do
        {acc, seen}
      else
        {[embedding | acc], MapSet.put(seen, id)}
      end
    end)
    |> elem(0)
    |> Enum.reverse()
  end

  @spec hybrid_rerank(t(), [float()], [Embedding.t()], hybrid_rerank(), pos_integer()) ::
          {:ok, [Result.t()]} | {:error, term()}
  defp hybrid_rerank(collection, query, candidates, :exact, limit) do
    exact_rerank(collection, query, candidates, limit)
  end

  defp hybrid_rerank(collection, _query, candidates, {:multi_vector, query_vectors}, limit) do
    multi_vector_hybrid_rerank(collection, candidates, query_vectors, [], limit)
  end

  defp hybrid_rerank(collection, _query, candidates, {:multi_vector, query_vectors, opts}, limit)
       when is_list(opts) do
    multi_vector_hybrid_rerank(collection, candidates, query_vectors, opts, limit)
  end

  defp hybrid_rerank(_collection, _query, _candidates, rerank, _limit),
    do: {:error, {:invalid_rerank, rerank}}

  @spec multi_vector_hybrid_rerank(t(), [Embedding.t()], [[number()]], keyword(), pos_integer()) ::
          {:ok, [Result.t()]} | {:error, term()}
  defp multi_vector_hybrid_rerank(collection, candidates, query_vectors, opts, limit) do
    with :ok <- validate_options(opts, [:metric]),
         metric = normalize_metric(Keyword.get(opts, :metric, collection.metric)),
         :ok <- validate_metric(metric),
         {:ok, query_vectors} <- prepare_query_vectors(collection, query_vectors) do
      multi_vector_results(collection, query_vectors, candidates, metric, limit)
    end
  end

  @spec funnel_stages(t(), keyword()) :: [pos_integer()]
  defp funnel_stages(%__MODULE__{} = collection, opts) do
    cond do
      Keyword.has_key?(opts, :stages) ->
        Keyword.fetch!(opts, :stages)

      Keyword.has_key?(opts, :dimensions) ->
        [Keyword.fetch!(opts, :dimensions)]

      true ->
        [min(collection.dimensions, 128)]
    end
  end

  @spec funnel_stage_embeddings(t(), [Embedding.t()], [float()], [pos_integer()], pos_integer()) ::
          {:ok, [Embedding.t()]} | {:error, term()}
  defp funnel_stage_embeddings(collection, embeddings, query, stages, candidates) do
    Enum.reduce_while(stages, {:ok, embeddings}, fn dimensions, {:ok, acc} ->
      case funnel_stage(collection, acc, query, dimensions, candidates) do
        {:ok, next} -> {:cont, {:ok, next}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
  end

  @spec funnel_stage(t(), [Embedding.t()], [float()], pos_integer(), pos_integer()) ::
          {:ok, [Embedding.t()]} | {:error, term()}
  defp funnel_stage(collection, embeddings, query, dimensions, candidates) do
    with {:ok, scored} <- score_embeddings(collection, embeddings, query, candidates, dimensions) do
      {:ok, Enum.map(scored, fn {_result, embedding} -> embedding end)}
    end
  end

  @spec binary_candidate_embeddings(
          [Embedding.t()],
          [non_neg_integer()],
          pos_integer(),
          pos_integer()
        ) :: {:ok, [Embedding.t()]} | {:error, term()}
  defp binary_candidate_embeddings(embeddings, query_bits, dimensions, candidates) do
    with :ok <- validate_runtime_embeddings(embeddings),
         {:ok, vectors} <- binary_vectors(embeddings, dimensions),
         {:ok, hits} <- Nifs.binary_top_k(vectors, query_bits, dimensions, candidates) do
      by_id = Map.new(embeddings, &{&1.id, &1})

      {:ok,
       Enum.flat_map(hits, fn {id, _distance} ->
         case Map.fetch(by_id, id) do
           {:ok, embedding} -> [embedding]
           :error -> []
         end
       end)}
    end
  end

  @spec binary_vectors([Embedding.t()], pos_integer()) ::
          {:ok, [{String.t(), [non_neg_integer()]}]} | {:error, term()}
  defp binary_vectors(embeddings, dimensions) do
    Enum.reduce_while(embeddings, {:ok, []}, fn embedding, {:ok, acc} ->
      case binary_vector(embedding, dimensions) do
        {:ok, words} -> {:cont, {:ok, [{embedding.id, words} | acc]}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, vectors} -> {:ok, Enum.reverse(vectors)}
      {:error, reason} -> {:error, reason}
    end
  end

  @spec binary_vector(Embedding.t(), pos_integer()) ::
          {:ok, [non_neg_integer()]}
          | {:error, :dimension_mismatch | :invalid_binary_vector | :invalid_vector}
  defp binary_vector(%Embedding{binary_vector: binary_vector}, dimensions)
       when is_list(binary_vector) do
    with :ok <- validate_binary_words(binary_vector, dimensions), do: {:ok, binary_vector}
  end

  defp binary_vector(%Embedding{vector: vector}, dimensions) do
    with :ok <- validate_vector(vector, dimensions), do: compress_vector(vector)
  end

  @spec multi_vector_results(t(), [[float()]], [Embedding.t()], atom(), pos_integer()) ::
          {:ok, [Result.t()]} | {:error, term()}
  defp multi_vector_results(collection, query_vectors, embeddings, metric, limit) do
    with :ok <- validate_runtime_embeddings(embeddings),
         {:ok, documents} <- multi_vector_documents(embeddings, collection.dimensions),
         {:ok, hits} <-
           Nifs.multi_vector_top_k(documents, query_vectors, metric_code(metric), limit)
           |> normalize_multi_vector_error() do
      by_id = Map.new(embeddings, &{&1.id, &1})

      {:ok,
       Enum.flat_map(hits, fn {id, score} ->
         case Map.fetch(by_id, id) do
           {:ok, embedding} -> [to_multi_vector_result(collection, embedding, score, metric)]
           :error -> []
         end
       end)}
    end
  end

  @spec normalize_multi_vector_error(term()) :: term()
  defp normalize_multi_vector_error({:error, "score overflow"}), do: {:error, :score_overflow}

  defp normalize_multi_vector_error({:error, "dimension mismatch"}),
    do: {:error, :dimension_mismatch}

  defp normalize_multi_vector_error({:error, "vector contains a non-finite value"}),
    do: {:error, :invalid_multi_vector}

  defp normalize_multi_vector_error(other), do: other

  @spec document_vectors(Embedding.t()) :: [[float()]]
  defp document_vectors(%Embedding{vectors: vectors}) when is_list(vectors) and vectors != [],
    do: vectors

  defp document_vectors(%Embedding{vector: vector}), do: [vector]

  @spec multi_vector_documents([Embedding.t()], pos_integer()) ::
          {:ok, [{String.t(), [[float()]]}]} | {:error, term()}
  defp multi_vector_documents(embeddings, dimensions) do
    Enum.reduce_while(embeddings, {:ok, []}, fn embedding, {:ok, acc} ->
      vectors = document_vectors(embedding)

      case validate_document_vectors(vectors, dimensions) do
        :ok -> {:cont, {:ok, [{embedding.id, vectors} | acc]}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, documents} -> {:ok, Enum.reverse(documents)}
      {:error, reason} -> {:error, reason}
    end
  end

  @spec validate_document_vectors([[term()]], pos_integer()) :: :ok | {:error, term()}
  defp validate_document_vectors(vectors, dimensions) do
    Enum.reduce_while(vectors, :ok, fn vector, :ok ->
      case validate_vector(vector, dimensions) do
        :ok -> {:cont, :ok}
        {:error, :invalid_vector} -> {:halt, {:error, :invalid_multi_vector}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
  end

  @spec to_multi_vector_result(t(), Embedding.t(), number(), atom()) :: Result.t()
  defp to_multi_vector_result(%__MODULE__{}, %Embedding{} = embedding, score, metric) do
    %Result{
      id: embedding.id,
      value: embedding.value,
      score: score / 1,
      distance: nil,
      metric: metric,
      metadata: embedding.metadata
    }
  end

  @spec exact_rerank(t(), [float()], [Embedding.t()], pos_integer()) ::
          {:ok, [Result.t()]} | {:error, term()}
  defp exact_rerank(%__MODULE__{} = collection, query, embeddings, limit) do
    with {:ok, scored} <-
           score_embeddings(collection, embeddings, query, limit, collection.dimensions) do
      {:ok, Enum.map(scored, fn {result, _embedding} -> result end)}
    end
  end

  @spec score_embeddings(t(), [Embedding.t()], [float()], pos_integer(), pos_integer()) ::
          {:ok, [{Result.t(), Embedding.t()}]} | {:error, term()}
  defp score_embeddings(%__MODULE__{} = collection, embeddings, query, limit, dimensions) do
    with :ok <- validate_runtime_embeddings(embeddings),
         {:ok, vectors} <- scoring_vectors(embeddings, collection.dimensions),
         {:ok, hits} <-
           Nifs.vector_top_k(
             vectors,
             query,
             metric_code(collection.metric),
             dimensions,
             limit
           ) do
      by_id = Map.new(embeddings, &{&1.id, &1})

      {:ok,
       Enum.flat_map(hits, fn {id, raw} ->
         case Map.fetch(by_id, id) do
           {:ok, embedding} -> [{to_result(collection, embedding, raw), embedding}]
           :error -> []
         end
       end)}
    end
  end

  @spec scoring_vectors([Embedding.t()], pos_integer()) ::
          {:ok, [{String.t(), [float()]}]} | {:error, term()}
  defp scoring_vectors(embeddings, dimensions) do
    Enum.reduce_while(embeddings, {:ok, []}, fn embedding, {:ok, acc} ->
      case validate_vector(embedding.vector, dimensions) do
        :ok -> {:cont, {:ok, [{embedding.id, embedding.vector} | acc]}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, vectors} -> {:ok, Enum.reverse(vectors)}
      {:error, reason} -> {:error, reason}
    end
  end

  @spec to_result(t(), Embedding.t(), number()) :: Result.t()
  defp to_result(%__MODULE__{} = collection, %Embedding{} = embedding, raw) do
    {score, distance} = Distance.result_values(collection.metric, raw, collection.score)

    %Result{
      id: embedding.id,
      value: embedding.value,
      score: score,
      distance: distance,
      metric: collection.metric,
      metadata: embedding.metadata
    }
  end

  @spec validate_limit(term()) :: :ok | {:error, :invalid_limit}
  defp validate_limit(limit)
       when is_integer(limit) and limit > 0 and limit <= @max_nif_usize,
       do: :ok

  defp validate_limit(_limit), do: {:error, :invalid_limit}

  @spec validate_candidates(term(), pos_integer()) :: :ok | {:error, :invalid_candidates}
  defp validate_candidates(candidates, limit)
       when is_integer(candidates) and candidates >= limit and candidates > 0 and
              candidates <= @max_nif_usize,
       do: :ok

  defp validate_candidates(_candidates, _limit), do: {:error, :invalid_candidates}

  @spec validate_generator_candidates(term()) :: :ok | {:error, :invalid_candidates}
  defp validate_generator_candidates(candidates)
       when is_integer(candidates) and candidates > 0 and candidates <= @max_nif_usize,
       do: :ok

  defp validate_generator_candidates(_candidates), do: {:error, :invalid_candidates}

  @spec validate_funnel_stages(term(), pos_integer()) :: :ok | {:error, :invalid_stages}
  defp validate_funnel_stages(stages, dimensions) when is_list(stages) and stages != [] do
    if Enum.all?(stages, &(is_integer(&1) and &1 > 0 and &1 <= dimensions)) do
      :ok
    else
      {:error, :invalid_stages}
    end
  end

  defp validate_funnel_stages(_stages, _dimensions), do: {:error, :invalid_stages}

  @spec config_option(map(), keyword(), atom(), term()) :: term()
  defp config_option(config, opts, key, default) do
    Keyword.get(opts, key, Map.get(config, key, default))
  end

  @spec prepare_embedding(t(), embedding_input()) :: {:ok, Embedding.t()} | {:error, term()}
  defp prepare_embedding(%__MODULE__{} = collection, embedding) do
    with {:ok, embedding} <- to_embedding(embedding),
         {:ok, id} <- embedding_id(embedding),
         {:ok, vectors} <- prepare_embedding_vectors(collection, embedding.vectors),
         {:ok, vector} <- prepare_primary_vector(collection, embedding.vector, vectors),
         {:ok, binary_vector} <- compress_vector(vector) do
      {:ok,
       %Embedding{
         embedding
         | id: id,
           value: embedding.value || id,
           vector: vector,
           vectors: vectors,
           binary_vector: binary_vector
       }}
    end
  end

  @spec compress_vector([number()]) ::
          {:ok, [non_neg_integer()]} | {:error, :invalid_vector}
  defp compress_vector(vector) do
    case Distance.compress_f32_vector(vector) do
      words when is_list(words) -> {:ok, words}
      {:error, :invalid_vector} = error -> error
    end
  end

  @spec prepare_embeddings(t(), [embedding_input()]) ::
          {:ok, [Embedding.t()]} | {:error, term()}
  defp prepare_embeddings(%__MODULE__{} = collection, embeddings) do
    Enum.reduce_while(embeddings, {:ok, []}, fn embedding, {:ok, acc} ->
      case prepare_embedding(collection, embedding) do
        {:ok, embedding} -> {:cont, {:ok, [embedding | acc]}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, prepared} -> {:ok, Enum.reverse(prepared)}
      {:error, reason} -> {:error, reason}
    end
  end

  @spec prepare_query_vectors(t(), term()) :: {:ok, [[float()]]} | {:error, term()}
  defp prepare_query_vectors(%__MODULE__{} = collection, vectors) do
    prepare_vectors(collection, vectors, :invalid_multi_vector)
  end

  @spec prepare_embedding_vectors(t(), term()) :: {:ok, [[float()]] | nil} | {:error, term()}
  defp prepare_embedding_vectors(_collection, nil), do: {:ok, nil}

  defp prepare_embedding_vectors(%__MODULE__{} = collection, vectors) do
    prepare_vectors(collection, vectors, :invalid_multi_vector)
  end

  @spec prepare_vectors(t(), term(), term()) :: {:ok, [[float()]]} | {:error, term()}
  defp prepare_vectors(%__MODULE__{} = collection, vectors, _invalid_reason)
       when is_list(vectors) and vectors != [] do
    Enum.reduce_while(vectors, {:ok, []}, fn vector, {:ok, acc} ->
      with :ok <- validate_vector(vector, collection.dimensions),
           {:ok, vector} <- Distance.normalize(vector, collection.normalize) do
        {:cont, {:ok, [vector | acc]}}
      else
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, vectors} -> {:ok, Enum.reverse(vectors)}
      {:error, reason} -> {:error, reason}
    end
  end

  defp prepare_vectors(_collection, _vectors, invalid_reason), do: {:error, invalid_reason}

  @spec prepare_primary_vector(t(), term(), [[float()]] | nil) ::
          {:ok, [float()]} | {:error, term()}
  defp prepare_primary_vector(%__MODULE__{} = collection, nil, vectors) when is_list(vectors) do
    vectors
    |> mean_vector(collection.dimensions)
    |> Distance.normalize(collection.normalize)
  end

  defp prepare_primary_vector(%__MODULE__{} = collection, vector, _vectors) do
    with :ok <- validate_vector(vector, collection.dimensions) do
      Distance.normalize(vector, collection.normalize)
    end
  end

  @spec mean_vector([[float()]], pos_integer()) :: [float()]
  defp mean_vector(vectors, dimensions) do
    count = length(vectors)

    vectors
    |> Enum.reduce(List.duplicate(0.0, dimensions), fn vector, acc ->
      Enum.zip_with(acc, vector, &(&1 + &2))
    end)
    |> Enum.map(&(&1 / count))
  end

  @spec to_embedding(embedding_input() | term()) ::
          {:ok, Embedding.t()} | {:error, :invalid_embedding}
  defp to_embedding(%Embedding{} = embedding), do: {:ok, embedding}

  defp to_embedding(%{id: id, vector: vector} = map) do
    {:ok,
     %Embedding{
       id: id,
       value: Map.get(map, :value, id),
       vector: vector,
       vectors: Map.get(map, :vectors),
       metadata: Map.get(map, :metadata)
     }}
  end

  defp to_embedding(%{id: id, vectors: vectors} = map) do
    {:ok,
     %Embedding{
       id: id,
       value: Map.get(map, :value, id),
       vector: nil,
       vectors: vectors,
       metadata: Map.get(map, :metadata)
     }}
  end

  defp to_embedding(%{value: value, vector: vector} = map) do
    {:ok,
     %Embedding{
       id: nil,
       value: value,
       vector: vector,
       vectors: Map.get(map, :vectors),
       metadata: Map.get(map, :metadata)
     }}
  end

  defp to_embedding(%{value: value, vectors: vectors} = map) do
    {:ok,
     %Embedding{
       id: nil,
       value: value,
       vector: nil,
       vectors: vectors,
       metadata: Map.get(map, :metadata)
     }}
  end

  defp to_embedding(_embedding), do: {:error, :invalid_embedding}

  @spec embedding_id(Embedding.t()) :: {:ok, String.t()} | {:error, :missing_id}
  defp embedding_id(%Embedding{id: id}) when is_binary(id) and id != "", do: {:ok, id}

  defp embedding_id(%Embedding{value: value}) when is_binary(value) and value != "",
    do: {:ok, value}

  defp embedding_id(_embedding), do: {:error, :missing_id}

  @spec validate_dimensions(term()) :: :ok | {:error, :invalid_dimensions}
  defp validate_dimensions(dimensions) when is_integer(dimensions) and dimensions > 0, do: :ok
  defp validate_dimensions(_dimensions), do: {:error, :invalid_dimensions}

  @spec validate_metric(atom()) :: :ok | {:error, :invalid_metric}
  defp validate_metric(metric) when metric in @metrics, do: :ok
  defp validate_metric(_metric), do: {:error, :invalid_metric}

  @spec validate_vector(term(), pos_integer()) ::
          :ok | {:error, :dimension_mismatch | :invalid_vector}
  defp validate_vector(vector, dimensions) when is_list(vector) do
    cond do
      length(vector) != dimensions -> {:error, :dimension_mismatch}
      Enum.all?(vector, &finite_f32?/1) -> :ok
      true -> {:error, :invalid_vector}
    end
  end

  defp validate_vector(_vector, _dimensions), do: {:error, :invalid_vector}

  @spec validate_normalization(term()) :: :ok | {:error, :invalid_normalization}
  defp validate_normalization(normalization) when normalization in @normalizations, do: :ok
  defp validate_normalization(_normalization), do: {:error, :invalid_normalization}

  @spec validate_score_mode(term()) :: :ok | {:error, :invalid_score_mode}
  defp validate_score_mode(score_mode) when score_mode in @score_modes, do: :ok
  defp validate_score_mode(_score_mode), do: {:error, :invalid_score_mode}

  @spec validate_boolean(term(), atom()) :: :ok | {:error, atom()}
  defp validate_boolean(value, _reason) when is_boolean(value), do: :ok
  defp validate_boolean(_value, reason), do: {:error, reason}

  @spec validate_keyword(term(), atom()) :: :ok | {:error, atom()}
  defp validate_keyword(value, reason) when is_list(value) do
    if Keyword.keyword?(value), do: :ok, else: {:error, reason}
  end

  defp validate_keyword(_value, reason), do: {:error, reason}

  @spec validate_options(term(), [atom()]) :: :ok | {:error, term()}
  defp validate_options(opts, allowed_keys) when is_list(opts) do
    cond do
      not Keyword.keyword?(opts) ->
        {:error, :invalid_options}

      duplicate = duplicate_key(opts) ->
        {:error, {:duplicate_option, duplicate}}

      unsupported = Enum.find(Keyword.keys(opts), &(&1 not in allowed_keys)) ->
        {:error, {:unsupported_option, unsupported}}

      true ->
        :ok
    end
  end

  defp validate_options(_opts, _allowed_keys), do: {:error, :invalid_options}

  @spec validate_generator_options(atom(), term()) :: :ok | {:error, term()}
  defp validate_generator_options(:funnel, opts),
    do: validate_options(opts, [:candidates, :stages, :dimensions])

  defp validate_generator_options(name, opts) when name in [:quantized, :search, :hnsw],
    do: validate_options(opts, [:candidates])

  defp validate_generator_options(name, _opts), do: {:error, {:unknown_generator, name}}

  @spec duplicate_key(keyword()) :: atom() | nil
  defp duplicate_key(opts) do
    opts
    |> Keyword.keys()
    |> Enum.reduce_while(MapSet.new(), fn key, seen ->
      if MapSet.member?(seen, key),
        do: {:halt, key},
        else: {:cont, MapSet.put(seen, key)}
    end)
    |> case do
      %MapSet{} -> nil
      key -> key
    end
  end

  @spec validate_snapshot_options(term()) :: :ok | {:error, term()}
  defp validate_snapshot_options(opts) when is_list(opts) do
    cond do
      not Keyword.keyword?(opts) ->
        {:error, :invalid_snapshot_options}

      duplicate = duplicate_key(opts) ->
        {:error, {:duplicate_snapshot_override, duplicate}}

      unsupported = Enum.find(Keyword.keys(opts), &(&1 not in @snapshot_override_keys)) ->
        {:error, {:unsupported_snapshot_override, unsupported}}

      true ->
        :ok
    end
  end

  @spec validate_snapshot_version(map()) :: :ok | {:error, :unsupported_snapshot_version}
  defp validate_snapshot_version(config) do
    case Map.get(config, :snapshot_version, 0) do
      version when version in [0, @snapshot_version] -> :ok
      _version -> {:error, :unsupported_snapshot_version}
    end
  end

  @spec validate_snapshot_embeddings(t(), term()) :: :ok | {:error, term()}
  defp validate_snapshot_embeddings(%__MODULE__{} = collection, embeddings)
       when is_list(embeddings) do
    Enum.reduce_while(embeddings, :ok, fn
      %Embedding{} = embedding, :ok ->
        with {:ok, _id} <- embedding_id(embedding),
             :ok <- validate_vector(embedding.vector, collection.dimensions),
             :ok <- validate_optional_vectors(embedding.vectors, collection.dimensions),
             :ok <- validate_binary_vector(embedding.binary_vector, collection.dimensions) do
          {:cont, :ok}
        else
          {:error, reason} -> {:halt, {:error, {:invalid_snapshot_record, reason}}}
        end

      _embedding, :ok ->
        {:halt, {:error, {:invalid_snapshot_record, :invalid_embedding}}}
    end)
  end

  defp validate_snapshot_embeddings(_collection, _embeddings), do: {:error, :invalid_snapshot}

  @spec validate_runtime_embeddings(term()) ::
          :ok | {:error, :duplicate_id | :invalid_embedding}
  defp validate_runtime_embeddings(embeddings) when is_list(embeddings) do
    Enum.reduce_while(embeddings, {:ok, MapSet.new()}, fn
      %Embedding{id: id}, {:ok, seen} when is_binary(id) and id != "" ->
        if MapSet.member?(seen, id) do
          {:halt, {:error, :duplicate_id}}
        else
          {:cont, {:ok, MapSet.put(seen, id)}}
        end

      _embedding, _acc ->
        {:halt, {:error, :invalid_embedding}}
    end)
    |> case do
      {:ok, _seen} -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  defp validate_runtime_embeddings(_embeddings), do: {:error, :invalid_embedding}

  @spec validate_optional_vectors(term(), pos_integer()) :: :ok | {:error, term()}
  defp validate_optional_vectors(nil, _dimensions), do: :ok

  defp validate_optional_vectors(vectors, dimensions) when is_list(vectors) and vectors != [] do
    if Enum.all?(vectors, &(validate_vector(&1, dimensions) == :ok)),
      do: :ok,
      else: {:error, :invalid_multi_vector}
  end

  defp validate_optional_vectors(_vectors, _dimensions), do: {:error, :invalid_multi_vector}

  @spec validate_binary_vector(term(), pos_integer()) :: :ok | {:error, :invalid_binary_vector}
  defp validate_binary_vector(words, dimensions) do
    cond do
      is_nil(words) ->
        :ok

      is_list(words) ->
        validate_binary_words(words, dimensions)

      true ->
        {:error, :invalid_binary_vector}
    end
  end

  @spec validate_binary_words([term()], pos_integer()) :: :ok | {:error, :invalid_binary_vector}
  defp validate_binary_words(words, dimensions) do
    expected_words = div(dimensions + 63, 64)

    if length(words) == expected_words and
         Enum.all?(words, &(is_integer(&1) and &1 >= 0 and &1 <= 18_446_744_073_709_551_615)) do
      :ok
    else
      {:error, :invalid_binary_vector}
    end
  end

  @spec finite_f32?(term()) :: boolean()
  defp finite_f32?(value) when is_integer(value), do: value >= -@f32_max and value <= @f32_max

  defp finite_f32?(value) when is_float(value),
    do: value >= -@f32_max and value <= @f32_max

  defp finite_f32?(_value), do: false

  @spec store_module(:ets | module() | term()) :: {:ok, module()} | {:error, :invalid_store}
  defp store_module(:ets), do: {:ok, Vettore.Store.ETS}

  defp store_module(module) when is_atom(module) do
    if valid_module?(module, @store_callbacks), do: {:ok, module}, else: {:error, :invalid_store}
  end

  defp store_module(_store), do: {:error, :invalid_store}

  @spec index_module(:flat | :hnsw | module() | term()) ::
          {:ok, module()} | {:error, :invalid_index}
  defp index_module(:flat), do: {:ok, Vettore.Index.Flat}
  defp index_module(:hnsw), do: {:ok, Vettore.Index.HNSW}

  defp index_module(module) when is_atom(module) do
    if valid_module?(module, @index_callbacks), do: {:ok, module}, else: {:error, :invalid_index}
  end

  defp index_module(_index), do: {:error, :invalid_index}

  @spec valid_module?(module(), keyword(pos_integer())) :: boolean()
  defp valid_module?(module, callbacks) do
    Code.ensure_loaded?(module) and
      Enum.all?(callbacks, fn {name, arity} ->
        function_exported?(module, name, arity)
      end)
  end

  @spec normalize_metric(atom()) :: atom()
  defp normalize_metric(:euclidean), do: :l2
  defp normalize_metric(:dot), do: :inner_product
  defp normalize_metric(:dot_product), do: :inner_product
  defp normalize_metric(metric), do: metric

  @spec metric_code(atom()) :: 0..8
  defp metric_code(:l2), do: 0
  defp metric_code(:l2_squared), do: 1
  defp metric_code(:cosine), do: 2
  defp metric_code(:inner_product), do: 3
  defp metric_code(:negative_inner_product), do: 4
  defp metric_code(:manhattan), do: 5
  defp metric_code(:chebyshev), do: 6
  defp metric_code(:hamming), do: 7
  defp metric_code(:jaccard), do: 8

  @spec default_normalize(atom()) :: :l2 | :none
  defp default_normalize(:cosine), do: :l2
  defp default_normalize(_metric), do: :none
end
