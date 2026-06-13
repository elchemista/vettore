defmodule Vettore.Collection do
  @moduledoc """
  Collection module for managing vector embeddings.

  This module provides functions for creating, managing, and querying vector collections.
  """

  alias Vettore.{Distance, Embedding, MultiVector, Result}

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
          index_options: keyword()
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
    :index_options
  ]

  @metrics ~w(l2 l2_squared cosine inner_product negative_inner_product manhattan chebyshev hamming jaccard)a

  @doc false
  @spec new(keyword()) :: {:ok, t()} | {:error, term()}
  def new(opts) when is_list(opts) do
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
         {:ok, store_mod} <- store_module(store),
         {:ok, index_mod} <- index_module(index),
         {:ok, store_state} <-
           store_mod.new(%{
             name: Keyword.get(opts, :name),
             dimensions: dimensions,
             metric: metric,
             normalize: normalize,
             score: score,
             index: index,
             index_options: index_options,
             compressed: compressed
           }),
         {:ok, index_state} <- index_mod.new(metric, index_options) do
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
         index_options: index_options
       }}
    end
  end

  @doc false
  @spec snapshot(t(), Path.t()) :: :ok | {:error, term()}
  def snapshot(%__MODULE__{} = collection, path) when is_binary(path) do
    collection.store_mod.snapshot(collection.store_state, path)
  end

  def snapshot(_collection, _path), do: {:error, :invalid_snapshot}

  @doc false
  @spec load_snapshot(Path.t(), keyword()) :: {:ok, t()} | {:error, term()}
  def load_snapshot(path, opts \\ [])

  def load_snapshot(path, opts) when is_binary(path) and is_list(opts) do
    store = Keyword.get(opts, :store, :ets)

    with {:ok, store_mod} <- store_module(store),
         {:ok, {store_state, config}} <- store_mod.load_snapshot(path),
         {:ok, collection} <- restore_collection(store_mod, store_state, config, opts) do
      rebuild_index(collection)
    end
  end

  def load_snapshot(_path, _opts), do: {:error, :invalid_snapshot}

  @doc false
  @spec put(t(), Embedding.t() | map()) :: :ok | {:error, term()}
  def put(%__MODULE__{} = collection, embedding) do
    with {:ok, embedding} <- prepare_embedding(collection, embedding),
         :ok <- collection.store_mod.put(collection.store_state, embedding) do
      collection.index_mod.put(collection, embedding)
    end
  end

  @doc false
  @spec put_many(t(), [Embedding.t() | map()]) :: :ok | {:error, term()}
  def put_many(%__MODULE__{} = collection, embeddings) when is_list(embeddings) do
    prepared =
      Enum.reduce_while(embeddings, [], fn embedding, acc ->
        case prepare_embedding(collection, embedding) do
          {:ok, embedding} -> {:cont, [embedding | acc]}
          {:error, reason} -> {:halt, {:error, reason}}
        end
      end)

    case prepared do
      {:error, reason} ->
        {:error, reason}

      prepared ->
        prepared = Enum.reverse(prepared)

        with :ok <- collection.store_mod.put_many(collection.store_state, prepared) do
          collection.index_mod.put_many(collection, prepared)
        end
    end
  end

  @doc false
  @spec get(t(), String.t()) :: {:ok, Embedding.t()} | {:error, term()}
  def get(%__MODULE__{} = collection, id) when is_binary(id) do
    collection.store_mod.get(collection.store_state, id)
  end

  @doc false
  @spec delete(t(), String.t()) :: :ok | {:error, term()}
  def delete(%__MODULE__{} = collection, id) when is_binary(id) do
    with :ok <- collection.store_mod.delete(collection.store_state, id) do
      collection.index_mod.delete(collection, id)
    end
  end

  @doc false
  @spec all(t()) :: {:ok, [Embedding.t()]} | {:error, term()}
  def all(%__MODULE__{} = collection), do: collection.store_mod.all(collection.store_state)

  @doc false
  @spec search(t(), [number()], keyword()) :: {:ok, [Vettore.Result.t()]} | {:error, term()}
  def search(%__MODULE__{} = collection, query, opts \\ []) do
    collection.index_mod.search(collection, query, opts)
  end

  @doc false
  @spec funnel_search(t(), [number()], keyword()) :: {:ok, [Result.t()]} | {:error, term()}
  def funnel_search(%__MODULE__{} = collection, query, opts \\ []) do
    limit = Keyword.get(opts, :limit, 10)
    candidates = candidate_count(opts, limit)
    stages = funnel_stages(collection, opts)

    with :ok <- validate_limit(limit),
         :ok <- validate_candidates(candidates, limit),
         :ok <- validate_funnel_stages(stages, collection.dimensions),
         {:ok, query} <- prepare_query(collection, query),
         {:ok, embeddings} <- collection.store_mod.all(collection.store_state) do
      stage_embeddings =
        funnel_stage_embeddings(collection, embeddings, query, stages, candidates)

      {:ok, exact_rerank(collection, query, stage_embeddings, limit)}
    end
  end

  @doc false
  @spec quantized_search(t(), [number()], keyword()) :: {:ok, [Result.t()]} | {:error, term()}
  def quantized_search(%__MODULE__{} = collection, query, opts \\ []) do
    limit = Keyword.get(opts, :limit, 10)
    candidates = candidate_count(opts, limit)

    with :ok <- validate_limit(limit),
         :ok <- validate_candidates(candidates, limit),
         {:ok, query} <- prepare_query(collection, query),
         {:ok, embeddings} <- collection.store_mod.all(collection.store_state) do
      query_bits = Distance.compress_f32_vector(query)

      stage_embeddings =
        embeddings
        |> Enum.flat_map(&binary_candidate(collection, query_bits, &1))
        |> Enum.sort_by(fn {distance, embedding} -> {distance, embedding.id} end)
        |> Enum.take(candidates)
        |> Enum.map(fn {_distance, embedding} -> embedding end)

      {:ok, exact_rerank(collection, query, stage_embeddings, limit)}
    end
  end

  @doc false
  @spec multi_vector_search(t(), [[number()]], keyword()) ::
          {:ok, [Result.t()]} | {:error, term()}
  def multi_vector_search(%__MODULE__{} = collection, query_vectors, opts \\ []) do
    limit = Keyword.get(opts, :limit, 10)
    metric = normalize_metric(Keyword.get(opts, :metric, collection.metric))

    with :ok <- validate_limit(limit),
         :ok <- validate_metric(metric),
         {:ok, query_vectors} <- prepare_query_vectors(collection, query_vectors),
         {:ok, embeddings} <- collection.store_mod.all(collection.store_state) do
      results =
        embeddings
        |> Enum.flat_map(&multi_vector_result(collection, query_vectors, &1, metric))
        |> Enum.sort_by(fn %Result{} = result -> {-result.score, result.id} end)
        |> Enum.take(limit)

      {:ok, results}
    end
  end

  @doc false
  @spec hybrid_search(t(), [number()], keyword()) :: {:ok, [Result.t()]} | {:error, term()}
  def hybrid_search(%__MODULE__{} = collection, query, opts \\ []) do
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
    with :ok <- validate_vector(query, collection.dimensions) do
      Distance.normalize(query, collection.normalize)
    end
  end

  @spec restore_collection(module(), term(), map(), keyword()) :: {:ok, t()} | {:error, term()}
  defp restore_collection(store_mod, store_state, config, opts) when is_map(config) do
    metric = config_option(config, opts, :metric, :cosine) |> normalize_metric()
    dimensions = config_option(config, opts, :dimensions, nil)
    normalize = config_option(config, opts, :normalize, default_normalize(metric))
    index = config_option(config, opts, :index, :flat)
    index_options = config_option(config, opts, :index_options, [])

    with :ok <- validate_dimensions(dimensions),
         :ok <- validate_metric(metric),
         {:ok, index_mod} <- index_module(index),
         {:ok, index_state} <- index_mod.new(metric, index_options) do
      {:ok,
       %__MODULE__{
         name: config_option(config, opts, :name, nil),
         dimensions: dimensions,
         metric: metric,
         normalize: normalize,
         score: config_option(config, opts, :score, :raw),
         store_mod: store_mod,
         store_state: store_state,
         index_mod: index_mod,
         index_state: index_state,
         index: index,
         index_options: index_options
       }}
    end
  end

  defp restore_collection(_store_mod, _store_state, _config, _opts),
    do: {:error, :invalid_snapshot}

  @spec rebuild_index(t()) :: {:ok, t()} | {:error, term()}
  defp rebuild_index(%__MODULE__{} = collection) do
    with {:ok, embeddings} <- collection.store_mod.all(collection.store_state),
         :ok <- collection.index_mod.put_many(collection, Enum.sort_by(embeddings, & &1.id)) do
      {:ok, collection}
    end
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
    opts = Keyword.put_new(opts, :candidates, max(limit * 10, limit))

    case name do
      :funnel -> funnel_candidates(collection, query, opts, limit)
      :quantized -> quantized_candidates(collection, query, opts, limit)
      :search -> index_candidates(collection, query, opts, limit)
      :hnsw -> hnsw_candidates(collection, query, opts, limit)
      _other -> {:error, {:unknown_generator, name}}
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
      {:ok, funnel_stage_embeddings(collection, embeddings, query, stages, candidates)}
    end
  end

  @spec quantized_candidates(t(), [float()], keyword(), pos_integer()) ::
          {:ok, [Embedding.t()]} | {:error, term()}
  defp quantized_candidates(collection, query, opts, limit) do
    candidates = candidate_count(opts, limit)

    with :ok <- validate_generator_candidates(candidates),
         {:ok, embeddings} <- collection.store_mod.all(collection.store_state) do
      query_bits = Distance.compress_f32_vector(query)

      embeddings =
        embeddings
        |> Enum.flat_map(&binary_candidate(collection, query_bits, &1))
        |> Enum.sort_by(fn {distance, embedding} -> {distance, embedding.id} end)
        |> Enum.take(candidates)
        |> Enum.map(fn {_distance, embedding} -> embedding end)

      {:ok, embeddings}
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
    {:ok, exact_rerank(collection, query, candidates, limit)}
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
    metric = normalize_metric(Keyword.get(opts, :metric, collection.metric))

    with :ok <- validate_metric(metric),
         {:ok, query_vectors} <- prepare_query_vectors(collection, query_vectors) do
      results =
        candidates
        |> Enum.flat_map(&multi_vector_result(collection, query_vectors, &1, metric))
        |> Enum.sort_by(fn %Result{} = result -> {-result.score, result.id} end)
        |> Enum.take(limit)

      {:ok, results}
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
          [Embedding.t()]
  defp funnel_stage_embeddings(collection, embeddings, query, stages, candidates) do
    Enum.reduce(stages, embeddings, fn dimensions, acc ->
      funnel_stage(collection, acc, query, dimensions, candidates)
    end)
  end

  @spec funnel_stage(t(), [Embedding.t()], [float()], pos_integer(), pos_integer()) ::
          [Embedding.t()]
  defp funnel_stage(collection, embeddings, query, dimensions, candidates) do
    query = take_dimensions(query, dimensions)

    collection
    |> score_embeddings(embeddings, query, candidates, dimensions)
    |> Enum.map(fn {_result, embedding} -> embedding end)
  end

  @spec binary_candidate(t(), [non_neg_integer()], Embedding.t()) ::
          [{float(), Embedding.t()}]
  defp binary_candidate(collection, query_bits, %Embedding{} = embedding) do
    embedding_bits = binary_vector(embedding)

    case Distance.packed_hamming(query_bits, embedding_bits, collection.dimensions) do
      {:ok, distance} -> [{distance, embedding}]
      {:error, _reason} -> []
    end
  end

  @spec binary_vector(Embedding.t()) :: [non_neg_integer()]
  defp binary_vector(%Embedding{binary_vector: binary_vector}) when is_list(binary_vector),
    do: binary_vector

  defp binary_vector(%Embedding{vector: vector}), do: Distance.compress_f32_vector(vector)

  @spec multi_vector_result(t(), [[float()]], Embedding.t(), atom()) :: [Result.t()]
  defp multi_vector_result(%__MODULE__{} = collection, query_vectors, embedding, metric) do
    case MultiVector.colbert_score(query_vectors, document_vectors(embedding), metric: metric) do
      {:ok, score} -> [to_multi_vector_result(collection, embedding, score, metric)]
      {:error, _reason} -> []
    end
  end

  @spec document_vectors(Embedding.t()) :: [[float()]]
  defp document_vectors(%Embedding{vectors: vectors}) when is_list(vectors) and vectors != [],
    do: vectors

  defp document_vectors(%Embedding{vector: vector}), do: [vector]

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

  @spec exact_rerank(t(), [float()], [Embedding.t()], pos_integer()) :: [Result.t()]
  defp exact_rerank(%__MODULE__{} = collection, query, embeddings, limit) do
    collection
    |> score_embeddings(embeddings, query, limit, collection.dimensions)
    |> Enum.map(fn {result, _embedding} -> result end)
  end

  @spec score_embeddings(t(), [Embedding.t()], [float()], pos_integer(), pos_integer()) ::
          [{Result.t(), Embedding.t()}]
  defp score_embeddings(%__MODULE__{} = collection, embeddings, query, limit, dimensions) do
    embeddings
    |> Enum.flat_map(&score_embedding(collection, query, &1, dimensions))
    |> Enum.sort_by(fn {%Result{} = result, _embedding} -> {-result.score, result.id} end)
    |> Enum.take(limit)
  end

  @spec score_embedding(t(), [float()], Embedding.t(), pos_integer()) ::
          [{Result.t(), Embedding.t()}]
  defp score_embedding(%__MODULE__{} = collection, query, %Embedding{} = embedding, dimensions) do
    vector = take_dimensions(embedding.vector, dimensions)

    case metric_value(collection, query, vector, dimensions) do
      {:ok, raw} -> [{to_result(collection, embedding, raw), embedding}]
      {:error, _reason} -> []
    end
  end

  @spec metric_value(t(), [float()], [float()], pos_integer()) ::
          {:ok, float()} | {:error, term()}
  defp metric_value(%__MODULE__{metric: :l2}, left, right, _dimensions),
    do: Distance.l2(left, right)

  defp metric_value(%__MODULE__{metric: :l2_squared}, left, right, _dimensions),
    do: Distance.l2_squared(left, right)

  defp metric_value(
         %__MODULE__{metric: :cosine, dimensions: dimensions},
         left,
         right,
         dimensions
       ),
       do: Distance.cosine(left, right, normalize: :none)

  defp metric_value(%__MODULE__{metric: :cosine}, left, right, _dimensions),
    do: Distance.cosine(left, right)

  defp metric_value(%__MODULE__{metric: :inner_product}, left, right, _dimensions),
    do: Distance.inner_product(left, right)

  defp metric_value(%__MODULE__{metric: :negative_inner_product}, left, right, _dimensions),
    do: Distance.negative_inner_product(left, right)

  defp metric_value(%__MODULE__{metric: :manhattan}, left, right, _dimensions),
    do: Distance.manhattan(left, right)

  defp metric_value(%__MODULE__{metric: :chebyshev}, left, right, _dimensions),
    do: Distance.chebyshev(left, right)

  defp metric_value(%__MODULE__{metric: :hamming}, left, right, _dimensions),
    do: Distance.hamming(left, right)

  defp metric_value(%__MODULE__{metric: :jaccard}, left, right, _dimensions),
    do: Distance.jaccard(left, right)

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

  @spec take_dimensions([number()], pos_integer()) :: [number()]
  defp take_dimensions(vector, dimensions), do: Enum.take(vector, dimensions)

  @spec validate_limit(term()) :: :ok | {:error, :invalid_limit}
  defp validate_limit(limit) when is_integer(limit) and limit > 0, do: :ok
  defp validate_limit(_limit), do: {:error, :invalid_limit}

  @spec validate_candidates(term(), pos_integer()) :: :ok | {:error, :invalid_candidates}
  defp validate_candidates(candidates, limit)
       when is_integer(candidates) and candidates >= limit and candidates > 0,
       do: :ok

  defp validate_candidates(_candidates, _limit), do: {:error, :invalid_candidates}

  @spec validate_generator_candidates(term()) :: :ok | {:error, :invalid_candidates}
  defp validate_generator_candidates(candidates) when is_integer(candidates) and candidates > 0,
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
    embedding = to_embedding(embedding)

    with {:ok, id} <- embedding_id(embedding),
         {:ok, vectors} <- prepare_embedding_vectors(collection, embedding.vectors),
         {:ok, vector} <- prepare_primary_vector(collection, embedding.vector, vectors) do
      {:ok,
       %Embedding{
         embedding
         | id: id,
           value: embedding.value || id,
           vector: vector,
           vectors: vectors,
           binary_vector: Distance.compress_f32_vector(vector)
       }}
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
    {:ok, mean_vector(vectors, collection.dimensions)}
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

  @spec to_embedding(embedding_input()) :: Embedding.t()
  defp to_embedding(%Embedding{} = embedding), do: embedding

  defp to_embedding(%{id: id, vector: vector} = map) do
    %Embedding{
      id: id,
      value: Map.get(map, :value, id),
      vector: vector,
      vectors: Map.get(map, :vectors),
      metadata: Map.get(map, :metadata)
    }
  end

  defp to_embedding(%{id: id, vectors: vectors} = map) do
    %Embedding{
      id: id,
      value: Map.get(map, :value, id),
      vector: nil,
      vectors: vectors,
      metadata: Map.get(map, :metadata)
    }
  end

  defp to_embedding(%{value: value, vector: vector} = map) do
    %Embedding{
      id: nil,
      value: value,
      vector: vector,
      vectors: Map.get(map, :vectors),
      metadata: Map.get(map, :metadata)
    }
  end

  defp to_embedding(%{value: value, vectors: vectors} = map) do
    %Embedding{
      id: nil,
      value: value,
      vector: nil,
      vectors: vectors,
      metadata: Map.get(map, :metadata)
    }
  end

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
      Enum.all?(vector, &is_number/1) -> :ok
      true -> {:error, :invalid_vector}
    end
  end

  defp validate_vector(_vector, _dimensions), do: {:error, :invalid_vector}

  @spec store_module(:ets | module() | term()) :: {:ok, module()} | {:error, :invalid_store}
  defp store_module(:ets), do: {:ok, Vettore.Store.ETS}
  defp store_module(module) when is_atom(module), do: {:ok, module}
  defp store_module(_store), do: {:error, :invalid_store}

  @spec index_module(:flat | :hnsw | module() | term()) ::
          {:ok, module()} | {:error, :invalid_index}
  defp index_module(:flat), do: {:ok, Vettore.Index.Flat}
  defp index_module(:hnsw), do: {:ok, Vettore.Index.HNSW}
  defp index_module(module) when is_atom(module), do: {:ok, module}
  defp index_module(_index), do: {:error, :invalid_index}

  @spec normalize_metric(atom()) :: atom()
  defp normalize_metric(:euclidean), do: :l2
  defp normalize_metric(:dot), do: :inner_product
  defp normalize_metric(:dot_product), do: :inner_product
  defp normalize_metric(metric), do: metric

  @spec default_normalize(atom()) :: :l2 | :none
  defp default_normalize(:cosine), do: :l2
  defp default_normalize(_metric), do: :none
end
