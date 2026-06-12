defmodule Vettore do
  @moduledoc """
  Vettore vNext.

  The primary API is `Vettore.Collection`, where ETS owns collection state and
  native code is reserved for optional acceleration. This module keeps a small
  compatibility surface for older `Vettore.new/create_collection/insert/search`
  workflows, implemented on top of the new ETS collections.
  """

  alias Vettore.{Collection, Distance, Embedding}

  defmodule DB do
    @moduledoc false
    defstruct [:table]
  end

  @doc """
  Creates a lightweight ETS-backed compatibility database.
  """
  @spec new() :: DB.t()
  def new do
    table =
      :ets.new(:vettore_db, [
        :set,
        :public,
        read_concurrency: true,
        write_concurrency: true
      ])

    %DB{table: table}
  end

  @doc """
  Compatibility collection creation.
  """
  def create_collection(db, name, dimensions, metric, opts \\ [])

  def create_collection(%DB{} = db, name, dimensions, metric, opts)
      when is_binary(name) and is_integer(dimensions) and dimensions > 0 do
    metric = normalize_metric(metric)
    index = Keyword.get(opts, :index, if(metric == :hnsw, do: :hnsw, else: :flat))
    metric = if metric == :hnsw, do: :l2, else: metric

    collection_opts = [
      name: name,
      dimensions: dimensions,
      metric: metric,
      index: index,
      store: Keyword.get(opts, :store, :ets),
      normalize: Keyword.get(opts, :normalize, default_normalize(metric)),
      score: Keyword.get(opts, :score, :similarity)
    ]

    with {:ok, collection} <- Collection.new(collection_opts) do
      true = :ets.insert(db.table, {{:collection, name}, collection})
      {:ok, name}
    end
  end

  def create_collection(_db, _name, _dimensions, _metric, _opts), do: {:error, :invalid_arguments}

  def delete_collection(%DB{} = db, name) when is_binary(name) do
    true = :ets.delete(db.table, {:collection, name})
    {:ok, name}
  end

  def delete_collection(_db, _name), do: {:error, :invalid_arguments}

  def insert(%DB{} = db, collection_name, %Embedding{} = embedding)
      when is_binary(collection_name) do
    with {:ok, collection} <- fetch_collection(db, collection_name),
         :ok <- Collection.put(collection, embedding) do
      {:ok, embedding.id || embedding.value}
    end
  end

  def insert(_db, _collection_name, _embedding), do: {:error, :invalid_arguments}

  def batch(%DB{} = db, collection_name, embeddings)
      when is_binary(collection_name) and is_list(embeddings) do
    with {:ok, collection} <- fetch_collection(db, collection_name),
         :ok <- Collection.put_many(collection, embeddings) do
      {:ok, Enum.map(embeddings, &(&1.id || &1.value))}
    end
  end

  def batch(_db, _collection_name, _embeddings), do: {:error, :invalid_arguments}

  def get_by_value(%DB{} = db, collection_name, id)
      when is_binary(collection_name) and is_binary(id) do
    with {:ok, collection} <- fetch_collection(db, collection_name) do
      Collection.get(collection, id)
    end
  end

  def get_by_value(_db, _collection_name, _id), do: {:error, :invalid_arguments}

  def get_by_vector(%DB{} = db, collection_name, vector)
      when is_binary(collection_name) and is_list(vector) do
    with {:ok, collection} <- fetch_collection(db, collection_name),
         {:ok, embeddings} <- Collection.all(collection),
         {:ok, prepared} <- Collection.prepare_query(collection, vector) do
      embeddings
      |> Enum.find(fn embedding -> embedding.vector == prepared end)
      |> case do
        nil -> {:error, :not_found}
        embedding -> {:ok, embedding}
      end
    end
  end

  def get_by_vector(_db, _collection_name, _vector), do: {:error, :invalid_arguments}

  def delete(%DB{} = db, collection_name, id) when is_binary(collection_name) and is_binary(id) do
    with {:ok, collection} <- fetch_collection(db, collection_name),
         :ok <- Collection.delete(collection, id) do
      {:ok, id}
    end
  end

  def delete(_db, _collection_name, _id), do: {:error, :invalid_arguments}

  def get_all(%DB{} = db, collection_name) when is_binary(collection_name) do
    with {:ok, collection} <- fetch_collection(db, collection_name),
         {:ok, embeddings} <- Collection.all(collection) do
      {:ok, Enum.map(embeddings, &{&1.id, &1.vector, &1.metadata})}
    end
  end

  def get_all(_db, _collection_name), do: {:error, :invalid_arguments}

  def similarity_search(db, collection_name, query, opts \\ [])

  def similarity_search(%DB{} = db, collection_name, query, opts)
      when is_binary(collection_name) and is_list(query) and is_list(opts) do
    with {:ok, collection} <- fetch_collection(db, collection_name),
         {:ok, results} <- Collection.search(collection, query, opts) do
      {:ok, Enum.map(results, &{&1.id, &1.score})}
    end
  end

  def similarity_search(_db, _collection_name, _query, _opts), do: {:error, :invalid_arguments}

  def rerank(db, collection_name, initial, opts \\ [])

  def rerank(%DB{} = db, collection_name, initial, opts)
      when is_binary(collection_name) and is_list(initial) and is_list(opts) do
    limit = Keyword.get(opts, :limit, 10)
    alpha = Keyword.get(opts, :alpha, 0.5)

    with {:ok, collection} <- fetch_collection(db, collection_name),
         {:ok, embeddings} <- Collection.all(collection) do
      pairs = Enum.map(embeddings, &{&1.id, &1.vector})
      Distance.mmr_rerank(initial, pairs, collection.metric, alpha, limit)
    end
  end

  def rerank(_db, _collection_name, _initial, _opts), do: {:error, :invalid_arguments}

  defp fetch_collection(%DB{} = db, name) do
    case :ets.lookup(db.table, {:collection, name}) do
      [{{:collection, ^name}, collection}] -> {:ok, collection}
      [] -> {:error, :collection_not_found}
    end
  end

  defp normalize_metric(:euclidean), do: :l2
  defp normalize_metric(:binary), do: :hamming
  defp normalize_metric(:dot), do: :inner_product
  defp normalize_metric(:hnsw), do: :hnsw
  defp normalize_metric(metric), do: metric

  defp default_normalize(:cosine), do: :l2
  defp default_normalize(_metric), do: :none
end
