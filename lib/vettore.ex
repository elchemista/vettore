defmodule Vettore do
  @moduledoc """
  Vettore vNext.

  The primary API is `Vettore.Collection`, where ETS owns collection state and
  native code is reserved for optional acceleration. This module keeps a small
  compatibility surface for older `Vettore.new/create_collection/insert/search`
  workflows, implemented on top of the new ETS collections.
  """

  alias Vettore.{Collection, Distance, Embedding}

  @doc """
  Creates a lightweight ETS-backed compatibility database.

  ## Examples

      iex> db = Vettore.new()
      iex> match?(%Vettore.DB{}, db)
      true
  """
  @spec new() :: Vettore.DB.t()
  def new do
    table =
      :ets.new(:vettore_db, [
        :set,
        :public,
        read_concurrency: true,
        write_concurrency: true
      ])

    %Vettore.DB{table: table}
  end

  @doc """
  Compatibility collection creation.

  ## Examples

      iex> db = Vettore.new()
      iex> Vettore.create_collection(db, "docs", 2, :cosine)
      {:ok, "docs"}

      iex> Vettore.create_collection(:bad_db, "docs", 2, :cosine)
      {:error, :invalid_arguments}
  """
  @spec create_collection(Vettore.DB.t(), String.t(), pos_integer(), atom(), keyword()) ::
          {:ok, String.t()} | {:error, term()}
  def create_collection(db, name, dimensions, metric, opts \\ [])

  def create_collection(%Vettore.DB{} = db, name, dimensions, metric, opts)
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

  @doc """
  Deletes a compatibility collection.

  ## Examples

      iex> db = Vettore.new()
      iex> {:ok, "docs"} = Vettore.create_collection(db, "docs", 2, :cosine)
      iex> Vettore.delete_collection(db, "docs")
      {:ok, "docs"}

      iex> Vettore.delete_collection(:bad_db, "docs")
      {:error, :invalid_arguments}
  """
  @spec delete_collection(Vettore.DB.t(), String.t()) ::
          {:ok, String.t()} | {:error, :invalid_arguments}
  def delete_collection(%Vettore.DB{} = db, name) when is_binary(name) do
    true = :ets.delete(db.table, {:collection, name})
    {:ok, name}
  end

  def delete_collection(_db, _name), do: {:error, :invalid_arguments}

  @doc """
  Inserts one embedding through the compatibility API.

  ## Examples

      iex> db = Vettore.new()
      iex> {:ok, "docs"} = Vettore.create_collection(db, "docs", 2, :cosine)
      iex> embedding = %Vettore.Embedding{id: "a", vector: [1.0, 0.0]}
      iex> Vettore.insert(db, "docs", embedding)
      {:ok, "a"}

      iex> Vettore.insert(:bad_db, "docs", %Vettore.Embedding{id: "a", vector: [1.0, 0.0]})
      {:error, :invalid_arguments}
  """
  @spec insert(Vettore.DB.t(), String.t(), Embedding.t()) ::
          {:ok, String.t() | nil} | {:error, term()}
  def insert(%Vettore.DB{} = db, collection_name, %Embedding{} = embedding)
      when is_binary(collection_name) do
    with {:ok, collection} <- fetch_collection(db, collection_name),
         :ok <- Collection.put(collection, embedding) do
      {:ok, embedding.id || embedding.value}
    end
  end

  def insert(_db, _collection_name, _embedding), do: {:error, :invalid_arguments}

  @doc """
  Inserts many embeddings through the compatibility API.

  ## Examples

      iex> db = Vettore.new()
      iex> {:ok, "docs"} = Vettore.create_collection(db, "docs", 2, :l2)
      iex> embeddings = [
      ...>   %Vettore.Embedding{id: "a", vector: [0.0, 0.0]},
      ...>   %Vettore.Embedding{id: "b", vector: [1.0, 1.0]}
      ...> ]
      iex> Vettore.batch(db, "docs", embeddings)
      {:ok, ["a", "b"]}

      iex> Vettore.batch(:bad_db, "docs", embeddings)
      {:error, :invalid_arguments}
  """
  @spec batch(Vettore.DB.t(), String.t(), [Embedding.t()]) ::
          {:ok, [String.t() | nil]} | {:error, term()}
  def batch(%Vettore.DB{} = db, collection_name, embeddings)
      when is_binary(collection_name) and is_list(embeddings) do
    with {:ok, collection} <- fetch_collection(db, collection_name),
         :ok <- Collection.put_many(collection, embeddings) do
      {:ok, Enum.map(embeddings, &(&1.id || &1.value))}
    end
  end

  def batch(_db, _collection_name, _embeddings), do: {:error, :invalid_arguments}

  @doc """
  Fetches one embedding by id through the compatibility API.

  ## Examples

      iex> db = Vettore.new()
      iex> {:ok, "docs"} = Vettore.create_collection(db, "docs", 2, :cosine)
      iex> {:ok, "a"} = Vettore.insert(db, "docs", %Vettore.Embedding{id: "a", vector: [1.0, 0.0]})
      iex> {:ok, %Vettore.Embedding{id: "a"}} = Vettore.get_by_value(db, "docs", "a")

      iex> Vettore.get_by_value(db, "docs", "missing")
      {:error, :not_found}
  """
  @spec get_by_value(Vettore.DB.t(), String.t(), String.t()) ::
          {:ok, Embedding.t()} | {:error, term()}
  def get_by_value(%Vettore.DB{} = db, collection_name, id)
      when is_binary(collection_name) and is_binary(id) do
    with {:ok, collection} <- fetch_collection(db, collection_name) do
      Collection.get(collection, id)
    end
  end

  def get_by_value(_db, _collection_name, _id), do: {:error, :invalid_arguments}

  @doc """
  Fetches the first embedding matching a normalized vector.

  ## Examples

      iex> db = Vettore.new()
      iex> {:ok, "docs"} = Vettore.create_collection(db, "docs", 2, :cosine)
      iex> {:ok, "a"} = Vettore.insert(db, "docs", %Vettore.Embedding{id: "a", vector: [1.0, 0.0]})
      iex> {:ok, %Vettore.Embedding{id: "a"}} = Vettore.get_by_vector(db, "docs", [1.0, 0.0])

      iex> Vettore.get_by_vector(db, "docs", [0.0, 1.0])
      {:error, :not_found}
  """
  @spec get_by_vector(Vettore.DB.t(), String.t(), [number()]) ::
          {:ok, Embedding.t()} | {:error, term()}
  def get_by_vector(%Vettore.DB{} = db, collection_name, vector)
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

  @doc """
  Deletes one embedding by id through the compatibility API.

  ## Examples

      iex> db = Vettore.new()
      iex> {:ok, "docs"} = Vettore.create_collection(db, "docs", 2, :l2)
      iex> {:ok, "a"} = Vettore.insert(db, "docs", %Vettore.Embedding{id: "a", vector: [0.0, 0.0]})
      iex> Vettore.delete(db, "docs", "a")
      {:ok, "a"}

      iex> Vettore.delete(:bad_db, "docs", "a")
      {:error, :invalid_arguments}
  """
  @spec delete(Vettore.DB.t(), String.t(), String.t()) ::
          {:ok, String.t()} | {:error, term()}
  def delete(%Vettore.DB{} = db, collection_name, id)
      when is_binary(collection_name) and is_binary(id) do
    with {:ok, collection} <- fetch_collection(db, collection_name),
         :ok <- Collection.delete(collection, id) do
      {:ok, id}
    end
  end

  def delete(_db, _collection_name, _id), do: {:error, :invalid_arguments}

  @doc """
  Returns all compatibility collection records as legacy tuples.

  ## Examples

      iex> db = Vettore.new()
      iex> {:ok, "docs"} = Vettore.create_collection(db, "docs", 2, :l2)
      iex> {:ok, "a"} = Vettore.insert(db, "docs", %Vettore.Embedding{id: "a", vector: [0.0, 0.0], metadata: %{kind: :origin}})
      iex> {:ok, [{"a", [0.0, 0.0], %{kind: :origin}}]} = Vettore.get_all(db, "docs")

      iex> Vettore.get_all(:bad_db, "docs")
      {:error, :invalid_arguments}
  """
  @spec get_all(Vettore.DB.t(), String.t()) ::
          {:ok, [{String.t(), [float()], map() | nil}]} | {:error, term()}
  def get_all(%Vettore.DB{} = db, collection_name) when is_binary(collection_name) do
    with {:ok, collection} <- fetch_collection(db, collection_name),
         {:ok, embeddings} <- Collection.all(collection) do
      {:ok, Enum.map(embeddings, &{&1.id, &1.vector, &1.metadata})}
    end
  end

  def get_all(_db, _collection_name), do: {:error, :invalid_arguments}

  @doc """
  Searches a compatibility collection.

  ## Examples

      iex> db = Vettore.new()
      iex> {:ok, "docs"} = Vettore.create_collection(db, "docs", 2, :cosine)
      iex> {:ok, "a"} = Vettore.insert(db, "docs", %Vettore.Embedding{id: "a", vector: [1.0, 0.0]})
      iex> {:ok, [{"a", score}]} = Vettore.similarity_search(db, "docs", [1.0, 0.0], limit: 1)
      iex> score
      1.0

      iex> Vettore.similarity_search(:bad_db, "docs", [1.0, 0.0])
      {:error, :invalid_arguments}
  """
  @spec similarity_search(Vettore.DB.t(), String.t(), [number()], keyword()) ::
          {:ok, [{String.t(), float()}]} | {:error, term()}
  def similarity_search(db, collection_name, query, opts \\ [])

  def similarity_search(%Vettore.DB{} = db, collection_name, query, opts)
      when is_binary(collection_name) and is_list(query) and is_list(opts) do
    with {:ok, collection} <- fetch_collection(db, collection_name),
         {:ok, results} <- Collection.search(collection, query, opts) do
      {:ok, Enum.map(results, &{&1.id, &1.score})}
    end
  end

  def similarity_search(_db, _collection_name, _query, _opts), do: {:error, :invalid_arguments}

  @doc """
  Applies MMR reranking to compatibility search results.

  ## Examples

      iex> db = Vettore.new()
      iex> {:ok, "docs"} = Vettore.create_collection(db, "docs", 2, :cosine)
      iex> {:ok, "a"} = Vettore.insert(db, "docs", %Vettore.Embedding{id: "a", vector: [1.0, 0.0]})
      iex> {:ok, "b"} = Vettore.insert(db, "docs", %Vettore.Embedding{id: "b", vector: [0.0, 1.0]})
      iex> Vettore.rerank(db, "docs", [{"a", 0.9}, {"b", 0.8}], limit: 1)
      {:ok, [{"a", 0.9}]}

      iex> Vettore.rerank(:bad_db, "docs", [{"a", 0.9}])
      {:error, :invalid_arguments}
  """
  @spec rerank(Vettore.DB.t(), String.t(), [{String.t(), float()}], keyword()) ::
          {:ok, [{String.t(), float()}]} | {:error, term()}
  def rerank(db, collection_name, initial, opts \\ [])

  def rerank(%Vettore.DB{} = db, collection_name, initial, opts)
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

  @spec fetch_collection(Vettore.DB.t(), String.t()) ::
          {:ok, Collection.t()} | {:error, :collection_not_found}
  defp fetch_collection(%Vettore.DB{} = db, name) do
    case :ets.lookup(db.table, {:collection, name}) do
      [{{:collection, ^name}, collection}] -> {:ok, collection}
      [] -> {:error, :collection_not_found}
    end
  end

  @spec normalize_metric(atom()) :: atom()
  defp normalize_metric(:euclidean), do: :l2
  defp normalize_metric(:binary), do: :hamming
  defp normalize_metric(:dot), do: :inner_product
  defp normalize_metric(:hnsw), do: :hnsw
  defp normalize_metric(metric), do: metric

  @spec default_normalize(atom()) :: :l2 | :none
  defp default_normalize(:cosine), do: :l2
  defp default_normalize(_metric), do: :none
end
