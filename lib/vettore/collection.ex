defmodule Vettore.Collection do
  @moduledoc """
  vNext collection API backed by ETS-owned storage.
  """

  alias Vettore.{Distance, Embedding}

  @type t :: %__MODULE__{
          name: atom() | String.t(),
          dimensions: pos_integer(),
          metric: atom(),
          normalize: atom(),
          score: atom(),
          store_mod: module(),
          store_state: term(),
          index_mod: module(),
          index: atom()
        }

  defstruct [
    :name,
    :dimensions,
    :metric,
    :normalize,
    :score,
    :store_mod,
    :store_state,
    :index_mod,
    :index
  ]

  @metrics ~w(l2 l2_squared cosine inner_product negative_inner_product manhattan chebyshev hamming jaccard)a

  @doc """
  Creates a collection.
  """
  @spec new(keyword()) :: {:ok, t()} | {:error, term()}
  def new(opts) when is_list(opts) do
    metric = normalize_metric(Keyword.get(opts, :metric, :cosine))
    dimensions = Keyword.get(opts, :dimensions)
    normalize = Keyword.get(opts, :normalize, default_normalize(metric))
    store = Keyword.get(opts, :store, :ets)
    index = Keyword.get(opts, :index, :flat)
    score = Keyword.get(opts, :score, :raw)

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
             index: index
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
         index: index
       }}
    end
  end

  @doc """
  Inserts or replaces one embedding by id.
  """
  @spec put(t(), Embedding.t() | map()) :: :ok | {:error, term()}
  def put(%__MODULE__{} = collection, embedding) do
    with {:ok, embedding} <- prepare_embedding(collection, embedding) do
      collection.store_mod.put(collection.store_state, embedding)
    end
  end

  @doc """
  Inserts or replaces many embeddings by id.
  """
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
      {:error, reason} -> {:error, reason}
      prepared -> collection.store_mod.put_many(collection.store_state, Enum.reverse(prepared))
    end
  end

  @doc """
  Fetches one embedding by id.
  """
  def get(%__MODULE__{} = collection, id) when is_binary(id) do
    collection.store_mod.get(collection.store_state, id)
  end

  @doc """
  Deletes one embedding by id.
  """
  def delete(%__MODULE__{} = collection, id) when is_binary(id) do
    collection.store_mod.delete(collection.store_state, id)
  end

  @doc """
  Returns all embeddings from the canonical store.
  """
  def all(%__MODULE__{} = collection), do: collection.store_mod.all(collection.store_state)

  @doc """
  Searches the collection.
  """
  def search(%__MODULE__{} = collection, query, opts \\ []) do
    collection.index_mod.search(collection, query, opts)
  end

  def prepare_query(%__MODULE__{} = collection, query) do
    with :ok <- validate_vector(query, collection.dimensions),
         {:ok, vector} <- Distance.normalize(query, collection.normalize) do
      {:ok, vector}
    end
  end

  defp prepare_embedding(%__MODULE__{} = collection, embedding) do
    embedding = to_embedding(embedding)

    with {:ok, id} <- embedding_id(embedding),
         :ok <- validate_vector(embedding.vector, collection.dimensions),
         {:ok, vector} <- Distance.normalize(embedding.vector, collection.normalize) do
      {:ok, %Embedding{embedding | id: id, value: embedding.value || id, vector: vector}}
    end
  end

  defp to_embedding(%Embedding{} = embedding), do: embedding

  defp to_embedding(%{id: id, vector: vector} = map) do
    %Embedding{
      id: id,
      value: Map.get(map, :value, id),
      vector: vector,
      metadata: Map.get(map, :metadata)
    }
  end

  defp to_embedding(%{value: value, vector: vector} = map) do
    %Embedding{
      id: nil,
      value: value,
      vector: vector,
      metadata: Map.get(map, :metadata)
    }
  end

  defp embedding_id(%Embedding{id: id}) when is_binary(id) and id != "", do: {:ok, id}

  defp embedding_id(%Embedding{value: value}) when is_binary(value) and value != "",
    do: {:ok, value}

  defp embedding_id(_embedding), do: {:error, :missing_id}

  defp validate_dimensions(dimensions) when is_integer(dimensions) and dimensions > 0, do: :ok
  defp validate_dimensions(_dimensions), do: {:error, :invalid_dimensions}

  defp validate_metric(metric) when metric in @metrics, do: :ok
  defp validate_metric(_metric), do: {:error, :invalid_metric}

  defp validate_vector(vector, dimensions) when is_list(vector) do
    cond do
      length(vector) != dimensions -> {:error, :dimension_mismatch}
      Enum.all?(vector, &is_number/1) -> :ok
      true -> {:error, :invalid_vector}
    end
  end

  defp validate_vector(_vector, _dimensions), do: {:error, :invalid_vector}

  defp store_module(:ets), do: {:ok, Vettore.Store.ETS}
  defp store_module(module) when is_atom(module), do: {:ok, module}
  defp store_module(_store), do: {:error, :invalid_store}

  defp index_module(:flat), do: {:ok, Vettore.Index.Flat}
  defp index_module(:hnsw), do: {:ok, Vettore.Index.HNSW}
  defp index_module(module) when is_atom(module), do: {:ok, module}
  defp index_module(_index), do: {:error, :invalid_index}

  defp normalize_metric(:euclidean), do: :l2
  defp normalize_metric(:dot), do: :inner_product
  defp normalize_metric(:dot_product), do: :inner_product
  defp normalize_metric(metric), do: metric

  defp default_normalize(:cosine), do: :l2
  defp default_normalize(_metric), do: :none
end
