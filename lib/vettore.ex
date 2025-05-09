defmodule Vettore do
  @moduledoc """
    The Vettore library is designed for fast, in-memory operations on vector (embedding) data.

    **All vectors (embeddings) are stored in a Rust data structure** (a `HashMap`), accessed via a shared resource
    (using Rustler’s `ResourceArc` with a `Mutex`). Core operations include:

      - **Creating a collection**:
        A named set of embeddings with a fixed dimension and a chosen similarity metric (:cosine, :euclidean, :dot,
        :hnsw, :binary).

      - **Inserting an embedding**:
        Add a new embedding (with ID, vector, and optional metadata) to a specific collection.

      - **Retrieving embeddings**:
        Fetch all embeddings from a collection or look up a single embedding by its unique ID.

      - **Similarity search**:
        Given a query vector, calculate a “score” for every embedding in the collection and return the top‑k results
        (e.g. the smallest distances or largest similarities).

    ## Usage Example

        db = Vettore.new()
        :ok = Vettore.create_collection(db, "my_collection", 3, :euclidean)

        # Insert an embedding via struct:
        embedding = %Vettore.Embedding{value: "my_id or text", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}
        :ok = Vettore.insert(db, "my_collection", embedding)

        # Retrieve it back:
        {:ok, returned_emb} = Vettore.get_by_value(db, "my_collection", "my_id")
        IO.inspect(returned_emb.vector, label: "Retrieved vector")

        # Perform a similarity search:
        {:ok, top_results} = Vettore.similarity_search(db, "my_collection", [1.5, 1.5, 1.5], 2)
        IO.inspect(top_results, label: "Top K search results")
  """

  alias Vettore.Nifs, as: N
  alias Vettore.Embedding

  @allowed_metrics ~w(euclidean cosine dot hnsw binary)a

  @doc """
  Allocate an **empty in‑memory DB** (owned by Rust).  Keep the returned
  reference around – every other call expects it.
  """
  @spec new() :: reference()
  def new, do: N.new_db()

  @doc """
  Create a collection.

  * `distance` must be one of the atoms: `:euclidean`, `:cosine`, `:dot`,
    `:hnsw`, or `:binary`.

  #Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean)
      {:ok, "my_collection"}
  """
  @spec create_collection(
          reference(),
          String.t(),
          pos_integer(),
          atom(),
          keyword()
        ) :: {:ok, String.t()} | {:error, String.t()}
  def create_collection(db, name, dimension, distance, opts \\ [])

  def create_collection(db, name, dimension, distance, opts)
      when is_reference(db) and
             is_bitstring(name) and byte_size(name) > 0 and
             is_integer(dimension) and dimension > 0 and
             is_atom(distance) and distance in @allowed_metrics and
             is_list(opts) do
    keep? = Keyword.get(opts, :keep_embeddings, true)
    N.create_collection(db, name, dimension, Atom.to_string(distance), keep?)
  end

  def create_collection(_, _, _, _, _),
    do: {:error, "invalid arguments (see @doc for correct types)"}

  @doc """
  Delete a collection.

  #Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.delete_collection("my_collection")
      {:ok, "my_collection"}
  """
  @spec delete_collection(reference(), String.t()) ::
          {:ok, String.t()} | {:error, String.t()}
  def delete_collection(db, name)
      when is_reference(db) and
             is_bitstring(name) do
    N.delete_collection(db, name)
  end

  def delete_collection(_, _),
    do: {:error, "invalid db reference or collection name"}

  @doc """
  Insert **one** `%Vettore.Embedding{}` into the collection.
  Returns `{:ok, value}` on success or `{:error, reason}`.

  #Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.insert("my_collection", %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}})
      {:ok, "my_id"}
  """
  @spec insert(reference(), String.t(), Embedding.t()) :: {:ok, String.t()} | {:error, String.t()}
  def insert(db, collection, %Embedding{value: value, vector: vec, metadata: meta})
      when is_reference(db) and is_bitstring(collection) and
             is_bitstring(value) and is_list(vec) do
    if Enum.all?(vec, &is_number/1) do
      N.insert_embedding(db, collection, value, vec, sanitize_meta(meta))
    else
      {:error, "vector must be a list of numbers"}
    end
  end

  def insert(_, _, _), do: {:error, "invalid arguments to insert/3"}

  @doc """
  Batch‑insert a list of embeddings. Reject elements that are not `%Vettore.Embedding{}`.

  #Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.batch("my_collection", [%Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}])
      {:ok, ["my_id"]}
  """
  @spec batch(reference(), String.t(), [Embedding.t()]) ::
          {:ok, [String.t()]} | {:error, String.t()}
  def batch(db, collection, embeddings) when is_list(embeddings) do
    with true <- is_reference(db),
         true <- is_bitstring(collection) and String.length(collection) > 0 do
      tuples = embeddings_to_tuples(embeddings)
      N.insert_embeddings(db, collection, tuples)
    else
      {:error, _} = err -> err
      _ -> {:error, "invalid db reference or collection name"}
    end
  end

  def batch(_, _, _), do: {:error, "embeddings must be a list"}

  @doc """
  Fetch a single embedding by *value (ID)* and return it as `%Vettore.Embedding{}`.

  #Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.insert("my_collection", %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}) |> Vettore.get_by_value("my_collection", "my_id")
      {:ok, %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}}
  """
  @spec get_by_value(reference(), String.t(), String.t()) ::
          {:ok, Embedding.t()} | {:error, String.t()}
  def get_by_value(db, collection, value)
      when is_reference(db) and is_bitstring(collection) and byte_size(collection) > 0 and
             is_bitstring(value) do
    with {:ok, {value, vec, meta}} <- N.get_embedding_by_value(db, collection, value) do
      {:ok, %Embedding{value: value, vector: vec, metadata: meta}}
    end
  end

  def get_by_value(_, _, _), do: {:error, "invalid arguments to get_by_value/3"}

  @doc """
  Fetch a single embedding by *vector* and return it as `%Vettore.Embedding{}`.

  #Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.insert("my_collection", %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}) |> Vettore.get_by_vector("my_collection", [1.0, 2.0, 3.0])
      {:ok, %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}}
  """
  @spec get_by_vector(reference(), String.t(), [number()]) ::
          {:ok, Embedding.t()} | {:error, String.t()}
  def get_by_vector(db, collection, vector)
      when is_reference(db) and is_bitstring(collection) and byte_size(collection) > 0 and
             is_list(vector) do
    with {:ok, {value, vec, meta}} <- N.get_embedding_by_vector(db, collection, vector) do
      {:ok, %Embedding{value: value, vector: vec, metadata: meta}}
    end
  end

  def get_by_vector(_, _, _), do: {:error, "invalid arguments to get_by_vector/3"}

  @doc """
  Delete a single embedding.

  #Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.insert("my_collection", %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}) |> Vettore.delete("my_collection", "my_id")
      {:ok, "my_id"}
  """
  @spec delete(reference(), String.t(), String.t()) :: {:ok, String.t()} | {:error, String.t()}
  def delete(db, collection, id)
      when is_reference(db) and is_bitstring(collection) and is_bitstring(id),
      do: N.delete_embedding_by_value(db, collection, id)

  def delete(_, _, _), do: {:error, "invalid arguments to delete/3"}

  @doc """
  Return all embeddings in *raw* form (`{value, vector, metadata}` tuples).

  #Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.insert("my_collection", %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}) |> Vettore.get_all("my_collection")
      {:ok, [{"my_id", [1.0, 2.0, 3.0], %{"note" => "hello"}}]}
  """
  @spec get_all(reference(), String.t()) ::
          {:ok, [{String.t(), [number()], map() | nil}]} | {:error, String.t()}
  def get_all(db, collection)
      when is_reference(db) and is_bitstring(collection),
      do: N.get_all_embeddings(db, collection)

  def get_all(_, _), do: {:error, "invalid arguments to get_all/2"}

  @doc """
  Similarity / nearest‑neighbour search.

  Options:
    * `:limit`  – number of results (default **10**)
    * `:filter` – metadata map; only embeddings whose metadata contains all
      key‑value pairs are considered.

  #Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.insert("my_collection", %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}) |> Vettore.similarity_search("my_collection", [1.0, 2.0, 3.0], limit: 1)
      {:ok, [{"my_id", 0.0}]}
  """
  @spec similarity_search(reference(), String.t(), [number()], keyword()) ::
          {:ok, [{String.t(), float()}]} | {:error, String.t()}
  def similarity_search(db, collection, query, opts \\ [])

  def similarity_search(db, collection, query, opts)
      when is_reference(db) and is_binary(collection) and byte_size(collection) > 0 and
             is_list(query) and is_list(opts) do
    limit = Keyword.get(opts, :limit, 10)
    filter = Keyword.get(opts, :filter)

    with :ok <- validate_limit(limit),
         :ok <- validate_filter(opts) do
      case filter do
        nil -> N.similarity_search(db, collection, query, limit)
        f -> N.similarity_search_with_filter(db, collection, query, limit, f)
      end
    else
      {:error, msg} -> {:error, msg}
    end
  end

  def similarity_search(_, _, _, _),
    do: {:error, "invalid arguments to similarity_search/4"}

  @doc """
   Re‑rank an existing result list with **Maximal Marginal Relevance**.

   Options:
     * `:limit` – desired output length (default **10**)
     * `:alpha` – relevance‑diversity balance **0.0..1.0** (default **0.5**)

   #Examples

       iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.insert("my_collection", %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}) |> Vettore.insert("my_collection", %Vettore.Embedding{value: "my_id2", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}) |> Vettore.insert("my_collection", %Vettore.Embedding{value: "my_id3", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}) |> Vettore.rerank("my_collection", [{"my_id", 0.0}, {"my_id2", 0.0}, {"my_id3", 0.0}], limit: 1)
       {:ok, [{"my_id", 0.0}]}
  """

  @spec rerank(reference(), String.t(), [{String.t(), number()}], keyword()) ::
          {:ok, [{String.t(), number()}]} | {:error, String.t()}
  def rerank(db, collection, initial, opts \\ [])

  def rerank(db, collection, initial, opts)
      when is_reference(db) and is_binary(collection) and is_list(initial) and is_list(opts) do
    # Validate initial list format
    if Enum.all?(initial, &valid_initial?/1) do
      limit = Keyword.get(opts, :limit, 10)
      alpha = Keyword.get(opts, :alpha, 0.5)

      # Validate limit and alpha
      with :ok <- validate_limit(limit),
           :ok <- validate_alpha(alpha) do
        N.mmr_rerank(db, collection, initial, alpha, limit)
      else
        {:error, msg} -> {:error, msg}
      end
    else
      {:error, "initial list must be [{String.t(), number()}]"}
    end
  end

  def rerank(_, _, _, _), do: {:error, "invalid arguments to rerank/4"}

  defp validate_limit(n) when is_integer(n) and n > 0, do: :ok
  defp validate_limit(_), do: {:error, ":limit must be a positive integer"}

  defp valid_initial?({id, score}) when is_binary(id) and is_number(score), do: true
  defp valid_initial?(_), do: false

  defp validate_alpha(a) when is_number(a) and a >= 0 and a <= 1, do: :ok
  defp validate_alpha(_), do: {:error, ":alpha must be between 0.0 and 1.0"}

  defp validate_filter(opts) do
    case Keyword.fetch(opts, :filter) do
      {:ok, f} when is_map(f) -> :ok
      {:ok, _} -> {:error, ":filter must be a map"}
      :error -> :ok
    end
  end

  #  Helpers (private)
  defp embeddings_to_tuples(list) do
    for %Embedding{value: v, vector: vec, metadata: m} <- list, is_bitstring(v), is_list(vec) do
      {v, vec, sanitize_meta(m)}
    end
  end

  defp sanitize_meta(nil), do: nil
  defp sanitize_meta(m) when is_map(m), do: m
  defp sanitize_meta(_), do: nil
end
