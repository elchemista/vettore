defmodule Vettore do
  @moduledoc """
    The Vettore library is designed for fast, in-memory operations on vector (embedding) data.

    **All vectors (embeddings) are stored in a Rust data structure** (a `HashMap`), accessed via a shared resource
    (using Rustler’s `ResourceArc` with a `Mutex`). Core operations include:

      - Creating a collection :
        A named set of embeddings with a fixed dimension and a chosen similarity metric (:cosine, :euclidean, :dot,
        :hnsw, :binary).

      - Inserting an embedding :
        Add a new embedding (with ID, vector, and optional metadata) to a specific collection.

      - Retrieving embeddings :
        Fetch all embeddings from a collection or look up a single embedding by its unique ID.

      - Similarity search :
        Given a query vector, calculate a “score” for every embedding in the collection and return the top‑k results
        (e.g. the smallest distances or largest similarities).

      # Usage Example

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

  alias Vettore.{Nifs, Embedding, Validator}

  import Vettore.Validator,
    only: [is_db: 1, is_col: 1, is_id: 1, is_vec: 1, is_embedding: 2]

  @allowed_metrics ~w(euclidean cosine dot hnsw binary)a

  @doc """
  Allocate an **empty in‑memory DB** (owned by Rust).  Keep the returned
  reference around – every other call expects it.
  """
  @spec new() :: reference()
  def new, do: Nifs.new_db()

  @doc """
  Create a collection.

  * `distance` must be one of the atoms: `:euclidean`, `:cosine`, `:dot`,
    `:hnsw`, or `:binary`.

  # Examples

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
  def create_collection(db, name, dim, dist, opts \\ [])

  def create_collection(db, name, dim, dist, opts)
      when is_db(db) and is_col(name) and dim > 0 and
             is_atom(dist) and dist in @allowed_metrics and is_list(opts) do
    keep? = Keyword.get(opts, :keep_embeddings, true)

    if is_boolean(keep?) do
      Nifs.create_collection(db, name, dim, Atom.to_string(dist), keep?)
    else
      {:error, "invalid arguments (keep_embeddings need to be a boolean)"}
    end
  end

  def create_collection(_, _, _, _, _),
    do: {:error, "invalid arguments (see @doc for correct types)"}

  @doc """
  Delete a collection.

  # Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.delete_collection("my_collection")
      {:ok, "my_collection"}
  """
  @spec delete_collection(reference(), String.t()) ::
          {:ok, String.t()} | {:error, String.t()}
  def delete_collection(db, name)
      when is_db(db) and is_col(name) do
    Nifs.delete_collection(db, name)
  end

  def delete_collection(_, _),
    do: {:error, "invalid db reference or collection name"}

  @doc """
  Insert **one** `%Vettore.Embedding{}` into the collection.
  Returns `{:ok, value}` on success or `{:error, reason}`.

  # Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.insert("my_collection", %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}})
      {:ok, "my_id"}
  """
  @spec insert(reference(), String.t(), Embedding.t()) :: {:ok, String.t()} | {:error, String.t()}
  def insert(db, col, %Embedding{value: val, vector: vec, metadata: meta})
      when is_db(db) and is_col(col) and is_embedding(val, vec) do
    with :ok <- Validator.numeric?(vec) |> ok?("vector must be numeric"),
         {:ok, clean_m} <- Validator.sanitize_meta(meta) do
      Nifs.insert_embedding(db, col, val, vec, clean_m)
    end
  end

  def insert(_, _, _), do: {:error, "invalid arguments to insert/3"}

  @doc """
  Insert! **one** `%Vettore.Embedding{}` into the collection.
  Raise error if the vector is not numeric or the metadata is invalid. Otherwise, return the inserted value.

  # Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.insert!("my_collection", %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => %{"hello" => "world"}}})
      ArgumentError[…]
  """
  @spec insert!(reference(), String.t(), Embedding.t()) ::
          {:ok, String.t()} | {:error, String.t()}
  def insert!(db, col, %Embedding{value: val, vector: vec, metadata: meta})
      when is_db(db) and is_col(col) and is_embedding(val, vec) do
    if Validator.numeric?(vec) do
      Nifs.insert_embedding(db, col, val, vec, Validator.sanitize_meta!(meta))
    else
      raise ArgumentError, "vector must be numeric"
    end
  end

  def insert!(_, _, _), do: raise(ArgumentError, "invalid arguments to insert/3")

  @doc """
  Batch‑insert a list of embeddings. Reject elements that are not `%Vettore.Embedding{}`.
  Batch is faster than `insert/3` for a large number of embeddings as it avoids to validate vector list on each embedding.

  # Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.batch("my_collection", [%Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}])
      {:ok, ["my_id"]}
  """
  @spec batch(reference(), String.t(), [Embedding.t()]) ::
          {:ok, [String.t()]} | {:error, String.t()}
  def batch(db, col, embs) when is_db(db) and is_col(col) and is_list(embs) do
    with {:ok, tuples} <- Validator.embeddings_to_tuples(embs) do
      Nifs.insert_embeddings(db, col, tuples)
    end
  end

  def batch(_, _, _), do: {:error, "embeddings must be a list"}

  @doc """
  Batch‑insert a list of embeddings. Reject elements that are not `%Vettore.Embedding{}`.
  Raise error if the metadata is invalid. Otherwise, return the list of inserted as in `batch/3`.

  # Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.batch("my_collection", [%Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => [1, 2, 4]}}])
      ArgumentError[…]
  """

  @spec batch!(reference(), String.t(), [Embedding.t()]) ::
          {:ok, [String.t()]} | {:error, String.t()}
  def batch!(db, col, embs) when is_db(db) and is_col(col) and is_list(embs) do
    with {:ok, tuples} <- Validator.embeddings_to_tuples!(embs) do
      Nifs.insert_embeddings(db, col, tuples)
    end
  end

  def batch!(_, _, _), do: raise(ArgumentError, "embeddings must be a list")

  @doc """
  Fetch a single embedding by *value (ID)* and return it as `%Vettore.Embedding{}`.

  # Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.insert("my_collection", %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}) |> Vettore.get_by_value("my_collection", "my_id")
      {:ok, %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}}
  """
  @spec get_by_value(reference(), String.t(), String.t()) ::
          {:ok, Embedding.t()} | {:error, String.t()}
  def get_by_value(db, col, val)
      when is_db(db) and is_col(col) and is_id(val) do
    with {:ok, {value, vec, meta}} <- Nifs.get_embedding_by_value(db, col, val) do
      {:ok, %Embedding{value: value, vector: vec, metadata: meta}}
    end
  end

  def get_by_value(_, _, _), do: {:error, "invalid arguments to get_by_value/3"}

  @doc """
  Fetch a single embedding by *vector* and return it as `%Vettore.Embedding{}`.

  # Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.insert("my_collection", %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}) |> Vettore.get_by_vector("my_collection", [1.0, 2.0, 3.0])
      {:ok, %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}}
  """
  @spec get_by_vector(reference(), String.t(), [number()]) ::
          {:ok, Embedding.t()} | {:error, String.t()}
  def get_by_vector(db, col, vector)
      when is_db(db) and is_col(col) and is_vec(vector) do
    with {:ok, {value, vec, meta}} <- Nifs.get_embedding_by_vector(db, col, vector) do
      {:ok, %Embedding{value: value, vector: vec, metadata: meta}}
    end
  end

  def get_by_vector(_, _, _), do: {:error, "invalid arguments to get_by_vector/3"}

  @doc """
  Delete a single embedding.

  # Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.insert("my_collection", %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}) |> Vettore.delete("my_collection", "my_id")
      {:ok, "my_id"}
  """
  @spec delete(reference(), String.t(), String.t()) :: {:ok, String.t()} | {:error, String.t()}
  def delete(db, col, id)
      when is_db(db) and is_col(col) and is_id(id),
      do: Nifs.delete_embedding_by_value(db, col, id)

  def delete(_, _, _), do: {:error, "invalid arguments to delete/3"}

  @doc """
  Return all embeddings in *raw* form (`{value, vector, metadata}` tuples).

  # Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.insert("my_collection", %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}) |> Vettore.get_all("my_collection")
      {:ok, [{"my_id", [1.0, 2.0, 3.0], %{"note" => "hello"}}]}
  """
  @spec get_all(reference(), String.t()) ::
          {:ok, [{String.t(), [number()], map() | nil}]} | {:error, String.t()}
  def get_all(db, col)
      when is_db(db) and is_col(col),
      do: Nifs.get_all_embeddings(db, col)

  def get_all(_, _), do: {:error, "invalid arguments to get_all/2"}

  @doc """
  Similarity / nearest‑neighbour search.

  Options:
    * `:limit`  – number of results (default **10**)
    * `:filter` – metadata map; only embeddings whose metadata contains all
      key‑value pairs are considered.

  # Examples

      iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.insert("my_collection", %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}) |> Vettore.similarity_search("my_collection", [1.0, 2.0, 3.0], limit: 1)
      {:ok, [{"my_id", 0.0}]}
  """
  @spec similarity_search(reference(), String.t(), [number()], keyword()) ::
          {:ok, [{String.t(), float()}]} | {:error, String.t()}
  def similarity_search(db, col, query, opts \\ [])

  def similarity_search(db, col, query, opts)
      when is_db(db) and is_col(col) and
             is_vec(query) and is_list(opts) do
    limit = Keyword.get(opts, :limit, 10)
    filter = Keyword.get(opts, :filter, nil)

    with :ok <- validate_limit(limit),
         :ok <- validate_filter(filter) do
      case filter do
        nil -> Nifs.similarity_search(db, col, query, limit)
        f -> Nifs.similarity_search_with_filter(db, col, query, limit, f)
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

   # Examples

       iex> Vettore.new() |> Vettore.create_collection("my_collection", 3, :euclidean) |> Vettore.insert("my_collection", %Vettore.Embedding{value: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}) |> Vettore.insert("my_collection", %Vettore.Embedding{value: "my_id2", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}) |> Vettore.insert("my_collection", %Vettore.Embedding{value: "my_id3", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}) |> Vettore.rerank("my_collection", [{"my_id", 0.0}, {"my_id2", 0.0}, {"my_id3", 0.0}], limit: 1)
       {:ok, [{"my_id", 0.0}]}
  """

  @spec rerank(reference(), String.t(), [{String.t(), number()}], keyword()) ::
          {:ok, [{String.t(), number()}]} | {:error, String.t()}
  def rerank(db, col, initial, opts \\ [])

  def rerank(db, col, initial, opts)
      when is_db(db) and is_col(col) and is_list(initial) and is_list(opts) do
    # Validate initial list format
    limit = Keyword.get(opts, :limit, 10)
    alpha = Keyword.get(opts, :alpha, 0.5)

    with :ok <- validate_limit(limit),
         :ok <- validate_alpha(alpha),
         :ok <-
           Enum.all?(initial, &match?({i, s} when is_binary(i) and is_number(s), &1))
           |> ok?("initial list format") do
      Nifs.mmr_rerank(db, col, initial, alpha, limit)
    end
  end

  def rerank(_, _, _, _), do: {:error, "invalid arguments to rerank/4"}

  # Internal micro-validators

  defp validate_limit(n) when is_integer(n) and n > 0, do: :ok
  defp validate_limit(_), do: {:error, ":limit must be a positive integer"}

  defp validate_alpha(a) when is_number(a) and a >= 0 and a <= 1, do: :ok
  defp validate_alpha(_), do: {:error, ":alpha must be between 0.0 and 1.0"}

  defp validate_filter(nil), do: :ok
  defp validate_filter(f) when is_map(f), do: :ok
  defp validate_filter(_), do: {:error, ":filter must be a map"}

  defp ok?(true, _), do: :ok
  defp ok?(false, m), do: {:error, m}
end
