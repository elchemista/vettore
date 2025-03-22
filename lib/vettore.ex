defmodule Vettore.Embedding do
  @moduledoc """
  Represents a single embedding entry for insertion into a collection.

  ## Fields

    * `:id` - A unique string identifier for this embedding (e.g. "emb123").
    * `:vector` - A list of floating‑point numbers representing the embedding (e.g. `[1.0, 2.0, 3.0]`).
    * `:metadata` - (Optional) A map with any additional information you want to store
      (e.g. `%{"info" => "my note"}`).
  """
  defstruct [:id, :vector, :metadata]

  @type t :: %__MODULE__{
          id: String.t(),
          vector: [float()],
          metadata: map() | nil
        }
end

defmodule Vettore do
  @moduledoc """
  The Vettore library is designed for fast, in-memory operations on vector (embedding) data.

  **All vectors (embeddings) are stored in a Rust data structure** (a `HashMap`), accessed via a shared resource
  (using Rustler’s `ResourceArc` with a `Mutex`). Core operations include:

    - **Creating a collection**:
      A named set of embeddings with a fixed dimension and a chosen similarity metric (`"hnsw"`, `"binary"`,
      `"euclidean"`, `"cosine"`, or `"dot"`).

    - **Inserting an embedding**:
      Add a new embedding (with ID, vector, and optional metadata) to a specific collection.

    - **Retrieving embeddings**:
      Fetch all embeddings from a collection or look up a single embedding by its unique ID.

    - **Similarity search**:
      Given a query vector, calculate a “score” for every embedding in the collection and return the top‑k results
      (e.g. the smallest distances or largest similarities).

  ## Usage Example

      db = Vettore.new_db()
      :ok = Vettore.create_collection(db, "my_collection", 3, "euclidean")

      # Insert an embedding via struct:
      embedding = %Vettore.Embedding{id: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{"note" => "hello"}}
      :ok = Vettore.insert_embedding(db, "my_collection", embedding)

      # Retrieve it back:
      {:ok, returned_emb} = Vettore.get_embedding_by_id(db, "my_collection", "my_id")
      IO.inspect(returned_emb.vector, label: "Retrieved vector")

      # Perform a similarity search:
      {:ok, top_results} = Vettore.similarity_search(db, "my_collection", [1.5, 1.5, 1.5], 2)
      IO.inspect(top_results, label: "Top K search results")
  """

  # version = Mix.Project.config()[:version]

  # use RustlerPrecompiled,
  #   otp_app: :vettore,
  #   crate: "vettore",
  #   base_url: "https://github.com/elchemista/vettore/releases/download/v#{version}",
  #   force_build: System.get_env("RUSTLER_PRECOMPILATION_EXAMPLE_BUILD") in ["1", "true"],
  #   version: version

  use Rustler,
    otp_app: :vettore,
    crate: "vettore"

  alias Vettore.Embedding

  #
  # Public API
  #

  @doc """
  Returns a new **database resource** (wrapped in a Rustler `ResourceArc`).

  The database resource is a handle for the underlying Rust data structure.

  ## Examples

      is_reference(db)
      # => true
  """
  @spec new_db() :: any()
  def new_db(), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Creates a new collection in the database with the given `name`, `dimension`, and `distance` metric.

  * `db` is the database resource (created with `new_db/0`).
  * `name` is the name of the collection.
  * `dimension` is the number of dimensions in the vector.
  * `distance` can be one of: `"euclidean"`, `"cosine"`, `"dot"`, `"hnsw"`, or `"binary"`.

  Returns `{:ok, name}` on success, or `{:error, reason}` if the collection already exists or if the distance is invalid.

  ## Examples

    {:ok, "my_collection"} = Vettore.create_collection(db, "my_collection", 3, "euclidean")
  """
  @spec create_collection(any(), String.t(), integer(), String.t()) ::
          {:ok, String.t()} | {:error, String.t()}
  def create_collection(db, name, dim, distance, opts \\ []) do
    nif_create_collection(db, name, dim, distance, Keyword.get(opts, :keep_embeddings, true))
  end

  @doc """
  Deletes a collection by its `name`.

  * `db` is the database resource (created with `new_db/0`).
  * `name` is the name of the collection.

  Returns `{:ok, name}` if the collection was found and deleted, or `{:error, reason}` otherwise.

  ## Examples

      {:ok, "my_collection"} = Vettore.delete_collection(db, "my_collection")
  """
  @spec delete_collection(any(), String.t()) :: {:ok, String.t()} | {:error, String.t()}
  def delete_collection(_db, _name),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Inserts a **single** embedding (as a `%Vettore.Embedding{}` struct) into a specified `collection`.

  If the collection doesn't exist, you'll get `{:error, "Collection '...' not found"}`.
  If another embedding with the same `:id` is already in the collection, you’ll get an error.
  Also, if the `:vector` length does not match the collection's configured dimension,
  you’ll get a dimension mismatch error.

  ## Examples

      embedding = %Vettore.Embedding{id: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{foo: "bar"}}
      {:ok, "my_id"} = Vettore.insert_embedding(db, "my_collection", embedding)
  """
  @spec insert_embedding(any(), String.t(), Embedding.t()) ::
          {:ok, String.t()} | {:error, String.t()}
  def insert_embedding(db, collection, %Embedding{id: id, vector: vec, metadata: meta}) do
    nif_insert_embedding(db, collection, id, vec, meta)
  end

  @doc """
  Batch-inserts a **list** of embeddings (each a `%Vettore.Embedding{}`) into the specified `collection`.

  It returns `{:ok, ["id1", "id2", ...]}` if all embeddings inserted successfully.
  If **any** embedding fails (e.g., dimension mismatch, ID conflict, or missing collection),
  the function returns `{:error, reason}` and stops immediately (does not insert the rest).

  ## Examples

      embs = [
        %Vettore.Embedding{id: "e1", vector: [1.0, 2.0, 3.0], metadata: nil},
        %Vettore.Embedding{id: "e2", vector: [4.5, 6.7, 8.9], metadata: %{"info" => "test"}}
      ]
      {:ok, ["e1", "e2"]} = Vettore.insert_embeddings(db, "my_collection", embs)
  """
  @spec insert_embeddings(any(), String.t(), [Embedding.t()]) ::
          {:ok, [String.t()]} | {:error, String.t()}
  def insert_embeddings(db, collection, embeddings) when is_list(embeddings) do
    # Convert each %Embedding{} into the tuple form the Rust NIF expects
    native_list =
      Enum.map(embeddings, fn %Embedding{id: i, vector: v, metadata: m} ->
        {i, v, m}
      end)

    nif_insert_embeddings(db, collection, native_list)
  end

  @doc """
  Looks up a single embedding by its `id`, within the given `collection`.

  If found, returns `{:ok, %Vettore.Embedding{}}`.
  If not found, returns `{:error, reason}`.

  ## Examples

      Vettore.get_embedding_by_id(db, "my_collection", "my_id")
      # => {:ok, %Vettore.Embedding{id: "my_id", vector: [1.0, 2.0, 3.0], metadata: %{foo: "bar"}}}
  """
  @spec get_embedding_by_id(any(), String.t(), String.t()) ::
          {:ok, Embedding.t()} | {:error, String.t()}
  def get_embedding_by_id(db, collection_name, id) do
    case nif_get_embedding_by_id(db, collection_name, id) do
      {:ok, {raw_id, raw_vector, raw_meta}} ->
        {:ok, %Embedding{id: raw_id, vector: raw_vector, metadata: raw_meta}}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Returns **all** embeddings from a given collection.

  Each embedding is returned as `(id, vector, metadata)` in a list—if you want to convert each
  item into a `%Vettore.Embedding{}`, you can do so manually or provide a helper function.

  ## Examples

      Vettore.get_embeddings(db, "my_collection")
      # => {:ok, [
      #   {"emb1", [1.0, 2.0, 3.0], %{"info" => "test"}},
      #   {"emb2", [3.14, 2.71, 1.62], nil},
      # ]}
  """
  @spec get_embeddings(any(), String.t()) ::
          {:ok, [{String.t(), [float()], map() | nil}]} | {:error, String.t()}
  def get_embeddings(_db, _collection),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Similarity search with optional `limit` (defaults to 10) and optional `filter` map.

  performs a similarity (or distance) search in the given `collection` using the provided `query` vector, returning
  the **top-k** results as a list of `{embedding_id, score}` tuples.

  - For `"euclidean"`, lower scores are better (distance).
  - For `"cosine"`, higher scores are better (dot product).
  - For `"dot"`, also higher is better.
  - For `"binary"`, the score is the Hamming distance—lower is more similar.
  - For `"hnsw"`, an approximate nearest neighbors is used.

  Examples:
      Vettore.similarity_search(db, "my_collection", [1.0, 2.0, 3.0], limit: 2, filter: %{"category" => "test"})
      Vettore.similarity_search(db, "my_collection", [1.0, 2.0, 3.0], limit: 2)
      Vettore.similarity_search(db, "my_collection", [1.0, 2.0, 3.0])

       # => {:ok, [{"emb1", 0.0}, {"emb2", 1.23}]}
  """
  @spec similarity_search(any(), String.t(), [float()], keyword()) ::
          {:ok, [{String.t(), float()}]} | {:error, String.t()}
  def similarity_search(db, collection, query, opts \\ []) do
    k = Keyword.get(opts, :limit, 10)
    filter = Keyword.get(opts, :filter, nil)

    if is_nil(filter) do
      nif_similarity_search(db, collection, query, k)
    else
      nif_similarity_search_with_filter(db, collection, query, k, filter)
    end
  end

  @doc """
  Re-rank a list of `{id, score}` results using **Maximal Marginal Relevance** (MMR).

  Given a database resource `db`, a `collection` name, and an `initial_results` list of
  `{id, score}` tuples (usually obtained from `similarity_search/4`), this function applies
  an MMR formula to select up to `:limit` items that maximize both relevance (the `score`)
  and diversity among the selected items.

  The `alpha` parameter (0.0 to 1.0) balances relevance vs. redundancy:
    - `alpha` close to `1.0` → prioritizes the raw score (similarity to query).
    - `alpha` close to `0.0` → heavily penalizes items similar to already-selected ones,
      thus promoting diversity.

  We automatically convert Euclidean or Binary `distance` into a “higher is better” similarity
  by negating the distance (i.e. `similarity = -distance`). For Cosine, Dot Product, or HNSW
  approaches, the `score` is already in a higher-is-better format.

  Returns `{:ok, [{id, mmr_score}, ...]}` on success or `{:error, reason}` if the collection
  is not found.

  ## Examples

  After calling `similarity_search/4`:

      {:ok, initial_results} = Vettore.similarity_search(db, "my_collection", query_vec, limit: 50)

  You can re-rank:

      {:ok, mmr_list} =
        Vettore.mmr_rerank(db, "my_collection", initial_results,
          limit: 10,
          alpha: 0.7
        )

  `mmr_list` then gives a smaller set (up to 10 items) in MMR order, each with a new score
  (`mmr_score`) that reflects their final MMR weighting.

  """
  @spec mmr_rerank(
          db :: any(),
          collection_name :: String.t(),
          initial_results :: [{String.t(), float()}],
          opts :: keyword()
        ) :: {:ok, [{String.t(), float()}]} | {:error, String.t()}
  def mmr_rerank(db, collection, initial_results, opts \\ []) do
    unless is_list(initial_results) and
             Enum.all?(initial_results, fn
               {id, score} when is_binary(id) and is_float(score) -> true
               _ -> false
             end) do
      {:error, "initial_results must be a list of {String.t(), float()} tuples"}
    else
      final_k = Keyword.get(opts, :limit, 10)
      alpha = Keyword.get(opts, :alpha, 0.5)
      nif_mmr_rerank(db, collection, initial_results, alpha, final_k)
    end
  end

  #
  # Internal (NIF) function signatures:
  #  - These do the direct Rust calls using the "raw" data form (id, vector, metadata).
  #  - The "public" functions above wrap them in your Elixir API.
  #
  @doc false
  @spec nif_create_collection(any(), String.t(), integer(), String.t(), boolean()) ::
          {:ok, String.t()} | {:error, String.t()}
  def nif_create_collection(_db, _name, _dim, _distance, _keep_embeddings),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec nif_similarity_search(any(), String.t(), any(), integer()) :: any()
  def nif_similarity_search(_db, _collection, _query, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec nif_insert_embedding(any(), String.t(), String.t(), [float()], map() | nil) ::
          :ok | {:error, String.t()}
  def nif_insert_embedding(_db, _collection, _id, _vector, _metadata \\ nil),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec nif_mmr_rerank(any(), String.t(), list(), float(), integer()) :: any()
  def nif_mmr_rerank(_db, _collection, _initial_results, _alpha, _final_k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec nif_insert_embeddings(
          any(),
          String.t(),
          [{String.t(), [float()], map() | nil}]
        ) :: :ok | {:error, String.t()}
  def nif_insert_embeddings(_db, _collection, _embeddings),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec nif_get_embedding_by_id(any(), String.t(), String.t()) ::
          {:ok, {String.t(), [float()], map() | nil}} | {:error, String.t()}
  def nif_get_embedding_by_id(_db, _collection, _id),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec nif_similarity_search_with_filter(
          any(),
          String.t(),
          [float()],
          non_neg_integer(),
          map() | nil
        ) ::
          {:ok, [{String.t(), float()}]} | {:error, String.t()}
  def nif_similarity_search_with_filter(_db, _collection, _query, _k, _filter),
    do: :erlang.nif_error(:nif_not_loaded)
end
