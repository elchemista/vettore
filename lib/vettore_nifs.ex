defmodule Vettore.Nifs do
  @moduledoc false
  # All heavy work lives in the Rust crate `vettore` (see /native/vettore/src/…).
  # This file only declares *stubs* so the BEAM can load the NIF at runtime.
  #
  # Naming scheme:
  #
  #   • Functions whose names **match** the Rust code (`new_db/0`, `insert_embedding/5`, …)
  #     are the real NIFs.
  #   • Thin Elixir wrappers (e.g. `new/0`) delegate to those NIFs so the public
  #     Elixir API stays tidy and backward-compatible.

  use Rustler, otp_app: :vettore, crate: "vettore"
  # version = Mix.Project.config()[:version]

  # use RustlerPrecompiled,
  #   otp_app: :vettore,
  #   crate: "vettore",
  #   base_url: "https://github.com/elchemista/vettore/releases/download/v#{version}",
  #   force_build: System.get_env("RUSTLER_PRECOMPILATION_EXAMPLE_BUILD") in ["1", "true"],
  #   version: version

  @doc false
  @spec new_db() :: reference()
  def new_db, do: :erlang.nif_error(:nif_not_loaded)

  #  Collection management

  @doc false
  @spec create_collection(
          reference(),
          String.t(),
          pos_integer(),
          String.t(),
          boolean()
        ) :: {:ok, String.t()} | {:error, String.t()}
  def create_collection(_db, _name, _dim, _dist, _keep?),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec delete_collection(reference(), String.t()) ::
          {:ok, String.t()} | {:error, String.t()}
  def delete_collection(_db, _name),
    do: :erlang.nif_error(:nif_not_loaded)

  #  Embedding CRUD────────────
  @doc false
  @spec insert_embedding(
          reference(),
          String.t(),
          String.t(),
          [number()],
          map() | nil
        ) ::
          {:ok, String.t()} | {:error, String.t()}
  def insert_embedding(_db, _col, _id, _vec, _meta \\ nil),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec insert_embeddings(
          reference(),
          String.t(),
          [{String.t(), [number()], map() | nil}]
        ) ::
          {:ok, [String.t()]} | {:error, String.t()}
  def insert_embeddings(_db, _col, _embeddings),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec get_embedding_by_value(
          reference(),
          String.t(),
          String.t()
        ) ::
          {:ok, {String.t(), [number()], map() | nil}} | {:error, String.t()}
  def get_embedding_by_value(_db, _col, _id),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec get_embedding_by_vector(
          reference(),
          String.t(),
          [number()]
        ) ::
          {:ok, {String.t(), [number()], map() | nil}} | {:error, String.t()}
  def get_embedding_by_vector(_db, _col, _vec),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec get_all_embeddings(reference(), String.t()) ::
          {:ok, [{String.t(), [number()], map() | nil}]} | {:error, String.t()}
  def get_all_embeddings(_db, _col),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec delete_embedding_by_value(reference(), String.t(), String.t()) ::
          {:ok, String.t()} | {:error, String.t()}
  def delete_embedding_by_value(_db, _col, _id),
    do: :erlang.nif_error(:nif_not_loaded)

  #  Search & filtering
  @doc false
  @spec similarity_search(reference(), String.t(), [number()], pos_integer()) ::
          {:ok, [{String.t(), float()}]} | {:error, String.t()}
  def similarity_search(_db, _col, _q, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec similarity_search_with_filter(
          reference(),
          String.t(),
          [number()],
          pos_integer(),
          map()
        ) ::
          {:ok, [{String.t(), float()}]} | {:error, String.t()}
  def similarity_search_with_filter(_db, _col, _q, _k, _filter),
    do: :erlang.nif_error(:nif_not_loaded)

  #  MMR re-rank (collection-aware)─────
  @doc false
  @spec mmr_rerank(reference(), String.t(), [{String.t(), float()}], float(), pos_integer()) ::
          {:ok, [{String.t(), float()}]} | {:error, String.t()}
  def mmr_rerank(_db, _col, _init, _alpha, _k),
    do: :erlang.nif_error(:nif_not_loaded)

  #  Stand-alone distance helpers (collection-agnostic)
  @doc false
  @spec euclidean_distance([float()], [float()]) :: float()
  def euclidean_distance(_a, _b), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec cosine_similarity([float()], [float()]) :: float()
  def cosine_similarity(_a, _b), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec dot_product([float()], [float()]) :: float()
  def dot_product(_a, _b), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec hamming_distance_bits([integer()], [integer()]) :: float()
  def hamming_distance_bits(_a, _b), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec compress_f32_vector([float()]) :: [integer()]
  def compress_f32_vector(_vec), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  @spec mmr_rerank_embeddings(
          [{String.t(), float()}],
          [{String.t(), [number()]}],
          String.t(),
          float(),
          pos_integer()
        ) ::
          {:ok, [{String.t(), float()}]} | {:error, String.t()}
  def mmr_rerank_embeddings(_init, _embeds, _dist, _alpha, _k),
    do: :erlang.nif_error(:nif_not_loaded)
end
