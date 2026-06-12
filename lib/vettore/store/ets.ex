defmodule Vettore.Store.ETS do
  @moduledoc """
  ETS-backed canonical store for Vettore collections.
  """

  @behaviour Vettore.Store

  alias Vettore.Embedding

  defstruct [:table]

  @type t :: %__MODULE__{table: :ets.tid()}

  @spec new(map()) :: {:ok, t()}
  @impl true
  def new(config) when is_map(config) do
    table = :ets.new(:vettore_collection, table_options(config))

    true = :ets.insert(table, {:__config__, config})
    {:ok, %__MODULE__{table: table}}
  end

  @spec snapshot(t(), Path.t()) :: :ok | {:error, term()}
  @impl true
  def snapshot(%__MODULE__{table: table}, path) when is_binary(path) and path != "" do
    with :ok <- ensure_snapshot_directory(path) do
      :ets.tab2file(table, String.to_charlist(path))
    end
  end

  def snapshot(_state, _path), do: {:error, :invalid_snapshot_path}

  @spec load_snapshot(Path.t()) :: {:ok, {t(), map()}} | {:error, term()}
  @impl true
  def load_snapshot(path) when is_binary(path) and path != "" do
    case :ets.file2tab(String.to_charlist(path)) do
      {:ok, table} -> load_config_or_delete(table)
      {:error, reason} -> {:error, reason}
    end
  end

  def load_snapshot(_path), do: {:error, :invalid_snapshot_path}

  @spec put(t(), Embedding.t()) :: :ok | {:error, :duplicate_id | :missing_id}
  @impl true
  def put(%__MODULE__{table: table}, %Embedding{} = embedding) do
    with {:ok, id} <- embedding_id(embedding) do
      record = {{:record, id}, normalize_value(embedding, id)}

      case :ets.insert_new(table, record) do
        true -> :ok
        false -> {:error, :duplicate_id}
      end
    end
  end

  @spec put_many(t(), [Embedding.t()]) :: :ok | {:error, :duplicate_id | :missing_id}
  @impl true
  def put_many(%__MODULE__{} = state, embeddings) when is_list(embeddings) do
    result =
      Enum.reduce_while(embeddings, {[], MapSet.new()}, &collect_insert_row/2)

    case result do
      {:error, reason} ->
        {:error, reason}

      {rows, _ids} ->
        case :ets.insert_new(state.table, rows) do
          true -> :ok
          false -> {:error, :duplicate_id}
        end
    end
  end

  @spec get(t(), String.t()) :: {:ok, Embedding.t()} | {:error, :not_found}
  @impl true
  def get(%__MODULE__{table: table}, id) when is_binary(id) do
    case :ets.lookup(table, {:record, id}) do
      [{{:record, ^id}, %Embedding{} = embedding}] -> {:ok, embedding}
      [] -> {:error, :not_found}
    end
  end

  @spec delete(t(), String.t()) :: :ok
  @impl true
  def delete(%__MODULE__{table: table}, id) when is_binary(id) do
    true = :ets.delete(table, {:record, id})
    :ok
  end

  @spec all(t()) :: {:ok, [Embedding.t()]}
  @impl true
  def all(%__MODULE__{table: table}) do
    rows =
      table
      |> :ets.tab2list()
      |> Enum.flat_map(fn
        {{:record, _id}, %Embedding{} = embedding} -> [embedding]
        _other -> []
      end)

    {:ok, rows}
  end

  @spec fold(t(), acc, (Embedding.t(), acc -> acc)) :: {:ok, acc} when acc: term()
  @impl true
  def fold(%__MODULE__{table: table}, acc, fun) when is_function(fun, 2) do
    folded =
      :ets.foldl(
        fn
          {{:record, _id}, %Embedding{} = embedding}, acc -> fun.(embedding, acc)
          _other, acc -> acc
        end,
        acc,
        table
      )

    {:ok, folded}
  end

  @spec count(t()) :: non_neg_integer()
  @impl true
  def count(%__MODULE__{table: table}) do
    table
    |> :ets.info(:size)
    |> Kernel.-(1)
    |> max(0)
  end

  @spec load_config_or_delete(:ets.tid()) :: {:ok, {t(), map()}} | {:error, term()}
  defp load_config_or_delete(table) do
    state = %__MODULE__{table: table}

    case config(state) do
      {:ok, config} ->
        {:ok, {state, config}}

      {:error, reason} ->
        true = :ets.delete(table)
        {:error, reason}
    end
  end

  @spec config(t()) :: {:ok, map()} | {:error, :missing_config}
  defp config(%__MODULE__{table: table}) do
    case :ets.lookup(table, :__config__) do
      [{:__config__, config}] when is_map(config) -> {:ok, config}
      _other -> {:error, :missing_config}
    end
  end

  @spec ensure_snapshot_directory(Path.t()) :: :ok | {:error, File.posix()}
  defp ensure_snapshot_directory(path) do
    path
    |> Path.dirname()
    |> File.mkdir_p()
  end

  @spec embedding_id(Embedding.t()) :: {:ok, String.t()} | {:error, :missing_id}
  defp embedding_id(%Embedding{id: id}) when is_binary(id) and id != "", do: {:ok, id}

  defp embedding_id(%Embedding{value: value}) when is_binary(value) and value != "",
    do: {:ok, value}

  defp embedding_id(_embedding), do: {:error, :missing_id}

  @spec collect_insert_row(Embedding.t(), {[tuple()], MapSet.t(String.t())}) ::
          {:cont, {[tuple()], MapSet.t(String.t())}}
          | {:halt, {:error, :duplicate_id | :missing_id}}
  defp collect_insert_row(embedding, {rows, ids}) do
    with {:ok, id} <- embedding_id(embedding),
         :ok <- validate_batch_id(ids, id) do
      row = {{:record, id}, normalize_value(embedding, id)}
      {:cont, {[row | rows], MapSet.put(ids, id)}}
    else
      {:error, reason} -> {:halt, {:error, reason}}
    end
  end

  @spec validate_batch_id(MapSet.t(String.t()), String.t()) :: :ok | {:error, :duplicate_id}
  defp validate_batch_id(ids, id) do
    if MapSet.member?(ids, id), do: {:error, :duplicate_id}, else: :ok
  end

  @spec table_options(map()) :: [
          :set | :public | :compressed | {:read_concurrency, true} | {:write_concurrency, true}
        ]
  defp table_options(config) do
    base = [
      :set,
      :public,
      read_concurrency: true,
      write_concurrency: true
    ]

    if Map.get(config, :compressed, false), do: [:compressed | base], else: base
  end

  @spec normalize_value(Embedding.t(), String.t()) :: Embedding.t()
  defp normalize_value(%Embedding{} = embedding, id) do
    value = embedding.value || id
    %Embedding{embedding | id: id, value: value}
  end
end
