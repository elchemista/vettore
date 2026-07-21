defmodule Vettore.Store.ETS do
  @moduledoc """
  ETS-backed canonical store for Vettore collections.
  """

  @behaviour Vettore.Store

  alias Vettore.Embedding

  defstruct [:table, :owner]

  @type t :: %__MODULE__{table: :ets.tid(), owner: pid()}

  @spec new(map()) :: {:ok, t()}
  @impl true
  def new(config) when is_map(config) do
    with {:ok, {owner, table}} <-
           Vettore.ETSOwner.start_table(
             :vettore_collection,
             table_options(config),
             [{:__config__, config}]
           ) do
      {:ok, %__MODULE__{table: table, owner: owner}}
    end
  end

  @spec snapshot(t(), Path.t()) :: :ok | {:error, term()}
  @impl true
  def snapshot(%__MODULE__{} = state, path) when is_binary(path) and path != "" do
    temporary_path = path <> ".tmp-#{System.unique_integer([:positive, :monotonic])}"

    try do
      with :ok <- ensure_snapshot_directory(path),
           :ok <-
             safe_table_call(state, fn table ->
               :ets.tab2file(table, String.to_charlist(temporary_path),
                 extended_info: [:object_count, :md5sum]
               )
             end) do
        File.rename(temporary_path, path)
      end
    after
      File.rm(temporary_path)
    end
  end

  def snapshot(_state, _path), do: {:error, :invalid_snapshot_path}

  @spec load_snapshot(Path.t()) :: {:ok, {t(), map()}} | {:error, term()}
  @impl true
  def load_snapshot(path) when is_binary(path) and path != "" do
    case Vettore.ETSOwner.load_table(path) do
      {:ok, {owner, table}} -> load_config_or_close(owner, table)
      {:error, reason} -> {:error, reason}
    end
  end

  def load_snapshot(_path), do: {:error, :invalid_snapshot_path}

  @spec put(t(), Embedding.t()) :: :ok | {:error, :closed | :duplicate_id | :missing_id}
  @impl true
  def put(%__MODULE__{} = state, %Embedding{} = embedding) do
    with {:ok, id} <- embedding_id(embedding) do
      record = {{:record, id}, normalize_value(embedding, id)}

      safe_owner_call(state, &insert_new(&1, record))
    end
  end

  @spec configure(t(), map()) :: :ok | {:error, :closed}
  @impl true
  def configure(%__MODULE__{} = state, config) when is_map(config) do
    safe_owner_call(state, fn owner ->
      case Vettore.ETSOwner.insert(owner, {:__config__, config}) do
        true -> :ok
        {:error, :closed} = error -> error
      end
    end)
  end

  @spec close(t()) :: :ok
  @impl true
  def close(%__MODULE__{owner: owner}) do
    Vettore.ETSOwner.close(owner)
  end

  @spec alive?(t()) :: boolean()
  @impl true
  def alive?(%__MODULE__{table: table, owner: owner}) do
    Vettore.ETSOwner.alive?(owner) and :ets.info(table) != :undefined
  rescue
    ArgumentError -> false
  end

  def alive?(_state), do: false

  @spec put_many(t(), [Embedding.t()]) ::
          :ok | {:error, :closed | :duplicate_id | :missing_id}
  @impl true
  def put_many(%__MODULE__{} = state, embeddings) when is_list(embeddings) do
    result =
      Enum.reduce_while(embeddings, {[], MapSet.new()}, &collect_insert_row/2)

    case result do
      {:error, reason} ->
        {:error, reason}

      {rows, _ids} ->
        safe_owner_call(state, &insert_new(&1, rows))
    end
  end

  @spec get(t(), String.t()) :: {:ok, Embedding.t()} | {:error, :closed | :not_found}
  @impl true
  def get(%__MODULE__{} = state, id) when is_binary(id) do
    safe_table_call(state, fn table ->
      case :ets.lookup(table, {:record, id}) do
        [{{:record, ^id}, %Embedding{} = embedding}] -> {:ok, embedding}
        [] -> {:error, :not_found}
      end
    end)
  end

  @spec delete(t(), String.t()) :: :ok | {:error, :closed}
  @impl true
  def delete(%__MODULE__{} = state, id) when is_binary(id) do
    safe_owner_call(state, fn owner ->
      case Vettore.ETSOwner.delete(owner, {:record, id}) do
        true -> :ok
        {:error, :closed} = error -> error
      end
    end)
  end

  @spec all(t()) :: {:ok, [Embedding.t()]} | {:error, :closed}
  @impl true
  def all(%__MODULE__{} = state) do
    safe_table_call(state, fn table ->
      rows =
        table
        |> :ets.tab2list()
        |> Enum.flat_map(fn
          {{:record, _id}, %Embedding{} = embedding} -> [embedding]
          _other -> []
        end)

      {:ok, rows}
    end)
  end

  @spec fold(t(), acc, (Embedding.t(), acc -> acc)) ::
          {:ok, acc} | {:error, :closed}
        when acc: term()
  @impl true
  def fold(%__MODULE__{} = state, acc, fun) when is_function(fun, 2) do
    safe_table_call(state, fn table ->
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
    end)
  end

  @spec count(t()) :: non_neg_integer()
  @impl true
  def count(%__MODULE__{} = state) do
    case safe_table_call(state, &:ets.info(&1, :size)) do
      size when is_integer(size) -> max(size - 1, 0)
      :undefined -> 0
      {:error, :closed} -> 0
    end
  end

  @spec load_config_or_close(pid(), :ets.tid()) :: {:ok, {t(), map()}} | {:error, term()}
  defp load_config_or_close(owner, table) do
    state = %__MODULE__{table: table, owner: owner}

    with {:ok, config} <- config(state),
         :ok <- validate_snapshot_rows(table) do
      {:ok, {state, config}}
    else
      {:error, reason} ->
        :ok = close(state)
        {:error, reason}
    end
  end

  @spec validate_snapshot_rows(:ets.tid()) :: :ok | {:error, term()}
  defp validate_snapshot_rows(table) do
    :ets.foldl(
      fn
        {:__config__, config}, :ok when is_map(config) ->
          :ok

        {{:record, key_id}, %Embedding{id: embedding_id}}, :ok
        when is_binary(key_id) and key_id != "" and key_id == embedding_id ->
          :ok

        {{:record, _key_id}, %Embedding{}}, :ok ->
          {:error, {:invalid_snapshot_record, :id_mismatch}}

        {{:record, _key_id}, _value}, :ok ->
          {:error, {:invalid_snapshot_record, :invalid_embedding}}

        _row, :ok ->
          {:error, :invalid_snapshot_row}

        _row, {:error, _reason} = error ->
          error
      end,
      :ok,
      table
    )
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

  @spec insert_new(pid(), tuple() | [tuple()]) :: :ok | {:error, :closed | :duplicate_id}
  defp insert_new(owner, objects) do
    case Vettore.ETSOwner.insert_new(owner, objects) do
      true -> :ok
      false -> {:error, :duplicate_id}
      {:error, :closed} = error -> error
    end
  end

  @spec table_options(map()) :: [:set | :protected | :compressed | {:read_concurrency, true}]
  defp table_options(config) do
    base = [
      :set,
      :protected,
      read_concurrency: true
    ]

    if Map.get(config, :compressed, false), do: [:compressed | base], else: base
  end

  @spec normalize_value(Embedding.t(), String.t()) :: Embedding.t()
  defp normalize_value(%Embedding{} = embedding, id) do
    value = embedding.value || id
    %Embedding{embedding | id: id, value: value}
  end

  @spec safe_table_call(t(), (:ets.tid() -> result)) :: result | {:error, :closed}
        when result: term()
  defp safe_table_call(%__MODULE__{} = state, fun) when is_function(fun, 1) do
    fun.(state.table)
  rescue
    ArgumentError -> {:error, :closed}
  end

  @spec safe_owner_call(t(), (pid() -> result)) :: result | {:error, :closed}
        when result: term()
  defp safe_owner_call(%__MODULE__{} = state, fun) when is_function(fun, 1) do
    fun.(state.owner)
  end
end
