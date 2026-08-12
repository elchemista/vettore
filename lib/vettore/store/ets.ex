defmodule Vettore.Store.ETS do
  @moduledoc """
  ETS-backed canonical store for Vettore collections.
  """

  @behaviour Vettore.Store

  alias Vettore.{Embedding, Identifier}

  defstruct [:table, :owner]

  @type t :: %__MODULE__{table: :ets.tid(), owner: pid()}

  @spec new(map()) :: {:ok, t()}
  @impl Vettore.Store
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
  @impl Vettore.Store
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
  @impl Vettore.Store
  def load_snapshot(path) when is_binary(path) and path != "" do
    case Vettore.ETSOwner.load_table(path) do
      {:ok, {owner, table}} -> load_config_or_close(owner, table)
      {:error, reason} -> {:error, reason}
    end
  end

  def load_snapshot(_path), do: {:error, :invalid_snapshot_path}

  @spec put(t(), Embedding.t()) ::
          :ok | {:error, :closed | :duplicate_id | :invalid_id | :missing_id}
  @impl Vettore.Store
  def put(%__MODULE__{} = state, %Embedding{} = embedding) do
    with {:ok, id} <- embedding_id(embedding) do
      record = {{:record, id}, normalize_value(embedding, id)}

      safe_owner_call(state, &insert_new(&1, record))
    end
  end

  @doc false
  @spec put_indexed(t(), Embedding.t(), (-> term()), (-> term())) ::
          :ok | {:error, term()}
  def put_indexed(%__MODULE__{} = state, %Embedding{} = embedding, index_fun, rollback_fun)
      when is_function(index_fun, 0) and is_function(rollback_fun, 0) do
    with {:ok, id} <- embedding_id(embedding) do
      record = {{:record, id}, normalize_value(embedding, id)}

      owner_transaction(
        state,
        &insert_indexed(&1, record, [id], index_fun, rollback_fun)
      )
    end
  end

  @spec configure(t(), map()) :: :ok | {:error, :closed}
  @impl Vettore.Store
  def configure(%__MODULE__{} = state, config) when is_map(config) do
    safe_owner_call(state, fn owner ->
      case Vettore.ETSOwner.insert(owner, {:__config__, config}) do
        true -> :ok
        {:error, :closed} = error -> error
      end
    end)
  end

  @spec close(t()) :: :ok
  @impl Vettore.Store
  def close(%__MODULE__{owner: owner}) do
    Vettore.ETSOwner.close(owner)
  end

  @spec alive?(t()) :: boolean()
  @impl Vettore.Store
  def alive?(%__MODULE__{table: table, owner: owner}) do
    Vettore.ETSOwner.alive?(owner) and :ets.info(table) != :undefined
  rescue
    ArgumentError -> false
  end

  def alive?(_state), do: false

  @spec put_many(t(), [Embedding.t()]) ::
          :ok | {:error, :closed | :duplicate_id | :invalid_id | :missing_id}
  @impl Vettore.Store
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

  @doc false
  @spec put_many_indexed(t(), [Embedding.t()], (-> term()), (-> term())) ::
          :ok | {:error, term()}
  def put_many_indexed(%__MODULE__{} = state, embeddings, index_fun, rollback_fun)
      when is_list(embeddings) and is_function(index_fun, 0) and is_function(rollback_fun, 0) do
    case Enum.reduce_while(embeddings, {[], MapSet.new()}, &collect_insert_row/2) do
      {:error, reason} ->
        {:error, reason}

      {rows, ids} ->
        owner_transaction(
          state,
          &insert_indexed(&1, rows, MapSet.to_list(ids), index_fun, rollback_fun)
        )
    end
  end

  @spec get(t(), String.t()) :: {:ok, Embedding.t()} | {:error, :closed | :not_found}
  @impl Vettore.Store
  def get(%__MODULE__{} = state, id) when is_binary(id) do
    with :ok <- Identifier.validate_utf8(id) do
      safe_table_call(state, &lookup_embedding(&1, id))
    end
  end

  @spec delete(t(), String.t()) :: :ok | {:error, :closed}
  @impl Vettore.Store
  def delete(%__MODULE__{} = state, id) when is_binary(id) do
    with :ok <- Identifier.validate_utf8(id) do
      safe_owner_call(state, &delete_record(&1, id))
    end
  end

  @doc false
  @spec delete_indexed(t(), String.t(), (-> term())) :: :ok | {:error, term()}
  def delete_indexed(%__MODULE__{} = state, id, index_fun)
      when is_binary(id) and is_function(index_fun, 0) do
    with :ok <- Identifier.validate_utf8(id) do
      owner_transaction(state, &delete_indexed_record(&1, id, index_fun))
    end
  end

  @spec all(t()) :: {:ok, [Embedding.t()]} | {:error, :closed}
  @impl Vettore.Store
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
  @impl Vettore.Store
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
  @impl Vettore.Store
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
          if String.valid?(key_id),
            do: :ok,
            else: {:error, {:invalid_snapshot_record, :invalid_id}}

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

  @spec embedding_id(Embedding.t()) ::
          {:ok, String.t()} | {:error, :missing_id | :invalid_id}
  defp embedding_id(%Embedding{} = embedding), do: Identifier.embedding_id(embedding)

  @spec collect_insert_row(Embedding.t(), {[tuple()], MapSet.t(String.t())}) ::
          {:cont, {[tuple()], MapSet.t(String.t())}}
          | {:halt, {:error, :duplicate_id | :invalid_id | :missing_id}}
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

  @spec insert_indexed(:ets.tid(), tuple() | [tuple()], [String.t()], (-> term()), (-> term())) ::
          :ok | {:error, term()}
  defp insert_indexed(table, objects, ids, index_fun, rollback_fun) do
    case :ets.insert_new(table, objects) do
      true -> finish_indexed_insert(table, ids, index_fun, rollback_fun)
      false -> {:error, :duplicate_id}
    end
  end

  @spec lookup_embedding(:ets.tid(), String.t()) ::
          {:ok, Embedding.t()} | {:error, :not_found}
  defp lookup_embedding(table, id) do
    case :ets.lookup(table, {:record, id}) do
      [{{:record, ^id}, %Embedding{} = embedding}] -> {:ok, embedding}
      [] -> {:error, :not_found}
    end
  end

  @spec delete_record(pid(), String.t()) :: :ok | {:error, :closed}
  defp delete_record(owner, id) do
    case Vettore.ETSOwner.delete(owner, {:record, id}) do
      true -> :ok
      {:error, :closed} = error -> error
    end
  end

  @spec delete_indexed_record(:ets.tid(), String.t(), (-> term())) ::
          :ok | {:error, term()}
  defp delete_indexed_record(table, id, index_fun) do
    case safe_index_call(index_fun) do
      :ok ->
        true = :ets.delete(table, {:record, id})
        :ok

      {:error, _reason} = error ->
        error
    end
  end

  @spec finish_indexed_insert(:ets.tid(), [String.t()], (-> term()), (-> term())) ::
          :ok | {:error, term()}
  defp finish_indexed_insert(table, ids, index_fun, rollback_fun) do
    case safe_index_call(index_fun) do
      :ok ->
        :ok

      {:error, _reason} = error ->
        Enum.each(ids, &:ets.delete(table, {:record, &1}))
        _rollback_result = safe_index_call(rollback_fun)
        error
    end
  end

  @spec safe_index_call((-> term())) :: :ok | {:error, term()}
  defp safe_index_call(fun) do
    case fun.() do
      :ok -> :ok
      {:error, _reason} = error -> error
      other -> {:error, {:invalid_index_result, other}}
    end
  rescue
    exception -> {:error, {:index_exception, exception}}
  catch
    kind, reason -> {:error, {:index_exception, {kind, reason}}}
  end

  @spec owner_transaction(t(), (:ets.tid() -> result)) :: result | {:error, :closed}
        when result: term()
  defp owner_transaction(%__MODULE__{} = state, fun) do
    safe_owner_call(state, &Vettore.ETSOwner.transaction(&1, fun))
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
