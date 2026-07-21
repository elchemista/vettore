defmodule Vettore.ETSOwner do
  @moduledoc false

  use GenServer

  @type start_result :: {:ok, {pid(), :ets.tid()}} | {:error, term()}

  @spec start_table(atom(), [term()]) :: start_result()
  def start_table(name, options), do: start_table(name, options, [])

  @spec start_table(atom(), [term()], [tuple()]) :: start_result()
  def start_table(name, options, initial_objects)
      when is_atom(name) and is_list(options) and is_list(initial_objects) do
    start_child({:new, name, options, initial_objects})
  end

  @spec load_table(Path.t()) :: start_result()
  def load_table(path) when is_binary(path) and path != "" do
    start_child({:load, path})
  end

  def load_table(_path), do: {:error, :invalid_snapshot_path}

  @spec close(pid()) :: :ok
  def close(owner) when is_pid(owner) do
    if Process.alive?(owner) and Process.whereis(Vettore.ETSSupervisor) do
      case DynamicSupervisor.terminate_child(Vettore.ETSSupervisor, owner) do
        :ok -> :ok
        {:error, :not_found} -> :ok
      end
    else
      :ok
    end
  end

  @spec alive?(pid()) :: boolean()
  def alive?(owner) when is_pid(owner), do: Process.alive?(owner)
  def alive?(_owner), do: false

  @spec insert(pid(), tuple() | [tuple()]) :: true | {:error, :closed}
  def insert(owner, objects), do: call(owner, {:insert, objects})

  @spec insert_new(pid(), tuple() | [tuple()]) :: boolean() | {:error, :closed}
  def insert_new(owner, objects), do: call(owner, {:insert_new, objects})

  @spec delete(pid(), term()) :: true | {:error, :closed}
  def delete(owner, key), do: call(owner, {:delete, key})

  @spec take(pid(), term()) :: [tuple()] | {:error, :closed}
  def take(owner, key), do: call(owner, {:take, key})

  @spec drain_and_close(pid()) :: [tuple()] | {:error, :closed}
  def drain_and_close(owner), do: call(owner, :drain_and_close)

  def child_spec(init_arg) do
    %{
      id: {__MODULE__, make_ref()},
      start: {__MODULE__, :start_link, [init_arg]},
      restart: :temporary,
      type: :worker
    }
  end

  @doc false
  def start_link(init_arg), do: GenServer.start_link(__MODULE__, init_arg)

  @impl true
  def init({:new, name, options, initial_objects}) do
    table = :ets.new(name, options)

    if initial_objects != [] do
      true = :ets.insert(table, initial_objects)
    end

    {:ok, table}
  end

  def init({:load, path}) do
    case :ets.file2tab(String.to_charlist(path), verify: true) do
      {:ok, table} -> normalize_loaded_table(table)
      {:error, reason} -> {:stop, reason}
    end
  end

  @impl true
  def handle_call(:table, _from, table), do: {:reply, table, table}

  def handle_call({:insert, objects}, _from, table),
    do: {:reply, :ets.insert(table, objects), table}

  def handle_call({:insert_new, objects}, _from, table),
    do: {:reply, :ets.insert_new(table, objects), table}

  def handle_call({:delete, key}, _from, table), do: {:reply, :ets.delete(table, key), table}
  def handle_call({:take, key}, _from, table), do: {:reply, :ets.take(table, key), table}

  def handle_call(:drain_and_close, _from, table) do
    rows = :ets.tab2list(table)
    true = :ets.delete(table)
    {:stop, :normal, rows, table}
  end

  @spec normalize_loaded_table(:ets.tid()) :: {:ok, :ets.tid()} | {:stop, term()}
  defp normalize_loaded_table(table) do
    if :ets.info(table, :type) == :set do
      if normalized_table?(table) do
        {:ok, table}
      else
        copy_to_normalized_table(table)
      end
    else
      true = :ets.delete(table)
      {:stop, :invalid_snapshot_table_type}
    end
  end

  @spec normalized_table?(:ets.tid()) :: boolean()
  defp normalized_table?(table) do
    :ets.info(table, :protection) == :protected and
      :ets.info(table, :read_concurrency) == true and
      :ets.info(table, :write_concurrency) == false and
      :ets.info(table, :named_table) == false and
      :ets.info(table, :keypos) == 1 and
      :ets.info(table, :heir) == :none
  end

  @spec copy_to_normalized_table(:ets.tid()) :: {:ok, :ets.tid()}
  defp copy_to_normalized_table(source) do
    target = :ets.new(:vettore_collection, normalized_table_options(source))

    true =
      :ets.foldl(
        fn object, true -> :ets.insert(target, object) end,
        true,
        source
      )

    true = :ets.delete(source)
    {:ok, target}
  end

  @spec normalized_table_options(:ets.tid()) :: [
          :set | :protected | :compressed | {:read_concurrency, true}
        ]
  defp normalized_table_options(source) do
    options = [:set, :protected, read_concurrency: true]

    if :ets.info(source, :compressed), do: [:compressed | options], else: options
  end

  @spec start_child(term()) :: start_result()
  defp start_child(init_arg) do
    with :ok <- ensure_supervisor_started(),
         {:ok, owner} <-
           DynamicSupervisor.start_child(Vettore.ETSSupervisor, child_spec(init_arg)) do
      {:ok, {owner, GenServer.call(owner, :table, :infinity)}}
    else
      {:error, {reason, _child}} -> {:error, reason}
      {:error, reason} -> {:error, reason}
    end
  end

  @spec ensure_supervisor_started() :: :ok | {:error, term()}
  defp ensure_supervisor_started do
    case Process.whereis(Vettore.ETSSupervisor) do
      pid when is_pid(pid) ->
        :ok

      nil ->
        case Application.ensure_all_started(:vettore) do
          {:ok, _apps} -> :ok
          {:error, reason} -> {:error, reason}
        end
    end
  end

  @spec call(pid(), term()) :: term() | {:error, :closed}
  defp call(owner, message) when is_pid(owner) do
    if Process.alive?(owner) do
      GenServer.call(owner, message, :infinity)
    else
      {:error, :closed}
    end
  catch
    :exit, _reason -> {:error, :closed}
  end
end
