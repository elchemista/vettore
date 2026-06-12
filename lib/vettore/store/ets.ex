defmodule Vettore.Store.ETS do
  @moduledoc """
  ETS-backed canonical store for Vettore collections.
  """

  @behaviour Vettore.Store

  alias Vettore.Embedding

  defstruct [:table]

  @impl true
  def new(config) when is_map(config) do
    table =
      :ets.new(:vettore_collection, [
        :set,
        :public,
        read_concurrency: true,
        write_concurrency: true
      ])

    true = :ets.insert(table, {:__config__, config})
    {:ok, %__MODULE__{table: table}}
  end

  @impl true
  def put(%__MODULE__{table: table}, %Embedding{} = embedding) do
    with {:ok, id} <- embedding_id(embedding) do
      if :ets.member(table, {:record, id}) do
        {:error, :duplicate_id}
      else
        record = {{:record, id}, normalize_value(embedding, id)}
        true = :ets.insert(table, record)
        :ok
      end
    end
  end

  @impl true
  def put_many(%__MODULE__{} = state, embeddings) when is_list(embeddings) do
    rows =
      Enum.reduce_while(embeddings, [], fn embedding, acc ->
        case embedding_id(embedding) do
          {:ok, id} ->
            cond do
              :ets.member(state.table, {:record, id}) ->
                {:halt, {:error, :duplicate_id}}

              Enum.any?(acc, fn {{:record, existing_id}, _embedding} -> existing_id == id end) ->
                {:halt, {:error, :duplicate_id}}

              true ->
                {:cont, [{{:record, id}, normalize_value(embedding, id)} | acc]}
            end

          {:error, reason} ->
            {:halt, {:error, reason}}
        end
      end)

    case rows do
      {:error, reason} ->
        {:error, reason}

      rows ->
        true = :ets.insert(state.table, rows)
        :ok
    end
  end

  @impl true
  def get(%__MODULE__{table: table}, id) when is_binary(id) do
    case :ets.lookup(table, {:record, id}) do
      [{{:record, ^id}, %Embedding{} = embedding}] -> {:ok, embedding}
      [] -> {:error, :not_found}
    end
  end

  @impl true
  def delete(%__MODULE__{table: table}, id) when is_binary(id) do
    true = :ets.delete(table, {:record, id})
    :ok
  end

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

  @impl true
  def count(%__MODULE__{table: table}) do
    table
    |> :ets.info(:size)
    |> Kernel.-(1)
    |> max(0)
  end

  defp embedding_id(%Embedding{id: id}) when is_binary(id) and id != "", do: {:ok, id}

  defp embedding_id(%Embedding{value: value}) when is_binary(value) and value != "",
    do: {:ok, value}

  defp embedding_id(_embedding), do: {:error, :missing_id}

  defp normalize_value(%Embedding{} = embedding, id) do
    value = embedding.value || id
    %Embedding{embedding | id: id, value: value}
  end
end
