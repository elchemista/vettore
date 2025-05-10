defmodule Vettore.Validator do
  @moduledoc false
  # Centralised, reusable validation helpers for the Vettore public API.

  alias Vettore.Embedding

  # Guards
  defguard is_db(d) when is_reference(d)
  defguard is_col(c) when is_binary(c) and byte_size(c) > 0
  defguard is_id(i) when is_binary(i)
  defguard is_vec(v) when is_list(v)

  defguard is_embedding(id, vec)
           when is_id(id) and is_vec(vec)

  # Vector helpers

  @doc false
  @spec numeric?([term]) :: boolean
  def numeric?([]), do: true
  def numeric?([h | t]) when is_number(h), do: numeric?(t)
  def numeric?(_), do: false

  # Metadata helpers

  @type meta :: %{required(String.t()) => String.t()} | nil

  @doc false
  @spec sanitize_meta!(meta) :: meta | no_return
  def sanitize_meta!(nil), do: nil

  def sanitize_meta!(m) when is_map(m) do
    # Convert once to list and walk it tail-recursively.
    m
    |> Map.to_list()
    |> validate_pairs()

    m
  end

  def sanitize_meta!(_),
    do: raise(ArgumentError, "metadata must be nil or map of string => string")

  # Tail-recursive validator (no allocations on success)
  defp validate_pairs([]), do: :ok

  defp validate_pairs([{k, v} | rest]) when is_binary(k) and is_binary(v),
    do: validate_pairs(rest)

  defp validate_pairs([{k, v} | _]),
    do:
      raise(
        ArgumentError,
        "invalid metadata entry #{inspect(k)} => #{inspect(v)}, " <>
          "metadata must be string => string"
      )

  @spec sanitize_meta(meta) :: {:ok, meta} | {:error, String.t()}
  def sanitize_meta(meta) do
    try do
      {:ok, sanitize_meta!(meta)}
    rescue
      e in ArgumentError -> {:error, e.message}
    end
  end

  # Embedding-list helpers (for batch)

  @type tupled :: {String.t(), [number()], meta}

  @doc false
  @spec embeddings_to_tuples!([Embedding.t()]) :: [tupled] | no_return
  def embeddings_to_tuples!(embs), do: et_loop(embs, [])

  defp et_loop([], acc), do: :lists.reverse(acc)

  defp et_loop([%Embedding{value: id, vector: vec, metadata: m} | rest], acc)
       when is_embedding(id, vec) do
    if numeric?(vec) do
      et_loop(rest, [{id, vec, sanitize_meta!(m)} | acc])
    else
      raise ArgumentError, "vector for #{inspect(id)} is not numeric"
    end
  end

  defp et_loop([bad | _], _),
    do:
      raise(
        ArgumentError,
        "each item must be %Vettore.Embedding{} with valid fields, got: #{inspect(bad)}"
      )

  @doc false
  @spec embeddings_to_tuples([Embedding.t()]) :: {:ok, [tupled]} | {:error, String.t()}
  def embeddings_to_tuples(list) do
    try do
      {:ok, embeddings_to_tuples!(list)}
    rescue
      e in ArgumentError -> {:error, e.message}
    end
  end
end
