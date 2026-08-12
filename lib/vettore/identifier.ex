defmodule Vettore.Identifier do
  @moduledoc false

  alias Vettore.Embedding

  @type error :: :missing_id | :invalid_id

  @spec valid?(term()) :: boolean()
  def valid?(id), do: is_binary(id) and id != "" and String.valid?(id)

  @spec embedding_id(Embedding.t()) :: {:ok, String.t()} | {:error, error()}
  def embedding_id(%Embedding{id: id} = embedding) do
    case validate_candidate(id) do
      {:ok, id} -> {:ok, id}
      :missing -> value_id(embedding.value)
      {:error, :invalid_id} = error -> error
    end
  end

  @spec validate(term()) :: :ok | {:error, :invalid_id}
  def validate(id) do
    if valid?(id), do: :ok, else: {:error, :invalid_id}
  end

  @spec validate_utf8(term()) :: :ok | {:error, :invalid_id}
  def validate_utf8(id) do
    if is_binary(id) and String.valid?(id), do: :ok, else: {:error, :invalid_id}
  end

  @spec validate_embeddings(term()) :: :ok | {:error, :invalid_id}
  def validate_embeddings(embeddings) when is_list(embeddings) do
    if Enum.all?(embeddings, fn
         %Embedding{id: id} -> valid?(id)
         _embedding -> false
       end),
       do: :ok,
       else: {:error, :invalid_id}
  end

  def validate_embeddings(_embeddings), do: {:error, :invalid_id}

  @spec validate_candidate(term()) :: {:ok, String.t()} | :missing | {:error, :invalid_id}
  defp validate_candidate(nil), do: :missing
  defp validate_candidate(""), do: :missing

  defp validate_candidate(id) when is_binary(id) do
    if String.valid?(id), do: {:ok, id}, else: {:error, :invalid_id}
  end

  defp validate_candidate(_id), do: :missing

  @spec value_id(term()) :: {:ok, String.t()} | {:error, error()}
  defp value_id(value) do
    case validate_candidate(value) do
      {:ok, value} -> {:ok, value}
      :missing -> {:error, :missing_id}
      {:error, :invalid_id} = error -> error
    end
  end
end
