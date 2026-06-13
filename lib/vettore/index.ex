defmodule Vettore.Index do
  @moduledoc """
  Search index behaviour.

  Indexes may keep acceleration state, but ETS remains the canonical record
  store. Implementations must return `Vettore.Result` structs.
  """

  alias Vettore.{Collection, Embedding}
  alias Vettore.Result

  @callback new(atom(), keyword()) :: {:ok, term()} | {:error, term()}
  @callback put(Collection.t(), Embedding.t()) :: :ok | {:error, term()}
  @callback put_many(Collection.t(), [Embedding.t()]) :: :ok | {:error, term()}
  @callback delete(Collection.t(), String.t()) :: :ok | {:error, term()}
  @callback search(Collection.t(), [number()], keyword()) ::
              {:ok, [Result.t()]} | {:error, term()}
end
