defmodule Vettore.Store do
  @moduledoc """
  Storage behaviour for Vettore collections.

  Stores own records and metadata. Indexes and native accelerators are separate
  from this layer so ETS can remain the canonical source of collection state.
  """

  alias Vettore.Embedding

  @type config :: map()
  @type state :: term()
  @type id :: String.t()

  @callback new(config()) :: {:ok, state()} | {:error, term()}
  @callback put(state(), Embedding.t()) :: :ok | {:error, term()}
  @callback put_many(state(), [Embedding.t()]) :: :ok | {:error, term()}
  @callback get(state(), id()) :: {:ok, Embedding.t()} | {:error, term()}
  @callback delete(state(), id()) :: :ok | {:error, term()}
  @callback all(state()) :: {:ok, [Embedding.t()]} | {:error, term()}
  @callback count(state()) :: non_neg_integer()
end
