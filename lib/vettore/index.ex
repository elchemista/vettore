defmodule Vettore.Index do
  @moduledoc """
  Search index behaviour.

  Indexes may keep acceleration state, but ETS remains the canonical record
  store. Implementations must return `Vettore.Result` structs.
  """

  alias Vettore.Collection
  alias Vettore.Result

  @callback search(Collection.t(), [number()], keyword()) ::
              {:ok, [Result.t()]} | {:error, term()}
end
