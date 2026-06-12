defmodule Vettore.Result do
  @moduledoc """
  Search result with explicit score and metric semantics.
  """

  @enforce_keys [:id, :score, :metric]
  defstruct [:id, :value, :score, :distance, :metric, :metadata]

  @type t :: %__MODULE__{
          id: String.t(),
          value: term(),
          score: float(),
          distance: float() | nil,
          metric: atom(),
          metadata: map() | nil
        }
end
