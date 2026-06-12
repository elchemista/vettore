defmodule Vettore.DB do
  @moduledoc false

  defstruct [:table]

  @type t :: %__MODULE__{table: :ets.tid()}
end
