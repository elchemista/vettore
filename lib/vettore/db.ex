defmodule Vettore.DB do
  @moduledoc """
  Compatibility database handle returned by `Vettore.new/0`.

  Most applications should use the collection API directly. This struct is
  public so compatibility API types and lifecycle ownership remain explicit.
  """

  defstruct [:table, :owner]

  @type t :: %__MODULE__{table: :ets.tid(), owner: pid()}
end
