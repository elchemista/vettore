defmodule Vettore.Embedding do
  @moduledoc """
  Represents a single embedding entry for insertion into a collection.

  ## Fields

    * `:value` - A string or content identifier for this embedding can be Id, or Text (e.g. "this is text data").
    * `:vector` - A list of floating‑point numbers representing the embedding (e.g. `[1.0, 2.0, 3.0]`).
    * `:metadata` - (Optional) A map with any additional information you want to store
      (e.g. `%{"info" => "my note"}`).
  """
  defstruct [:value, :vector, :metadata]

  @type t :: %__MODULE__{
          value: String.t(),
          vector: [float()],
          metadata: map() | nil
        }
end
