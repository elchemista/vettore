defmodule Vettore.Embedding do
  @moduledoc """
  Represents a single embedding entry for insertion into a collection.

  ## Fields

    * `:id` - Stable identifier for this embedding. Preferred in vNext.
    * `:value` - A string or content identifier for this embedding can be Id, or Text (e.g. "this is text data").
    * `:vector` - A list of floating‑point numbers representing the embedding (e.g. `[1.0, 2.0, 3.0]`).
    * `:vectors` - Optional token/document vectors for late interaction search.
    * `:binary_vector` - Sign-bit compressed vector used for binary quantized candidate search.
    * `:metadata` - (Optional) A map with any additional information you want to store
      (e.g. `%{"info" => "my note"}`).
  """
  defstruct [:id, :value, :vector, :vectors, :binary_vector, :metadata]

  @type t :: %__MODULE__{
          id: String.t() | nil,
          value: String.t() | nil,
          vector: [float()] | nil,
          vectors: [[float()]] | nil,
          binary_vector: [non_neg_integer()] | nil,
          metadata: map() | nil
        }
end
