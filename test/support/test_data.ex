defmodule TestData do
  @moduledoc false

  @data_file Path.expand("./data.json", __DIR__)

  @doc """
  Returns a list of `%Vettore.Embedding{}` structs, one per JSON
  object in test/data.json.
  """
  def load_embeddings do
    @data_file
    |> File.read!()
    |> Jason.decode!()
    |> Enum.map(fn %{"embedding" => vec, "tags" => tags} ->
      %Vettore.Embedding{
        value: Enum.join(tags, ","),
        vector: vec,
        metadata: %{"tag" => "test"}
      }
    end)
  end
end
