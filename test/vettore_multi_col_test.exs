defmodule VettoreMultiColTest do
  use ExUnit.Case, async: true
  import TestData
  alias Vettore.Embedding

  @metrics [:euclidean, :cosine, :dot, :hnsw, :binary]

  setup_all do
    {:ok, embeddings: load_embeddings()}
  end

  @moduletag :distance

  describe "concurrent collections test" do
    test "concurrent creation, insertion and get_by_value in multiple collections",
         %{embeddings: [sample | _]} do
      db = Vettore.new()
      vec = sample.vector
      dim = length(vec)

      results =
        @metrics
        |> Task.async_stream(
          fn metric ->
            coll = "#{metric}_coll"
            assert {:ok, ^coll} = Vettore.create_collection(db, coll, dim, metric)

            id = "id_#{metric}"
            emb = %Embedding{value: id, vector: vec, metadata: %{"metric" => to_string(metric)}}

            assert {:ok, ^id} = Vettore.insert(db, coll, emb)
            {:ok, %Embedding{value: val, metadata: meta}} = Vettore.get_by_value(db, coll, id)

            {metric, val, meta}
          end,
          max_concurrency: length(@metrics),
          timeout: 10_000
        )
        |> Enum.map(&elem(&1, 1))

      for {metric, val, meta} <- results do
        assert val == "id_#{metric}"
        assert meta == %{"metric" => to_string(metric)}
      end
    end
  end
end
