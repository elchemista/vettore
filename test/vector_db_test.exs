defmodule VettoreDBTest do
  use ExUnit.Case, async: true
  alias Vettore.Embedding

  @metrics [:euclidean, :cosine, :dot, :hnsw, :binary]

  for metric <- @metrics do
    coll = "#{metric}_coll"

    describe "#{metric} collection end-to-end" do
      test "full workflow for #{metric}" do
        coll = unquote(coll)
        metric = unquote(metric)

        db = Vettore.new()
        assert {:ok, ^coll} = Vettore.create_collection(db, coll, 3, metric)

        v1 = [1.0, 2.0, 3.0]
        v2 = [-1.3, 3.2, 5.6]

        assert {:ok, "e1"} =
                 Vettore.insert(db, coll, %Embedding{
                   value: "e1",
                   vector: v1,
                   metadata: %{"tag" => "A"}
                 })

        # duplicate value → error
        assert {:error, _} =
                 Vettore.insert(db, coll, %Embedding{value: "e1", vector: v2, metadata: nil})

        # duplicate vector → error
        assert {:error, "duplicate vector"} =
                 Vettore.insert(db, coll, %Embedding{value: "ex", vector: v1, metadata: nil})

        # insert second distinct
        assert {:ok, "e2"} =
                 Vettore.insert(db, coll, %Embedding{value: "e2", vector: v2, metadata: nil})

        #  get_all
        assert {:ok, all2} = Vettore.get_all(db, coll)
        assert length(all2) == 2

        # get_by_value
        assert {:ok, %Embedding{value: "e1", vector: _, metadata: %{"tag" => "A"}}} =
                 Vettore.get_by_value(db, coll, "e1")

        # get_by_vector
        assert {:ok, %Embedding{value: "e2", vector: _, metadata: nil}} =
                 Vettore.get_by_vector(db, coll, v2)

        # batch insert two more distinct
        # negative second element
        v3 = [7.1, -8.2, 9.3]
        # negatives in two positions
        v4 = [-10.4, 11.5, -12.6]

        b1 = %Embedding{value: "b1", vector: v3, metadata: nil}
        b2 = %Embedding{value: "b2", vector: v4, metadata: %{"foo" => "bar"}}
        assert {:ok, ["b1", "b2"]} = Vettore.batch(db, coll, [b1, b2])
        assert {:ok, all4} = Vettore.get_all(db, coll)
        assert length(all4) == 4

        # delete one
        assert {:ok, "e1"} = Vettore.delete(db, coll, "e1")
        assert {:error, _} = Vettore.get_by_value(db, coll, "e1")

        # similarity_search limit 2
        query = [1.0, 2.0, 3.0]
        assert {:ok, results} = Vettore.similarity_search(db, coll, query, limit: 2)
        assert length(results) <= 2

        for {val, score} <- results do
          assert is_binary(val)
          assert is_float(score)
        end

        # rerank limit 2, alpha 0.5
        assert {:ok, reranked} = Vettore.rerank(db, coll, results, limit: 2, alpha: 0.5)
        assert length(reranked) <= 2
      end
    end
  end
end
