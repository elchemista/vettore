defmodule VettoreDBTest do
  use ExUnit.Case, async: true
  import TestData

  @metrics [:euclidean, :cosine, :dot, :hnsw, :binary]

  setup_all do
    # load JSON file once
    embeddings = load_embeddings()
    dim = length(hd(embeddings).vector)
    {:ok, embeddings: embeddings, dimension: dim}
  end

  for metric <- @metrics do
    coll = "#{metric}_coll"

    describe "#{metric} collection end-to-end" do
      test "full workflow for #{metric}", %{embeddings: embeddings, dimension: dim} do
        coll = unquote(coll)
        metric = unquote(metric)

        db = Vettore.new()
        assert {:ok, ^coll} = Vettore.create_collection(db, coll, dim, metric)

        # batch-insert all embeddings from data.json
        assert {:ok, inserted_values} = Vettore.batch(db, coll, embeddings)
        # make sure we inserted exactly as many as we loaded
        assert length(inserted_values) == length(embeddings)

        # get_all should return them all
        assert {:ok, all} = Vettore.get_all(db, coll)
        assert length(all) == length(embeddings)

        # pick the first vector as our query
        [first_embedding | _] = embeddings
        query_vector = first_embedding.vector

        # similarity_search limit 5
        assert {:ok, results} = Vettore.similarity_search(db, coll, query_vector, limit: 5)
        assert length(results) <= 5

        for {val, score} <- results do
          assert is_binary(val)
          assert is_float(score)
          # scores are normalized between 0 and 1, and should be at least 0.1
          assert score >= 0.1
          assert score <= 1.0
        end

        # rerank limit 5, alpha 0.5
        assert {:ok, reranked} =
                 Vettore.rerank(db, coll, results, limit: 5, alpha: 0.5)

        assert length(reranked) <= 5
      end
    end
  end
end
