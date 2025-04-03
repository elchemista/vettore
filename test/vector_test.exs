defmodule VettoreTest do
  use ExUnit.Case, async: true
  alias Vettore.Embedding

  @moduletag :vettore

  test "CRUD operations with Euclidean" do
    db = Vettore.new_db()

    assert {:ok, "euclidean_coll"} =
             Vettore.create_collection(db, "euclidean_coll", 3, "euclidean")

    assert {:ok, "emb1"} =
             Vettore.insert_embedding(db, "euclidean_coll", %Embedding{
               id: "emb1",
               vector: [1.0, 2.0, 3.0],
               metadata: %{"info" => "test"}
             })

    assert {:ok, "emb2"} =
             Vettore.insert_embedding(db, "euclidean_coll", %Embedding{
               id: "emb2",
               vector: [2.0, 3.0, 4.0],
               metadata: nil
             })

    # Retrieve all embeddings
    assert {:ok, all_embs} = Vettore.get_embeddings(db, "euclidean_coll")
    assert length(all_embs) == 2

    # Confirm "emb1" is present
    assert Enum.any?(all_embs, fn
             {"emb1", [1.0, 2.0, 3.0], %{"info" => "test"}} -> true
             _ -> false
           end)

    # Get specific embedding
    assert {:ok, %Embedding{id: id, vector: vec, metadata: meta}} =
             Vettore.get_embedding_by_id(db, "euclidean_coll", "emb1")

    assert id == "emb1"
    assert vec == [1.0, 2.0, 3.0]
    assert meta == %{"info" => "test"}

    # Similarity search (Euclidean => smaller is better)
    assert {:ok, top2} =
             Vettore.similarity_search(db, "euclidean_coll", [1.0, 2.0, 3.0], limit: 2)

    assert length(top2) == 2
    [{"emb1", score1}, {"emb2", score2}] = top2
    assert score1 >= score2

    assert {:ok, "emb1"} = Vettore.delete_embedding_by_id(db, "euclidean_coll", "emb1")
    assert {:error, _} = Vettore.get_embedding_by_id(db, "euclidean_coll", "emb1")
  end

  test "metadata filtering (Euclidean example)" do
    db = Vettore.new_db()

    assert {:ok, "filter_coll"} =
             Vettore.create_collection(db, "filter_coll", 2, "euclidean")

    # Insert 2 embeddings, only one has category="special"
    assert {:ok, "f1"} =
             Vettore.insert_embedding(db, "filter_coll", %Embedding{
               id: "f1",
               vector: [0.1, 0.1],
               metadata: %{"category" => "special"}
             })

    assert {:ok, "f2"} =
             Vettore.insert_embedding(db, "filter_coll", %Embedding{
               id: "f2",
               vector: [5.0, 5.0],
               metadata: %{"category" => "other"}
             })

    # Normal search (limit=2)
    assert {:ok, overall} =
             Vettore.similarity_search(db, "filter_coll", [0.0, 0.0], limit: 2)

    # Expect 2 results
    assert length(overall) == 2

    # Filter => only the "special" one
    assert {:ok, only_special} =
             Vettore.similarity_search(
               db,
               "filter_coll",
               [0.0, 0.0],
               limit: 2,
               filter: %{"category" => "special"}
             )

    assert [{"f1", _score}] = only_special

    # Filter => "missing" key => expect empty
    assert {:ok, []} =
             Vettore.similarity_search(
               db,
               "filter_coll",
               [0.0, 0.0],
               limit: 2,
               filter: %{"unknown" => "does_not_exist"}
             )
  end

  test "all embeddings without metadata" do
    db = Vettore.new_db()

    assert {:ok, "no_meta_coll"} =
             Vettore.create_collection(db, "no_meta_coll", 2, "euclidean")

    # Insert embeddings that have no metadata
    for i <- 1..3 do
      id = "nm#{i}"

      assert {:ok, ^id} =
               Vettore.insert_embedding(db, "no_meta_coll", %Embedding{
                 id: id,
                 vector: [i * 1.0, i * 2.0],
                 metadata: nil
               })
    end

    # Try to filter => should be empty since none has metadata
    assert {:ok, []} =
             Vettore.similarity_search(
               db,
               "no_meta_coll",
               [1.0, 1.0],
               filter: %{"irrelevant" => "something"}
             )
  end

  test "checking limit is respected" do
    db = Vettore.new_db()

    assert {:ok, "many_coll"} =
             Vettore.create_collection(db, "many_coll", 2, "euclidean")

    # Insert 5 embeddings
    for i <- 1..5 do
      id = "m#{i}"

      assert {:ok, ^id} =
               Vettore.insert_embedding(db, "many_coll", %Embedding{
                 id: id,
                 vector: [i * 1.0, i * 1.0],
                 metadata: %{"kind" => "test"}
               })
    end

    # Request limit=3
    assert {:ok, top3} =
             Vettore.similarity_search(
               db,
               "many_coll",
               [0.0, 0.0],
               limit: 3,
               filter: %{"kind" => "test"}
             )

    # Expect exactly 3 results
    assert length(top3) == 3
  end

  test "HNSW operations" do
    db = Vettore.new_db()
    assert {:ok, "hnsw_coll"} = Vettore.create_collection(db, "hnsw_coll", 3, "hnsw")

    assert {:ok, "vec1"} =
             Vettore.insert_embedding(db, "hnsw_coll", %Embedding{
               id: "vec1",
               vector: [1.0, 2.0, 3.0],
               metadata: %{"meta" => "test"}
             })

    assert {:ok, "vec2"} =
             Vettore.insert_embedding(db, "hnsw_coll", %Embedding{
               id: "vec2",
               vector: [2.0, 3.0, 4.0],
               metadata: nil
             })

    assert {:ok, "vec3"} =
             Vettore.insert_embedding(db, "hnsw_coll", %Embedding{
               id: "vec3",
               vector: [3.0, 4.0, 5.0],
               metadata: nil
             })

    # Normal HNSW search
    assert {:ok, top2} =
             Vettore.similarity_search(db, "hnsw_coll", [1.0, 2.0, 3.0], limit: 2)

    assert length(top2) == 2

    [{"vec1", score1}, {"vec2", score2}] = top2
    assert score1 >= score2

    # Attempt filter => error
    assert {:error, _} =
             Vettore.similarity_search(
               db,
               "hnsw_coll",
               [1.0, 2.0, 3.0],
               limit: 2,
               filter: %{"meta" => "test"}
             )
  end

  test "Binary operations" do
    db = Vettore.new_db()
    assert {:ok, "binary_coll"} = Vettore.create_collection(db, "binary_coll", 3, "binary")

    assert {:ok, "vec1"} =
             Vettore.insert_embedding(db, "binary_coll", %Embedding{
               id: "vec1",
               vector: [1.0, 2.0, 3.0],
               metadata: %{"meta" => "test"}
             })

    assert {:ok, "vec2"} =
             Vettore.insert_embedding(db, "binary_coll", %Embedding{
               id: "vec2",
               vector: [2.0, 3.0, 4.0],
               metadata: nil
             })

    assert {:ok, "vec3"} =
             Vettore.insert_embedding(db, "binary_coll", %Embedding{
               id: "vec3",
               vector: [3.0, 4.0, 5.0],
               metadata: nil
             })

    # Hamming distance => lower is better
    assert {:ok, top2} = Vettore.similarity_search(db, "binary_coll", [1.0, 2.0, 3.0], limit: 2)
    assert length(top2) == 2

    [{"vec1", score1}, {"vec2", score2}] = top2
    assert score1 <= score2
  end

  test "Cosine operations" do
    db = Vettore.new_db()
    assert {:ok, "cosine_coll"} = Vettore.create_collection(db, "cosine_coll", 3, "cosine")

    assert {:ok, "cos1"} =
             Vettore.insert_embedding(db, "cosine_coll", %Embedding{
               id: "cos1",
               vector: [1.0, 2.0, 3.0],
               metadata: %{"desc" => "test"}
             })

    assert {:ok, "cos2"} =
             Vettore.insert_embedding(db, "cosine_coll", %Embedding{
               id: "cos2",
               vector: [2.0, 3.0, 4.0],
               metadata: nil
             })

    # Cosine => bigger dot product is better
    assert {:ok, results} =
             Vettore.similarity_search(db, "cosine_coll", [1.0, 2.0, 3.0], limit: 2)

    [{"cos1", dp1}, {"cos2", dp2}] = results
    assert dp1 >= dp2
  end

  test "Dot operations" do
    db = Vettore.new_db()
    assert {:ok, "dot_coll"} = Vettore.create_collection(db, "dot_coll", 3, "dot")

    assert {:ok, "dot1"} =
             Vettore.insert_embedding(db, "dot_coll", %Embedding{
               id: "dot1",
               vector: [1.0, 2.0, 3.0],
               metadata: %{"desc" => "test"}
             })

    assert {:ok, "dot2"} =
             Vettore.insert_embedding(db, "dot_coll", %Embedding{
               id: "dot2",
               vector: [2.0, 3.0, 4.0],
               metadata: nil
             })

    # Dot => bigger is better
    assert {:ok, top2} = Vettore.similarity_search(db, "dot_coll", [1.0, 2.0, 3.0], limit: 2)
    [{"dot2", score1}, {"dot1", score2}] = top2
    assert score1 >= score2
  end

  test "Binary with keep_embeddings option" do
    db = Vettore.new_db()

    # A "binary" collection that does NOT keep float vectors
    assert {:ok, "bin_no_keep"} =
             Vettore.create_collection(
               db,
               "bin_no_keep",
               3,
               "binary",
               keep_embeddings: false
             )

    assert {:ok, "nokey1"} =
             Vettore.insert_embedding(db, "bin_no_keep", %Embedding{
               id: "nokey1",
               vector: [1.0, 2.0, 3.0],
               metadata: nil
             })

    # If keep_embeddings is false (and distance="binary"), we expect the float vector is cleared
    assert {:ok, no_keep_embs} = Vettore.get_embeddings(db, "bin_no_keep")
    # There's only one embedding. The float vector should be empty.
    assert [{"nokey1", [], nil}] = no_keep_embs

    # "binary" collection that DOES keep float vectors
    assert {:ok, "bin_keep"} =
             Vettore.create_collection(
               db,
               "bin_keep",
               3,
               "binary",
               keep_embeddings: true
             )

    assert {:ok, "key1"} =
             Vettore.insert_embedding(db, "bin_keep", %Embedding{
               id: "key1",
               vector: [9.9, 8.8, 7.7],
               metadata: %{"foo" => "bar"}
             })

    assert {:ok, keep_embs} = Vettore.get_embeddings(db, "bin_keep")
    assert [{"key1", vec, %{"foo" => "bar"}}] = keep_embs

    expected = [9.9, 8.8, 7.7]

    for {exp, act} <- Enum.zip(expected, vec) do
      assert_in_delta(exp, act, 1.0e-4)
    end
  end

  test "MMR re-rank with Euclidean" do
    db = Vettore.new_db()
    assert {:ok, "mmr_coll"} = Vettore.create_collection(db, "mmr_coll", 3, "euclidean")

    assert {:ok, "m1"} =
             Vettore.insert_embedding(db, "mmr_coll", %Vettore.Embedding{
               id: "m1",
               vector: [1.0, 2.0, 3.0],
               metadata: %{"tag" => "A"}
             })

    assert {:ok, "m2"} =
             Vettore.insert_embedding(db, "mmr_coll", %Vettore.Embedding{
               id: "m2",
               vector: [2.0, 3.0, 4.0],
               metadata: %{"tag" => "B"}
             })

    assert {:ok, "m3"} =
             Vettore.insert_embedding(db, "mmr_coll", %Vettore.Embedding{
               id: "m3",
               vector: [3.0, 2.0, 1.0],
               metadata: nil
             })

    assert {:ok, top3} = Vettore.similarity_search(db, "mmr_coll", [2.1, 2.1, 2.1], limit: 3)

    assert {:ok, mmr_list} =
             Vettore.mmr_rerank(db, "mmr_coll", top3,
               limit: 2,
               alpha: 0.5
             )

    assert length(mmr_list) == 2
    [{id1, mmr_score1}, {id2, mmr_score2}] = mmr_list

    assert Enum.all?([id1, id2], &(&1 in ["m1", "m2", "m3"]))

    assert is_float(mmr_score1)
    assert is_float(mmr_score2)
  end
end
