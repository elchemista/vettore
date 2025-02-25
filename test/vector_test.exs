defmodule VettoreTest do
  use ExUnit.Case, async: true
  # doctest Vettore   # <-- remove or comment out if you don't want actual doctests

  alias Vettore.Embedding

  @moduletag :vettore

  test "CRUD operations with Euclidean" do
    db = Vettore.new_db()

    # Create a Euclidean-based collection
    assert {:ok, "euclidean_coll"} =
             Vettore.create_collection(db, "euclidean_coll", 3, "euclidean")

    # Insert embeddings into the collection
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

    # Similarity search (for Euclidean, smaller distance = better)
    assert {:ok, top2} = Vettore.similarity_search(db, "euclidean_coll", [1.0, 2.0, 3.0], 2)
    assert length(top2) == 2

    # top2 should be [{"emb1", dist1}, {"emb2", dist2}] with dist1 <= dist2
    [{"emb1", score1}, {"emb2", score2}] = top2
    assert score1 <= score2
  end

  test "HNSW operations" do
    db = Vettore.new_db()
    # Create an HNSW-based collection
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

    # Perform a similarity search
    assert {:ok, top2} = Vettore.similarity_search(db, "hnsw_coll", [1.0, 2.0, 3.0], 2)
    assert length(top2) == 2

    [{"vec1", score1}, {"vec2", score2}] = top2
    # Usually vec1 is closer
    assert score1 <= score2
  end

  test "Binary operations" do
    db = Vettore.new_db()
    # Create a collection with "binary" distance
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

    # Hamming distance search (lower = more similar)
    assert {:ok, top2} = Vettore.similarity_search(db, "binary_coll", [1.0, 2.0, 3.0], 2)
    assert length(top2) == 2

    [{"vec1", score1}, {"vec2", score2}] = top2
    # Typically, (1.0,2.0,3.0) will have distance 0 or lower to itself
    assert score1 <= score2
  end

  test "Cosine operations" do
    db = Vettore.new_db()
    # Create a collection with "cosine" distance
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

    # Cosine similarity => bigger dot product is better
    assert {:ok, results} = Vettore.similarity_search(db, "cosine_coll", [1.0, 2.0, 3.0], 2)
    # results might be [{"cos1", dp1}, {"cos2", dp2}] with dp1 >= dp2
    [{"cos1", dp1}, {"cos2", dp2}] = results
    assert dp1 >= dp2
  end

  test "Dot operations" do
    db = Vettore.new_db()
    # Create a collection with "dot" distance
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

    # Dot product => bigger is better
    assert {:ok, top2} = Vettore.similarity_search(db, "dot_coll", [1.0, 2.0, 3.0], 2)
    [{"dot2", score1}, {"dot1", score2}] = top2

    # Typically, "dot1" with itself is largest
    assert score1 >= score2
  end
end
