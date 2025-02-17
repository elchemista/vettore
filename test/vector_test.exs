defmodule VettoreTest do
  use ExUnit.Case
  doctest Vettore

  test "CRUD operations" do
    # Create a new in-memory DB
    db = Vettore.new_db()

    # Create a new collection
    assert {:ok, {}} = Vettore.create_collection(db, "my_collection", 3, "euclidean")

    # Insert embeddings into the collection
    assert {:ok, {}} =
             Vettore.insert_embedding(db, "my_collection", "emb1", [1.0, 2.0, 3.0], %{
               "info" => "test"
             })

    assert {:ok, {}} = Vettore.insert_embedding(db, "my_collection", "emb2", [2.0, 3.0, 4.0], nil)

    # Retrieve all embeddings from the collection
    assert {:ok, embeddings} = Vettore.get_embeddings(db, "my_collection")

    # Assert that exactly two embeddings are present
    assert length(embeddings) == 2

    # Assert that the embeddings contain the expected data
    assert Enum.any?(embeddings, fn
             {"emb1", [1.0, 2.0, 3.0], %{"info" => "test"}} -> true
             _ -> false
           end)

    assert Enum.any?(embeddings, fn
             {"emb2", [2.0, 3.0, 4.0], nil} -> true
             _ -> false
           end)

    # Retrieve a specific embedding by ID
    assert {:ok, {id, vector, metadata}} =
             Vettore.get_embedding_by_id(db, "my_collection", "emb1")

    assert id == "emb1"
    assert vector == [1.0, 2.0, 3.0]
    assert metadata == %{"info" => "test"}

    # Perform a similarity search
    assert {:ok, top_results} = Vettore.similarity_search(db, "my_collection", [1.0, 2.0, 3.0], 2)

    # Assert that the top result is the embedding with the same vector
    [{"emb1", score1}, {"emb2", score2}] = top_results

    assert length(top_results) == 2
    assert score1 <= score2
    assert "emb1" == "emb1"
    assert "emb2" == "emb2"
  end

  test "HNSW operations" do
    # Create a new in-memory DB

    db = Vettore.new_db()

    # Create a HNSW-based collection
    assert {:ok, {}} = Vettore.create_collection(db, "mycoll", 3, "hnsw")

    assert {:ok, {}} =
             Vettore.insert_embedding(db, "mycoll", "vec1", [1.0, 2.0, 3.0], %{"meta" => "test"})

    assert {:ok, {}} = Vettore.insert_embedding(db, "mycoll", "vec2", [2.0, 3.0, 4.0], nil)
    assert {:ok, {}} = Vettore.insert_embedding(db, "mycoll", "vec3", [3.0, 4.0, 5.0], nil)

    assert {:ok, top_results} = Vettore.similarity_search(db, "mycoll", [1.0, 2.0, 3.0], 2)

    [{"vec1", score1}, {"vec2", score2}] = top_results

    assert length(top_results) == 2
    assert score1 <= score2
  end

  test "Binary operations" do
    # Create a new in-memory DB

    db = Vettore.new_db()

    # Create a HNSW-based collection
    assert {:ok, {}} = Vettore.create_collection(db, "mycoll", 3, "binary")

    assert {:ok, {}} =
             Vettore.insert_embedding(db, "mycoll", "vec1", [1.0, 2.0, 3.0], %{"meta" => "test"})

    assert {:ok, {}} = Vettore.insert_embedding(db, "mycoll", "vec2", [2.0, 3.0, 4.0], nil)
    assert {:ok, {}} = Vettore.insert_embedding(db, "mycoll", "vec3", [3.0, 4.0, 5.0], nil)

    assert {:ok, top_results} = Vettore.similarity_search(db, "mycoll", [1.0, 2.0, 3.0], 2)

    [{"vec1", score1}, {"vec2", score2}] = top_results

    assert length(top_results) == 2
    assert score1 <= score2
  end
end
