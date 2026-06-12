defmodule VettoreDBTest do
  use ExUnit.Case, async: true

  alias Vettore.{Collection, Embedding, Result}

  describe "Vettore.Collection" do
    test "stores records in ETS and searches with explicit result semantics" do
      assert {:ok, collection} =
               Collection.new(
                 name: :test_vectors,
                 dimensions: 2,
                 metric: :cosine,
                 normalize: :l2,
                 score: :raw
               )

      assert :ok =
               Collection.put_many(collection, [
                 %Embedding{id: "right", vector: [1.0, 0.0], metadata: %{"tag" => "axis"}},
                 %Embedding{id: "up", vector: [0.0, 1.0]},
                 %Embedding{id: "left", vector: [-1.0, 0.0]}
               ])

      assert {:error, :duplicate_id} =
               Collection.put(collection, %Embedding{id: "right", vector: [0.5, 0.5]})

      assert {:ok, %Embedding{id: "right", metadata: %{"tag" => "axis"}}} =
               Collection.get(collection, "right")

      assert {:ok, [%Result{id: "right", score: 1.0, distance: distance, metric: :cosine} | _]} =
               Collection.search(collection, [1.0, 0.0], limit: 2)

      assert distance == 0.0
    end

    test "duplicate vectors are allowed when ids are unique" do
      {:ok, collection} = Collection.new(name: :dupes, dimensions: 2, metric: :l2)

      assert :ok =
               Collection.put_many(collection, [
                 %Embedding{id: "a", vector: [1.0, 1.0]},
                 %Embedding{id: "b", vector: [1.0, 1.0]}
               ])

      assert {:ok, embeddings} = Collection.all(collection)
      assert Enum.map(embeddings, & &1.id) |> Enum.sort() == ["a", "b"]
    end

    test "flat and hnsw boundaries return compatible result shapes" do
      embeddings = [
        %Embedding{id: "near", vector: [0.0, 0.0]},
        %Embedding{id: "far", vector: [10.0, 10.0]}
      ]

      {:ok, flat} = Collection.new(name: :flat, dimensions: 2, metric: :l2, index: :flat)
      {:ok, hnsw} = Collection.new(name: :hnsw, dimensions: 2, metric: :l2, index: :hnsw)

      assert :ok = Collection.put_many(flat, embeddings)
      assert :ok = Collection.put_many(hnsw, embeddings)

      assert {:ok, [%Result{id: "near"}]} = Collection.search(flat, [0.0, 0.0], limit: 1)
      assert {:ok, [%Result{id: "near"}]} = Collection.search(hnsw, [0.0, 0.0], limit: 1)
    end

    test "collections are isolated" do
      {:ok, first} = Collection.new(name: :first, dimensions: 1, metric: :l2)
      {:ok, second} = Collection.new(name: :second, dimensions: 1, metric: :l2)

      assert :ok = Collection.put(first, %Embedding{id: "same", vector: [1.0]})
      assert :ok = Collection.put(second, %Embedding{id: "same", vector: [2.0]})

      assert {:ok, %Embedding{vector: [1.0]}} = Collection.get(first, "same")
      assert {:ok, %Embedding{vector: [2.0]}} = Collection.get(second, "same")
    end
  end

  describe "compat Vettore API" do
    test "legacy wrapper is backed by ETS collections" do
      db = Vettore.new()
      assert {:ok, "legacy"} = Vettore.create_collection(db, "legacy", 2, :cosine)
      assert {:ok, "a"} = Vettore.insert(db, "legacy", %Embedding{value: "a", vector: [1.0, 0.0]})
      assert {:ok, "b"} = Vettore.insert(db, "legacy", %Embedding{value: "b", vector: [0.0, 1.0]})

      assert {:error, :duplicate_id} =
               Vettore.insert(db, "legacy", %Embedding{value: "a", vector: [0.5, 0.5]})

      assert {:ok, [{"a", score} | _]} =
               Vettore.similarity_search(db, "legacy", [1.0, 0.0], limit: 1)

      assert score == 1.0
    end
  end
end
