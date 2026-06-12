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

    test "funnel search uses prefix candidates and reranks with full vectors" do
      {:ok, collection} =
        Collection.new(name: :funnel, dimensions: 3, metric: :l2, index: :flat)

      assert :ok =
               Collection.put_many(collection, [
                 %Embedding{id: "exact", vector: [1.0, 0.0, 0.0], metadata: %{kind: :best}},
                 %Embedding{id: "prefix", vector: [1.0, 5.0, 0.0]},
                 %Embedding{id: "far", vector: [-1.0, 0.0, 0.0]}
               ])

      assert {:ok, [%Result{id: "exact", metadata: %{kind: :best}}]} =
               Collection.funnel_search(collection, [1.0, 0.0, 0.0],
                 stages: [1],
                 candidates: 2,
                 limit: 1
               )
    end

    test "binary quantized search uses sign-bit candidates and exact reranking" do
      {:ok, collection} =
        Collection.new(name: :quantized, dimensions: 2, metric: :l2, index: :flat)

      assert :ok =
               Collection.put_many(collection, [
                 %Embedding{id: "exact", vector: [1.0, 1.0]},
                 %Embedding{id: "same_bits_far", vector: [100.0, 100.0]},
                 %Embedding{id: "opposite", vector: [-1.0, -1.0]}
               ])

      assert {:ok, %Embedding{binary_vector: [1, 1]}} = Collection.get(collection, "exact")

      assert {:ok, [%Result{id: "exact", distance: distance}]} =
               Collection.quantized_search(collection, [1.0, 1.0],
                 candidates: 2,
                 limit: 1
               )

      assert distance == 0.0
    end

    test "collections are isolated" do
      {:ok, first} = Collection.new(name: :first, dimensions: 1, metric: :l2)
      {:ok, second} = Collection.new(name: :second, dimensions: 1, metric: :l2)

      assert :ok = Collection.put(first, %Embedding{id: "same", vector: [1.0]})
      assert :ok = Collection.put(second, %Embedding{id: "same", vector: [2.0]})

      assert {:ok, %Embedding{vector: [1.0]}} = Collection.get(first, "same")
      assert {:ok, %Embedding{vector: [2.0]}} = Collection.get(second, "same")
    end

    test "ets store can be created with compression enabled" do
      {:ok, collection} =
        Collection.new(
          name: :compressed,
          dimensions: 2,
          metric: :l2,
          compressed: true
        )

      assert :ets.info(collection.store_state.table, :compressed) == true
      assert :ok = Collection.put(collection, %Embedding{id: "a", vector: [1.0, 2.0]})
      assert {:ok, %Embedding{id: "a"}} = Collection.get(collection, "a")
    end

    test "snapshot saves and loads the canonical ETS store" do
      path =
        Path.join(System.tmp_dir!(), "vettore-flat-#{System.unique_integer([:positive])}.ets")

      on_exit(fn -> File.rm(path) end)

      {:ok, collection} =
        Collection.new(
          name: :snapshot_flat,
          dimensions: 2,
          metric: :cosine,
          normalize: :l2,
          compressed: true
        )

      assert :ok =
               Collection.put_many(collection, [
                 %Embedding{id: "right", vector: [1.0, 0.0], metadata: %{axis: :x}},
                 %Embedding{id: "up", vector: [0.0, 1.0]}
               ])

      assert :ok = Collection.snapshot(collection, path)
      assert File.exists?(path)

      assert {:ok, loaded} = Collection.load_snapshot(path)
      assert loaded.name == :snapshot_flat
      assert loaded.metric == :cosine
      assert loaded.normalize == :l2
      assert :ets.info(loaded.store_state.table, :compressed) == true

      assert {:ok, %Embedding{id: "right", metadata: %{axis: :x}}} =
               Collection.get(loaded, "right")

      assert {:ok, [%Result{id: "right", score: 1.0}]} =
               Collection.search(loaded, [1.0, 0.0], limit: 1)
    end

    test "snapshot load rebuilds hnsw index from ETS records" do
      path =
        Path.join(System.tmp_dir!(), "vettore-hnsw-#{System.unique_integer([:positive])}.ets")

      on_exit(fn -> File.rm(path) end)

      {:ok, collection} =
        Collection.new(
          name: :snapshot_hnsw,
          dimensions: 2,
          metric: :l2,
          index: :hnsw
        )

      assert :ok =
               Collection.put_many(collection, [
                 %Embedding{id: "a_near", vector: [1.0, 1.0]},
                 %Embedding{id: "b_far", vector: [5.0, 5.0]}
               ])

      assert :ok = Collection.snapshot(collection, path)
      assert {:ok, loaded} = Collection.load_snapshot(path)
      assert loaded.index == :hnsw

      assert {:ok, [%Result{id: "a_near"}]} =
               Collection.search(loaded, [1.0, 1.0], limit: 1)
    end

    test "snapshot load can override the restored index" do
      path =
        Path.join(System.tmp_dir!(), "vettore-override-#{System.unique_integer([:positive])}.ets")

      on_exit(fn -> File.rm(path) end)

      {:ok, collection} =
        Collection.new(
          name: :snapshot_override,
          dimensions: 2,
          metric: :l2,
          index: :hnsw
        )

      assert :ok = Collection.put(collection, %Embedding{id: "a", vector: [1.0, 2.0]})
      assert :ok = Collection.snapshot(collection, path)

      assert {:ok, loaded} = Collection.load_snapshot(path, index: :flat)
      assert loaded.index == :flat
      assert loaded.index_mod == Vettore.Index.Flat
      assert {:ok, %Embedding{id: "a"}} = Collection.get(loaded, "a")
    end

    test "snapshot load returns an error for missing files" do
      path =
        Path.join(System.tmp_dir!(), "vettore-missing-#{System.unique_integer([:positive])}.ets")

      assert {:error, _reason} = Collection.load_snapshot(path)
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
