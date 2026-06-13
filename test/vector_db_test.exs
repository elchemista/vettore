defmodule VettoreDBTest do
  use ExUnit.Case, async: true

  alias Vettore.{Collection, Embedding, Result}

  describe "Vettore.Collection" do
    test "top-level Vettore API creates and searches collections" do
      assert {:ok, collection} = Vettore.new(dimensions: 2, metric: :l2)

      assert :ok =
               Vettore.put_many(collection, [
                 %{id: "near", vector: [0.0, 0.0]},
                 %{id: "far", vector: [10.0, 10.0]}
               ])

      assert {:ok, %Embedding{id: "near"}} = Vettore.get(collection, "near")
      assert {:ok, [%Result{id: "near"}]} = Vettore.search(collection, [0.0, 0.0], limit: 1)

      assert {:ok, [%Result{id: "near"}]} =
               Vettore.hybrid_search(collection, [0.0, 0.0],
                 generators: [quantized: [candidates: 2]],
                 limit: 1
               )
    end

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

      assert is_reference(flat.index_state)
      assert is_reference(hnsw.index_state)

      assert :ok = Collection.put_many(flat, embeddings)
      assert :ok = Collection.put_many(hnsw, embeddings)

      assert {:ok, [%Result{id: "near"}]} = Collection.search(flat, [0.0, 0.0], limit: 1)
      assert {:ok, [%Result{id: "near"}]} = Collection.search(hnsw, [0.0, 0.0], limit: 1)
    end

    test "flat native index mirrors deletes from the canonical ETS store" do
      {:ok, collection} = Collection.new(name: :flat_delete, dimensions: 2, metric: :l2)

      assert :ok =
               Collection.put_many(collection, [
                 %Embedding{id: "delete_me", vector: [0.0, 0.0]},
                 %Embedding{id: "keep_me", vector: [1.0, 1.0]}
               ])

      assert :ok = Collection.delete(collection, "delete_me")

      assert {:ok, [%Result{id: "keep_me"}]} =
               Collection.search(collection, [0.0, 0.0], limit: 1)
    end

    test "hnsw index accepts configurable graph and search parameters" do
      options = [m: 4, m0: 8, ef_construction: 16, ef_search: 8, max_level: 4]

      assert {:ok, collection} =
               Collection.new(
                 name: :hnsw_options,
                 dimensions: 2,
                 metric: :l2,
                 index: :hnsw,
                 index_options: options
               )

      assert collection.index_options == options

      assert :ok =
               Collection.put_many(collection, [
                 %Embedding{id: "near", vector: [0.0, 0.0]},
                 %Embedding{id: "far", vector: [5.0, 5.0]}
               ])

      assert {:ok, [%Result{id: "near"}]} =
               Collection.search(collection, [0.0, 0.0], limit: 1)

      assert {:error, :invalid_hnsw_options} =
               Collection.new(
                 name: :bad_hnsw_options,
                 dimensions: 2,
                 metric: :l2,
                 index: :hnsw,
                 index_options: [m: 0]
               )
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

      assert {:ok, %Embedding{binary_vector: [3]}} = Collection.get(collection, "exact")

      assert {:ok, [%Result{id: "exact", distance: distance}]} =
               Collection.quantized_search(collection, [1.0, 1.0],
                 candidates: 2,
                 limit: 1
               )

      assert distance == 0.0
    end

    test "multi-vector search ranks records with ColBERT-style late interaction" do
      {:ok, collection} =
        Collection.new(
          name: :late_interaction,
          dimensions: 2,
          metric: :inner_product,
          index: :flat
        )

      assert :ok =
               Collection.put_many(collection, [
                 %Embedding{
                   id: "both_axes",
                   vectors: [[1.0, 0.0], [0.0, 1.0]],
                   metadata: %{kind: :best}
                 },
                 %Embedding{
                   id: "one_axis",
                   vectors: [[1.0, 0.0], [-1.0, 0.0]]
                 },
                 %Embedding{
                   id: "opposite",
                   vectors: [[-1.0, 0.0], [0.0, -1.0]]
                 }
               ])

      assert {:ok, %Embedding{} = embedding} = Collection.get(collection, "both_axes")
      assert embedding.vector == [0.5, 0.5]
      assert embedding.vectors == [[1.0, 0.0], [0.0, 1.0]]

      assert {:ok,
              [
                %Result{
                  id: "both_axes",
                  score: 2.0,
                  distance: nil,
                  metric: :inner_product,
                  metadata: %{kind: :best}
                },
                %Result{id: "one_axis", score: 1.0}
              ]} =
               Collection.multi_vector_search(collection, [[1.0, 0.0], [0.0, 1.0]], limit: 2)
    end

    test "hybrid search combines generators and exact reranks unique candidates" do
      {:ok, collection} =
        Collection.new(name: :hybrid_exact, dimensions: 3, metric: :l2, index: :flat)

      assert :ok =
               Collection.put_many(collection, [
                 %Embedding{id: "exact", vector: [1.0, 0.0, 0.0]},
                 %Embedding{id: "same_prefix", vector: [1.0, 5.0, 0.0]},
                 %Embedding{id: "same_bits", vector: [10.0, 10.0, 10.0]},
                 %Embedding{id: "opposite", vector: [-1.0, -1.0, -1.0]}
               ])

      assert {:ok, [%Result{id: "exact", distance: distance} | _]} =
               Collection.hybrid_search(collection, [1.0, 0.0, 0.0],
                 generators: [
                   funnel: [stages: [1], candidates: 2],
                   quantized: [candidates: 3]
                 ],
                 rerank: :exact,
                 limit: 3
               )

      assert distance == 0.0
    end

    test "hybrid search can rerank generated candidates with multi-vector scoring" do
      {:ok, collection} =
        Collection.new(
          name: :hybrid_multi_vector,
          dimensions: 2,
          metric: :inner_product,
          index: :flat
        )

      assert :ok =
               Collection.put_many(collection, [
                 %Embedding{id: "both_axes", vectors: [[1.0, 0.0], [0.0, 1.0]]},
                 %Embedding{id: "one_axis", vectors: [[1.0, 0.0], [-1.0, 0.0]]},
                 %Embedding{id: "opposite", vectors: [[-1.0, 0.0], [0.0, -1.0]]}
               ])

      query_vectors = [[1.0, 0.0], [0.0, 1.0]]

      assert {:ok, [%Result{id: "both_axes", score: 2.0} | _]} =
               Collection.hybrid_search(collection, [1.0, 1.0],
                 generators: [quantized: [candidates: 3]],
                 rerank: {:multi_vector, query_vectors},
                 limit: 2
               )
    end

    test "hybrid search hnsw generator requires an hnsw collection" do
      {:ok, collection} =
        Collection.new(name: :hybrid_hnsw_guard, dimensions: 2, metric: :l2, index: :flat)

      assert :ok = Collection.put(collection, %Embedding{id: "near", vector: [0.0, 0.0]})

      assert {:error, :hnsw_index_required} =
               Collection.hybrid_search(collection, [0.0, 0.0],
                 generators: [:hnsw],
                 limit: 1
               )
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
          index: :hnsw,
          index_options: [m: 4, m0: 8, ef_construction: 16, ef_search: 8, max_level: 4]
        )

      assert :ok =
               Collection.put_many(collection, [
                 %Embedding{id: "a_near", vector: [1.0, 1.0]},
                 %Embedding{id: "b_far", vector: [5.0, 5.0]}
               ])

      assert :ok = Collection.snapshot(collection, path)
      assert {:ok, loaded} = Collection.load_snapshot(path)
      assert loaded.index == :hnsw

      assert loaded.index_options == [
               m: 4,
               m0: 8,
               ef_construction: 16,
               ef_search: 8,
               max_level: 4
             ]

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
