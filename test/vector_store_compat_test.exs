defmodule VettoreStoreCompatTest do
  use ExUnit.Case, async: false

  alias Vettore.{Embedding, Store.ETS}

  describe "ETS store boundary" do
    test "owner helpers validate paths and tolerate foreign or stopped processes" do
      assert {:error, :invalid_snapshot_path} = Vettore.ETSOwner.load_table("")
      refute Vettore.ETSOwner.alive?(:bad)

      foreign =
        spawn(fn ->
          receive do
            :stop -> :ok
          end
        end)

      assert :ok = Vettore.ETSOwner.close(foreign)
      assert Process.alive?(foreign)
      send(foreign, :stop)

      assert {:ok, {owner, _table}} = Vettore.ETSOwner.start_table(:owner_test, [:set])
      assert :ok = Vettore.ETSOwner.close(owner)
      assert :ok = Vettore.ETSOwner.close(owner)
    end

    test "supports canonical CRUD, fold, count, configuration, and close" do
      assert {:ok, state} = ETS.new(%{compressed: false, dimensions: 2})
      assert ETS.alive?(state)
      assert ETS.count(state) == 0

      assert :ok = ETS.put(state, %Embedding{value: "value-id", vector: [1.0, 2.0]})

      assert {:ok, %Embedding{id: "value-id", value: "value-id"}} =
               ETS.get(state, "value-id")

      assert :ok =
               ETS.put_many(state, [
                 %Embedding{id: "a", vector: [0.0, 0.0]},
                 %Embedding{id: "b", vector: [1.0, 1.0]}
               ])

      assert ETS.count(state) == 3
      assert {:ok, ids} = ETS.fold(state, [], fn embedding, acc -> [embedding.id | acc] end)
      assert Enum.sort(ids) == ["a", "b", "value-id"]
      assert {:ok, rows} = ETS.all(state)
      assert length(rows) == 3
      assert :ok = ETS.configure(state, %{dimensions: 2, metric: :l2})
      assert :ok = ETS.delete(state, "a")
      assert {:error, :not_found} = ETS.get(state, "a")
      assert ETS.count(state) == 2

      assert :ok = ETS.close(state)
      refute ETS.alive?(state)
      refute ETS.alive?(:bad)
      assert ETS.count(state) == 0
      assert {:error, :closed} = ETS.get(state, "b")
      assert {:error, :closed} = ETS.put(state, %Embedding{id: "c", vector: [0.0, 0.0]})
      assert {:error, :closed} = ETS.delete(state, "b")
      assert {:error, :closed} = ETS.fold(state, 0, fn _embedding, count -> count + 1 end)
    end

    test "batch insert validates identifiers and duplicates atomically" do
      assert {:ok, state} = ETS.new(%{})

      assert {:error, :missing_id} = ETS.put(state, %Embedding{vector: [1.0]})

      assert {:error, :missing_id} =
               ETS.put_many(state, [%Embedding{id: "a", vector: [1.0]}, %Embedding{vector: [2.0]}])

      assert {:ok, []} = ETS.all(state)

      assert {:error, :duplicate_id} =
               ETS.put_many(state, [
                 %Embedding{id: "a", vector: [1.0]},
                 %Embedding{id: "a", vector: [2.0]}
               ])

      assert {:ok, []} = ETS.all(state)
      assert :ok = ETS.put(state, %Embedding{id: "a", vector: [1.0]})
      assert {:error, :duplicate_id} = ETS.put(state, %Embedding{id: "a", vector: [2.0]})

      assert {:error, :duplicate_id} =
               ETS.put_many(state, [%Embedding{id: "b"}, %Embedding{id: "a"}])

      assert {:error, :not_found} = ETS.get(state, "b")
      assert :ok = ETS.close(state)
    end

    test "snapshot creates parent directories and verifies round trips" do
      directory =
        Path.join(
          System.tmp_dir!(),
          "vettore-store-#{System.unique_integer([:positive, :monotonic])}"
        )

      path = Path.join(directory, "nested/store.ets")
      on_exit(fn -> File.rm_rf(directory) end)

      assert {:ok, state} = ETS.new(%{dimensions: 1, metric: :l2})
      assert :ok = ETS.put(state, %Embedding{id: "a", vector: [1.0]})
      assert :ok = ETS.snapshot(state, path)
      assert File.exists?(path)
      assert {:ok, {loaded, %{dimensions: 1, metric: :l2}}} = ETS.load_snapshot(path)
      assert {:ok, %Embedding{id: "a"}} = ETS.get(loaded, "a")
      assert :ok = ETS.close(state)
      assert :ok = ETS.close(loaded)

      assert {:error, :invalid_snapshot_path} = ETS.snapshot(state, "")
      assert {:error, :invalid_snapshot_path} = ETS.load_snapshot("")
      assert {:error, _reason} = ETS.load_snapshot(Path.join(directory, "missing.ets"))
    end
  end

  describe "compatibility database API" do
    test "top-level adaptive wrappers and default arguments delegate safely" do
      assert {:ok, collection} = Vettore.new(dimensions: 2, metric: :l2)
      assert :ok = Vettore.put(collection, %{id: "origin", vector: [0.0, 0.0]})
      assert {:ok, prepared} = Vettore.prepare_query(collection, [0.0, 0.0])
      assert prepared == [0.0, 0.0]
      assert {:ok, [_]} = Vettore.search(collection, [0.0, 0.0])
      assert {:ok, [_]} = Vettore.funnel_search(collection, [0.0, 0.0])
      assert {:ok, [_]} = Vettore.quantized_search(collection, [0.0, 0.0])
      assert {:ok, [_]} = Vettore.multi_vector_search(collection, [[0.0, 0.0]])
      assert {:ok, [_]} = Vettore.hybrid_search(collection, [0.0, 0.0])
      assert :ok = Vettore.close(collection)
    end

    test "covers collection lifecycle, CRUD, search, and reranking" do
      db = Vettore.new()

      assert {:ok, "docs"} =
               Vettore.create_collection(db, "docs", 2, :hnsw,
                 index_options: [m: 4, m0: 8, ef_construction: 16, ef_search: 16]
               )

      assert {:error, :collection_already_exists} =
               Vettore.create_collection(db, "docs", 2, :l2)

      assert {:ok, ["a", "b"]} =
               Vettore.batch(db, "docs", [
                 %Embedding{id: "a", vector: [0.0, 0.0], metadata: %{kind: :origin}},
                 %Embedding{id: "b", vector: [1.0, 1.0], metadata: %{kind: :unit}}
               ])

      assert {:ok, %Embedding{id: "a"}} = Vettore.get_by_value(db, "docs", "a")
      assert {:ok, %Embedding{id: "b"}} = Vettore.get_by_vector(db, "docs", [1.0, 1.0])
      assert {:error, :not_found} = Vettore.get_by_vector(db, "docs", [9.0, 9.0])
      assert {:ok, records} = Vettore.get_all(db, "docs")

      assert Enum.sort_by(records, &elem(&1, 0)) == [
               {"a", [0.0, 0.0], %{kind: :origin}},
               {"b", [1.0, 1.0], %{kind: :unit}}
             ]

      assert {:ok, [{"a", _score}]} =
               Vettore.similarity_search(db, "docs", [0.0, 0.0], limit: 1)

      assert {:ok, [{"a", 0.9}]} =
               Vettore.rerank(db, "docs", [{"a", 0.9}, {"b", 0.8}], limit: 1, alpha: 0.5)

      assert {:error, :invalid_options} =
               Vettore.rerank(db, "docs", [{"a", 0.9}], unknown: true)

      assert {:ok, "a"} = Vettore.delete(db, "docs", "a")
      assert {:error, :not_found} = Vettore.get_by_value(db, "docs", "a")
      assert {:ok, "docs"} = Vettore.delete_collection(db, "docs")
      assert {:error, :collection_not_found} = Vettore.delete_collection(db, "docs")
      assert {:error, :collection_not_found} = Vettore.get_all(db, "missing")
      assert :ok = Vettore.close(db)
      assert {:error, :closed} = Vettore.get_all(db, "docs")
    end

    test "invalid compatibility arguments return tagged errors" do
      db = Vettore.new()
      embedding = %Embedding{id: "a", vector: [1.0]}

      assert {:error, :invalid_options} =
               Vettore.create_collection(db, "docs", 1, :l2, [:bad])

      assert {:error, :invalid_options} =
               Vettore.create_collection(db, "docs", 1, :l2, unknown: true)

      assert {:error, :invalid_options} =
               Vettore.create_collection(db, "docs", 1, :l2, score: :raw, score: :similarity)

      assert {:error, :invalid_options} =
               Vettore.rerank(db, "docs", [], limit: 1, limit: 2)

      assert {:error, :invalid_arguments} = Vettore.create_collection(:bad, "docs", 1, :l2)
      assert {:error, :invalid_options} = Vettore.new(:bad)
      assert {:error, :invalid_arguments} = Vettore.insert(:bad, "docs", embedding)
      assert {:error, :invalid_arguments} = Vettore.batch(:bad, "docs", [embedding])
      assert {:error, :invalid_arguments} = Vettore.get_by_value(:bad, "docs", "a")
      assert {:error, :invalid_arguments} = Vettore.get_by_vector(:bad, "docs", [1.0])
      assert {:error, :invalid_arguments} = Vettore.get_all(:bad, "docs")
      assert {:error, :invalid_arguments} = Vettore.delete(:bad, "docs", "a")
      assert {:error, :invalid_arguments} = Vettore.similarity_search(:bad, "docs", [1.0])
      assert {:error, :invalid_arguments} = Vettore.rerank(:bad, "docs", [])
      assert {:error, :invalid_arguments} = Vettore.delete_collection(:bad, "docs")
      assert {:error, :invalid_resource} = Vettore.close(:bad)
      assert {:error, :invalid_snapshot} = Vettore.snapshot(:bad, "path")
      assert true = Vettore.ETSOwner.insert(db.owner, {:unrelated, :row})
      assert :ok = Vettore.close(db)
      assert :ok = Vettore.close(db)
    end

    test "legacy metric aliases map to supported canonical metrics" do
      db = Vettore.new()

      for {name, metric} <- [{"euclidean", :euclidean}, {"binary", :binary}, {"dot", :dot}] do
        assert {:ok, ^name} = Vettore.create_collection(db, name, 2, metric)
      end

      assert :ok = Vettore.close(db)
    end
  end
end
