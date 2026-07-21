defmodule Vettore.Test.FailingIndex do
  @moduledoc false

  def new(_metric, _opts), do: {:ok, make_ref()}
  def put(_collection, _embedding), do: {:error, :forced_index_failure}
  def put_many(_collection, _embeddings), do: {:error, :forced_index_failure}
  def delete(_collection, _id), do: :ok
  def search(_collection, _query, _opts), do: {:ok, []}
end

defmodule Vettore.Test.FailingDeleteStore do
  @moduledoc false

  alias Vettore.Store.ETS

  def new(config), do: ETS.new(config)
  def put(state, embedding), do: ETS.put(state, embedding)
  def put_many(state, embeddings), do: ETS.put_many(state, embeddings)
  def get(state, id), do: ETS.get(state, id)
  def delete(_state, _id), do: {:error, :forced_delete_failure}
  def all(state), do: ETS.all(state)
  def snapshot(state, path), do: ETS.snapshot(state, path)
  def load_snapshot(path), do: ETS.load_snapshot(path)
  def configure(state, config), do: ETS.configure(state, config)
  def close(state), do: ETS.close(state)
  def alive?(state), do: ETS.alive?(state)
end

defmodule VettoreHardeningTest do
  use ExUnit.Case, async: false

  alias Vettore.{Collection, Embedding, Result}

  @hnsw_options [m: 8, m0: 16, ef_construction: 200, ef_search: 200, max_level: 12]

  describe "HNSW correctness" do
    test "every inserted node remains reachable and results hydrate canonical fields" do
      assert {:ok, collection} =
               Collection.new(
                 dimensions: 1,
                 metric: :l2,
                 index: :hnsw,
                 index_options: @hnsw_options
               )

      on_exit(fn -> Vettore.close(collection) end)

      embeddings =
        Enum.map(0..99, fn value ->
          %Embedding{
            id: String.pad_leading(Integer.to_string(value), 3, "0"),
            value: "value-#{value}",
            vector: [value / 1],
            metadata: %{position: value}
          }
        end)

      assert :ok = Collection.put_many(collection, embeddings)
      assert {:ok, all_hits} = Collection.search(collection, [99.0], limit: 100)
      assert length(all_hits) == 100
      assert MapSet.size(MapSet.new(all_hits, & &1.id)) == 100

      for value <- 0..99 do
        id = String.pad_leading(Integer.to_string(value), 3, "0")

        assert {:ok,
                [
                  %Result{
                    id: ^id,
                    value: "value-" <> value_text,
                    metadata: %{position: ^value}
                  }
                ]} = Collection.search(collection, [value / 1], limit: 1)

        assert value_text == Integer.to_string(value)
      end
    end

    test "validates construction options and does not leak a store on index failure" do
      before_count = active_ets_owners()

      assert {:error, {:unsupported_hnsw_metric, :manhattan}} =
               Collection.new(dimensions: 2, metric: :manhattan, index: :hnsw)

      assert active_ets_owners() == before_count

      for options <- [
            [unknown: 1],
            [m: 8, m: 16],
            [m: 16, m0: 8],
            [m: 1_025],
            [ef_search: 1_000_001],
            [max_level: 65]
          ] do
        assert {:error, :invalid_hnsw_options} =
                 Collection.new(
                   dimensions: 2,
                   metric: :l2,
                   index: :hnsw,
                   index_options: options
                 )
      end
    end

    test "supports cosine and inner-product graph ordering" do
      for metric <- [:cosine, :inner_product] do
        assert {:ok, collection} =
                 Collection.new(
                   dimensions: 2,
                   metric: metric,
                   index: :hnsw,
                   index_options: [m: 4, m0: 8, ef_construction: 16, ef_search: 16]
                 )

        assert :ok =
                 Collection.put_many(collection, [
                   %{id: "best", vector: [1.0, 0.0]},
                   %{id: "other", vector: [0.0, 1.0]}
                 ])

        assert {:ok, [%Result{id: "best"}]} =
                 Collection.search(collection, [1.0, 0.0], limit: 1)

        assert :ok = Vettore.close(collection)
      end
    end
  end

  describe "ETS ownership and explicit lifecycle" do
    test "a collection survives the process that created it" do
      parent = self()

      {pid, monitor} =
        spawn_monitor(fn ->
          {:ok, collection} = Collection.new(dimensions: 2, metric: :l2)
          :ok = Collection.put(collection, %{id: "alive", vector: [1.0, 2.0]})
          send(parent, {:collection, collection})
        end)

      assert_receive {:collection, collection}
      assert_receive {:DOWN, ^monitor, :process, ^pid, :normal}
      assert Process.alive?(collection.store_state.owner)
      assert {:ok, %Embedding{id: "alive"}} = Collection.get(collection, "alive")
      assert :ok = Vettore.close(collection)
    end

    test "protected tables reject outside writes and close is idempotent" do
      assert {:ok, collection} = Collection.new(dimensions: 2, metric: :l2)
      assert :protected = :ets.info(collection.store_state.table, :protection)
      assert true = :ets.info(collection.store_state.table, :read_concurrency)
      refute :ets.info(collection.store_state.table, :write_concurrency)

      assert_raise ArgumentError, fn ->
        :ets.insert(
          collection.store_state.table,
          {{:record, "bypass"}, %Embedding{id: "bypass", vector: [0.0, 0.0]}}
        )
      end

      assert :ok = Vettore.close(collection)
      assert :ok = Vettore.close(collection)
      refute Process.alive?(collection.store_state.owner)
      assert {:error, :closed} = Collection.get(collection, "missing")
      assert {:error, :closed} = Collection.all(collection)
      assert {:error, :closed} = Collection.put(collection, %{id: "a", vector: [0.0, 0.0]})
      assert {:error, :closed} = Collection.delete(collection, "a")
      assert {:error, :closed} = Collection.search(collection, [0.0, 0.0], limit: 1)
      assert {:error, :closed} = Collection.snapshot(collection, temporary_path("closed"))
    end

    test "readers bypass the owner and read protected ETS concurrently" do
      assert {:ok, collection} = Collection.new(dimensions: 2, metric: :l2)

      embeddings =
        for index <- 1..100 do
          %{id: Integer.to_string(index), vector: [index * 1.0, 0.0]}
        end

      assert :ok = Collection.put_many(collection, embeddings)
      assert :ok = :sys.suspend(collection.store_state.owner)

      on_exit(fn ->
        if Process.alive?(collection.store_state.owner) do
          if Process.info(collection.store_state.owner, :status) == {:status, :suspended} do
            :sys.resume(collection.store_state.owner)
          end

          Vettore.close(collection)
        end
      end)

      results =
        1..100
        |> Task.async_stream(
          fn index -> Collection.get(collection, Integer.to_string(index)) end,
          max_concurrency: System.schedulers_online(),
          ordered: false
        )
        |> Enum.to_list()

      assert Enum.all?(results, fn
               {:ok, {:ok, %Embedding{}}} -> true
               _other -> false
             end)

      assert :ok = :sys.resume(collection.store_state.owner)
      assert :ok = Vettore.close(collection)
    end

    test "closing a compatibility database closes its child collections" do
      db = Vettore.new()
      assert {:ok, "docs"} = Vettore.create_collection(db, "docs", 2, :l2)

      [{{:collection, "docs"}, collection}] = :ets.lookup(db.table, {:collection, "docs"})
      assert Process.alive?(collection.store_state.owner)

      assert :ok = Vettore.close(db)
      refute Process.alive?(db.owner)
      refute Process.alive?(collection.store_state.owner)
      assert {:error, :closed} = Vettore.create_collection(db, "new", 2, :l2)
      assert {:error, :closed} = Vettore.delete_collection(db, "docs")
    end

    test "compatibility shutdown drains collections racing with creation" do
      before_count = active_ets_owners()
      db = Vettore.new()
      parent = self()

      creators =
        for index <- 1..40 do
          Task.async(fn ->
            send(parent, {:ready, self()})

            receive do
              :go -> Vettore.create_collection(db, "racing-#{index}", 1, :l2)
            end
          end)
        end

      Enum.each(creators, fn task ->
        assert_receive {:ready, pid} when pid == task.pid
      end)

      Enum.each(creators, &send(&1.pid, :go))
      closer = Task.async(fn -> Vettore.close(db) end)

      assert :ok = Task.await(closer)
      results = Enum.map(creators, &Task.await/1)

      assert Enum.all?(results, fn
               {:ok, "racing-" <> _suffix} -> true
               {:error, :closed} -> true
               _other -> false
             end)

      assert active_ets_owners() == before_count
    end
  end

  describe "atomic canonical/index changes" do
    test "rolls canonical inserts back when an index rejects them" do
      assert {:ok, collection} =
               Collection.new(
                 dimensions: 2,
                 metric: :l2,
                 index: Vettore.Test.FailingIndex
               )

      assert {:error, :forced_index_failure} =
               Collection.put(collection, %{id: "one", vector: [1.0, 1.0]})

      assert {:ok, []} = Collection.all(collection)

      assert {:error, :forced_index_failure} =
               Collection.put_many(collection, [
                 %{id: "two", vector: [2.0, 2.0]},
                 %{id: "three", vector: [3.0, 3.0]}
               ])

      assert {:ok, []} = Collection.all(collection)
      assert :ok = Vettore.close(collection)
    end

    test "restores the index when canonical deletion fails" do
      assert {:ok, collection} =
               Collection.new(
                 dimensions: 2,
                 metric: :l2,
                 store: Vettore.Test.FailingDeleteStore
               )

      assert :ok = Collection.put(collection, %{id: "kept", vector: [0.0, 0.0]})
      assert {:error, :forced_delete_failure} = Collection.delete(collection, "kept")
      assert {:ok, [%Result{id: "kept"}]} = Collection.search(collection, [0.0, 0.0], limit: 1)
      assert :ok = Vettore.close(collection)
    end

    test "batch validation and duplicate detection are atomic" do
      assert {:ok, collection} = Collection.new(dimensions: 2, metric: :l2)

      assert {:error, :dimension_mismatch} =
               Collection.put_many(collection, [
                 %{id: "valid", vector: [1.0, 1.0]},
                 %{id: "invalid", vector: [1.0]}
               ])

      assert {:ok, []} = Collection.all(collection)

      assert {:error, :duplicate_id} =
               Collection.put_many(collection, [
                 %{id: "duplicate", vector: [1.0, 1.0]},
                 %{id: "duplicate", vector: [2.0, 2.0]}
               ])

      assert {:ok, []} = Collection.all(collection)
      assert :ok = Vettore.close(collection)
    end
  end

  describe "snapshot durability and schema" do
    test "an index override persists through a second snapshot" do
      first_path = temporary_path("override-source")
      second_path = temporary_path("override-resnapshot")
      on_exit(fn -> Enum.each([first_path, second_path], &File.rm/1) end)

      assert {:ok, original} =
               Collection.new(dimensions: 2, metric: :l2, index: :hnsw)

      assert :ok = Collection.put(original, %{id: "a", vector: [1.0, 2.0]})
      assert :ok = Collection.snapshot(original, first_path)
      assert {:ok, overridden} = Collection.load_snapshot(first_path, index: :flat)
      assert overridden.index == :flat
      assert :ok = Collection.snapshot(overridden, second_path)
      assert {:ok, reloaded} = Collection.load_snapshot(second_path)
      assert reloaded.index == :flat
      assert reloaded.index_mod == Vettore.Index.Flat
      assert {:ok, %Embedding{id: "a"}} = Collection.get(reloaded, "a")

      Enum.each([original, overridden, reloaded], &Vettore.close/1)
    end

    test "rejects structural overrides and invalid snapshot versions without owner leaks" do
      path = temporary_path("invalid-schema")
      table = :ets.new(:vettore_invalid_snapshot, [:set])
      on_exit(fn -> File.rm(path) end)

      true =
        :ets.insert(table, {
          :__config__,
          %{
            snapshot_version: 99,
            dimensions: 2,
            metric: :l2,
            normalize: :none,
            score: :raw,
            index: :flat,
            index_options: [],
            compressed: false
          }
        })

      assert :ok =
               :ets.tab2file(table, String.to_charlist(path),
                 extended_info: [:object_count, :md5sum]
               )

      true = :ets.delete(table)
      before_count = active_ets_owners()
      assert {:error, :unsupported_snapshot_version} = Collection.load_snapshot(path)
      assert active_ets_owners() == before_count

      assert {:error, {:unsupported_snapshot_override, :dimensions}} =
               Collection.load_snapshot(path, dimensions: 3)
    end

    test "detects corrupted snapshots" do
      path = temporary_path("corrupted")
      on_exit(fn -> File.rm(path) end)
      File.write!(path, "not an ETS snapshot")
      assert {:error, _reason} = Collection.load_snapshot(path)
    end

    test "rejects snapshot table types that cannot provide unique ids" do
      path = temporary_path("bag-table")
      table = :ets.new(:vettore_bag_snapshot, [:bag, :public])
      on_exit(fn -> File.rm(path) end)

      true =
        :ets.insert(table, {
          :__config__,
          %{
            snapshot_version: 1,
            dimensions: 1,
            metric: :l2,
            normalize: :none,
            score: :raw,
            index: :flat,
            index_options: [],
            compressed: false
          }
        })

      assert :ok =
               :ets.tab2file(table, String.to_charlist(path),
                 extended_info: [:object_count, :md5sum]
               )

      true = :ets.delete(table)
      assert {:error, :invalid_snapshot_table_type} = Collection.load_snapshot(path)
    end

    test "loads legacy records without binary vectors and rejects malformed records" do
      legacy_path = temporary_path("legacy-record")
      invalid_path = temporary_path("invalid-record")
      mismatch_path = temporary_path("mismatched-record")
      on_exit(fn -> Enum.each([legacy_path, invalid_path, mismatch_path], &File.rm/1) end)

      config = %{
        snapshot_version: 0,
        name: :legacy,
        dimensions: 2,
        metric: :l2,
        normalize: :none,
        score: :raw,
        index: :flat,
        index_options: [],
        compressed: false
      }

      write_manual_snapshot(legacy_path, config, [
        {{:record, "legacy"},
         %Embedding{id: "legacy", value: "legacy", vector: [1.0, 1.0], binary_vector: nil}}
      ])

      assert {:ok, loaded} = Collection.load_snapshot(legacy_path)
      assert :protected = :ets.info(loaded.store_state.table, :protection)
      assert true = :ets.info(loaded.store_state.table, :read_concurrency)
      refute :ets.info(loaded.store_state.table, :write_concurrency)

      assert {:ok, [%Result{id: "legacy"}]} =
               Collection.quantized_search(loaded, [1.0, 1.0], candidates: 1, limit: 1)

      assert :ok = Vettore.close(loaded)

      write_manual_snapshot(invalid_path, config, [
        {{:record, "invalid"}, %Embedding{id: "invalid", vector: [1.0]}}
      ])

      assert {:error, {:invalid_snapshot_record, :dimension_mismatch}} =
               Collection.load_snapshot(invalid_path)

      write_manual_snapshot(mismatch_path, config, [
        {{:record, "key"}, %Embedding{id: "different", vector: [1.0, 1.0]}}
      ])

      assert {:error, {:invalid_snapshot_record, :id_mismatch}} =
               Collection.load_snapshot(mismatch_path)
    end

    test "snapshot public guards and override validation return tagged errors" do
      path = temporary_path("guard")
      assert {:error, :invalid_snapshot} = Collection.snapshot(:bad, path)
      assert {:error, :invalid_snapshot} = Collection.load_snapshot(123)
      assert {:error, :invalid_snapshot} = Collection.load_snapshot(path, :bad)
      assert {:error, :invalid_snapshot_options} = Collection.load_snapshot(path, [:bad])

      assert {:error, {:duplicate_snapshot_override, :index}} =
               Collection.load_snapshot(path, index: :flat, index: :hnsw)
    end
  end

  describe "validation and score semantics" do
    test "validates collection and search options without raising" do
      assert {:error, :invalid_options} = Collection.new([:not_a_keyword])

      assert {:error, {:unsupported_option, :unknown}} =
               Collection.new(dimensions: 2, unknown: true)

      assert {:error, {:duplicate_option, :dimensions}} =
               Collection.new([{:dimensions, 2}, {:dimensions, 3}])

      for {options, reason} <- [
            {[dimensions: 0], :invalid_dimensions},
            {[dimensions: 2, metric: :unknown], :invalid_metric},
            {[dimensions: 2, normalize: :unknown], :invalid_normalization},
            {[dimensions: 2, score: :unknown], :invalid_score_mode},
            {[dimensions: 2, compressed: :yes], :invalid_compressed},
            {[dimensions: 2, store: :unknown], :invalid_store},
            {[dimensions: 2, index: :unknown], :invalid_index},
            {[dimensions: 2, index_options: :bad], :invalid_index_options}
          ] do
        assert {:error, ^reason} = Collection.new(options)
      end

      assert {:ok, collection} = Collection.new(dimensions: 2, metric: :l2)
      assert {:error, :invalid_options} = Collection.search(collection, [0.0, 0.0], [:bad])

      assert {:error, {:unsupported_option, :unknown}} =
               Collection.search(collection, [0.0, 0.0], unknown: true)

      assert {:error, :invalid_limit} = Collection.search(collection, [0.0, 0.0], limit: 0)

      assert {:error, :invalid_limit} =
               Collection.search(collection, [0.0, 0.0], limit: 4_294_967_296)

      assert {:error, :dimension_mismatch} = Collection.search(collection, [0.0], limit: 1)

      assert {:error, :invalid_vector} =
               Collection.put(collection, %{id: "nan", vector: [:bad, 0.0]})

      assert {:error, :missing_id} = Collection.put(collection, %{id: "", vector: [0.0, 0.0]})
      assert {:error, :invalid_embedding} = Collection.put(collection, :bad)
      assert :ok = Vettore.close(collection)
    end

    test "normalizes a derived multi-vector mean before cosine indexing" do
      assert {:ok, collection} = Collection.new(dimensions: 2, metric: :cosine)

      assert :ok =
               Collection.put(collection, %{
                 id: "axes",
                 vectors: [[1.0, 0.0], [0.0, 1.0]]
               })

      assert {:ok, %Embedding{vector: [x, y]}} = Collection.get(collection, "axes")
      assert_in_delta x, :math.sqrt(0.5), 1.0e-6
      assert_in_delta y, :math.sqrt(0.5), 1.0e-6
      assert {:ok, [%Result{score: score}]} = Collection.search(collection, [1.0, 1.0], limit: 1)
      assert_in_delta score, 1.0, 1.0e-6
      assert :ok = Vettore.close(collection)
    end

    test "negative inner product similarity mode has finite explicit semantics" do
      assert Vettore.Distance.result_values(:negative_inner_product, -1.0, :raw) == {1.0, -1.0}

      assert Vettore.Distance.result_values(:negative_inner_product, -1.0, :similarity) ==
               {1.0, -1.0}

      assert {:ok, collection} =
               Collection.new(
                 dimensions: 1,
                 metric: :negative_inner_product,
                 score: :similarity
               )

      assert :ok = Collection.put(collection, %{id: "positive", vector: [1.0]})

      assert {:ok, [%Result{id: "positive", score: 1.0, distance: -1.0}]} =
               Collection.search(collection, [1.0], limit: 1)

      assert :ok = Vettore.close(collection)
    end

    test "default and malformed adaptive-search options stay total" do
      assert {:ok, collection} = Collection.new(dimensions: 2, metric: :l2)
      assert :ok = Collection.put(collection, %{id: "origin", vector: [0.0, 0.0]})

      assert {:ok, [%Result{id: "origin"}]} = Collection.search(collection, [0.0, 0.0])
      assert {:ok, [%Result{id: "origin"}]} = Collection.funnel_search(collection, [0.0, 0.0])

      assert {:ok, [%Result{id: "origin"}]} =
               Collection.quantized_search(collection, [0.0, 0.0])

      assert {:ok, [%Result{id: "origin"}]} =
               Collection.multi_vector_search(collection, [[0.0, 0.0]])

      assert {:ok, [%Result{id: "origin"}]} = Collection.hybrid_search(collection, [0.0, 0.0])

      for {search, query} <- [
            {&Collection.search/3, [0.0, 0.0]},
            {&Collection.funnel_search/3, [0.0, 0.0]},
            {&Collection.quantized_search/3, [0.0, 0.0]},
            {&Collection.multi_vector_search/3, [[0.0, 0.0]]},
            {&Collection.hybrid_search/3, [0.0, 0.0]}
          ] do
        assert {:error, :invalid_options} = search.(collection, query, :bad)
      end

      assert {:error, :invalid_candidates} =
               Collection.funnel_search(collection, [0.0, 0.0], limit: 2, candidates: 1)

      assert {:error, :invalid_stages} =
               Collection.funnel_search(collection, [0.0, 0.0], stages: [])

      assert {:ok, [%Result{id: "origin"}]} =
               Collection.funnel_search(collection, [0.0, 0.0], dimensions: 1)

      assert {:error, :invalid_candidates} =
               Collection.quantized_search(collection, [0.0, 0.0], candidates: 0)

      assert {:error, :invalid_candidates} =
               Collection.quantized_search(collection, [0.0, 0.0], candidates: 4_294_967_296)

      assert {:error, :invalid_multi_vector} = Collection.multi_vector_search(collection, [])

      assert {:error, :invalid_generators} =
               Collection.hybrid_search(collection, [0.0, 0.0], generators: [])

      assert {:error, {:unknown_generator, :unknown}} =
               Collection.hybrid_search(collection, [0.0, 0.0], generators: [:unknown])

      assert {:error, {:invalid_generator, {:funnel, :bad}}} =
               Collection.hybrid_search(collection, [0.0, 0.0], generators: [{:funnel, :bad}])

      assert {:ok, [%Result{id: "origin"}]} =
               Collection.hybrid_search(collection, [0.0, 0.0], generators: [:search])

      assert {:error, {:invalid_rerank, :unknown}} =
               Collection.hybrid_search(collection, [0.0, 0.0],
                 generators: [:search],
                 rerank: :unknown
               )

      assert {:error, {:unsupported_option, :unknown}} =
               Collection.hybrid_search(collection, [0.0, 0.0],
                 generators: [:search],
                 rerank: {:multi_vector, [[0.0, 0.0]], [unknown: true]}
               )

      assert :ok = Collection.delete(collection, "missing")
      assert {:error, :invalid_id} = Collection.delete(collection, :bad)
      assert {:error, :invalid_embeddings} = Collection.put_many(collection, :bad)
      assert :ok = Vettore.close(collection)
    end

    test "value-based embeddings, metric aliases, and every native metric code are supported" do
      assert {:ok, collection} = Collection.new(dimensions: 2, metric: :euclidean)
      assert collection.metric == :l2
      assert :ok = Collection.put(collection, %{value: "value-id", vector: [1, 0]})
      assert :ok = Collection.put(collection, %{value: "multi-id", vectors: [[0, 1], [1, 0]]})
      assert {:ok, %Embedding{id: "value-id"}} = Collection.get(collection, "value-id")

      for metric <- [
            :l2,
            :l2_squared,
            :cosine,
            :dot,
            :dot_product,
            :negative_inner_product,
            :manhattan,
            :chebyshev,
            :hamming,
            :jaccard
          ] do
        assert {:ok, [_ | _]} =
                 Collection.multi_vector_search(collection, [[1.0, 0.0]],
                   metric: metric,
                   limit: 1
                 )
      end

      assert :ok = Vettore.close(collection)

      assert {:error, :invalid_options} = Collection.new(:bad)
      assert {:error, :invalid_store} = Collection.new(dimensions: 2, store: "bad")
      assert {:error, :invalid_index} = Collection.new(dimensions: 2, index: "bad")
    end
  end

  defp active_ets_owners do
    DynamicSupervisor.count_children(Vettore.ETSSupervisor).active
  end

  defp temporary_path(label) do
    Path.join(
      System.tmp_dir!(),
      "vettore-#{label}-#{System.unique_integer([:positive, :monotonic])}.ets"
    )
  end

  defp write_manual_snapshot(path, config, records) do
    table =
      :ets.new(:vettore_manual_snapshot, [
        :set,
        :public,
        read_concurrency: true,
        write_concurrency: true
      ])

    true = :ets.insert(table, [{:__config__, config} | records])

    :ok =
      :ets.tab2file(table, String.to_charlist(path), extended_info: [:object_count, :md5sum])

    true = :ets.delete(table)
    :ok
  end
end
