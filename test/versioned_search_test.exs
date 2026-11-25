defmodule Vettore.Test.UpsertIndex do
  alias Vettore.Index.Flat

  defdelegate new(metric, opts), to: Flat
  defdelegate put_many(collection, embeddings), to: Flat
  defdelegate delete(collection, id), to: Flat
  defdelegate search(collection, query, opts), to: Flat
  defdelegate close(collection), to: Flat

  def put(collection, embedding) do
    with :ok <- Flat.put(collection, embedding) do
      case embedding.metadata do
        %{fail: true} ->
          {:error, :forced_index_failure}

        %{pause: caller} ->
          send(caller, {:indexed, self()})

          receive do
            :commit -> :ok
          end

        _metadata ->
          :ok
      end
    end
  end
end

defmodule Vettore.VersionedSearchTest do
  use ExUnit.Case, async: true

  alias Vettore.{Index, Nifs, Result}
  alias Vettore.Store.ETS

  for index <- [:flat, :hnsw] do
    @index index

    test "#{index} rejects hits for replaced or reinserted IDs" do
      {:ok, collection} = Vettore.new(dimensions: 1, metric: :l2, index: @index)
      old = %{id: "doc", vector: [0.0], metadata: %{edition: 1}}
      replacement = %{old | vector: [3.0], metadata: %{edition: 2}}
      assert :ok = Vettore.put(collection, old)

      native_search =
        case @index do
          :flat -> &Nifs.flat_search_versioned/4
          :hnsw -> &Nifs.hnsw_search_versioned/4
        end

      assert {:ok, [{"doc", first, raw}] = hits} =
               native_search.(collection.index_state, [0.0], 1, nil)

      assert raw == 0.0

      assert :ok = Vettore.delete(collection, "doc")
      assert :ok = Vettore.put(collection, replacement)
      assert [] == Index.hydrate_results(collection, hits)

      assert {:ok, [%Result{distance: 3.0, metadata: %{edition: 2}}]} =
               Vettore.search(collection, [0.0])

      assert {:error, :duplicate_id} = Vettore.put(collection, old)
      assert {:error, :dimension_mismatch} = Vettore.upsert(collection, %{old | vector: [0, 0]})
      assert {:ok, [%Result{distance: 3.0}]} = Vettore.search(collection, [0.0])
      assert :ok = Vettore.upsert(collection, old)
      assert :ok = Vettore.upsert(collection, %{old | id: "new"})
      assert {:ok, {_, current}} = ETS.get_versioned(collection.store_state, "doc")
      assert current != first

      assert {:ok, [%Result{distance: distance, metadata: %{edition: 1}}]} =
               Vettore.search(collection, [0.0], filter: %{edition: 1}, limit: 1)

      assert distance == 0.0
    end

    test "#{index} gets its top-k from records matching all supplied entries" do
      {:ok, collection} = Vettore.new(dimensions: 1, metric: :l2, index: @index)

      records =
        for i <- 0..21 do
          metadata =
            case i do
              10 -> %{lingua: "it", edition: 1.0}
              i when i in 11..20 -> %{lingua: "it", edition: 1}
              21 -> nil
              _ -> %{lingua: "en", edition: 1}
            end

          %{id: Integer.to_string(i), vector: [i / 1], metadata: metadata}
        end

      assert :ok = Vettore.put_many(collection, records)

      assert {:ok, hits} =
               Vettore.search(collection, [0.0], filter: %{lingua: "it", edition: 1}, limit: 10)

      assert Enum.map(hits, & &1.distance) == Enum.map(11..20, &(&1 / 1))
      assert {:ok, []} = Vettore.search(collection, [0.0], filter: %{lingua: nil})
      assert {:error, :invalid_filter} = Vettore.search(collection, [0.0], filter: [])

      assert Vettore.search(collection, [0.0], filter: %{}, limit: 2) ==
               Vettore.search(collection, [0.0], limit: 2)
    end
  end

  test "a failed upsert restores the old indexed revision and removes an unsuccessful new ID" do
    {:ok, collection} = Vettore.new(dimensions: 1, metric: :l2, index: Vettore.Test.UpsertIndex)
    old = %{id: "doc", vector: [2.0], metadata: %{edition: 1}}
    failed = %{old | vector: [0.0], metadata: %{fail: true}}

    assert :ok = Vettore.put(collection, old)
    assert {:ok, {_, revision}} = ETS.get_versioned(collection.store_state, "doc")
    assert {:error, :forced_index_failure} = Vettore.upsert(collection, failed)
    assert {:error, :forced_index_failure} = Vettore.upsert(collection, %{failed | id: "new"})
    assert {:error, :not_found} = Vettore.get(collection, "new")

    assert {:ok, {%{metadata: %{edition: 1}}, ^revision}} =
             ETS.get_versioned(collection.store_state, "doc")

    assert {:ok, [%Result{id: "doc", metadata: %{edition: 1}, distance: 2.0}]} =
             Vettore.search(collection, [0.0])

    assert {:ok, [{"doc", 2.0}]} = Nifs.flat_search(collection.index_state, [0.0], 2)
  end

  test "upsert reports an index rollback failure and retains the canonical record" do
    {:ok, collection} = Vettore.new(dimensions: 1, metric: :l2)
    assert :ok = Vettore.put(collection, %{id: "doc", vector: [2.0]})
    assert {:ok, record} = Vettore.get(collection, "doc")

    assert {:error, {:index_restore_failed, :update_failed, :restore_failed}} =
             ETS.upsert_indexed(
               collection.store_state,
               %{record | vector: [3.0]},
               ETS.revision(),
               fn
                 %{vector: [3.0]}, _revision -> {:error, :update_failed}
                 ^record, _revision -> {:error, :restore_failed}
               end,
               fn _id -> :ok end
             )

    assert {:ok, ^record} = Vettore.get(collection, "doc")
  end

  test "concurrent search observes the committed upsert while get can read the old row" do
    {:ok, collection} = Vettore.new(dimensions: 1, metric: :l2, index: Vettore.Test.UpsertIndex)
    assert :ok = Vettore.put(collection, %{id: "doc", vector: [0.0], metadata: %{edition: 1}})
    record = %{id: "doc", vector: [2.0], metadata: %{edition: 2, pause: self()}}
    writer = Task.async(fn -> Vettore.upsert(collection, record) end)

    assert_receive {:indexed, owner}
    assert {:ok, %{metadata: %{edition: 1}}} = Vettore.get(collection, "doc")

    reader =
      Task.async(fn -> Vettore.search(collection, [0.0], filter: %{edition: 2}, limit: 1) end)

    yielded = Task.yield(reader, 50)
    send(owner, :commit)

    refute yielded
    assert :ok = Task.await(writer)
    assert {:ok, [%Result{distance: 2.0, metadata: %{edition: 2}}]} = Task.await(reader)
  end

  test "a search that started before an upsert retries against the committed record" do
    {:ok, collection} = Vettore.new(dimensions: 1, metric: :l2)
    state = collection.store_state
    initial = %{id: "doc", vector: [1.0], metadata: %{edition: 1}}
    assert :ok = Vettore.put(collection, initial)
    parent = self()

    reader =
      Task.async(fn ->
        ETS.consistent_search(state, fn ->
          results = Index.Flat.search(collection, [0.0], limit: 1)
          send(parent, {:searched, self(), results})

          if self() != state.owner do
            receive do
              :resume -> :ok
            end
          end

          results
        end)
      end)

    assert_receive {:searched, pid, {:ok, [%Result{distance: 1.0, metadata: %{edition: 1}}]}},
                   1_000

    assert :ok = Vettore.upsert(collection, %{initial | vector: [2.0], metadata: %{edition: 2}})
    send(pid, :resume)

    committed = Task.await(reader)
    assert {:ok, [%Result{distance: 2.0, metadata: %{edition: 2}}]} = committed
    owner = state.owner
    assert_receive {:searched, ^owner, ^committed}, 1_000

    assert ^committed =
             Vettore.ETSOwner.transaction(owner, fn _table ->
               Vettore.search(collection, [0.0], limit: 1)
             end)
  end

  test "snapshot rebuild refreshes revisions and supports filtered results" do
    path = Path.join(System.tmp_dir!(), "vettore-revision-#{System.unique_integer([:positive])}")
    on_exit(fn -> File.rm(path) end)

    {:ok, collection} = Vettore.new(dimensions: 1, metric: :l2)
    assert :ok = Vettore.put(collection, %{id: "doc", vector: [2.0], metadata: %{edition: 1}})
    assert {:ok, {_, before}} = ETS.get_versioned(collection.store_state, "doc")
    assert :ok = Vettore.snapshot(collection, path)
    assert {:ok, loaded} = Vettore.load_snapshot(path)
    assert {:ok, {_, later}} = ETS.get_versioned(loaded.store_state, "doc")
    assert later != before

    assert {:ok, [%Result{distance: 2.0}]} =
             Vettore.search(loaded, [0.0], filter: %{edition: 1})
  end
end
