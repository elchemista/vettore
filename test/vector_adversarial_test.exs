defmodule Vettore.Test.RestoreFailingIndex do
  @moduledoc false

  def new(_metric, _opts), do: {:ok, :ets.new(:vettore_restore_failing_index, [:set, :private])}

  def put(collection, embedding) do
    attempts =
      :ets.update_counter(collection.index_state, embedding.id, {2, 1}, {embedding.id, 0})

    if attempts == 1, do: :ok, else: {:error, :forced_restore_failure}
  end

  def put_many(collection, embeddings) do
    Enum.each(embeddings, fn embedding ->
      :ets.insert(collection.index_state, {embedding.id, 1})
    end)

    :ok
  end

  def delete(_collection, _id), do: :ok
  def search(_collection, _query, _opts), do: {:ok, []}
end

defmodule Vettore.Test.DeleteFailingStore do
  @moduledoc false

  alias Vettore.Store.ETS

  def new(config), do: ETS.new(config)
  def put(state, embedding), do: ETS.put(state, embedding)
  def put_many(state, embeddings), do: ETS.put_many(state, embeddings)
  def get(state, id), do: ETS.get(state, id)
  def delete(_state, _id), do: {:error, :forced_store_delete_failure}
  def all(state), do: ETS.all(state)
  def snapshot(state, path), do: ETS.snapshot(state, path)
  def load_snapshot(path), do: ETS.load_snapshot(path)
  def configure(state, config), do: ETS.configure(state, config)
  def close(state), do: ETS.close(state)
  def alive?(state), do: ETS.alive?(state)
end

defmodule Vettore.Test.ScriptedIndex do
  @moduledoc false

  def new(_metric, _opts), do: {:ok, {:results, []}}
  def put(_collection, _embedding), do: :ok
  def put_many(_collection, _embeddings), do: :ok
  def delete(_collection, _id), do: :ok
  def search(%{index_state: {:results, results}}, _query, _opts), do: {:ok, results}
end

defmodule Vettore.Test.ScriptedStore do
  @moduledoc false

  alias Vettore.Embedding

  @config %{
    snapshot_version: 1,
    name: :adversarial_snapshot,
    dimensions: 1,
    metric: :l2,
    normalize: :none,
    score: :raw,
    index: :flat,
    index_options: [],
    compressed: false
  }

  def new(_config), do: {:ok, {:all, []}}
  def put(_state, _embedding), do: :ok
  def put_many(_state, _embeddings), do: :ok

  def get({:get, response}, _id), do: response

  def get({:all, embeddings}, id) when is_list(embeddings) do
    case Enum.find(embeddings, &match?(%Embedding{id: ^id}, &1)) do
      nil -> {:error, :not_found}
      embedding -> {:ok, embedding}
    end
  end

  def delete(_state, _id), do: :ok
  def all({:all, embeddings}), do: {:ok, embeddings}
  def all({:get, _response}), do: {:ok, []}
  def snapshot(_state, _path), do: :ok

  def load_snapshot("bad-config"), do: {:ok, {{:all, []}, :not_a_config}}
  def load_snapshot("non-list"), do: {:ok, {{:all, :not_a_list}, @config}}
  def load_snapshot("non-embedding"), do: {:ok, {{:all, [:bad]}, @config}}

  def load_snapshot("empty-vectors") do
    {:ok,
     {{:all, [%Embedding{id: "a", vector: [1.0], vectors: [], binary_vector: nil}]}, @config}}
  end

  def load_snapshot("bad-vectors") do
    {:ok, {{:all, [%Embedding{id: "a", vector: [1.0], vectors: [[1.0, 2.0]]}]}, @config}}
  end

  def load_snapshot("bad-binary") do
    {:ok, {{:all, [%Embedding{id: "a", vector: [1.0], binary_vector: :bad}]}, @config}}
  end

  def configure(_state, _config), do: :ok
  def close(_state), do: :ok
  def alive?(_state), do: true
end

defmodule VettoreAdversarialTest do
  use ExUnit.Case, async: false

  alias Vettore.{Collection, Distance, Embedding, MultiVector, Result, Store.ETS}
  alias Vettore.Test.{DeleteFailingStore, RestoreFailingIndex, ScriptedIndex, ScriptedStore}

  @max_f32 3.402_823_466_385_288_6e38

  describe "numerical failure propagation" do
    test "large representable L2 and cancelling products remain correct" do
      assert {:ok, distance} = Distance.l2([1.0e20], [0.0])
      assert_in_delta distance, 1.0e20, 1.0e14

      assert {:ok, product} = Distance.inner_product([@max_f32, @max_f32], [2.0, -2.0])
      assert product == 0.0
      assert {:error, :metric_overflow} = Distance.l2_squared([1.0e20], [0.0])
    end

    test "invalid compression inputs and MaxSim accumulation never raise" do
      assert {:error, :invalid_vector} = Distance.compress_f32_vector(:bad)
      assert {:error, :invalid_vector} = Distance.compress_f32_vector([:bad])

      assert {:error, :score_overflow} =
               MultiVector.chamfer(List.duplicate([1.0e19], 4), [[1.0e19]],
                 metric: :inner_product
               )

      assert {:ok, 1.0} = MultiVector.colbert_score([[1]], [[1]])
      assert {:error, :invalid_multi_vector} = MultiVector.chamfer([1.0], [[1.0]])
    end

    test "MMR rejects malformed records and propagates metric overflow" do
      assert Distance.result_values(:l2, 2.0) == {-2.0, 2.0}
      assert {:ok, []} = Distance.mmr_rerank([], [], :l2, 0.5, 10)

      assert {:error, :invalid_mmr_args} =
               Distance.mmr_rerank(
                 [{"a", 1.0}],
                 [{"a", [1.0]}, {"a", [1.0]}],
                 :l2,
                 0.5,
                 1
               )

      assert {:error, :invalid_mmr_args} =
               Distance.mmr_rerank([{"a", 1.0}], [:bad], :l2, 0.5, 1)

      assert {:error, :invalid_mmr_args} =
               Distance.mmr_rerank([:bad], [{"a", [1.0]}], :l2, 0.5, 1)

      initial = [{"a", 1.0}, {"b", 0.5}]
      huge = [{"a", [@max_f32]}, {"b", [-@max_f32]}]

      assert {:error, :metric_overflow} =
               Distance.mmr_rerank(initial, huge, :l2_squared, 0.5, 2)

      huge_product = [{"a", [@max_f32]}, {"b", [2.0]}]

      assert {:error, :metric_overflow} =
               Distance.mmr_rerank(initial, huge_product, :inner_product, 0.5, 2)
    end
  end

  describe "fault-injected collection boundaries" do
    test "reports both the store deletion and failed index restoration" do
      assert {:ok, collection} =
               Collection.new(
                 dimensions: 1,
                 metric: :l2,
                 store: DeleteFailingStore,
                 index: RestoreFailingIndex
               )

      on_exit(fn ->
        Vettore.close(collection)

        if :ets.info(collection.index_state) != :undefined do
          :ets.delete(collection.index_state)
        end
      end)

      assert :ok = Collection.put(collection, %{id: "a", vector: [1.0]})

      assert {:error,
              {:index_restore_failed, :forced_store_delete_failure, :forced_restore_failure}} =
               Collection.delete(collection, "a")
    end

    test "hybrid search tolerates stale ids but returns real store errors" do
      stale =
        scripted_collection({:get, {:error, :not_found}}, [
          %Result{id: "missing", score: 1.0, metric: :l2}
        ])

      assert {:ok, []} =
               Collection.hybrid_search(stale, [0.0], generators: [:search], limit: 1)

      failed =
        scripted_collection({:get, {:error, :forced_get_failure}}, [
          %Result{id: "broken", score: 1.0, metric: :l2}
        ])

      assert {:error, :forced_get_failure} =
               Collection.hybrid_search(failed, [0.0], generators: [:search], limit: 1)
    end

    test "adaptive paths reject malformed custom-store records without raising" do
      invalid_vector = scripted_collection({:all, [%Embedding{id: "bad", vector: [:bad]}]})

      assert {:error, :invalid_vector} =
               Collection.quantized_search(invalid_vector, [0.0], candidates: 1, limit: 1)

      wrong_funnel_dimension =
        scripted_collection({:all, [%Embedding{id: "bad", vector: []}]})

      assert {:error, :dimension_mismatch} =
               Collection.funnel_search(wrong_funnel_dimension, [0.0],
                 stages: [1],
                 candidates: 1,
                 limit: 1
               )

      overflow_document =
        scripted_collection({:all, [%Embedding{id: "huge", vectors: [[1.0e19]]}]})

      assert {:error, :score_overflow} =
               Collection.multi_vector_search(
                 overflow_document,
                 List.duplicate([1.0e19], 4),
                 metric: :inner_product,
                 limit: 1
               )

      non_finite_document =
        scripted_collection({:all, [%Embedding{id: "infinite", vectors: [[1.0e39]]}]})

      assert {:error, :invalid_multi_vector} =
               Collection.multi_vector_search(non_finite_document, [[1.0]], limit: 1)

      mismatched =
        scripted_collection({:all, [%Embedding{id: "bad", vectors: [[1.0, 2.0]]}]})

      assert {:error, :dimension_mismatch} =
               Collection.multi_vector_search(mismatched, [[1.0]], limit: 1)

      malformed_record = scripted_collection({:all, [:not_an_embedding]})

      assert {:error, :invalid_embedding} =
               Collection.funnel_search(malformed_record, [0.0],
                 stages: [1],
                 candidates: 1,
                 limit: 1
               )

      duplicate_ids =
        scripted_collection(
          {:all,
           [%Embedding{id: "same", vectors: [[1.0]]}, %Embedding{id: "same", vectors: [[0.0]]}]}
        )

      assert {:error, :duplicate_id} =
               Collection.multi_vector_search(duplicate_ids, [[1.0]], limit: 1)
    end

    test "adaptive option and nested-vector failures are tagged" do
      collection = scripted_collection({:all, []})

      assert {:error, :invalid_limit} = Collection.hybrid_search(collection, [0.0], limit: 0)

      assert {:error, :invalid_candidates} =
               Collection.hybrid_search(collection, [0.0],
                 generators: [quantized: [candidates: 0]],
                 limit: 1
               )

      assert {:error, :dimension_mismatch} =
               Collection.multi_vector_search(collection, [[0.0], [0.0, 1.0]], limit: 1)

      assert {:error, :invalid_vector} =
               Collection.put(collection, %{id: "bad", vector: :not_a_vector})
    end
  end

  describe "snapshot and store corruption guards" do
    test "custom snapshot stores cannot bypass restored-record validation" do
      for {path, expected} <- [
            {"bad-config", :invalid_snapshot},
            {"non-list", :invalid_snapshot},
            {"non-embedding", {:invalid_snapshot_record, :invalid_embedding}},
            {"empty-vectors", {:invalid_snapshot_record, :invalid_multi_vector}},
            {"bad-vectors", {:invalid_snapshot_record, :invalid_multi_vector}},
            {"bad-binary", {:invalid_snapshot_record, :invalid_binary_vector}}
          ] do
        assert {:error, ^expected} =
                 Collection.load_snapshot(path, store: ScriptedStore, index: :flat)
      end
    end

    test "ETS snapshots reject missing configuration and arbitrary rows" do
      cases = [
        {[%Embedding{id: "unused"}], :missing_config},
        {[{:__config__, %{}}, {{:record, "a"}, :not_an_embedding}],
         {:invalid_snapshot_record, :invalid_embedding}},
        {[{:__config__, %{}}, {:unexpected, :row}], :invalid_snapshot_row}
      ]

      Enum.each(cases, fn {rows, expected} ->
        path = temporary_path("adversarial-store")
        on_exit(fn -> File.rm(path) end)

        snapshot_rows =
          Enum.map(rows, fn
            %Embedding{id: id} = embedding -> {{:record, id}, embedding}
            row -> row
          end)

        write_manual_snapshot(path, snapshot_rows)
        assert {:error, ^expected} = ETS.load_snapshot(path)
      end)
    end

    test "closed and malformed ETS states keep every boundary total" do
      assert {:ok, state} = ETS.new(%{dimensions: 1})
      assert :ok = ETS.close(state)
      assert {:error, :closed} = ETS.configure(state, %{dimensions: 2})

      malformed = %ETS{table: 123, owner: self()}
      refute ETS.alive?(malformed)
      assert ETS.count(malformed) == 0
      assert {:error, :invalid_vector} = Distance.packed_hamming(:bad, [], 1)
    end
  end

  describe "concurrency and search-mode equivalence" do
    test "parallel callers serialize writes while direct readers keep progressing" do
      assert {:ok, collection} = Collection.new(dimensions: 2, metric: :l2)
      on_exit(fn -> Vettore.close(collection) end)
      assert :ok = Collection.put(collection, %{id: "stable", vector: [0.0, 0.0]})

      writers =
        for worker <- 1..8 do
          Task.async(fn ->
            embeddings =
              for item <- 1..25 do
                %{id: "#{worker}-#{item}", vector: [worker / 1, item / 1]}
              end

            Collection.put_many(collection, embeddings)
          end)
        end

      readers =
        for _reader <- 1..16 do
          Task.async(fn ->
            Enum.all?(1..250, fn _ ->
              match?({:ok, %Embedding{id: "stable"}}, Collection.get(collection, "stable"))
            end)
          end)
        end

      assert Enum.all?(Task.await_many(writers, 30_000), &(&1 == :ok))
      assert Enum.all?(Task.await_many(readers, 30_000))
      assert {:ok, all} = Collection.all(collection)
      assert length(all) == 201
    end

    test "full-candidate adaptive modes agree with exact flat search" do
      assert {:ok, collection} = Collection.new(dimensions: 4, metric: :l2, index: :flat)
      on_exit(fn -> Vettore.close(collection) end)

      embeddings =
        for index <- 0..63 do
          %{
            id: "id-#{String.pad_leading(Integer.to_string(index), 2, "0")}",
            vector: [
              index / 10,
              rem(index * 7, 17) / 5,
              rem(index * 11, 19) / 7,
              rem(index, 3) / 1
            ]
          }
        end

      assert :ok = Collection.put_many(collection, embeddings)
      query = [2.25, 1.5, 0.75, 1.0]
      assert {:ok, exact} = Collection.search(collection, query, limit: 10)
      expected_ids = Enum.map(exact, & &1.id)

      assert {:ok, funnel} =
               Collection.funnel_search(collection, query,
                 stages: [2, 4],
                 candidates: 64,
                 limit: 10
               )

      assert {:ok, quantized} =
               Collection.quantized_search(collection, query, candidates: 64, limit: 10)

      assert {:ok, hybrid} =
               Collection.hybrid_search(collection, query,
                 generators: [
                   funnel: [stages: [2, 4], candidates: 64],
                   quantized: [candidates: 64],
                   search: [candidates: 64]
                 ],
                 limit: 10
               )

      assert Enum.map(funnel, & &1.id) == expected_ids
      assert Enum.map(quantized, & &1.id) == expected_ids
      assert Enum.map(hybrid, & &1.id) == expected_ids
    end
  end

  test "the ETS application can restart itself and owner calls catch process exits" do
    assert :ok = Application.stop(:vettore)
    on_exit(fn -> Application.ensure_all_started(:vettore) end)

    assert {:ok, {owner, _table}} = Vettore.ETSOwner.start_table(:restart_probe, [:set])
    assert :ok = Vettore.ETSOwner.close(owner)

    foreign =
      spawn(fn ->
        receive do
          {:"$gen_call", _from, _message} -> :ok
        end
      end)

    assert {:error, :closed} = Vettore.ETSOwner.insert(foreign, {:key, :value})
  end

  defp scripted_collection(state, results \\ []) do
    %Collection{
      name: :scripted,
      dimensions: 1,
      metric: :l2,
      normalize: :none,
      score: :raw,
      store_mod: ScriptedStore,
      store_state: state,
      index_mod: ScriptedIndex,
      index_state: {:results, results},
      index: :flat,
      index_options: [],
      compressed: false
    }
  end

  defp temporary_path(label) do
    Path.join(
      System.tmp_dir!(),
      "vettore-#{label}-#{System.unique_integer([:positive, :monotonic])}.ets"
    )
  end

  defp write_manual_snapshot(path, rows) do
    table = :ets.new(:vettore_adversarial_snapshot, [:set, :public])
    true = :ets.insert(table, rows)

    :ok =
      :ets.tab2file(table, String.to_charlist(path), extended_info: [:object_count, :md5sum])

    true = :ets.delete(table)
    :ok
  end
end
