Code.require_file("support/fastembed_fixture.ex", __DIR__)

defmodule VettoreIndexRecoveryTest do
  use ExUnit.Case, async: true

  alias Vettore.{Collection, Embedding}
  alias Vettore.Test.FastembedFixture

  # Simulate the public error boundary; Rust tests poison a real write lock.
  # Snapshot recovery must never call the unavailable native index.
  defmodule PoisonedIndex do
    def search(context, _query, _opts), do: {:error, "#{context.index} lock poisoned"}
    def close(context), do: {:error, "#{context.index} lock poisoned"}
  end

  setup do
    {:ok, fixture: FastembedFixture.load!()}
  end

  for {index, options} <- [flat: [gpu: false], hnsw: [m: 8]] do
    @tag :tmp_dir
    test "canonical ETS snapshot rebuilds a failed #{index} index with real embeddings", %{
      fixture: fixture,
      tmp_dir: tmp_dir
    } do
      index = unquote(index)
      reason = "#{index} lock poisoned"
      index_options = unquote(options)

      {:ok, original} =
        Collection.new(
          dimensions: fixture.dimensions,
          metric: :cosine,
          index: index,
          normalize: :none,
          score: :raw,
          index_options: index_options
        )

      on_exit(fn -> Collection.close(original) end)
      :ok = Collection.put_many(original, embeddings(fixture))
      query = hd(fixture.queries).vector
      assert {:ok, expected_hits} = Collection.search(original, query, limit: 5)
      assert {:ok, expected_records} = Collection.all(original)
      failed = %{original | index_mod: PoisonedIndex}
      assert {:error, ^reason} = Collection.search(failed, query, limit: 5)

      path = Path.join(tmp_dir, "recovery.ets")
      assert :ok = Vettore.snapshot(failed, path)
      assert {:ok, recovered} = Vettore.load_snapshot(path)
      on_exit(fn -> Vettore.close(recovered) end)
      assert recovered.index_state != failed.index_state
      assert recovered.store_state.table != failed.store_state.table
      config_keys = [:dimensions, :metric, :index, :normalize, :score, :index_options]
      assert Map.take(recovered, config_keys) == Map.take(original, config_keys)
      assert {:ok, ^expected_hits} = Collection.search(recovered, query, limit: 5)
      assert {:ok, restored_records} = Collection.all(recovered)
      assert Enum.sort_by(restored_records, & &1.id) == Enum.sort_by(expected_records, & &1.id)

      assert {:error, ^reason} = Vettore.close(failed)
      refute failed.store_mod.alive?(failed.store_state)
      assert {:ok, ^expected_hits} = Collection.search(recovered, query, limit: 5)
    end
  end

  defp embeddings(fixture) do
    Enum.map(fixture.documents, fn document ->
      %Embedding{
        id: document.id,
        value: document.text,
        vector: document.vector,
        metadata: %{category: document.category}
      }
    end)
  end
end
