defmodule VettoreQuantizedProjectionTest do
  use ExUnit.Case, async: true

  alias Vettore.Embedding
  alias Vettore.Store.ETS

  defmodule FullRecordStore do
    defdelegate all(state), to: ETS
  end

  test "adaptive searches validate limit before calculating candidates" do
    {:ok, collection} = Vettore.new(dimensions: 1)

    for search <- [&Vettore.funnel_search/3, &Vettore.quantized_search/3],
        limit <- ["10", 1.5, nil, 0, -1, 4_294_967_296],
        opts <- [[limit: limit], [limit: limit, candidates: 10]] do
      assert {:error, :invalid_limit} = search.(collection, [0.0], opts)
    end
  end

  test "projected sign candidates match the full-record path across packed words" do
    tail = List.duplicate(1.0, 64)
    query = [1.0 | tail]
    opts = [candidates: 3, limit: 2]

    for compressed <- [false, true] do
      {:ok, collection} = Vettore.new(dimensions: 65, metric: :l2, compressed: compressed)

      assert :ok =
               Vettore.put_many(collection, [
                 %{id: "same", vector: query, metadata: %{payload: tail}},
                 %{id: "near", vector: [1.125 | tail]},
                 %{id: "far", vector: [2.0 | tail]},
                 %{id: "opposite", vector: List.replace_at(query, 64, -1.0)}
               ])

      assert {:ok, rows} = ETS.binary_rows(collection.store_state)
      assert length(rows) == 4

      for {id, revision, words} <- rows do
        assert {:ok, {%Embedding{binary_vector: ^words}, ^revision}} =
                 ETS.get_versioned(collection.store_state, id)

        assert revision > 0
        assert length(words) == 2
      end

      full_record_collection = %{collection | store_mod: FullRecordStore}

      assert {:ok, results} = Vettore.quantized_search(full_record_collection, query, opts)
      assert {:ok, ^results} = Vettore.quantized_search(collection, query, opts)
      assert Enum.map(results, &{&1.id, &1.distance}) == [{"same", 0.0}, {"near", 0.125}]

      assert {:ok, ^results} =
               Vettore.hybrid_search(collection, query,
                 limit: 2,
                 generators: [quantized: [candidates: 3]]
               )
    end
  end

  test "projection computes missing legacy words and validates cached words" do
    {:ok, collection} = Vettore.new(dimensions: 1, metric: :l2)
    state = collection.store_state

    assert true =
             Vettore.ETSOwner.insert(
               state.owner,
               {{:record, "legacy"}, %Embedding{id: "legacy", vector: [1.0]}}
             )

    assert {:ok, [{"legacy", 0, nil}]} = ETS.binary_rows(state)
    assert {:ok, [%{id: "legacy"}]} = Vettore.quantized_search(collection, [1.0], limit: 1)
    assert {:ok, ["legacy"]} = ETS.fold(state, [], &[&1.id | &2])

    assert :ok = ETS.put(state, %Embedding{id: "bad", vector: [0.0], binary_vector: []})

    assert {:error, :invalid_binary_vector} =
             Vettore.quantized_search(collection, [1.0], limit: 1)

    assert :ok = Vettore.close(collection)
    assert {:error, :closed} = ETS.binary_rows(state)
  end

  test "a projected revision cannot hydrate a deleted or reinserted id" do
    {:ok, collection} = Vettore.new(dimensions: 1, metric: :l2)
    state = collection.store_state

    assert :ok = Vettore.put(collection, %{id: "doc", vector: [-1.0]})
    assert {:ok, [{"doc", revision, [0]}]} = ETS.binary_rows(state)
    assert {:ok, %Embedding{vector: [-1.0]}} = ETS.fetch_revision(state, "doc", revision)

    assert :ok = Vettore.delete(collection, "doc")
    assert {:error, :not_found} = ETS.fetch_revision(state, "doc", revision)

    assert :ok = Vettore.put(collection, %{id: "doc", vector: [1.0]})
    assert {:error, :not_found} = ETS.fetch_revision(state, "doc", revision)
    assert {:ok, [{"doc", updated_revision, [1]}]} = ETS.binary_rows(state)
    assert {:ok, %Embedding{vector: [1.0]}} = ETS.fetch_revision(state, "doc", updated_revision)
  end
end
