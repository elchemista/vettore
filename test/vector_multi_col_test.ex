defmodule VettoreMultiColTest do
  use ExUnit.Case, async: true
  alias Vettore.Embedding

  @metrics [:euclidean, :cosine, :dot, :hnsw, :binary]
  @vector [-1.3, 2.7, 3.1, 4.3, 5.7, 5.2, 6.1, 7.9, 8.2, 9.4]
  @dim length(@vector)

  # 1) For float‐based metrics, shift *every* coordinate by a large multiple
  #    so each vector is radically different.
  # defp gen_vector(i) do
  #   Enum.map(@vector, fn v -> v + i * 100.0 end)
  # end

  # # 2) For binary, flip two different bits per vector so sign‐bit pattern always changes.
  # defp gen_binary_vector(i) do
  #   # flip bit at pos i and pos (i+1)
  #   idx1 = rem(i - 1, @dim)
  #   idx2 = rem(i, @dim)

  #   @vector
  #   |> List.update_at(idx1, &(-&1))
  #   |> List.update_at(idx2, &(-&1))
  # end

  describe "concurrent collections test" do
    test "concurrent creation, insertion and get_by_value in multiple collections" do
      db = Vettore.new()

      results =
        @metrics
        |> Task.async_stream(
          fn metric ->
            coll = "#{metric}_coll"
            assert {:ok, ^coll} = Vettore.create_collection(db, coll, @dim, metric)

            id = "id_#{metric}"

            emb = %Embedding{
              value: id,
              vector: @vector,
              metadata: %{"metric" => to_string(metric)}
            }

            assert {:ok, ^id} = Vettore.insert(db, coll, emb)
            {:ok, %Embedding{value: val, metadata: meta}} = Vettore.get_by_value(db, coll, id)

            {metric, val, meta}
          end,
          max_concurrency: length(@metrics),
          timeout: 10_000
        )
        |> Enum.map(&elem(&1, 1))

      for {metric, val, meta} <- results do
        assert val == "id_#{metric}"
        assert meta == %{"metric" => to_string(metric)}
      end
    end

    # test "concurrent batch inserts and similarity searches across multiple collections" do
    #   db = Vettore.new()

    #   tasks =
    #     @metrics
    #     |> Enum.map(fn metric ->
    #       Task.async(fn ->
    #         coll = "#{metric}_batch"
    #         assert {:ok, ^coll} = Vettore.create_collection(db, coll, @dim, metric)

    #         bs =
    #           for i <- 1..5 do
    #             vec =
    #               case metric do
    #                 :binary -> gen_binary_vector(i)
    #                 _ -> gen_vector(i)
    #               end

    #             %Embedding{
    #               value: "b#{i}_#{metric}",
    #               vector: vec |> Enum.map(fn x -> x + Enum.random(1..50) * 0.4 end),
    #               metadata: %{"idx" => "#{i}"}
    #             }
    #           end

    #         # now all 5 vectors are very distinct
    #         assert {:ok, ids} = Vettore.batch(db, coll, bs)
    #         assert length(ids) == 5

    #         # pick one of our inserted vectors as the query
    #         query =
    #           case metric do
    #             :binary -> gen_binary_vector(3)
    #             _ -> gen_vector(3)
    #           end

    #         assert {:ok, results} = Vettore.similarity_search(db, coll, query, limit: 3)
    #         assert length(results) == 3

    #         for {id, score} <- results do
    #           assert id in ids
    #           assert is_float(score)
    #         end

    #         {metric, ids}
    #       end)
    #     end)

    #   outs = Enum.map(tasks, &Task.await(&1, 10_000))

    #   for {_metric, ids} <- outs do
    #     assert length(ids) == 5
    #   end
    # end
  end
end
