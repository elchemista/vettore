defmodule VettoreNumericalRegressionTest do
  use ExUnit.Case, async: true

  alias Vettore.{Collection, Compute, Distance, Nifs, Vector}

  test "signed-zero dot scores use the id tiebreaker after final f32 rounding" do
    for {index, metric} <- [
          flat: :inner_product,
          flat: :negative_inner_product,
          hnsw: :inner_product
        ] do
      {:ok, collection} =
        Collection.new(dimensions: 1, metric: metric, normalize: :none, index: index)

      on_exit(fn -> Collection.close(collection) end)

      :ok =
        Collection.put_many(collection, [
          %{id: "a-negative", vector: [-1.0e-30]},
          %{id: "z-positive", vector: [1.0e-30]}
        ])

      assert {:ok, hits} = Collection.search(collection, [1.0e-30], limit: 2)
      assert Enum.map(hits, & &1.id) == ["a-negative", "z-positive"]
    end
  end

  test "resident GPU falls back for tiny product scores and strict mode reports the range" do
    if Compute.gpu_detected?() do
      for metric <- [:inner_product, :negative_inner_product, :l2_squared] do
        {:ok, collection} =
          Collection.new(
            dimensions: 768,
            metric: metric,
            normalize: :none,
            score: :raw,
            index_options: [gpu: true, gpu_fallback: :cpu]
          )

        on_exit(fn -> Collection.close(collection) end)

        :ok =
          Collection.put_many(collection, [
            %{id: "a-small", vector: List.duplicate(1.0e-23, 768)},
            %{id: "z-large", vector: List.duplicate(2.0e-23, 768)}
          ])

        query = List.duplicate(1.0e-23, 768)
        cpu = %{collection | index_options: [gpu: false]}
        strict = %{collection | index_options: [gpu: true, gpu_fallback: :error]}
        assert {:ok, expected} = Collection.search(cpu, query, limit: 2)
        assert {:ok, ^expected} = Collection.search(collection, query, limit: 2)

        assert {:error, :gpu_failed} = Collection.search(strict, query, limit: 2)

        assert {:error, "gpu numeric range unsupported"} =
                 Nifs.flat_gpu_search(collection.index_state, query, 2)

        # A normal query must also reject tiny rows at matrix upload time.
        normal_query = List.duplicate(1.0, 768)
        assert {:error, :gpu_failed} = Collection.search(strict, normal_query, limit: 2)

        assert {:error, "gpu numeric range unsupported"} =
                 Nifs.flat_gpu_search(collection.index_state, normal_query, 2)
      end
    else
      refute System.get_env("VETTORE_REQUIRE_GPU") == "1"
    end
  end

  test "dimensions above 786 work across Flat, HNSW and resident GPU search" do
    gpu? = Compute.gpu_detected?()
    assert gpu? or System.get_env("VETTORE_REQUIRE_GPU") != "1"

    for dimensions <- [787, 1024, 1536, 3072], metric <- [:l2, :cosine, :inner_product] do
      # Put signal at the end so truncating at 768/786 cannot pass this test.
      prefix = List.duplicate(0.0, dimensions - 2)
      query = prefix ++ [0.0, 1.0]

      rows = [
        %{id: "z-best", vector: query},
        %{id: "m-near", vector: prefix ++ [0.6, 0.8]},
        %{id: "a-far", vector: prefix ++ [0.0, -1.0]}
      ]

      backends = [{:flat, [gpu: false]}, {:hnsw, []}]

      backends =
        if gpu?, do: backends ++ [{:flat, [gpu: true, gpu_fallback: :error]}], else: backends

      for {index, index_options} <- backends do
        {:ok, collection} =
          Collection.new(
            dimensions: dimensions,
            metric: metric,
            normalize: :none,
            index: index,
            index_options: index_options
          )

        on_exit(fn -> Collection.close(collection) end)
        assert :ok = Collection.put_many(collection, rows)
        assert {:ok, hits} = Collection.search(collection, query, limit: 3)
        assert Enum.map(hits, & &1.id) == ["z-best", "m-near", "a-far"]
      end
    end
  end

  test "dot rounds the final sum at list, binary, and collection boundaries" do
    left = List.duplicate(1.0e-23, 768)
    expected = 768 * 1.0e-46

    for metric <- [:inner_product, :negative_inner_product], sign <- [-1.0, 1.0] do
      right = List.duplicate(sign * 1.0e-23, 768)
      expected_sign = if metric == :inner_product, do: sign, else: -sign
      assert {:ok, value} = apply(Distance, metric, [left, right, [gpu: false]])
      assert_in_delta value, expected_sign * expected, 1.5e-45
      assert {:ok, ^value} = apply(Vector, metric, [binary(left), binary(right), [gpu: false]])
    end

    for {index, metric} <- [
          {:flat, :inner_product},
          {:flat, :negative_inner_product},
          {:hnsw, :inner_product}
        ] do
      assert {:ok, collection} =
               Collection.new(
                 dimensions: 768,
                 metric: metric,
                 normalize: :none,
                 index: index,
                 score: :raw
               )

      on_exit(fn -> Collection.close(collection) end)

      assert :ok =
               Collection.put_many(collection, [
                 %{id: "a-worse", vector: left},
                 %{id: "z-better", vector: List.duplicate(2.0e-23, 768)}
               ])

      assert {:ok, hits} = Collection.search(collection, left, limit: 2)
      assert Enum.map(hits, & &1.id) == ["z-better", "a-worse"]
      assert Enum.all?(hits, &(&1.score != 0.0))
    end
  end

  test "tiny L2 survives the list and binary NIF boundaries" do
    for dimensions <- [1, 7, 9, 383, 769], scale <- [1.0e-30, 1.0e-22, 1.0e-20] do
      left = List.duplicate(scale, dimensions)
      right = List.duplicate(0.0, dimensions)
      expected = scale * :math.sqrt(dimensions)
      assert {:ok, distance} = Distance.l2(left, right, gpu: false)
      assert_in_delta distance, expected, expected * 1.0e-6
      assert {:ok, binary_distance} = Vector.l2(binary(left), binary(right), gpu: false)
      assert binary_distance == distance
    end
  end

  test "squared L2 rounds only the final result to f32" do
    assert {:ok, tiny} = Distance.l2_squared([1.0e-22], [0.0], gpu: false)
    assert tiny > 0.0
    assert_in_delta tiny, 1.0e-44, 1.5e-45
    # The exact 1e-60 square is below the output type's representable range.
    assert {:ok, zero} = Distance.l2_squared([1.0e-30], [0.0], gpu: false)
    assert zero == 0.0
  end

  test "tiny vectors rank by distance through Flat and HNSW collections" do
    for index <- [:flat, :hnsw] do
      assert {:ok, collection} =
               Collection.new(
                 dimensions: 9,
                 metric: :l2,
                 normalize: :none,
                 index: index,
                 score: :raw
               )

      on_exit(fn -> Collection.close(collection) end)

      assert :ok =
               Collection.put_many(collection, [
                 %{id: "a-far", vector: List.duplicate(3.0e-30, 9)},
                 %{id: "z-near", vector: List.duplicate(1.0e-30, 9)}
               ])

      assert {:ok, results} = Collection.search(collection, List.duplicate(0.0, 9), limit: 2)
      assert Enum.map(results, & &1.id) == ["z-near", "a-far"]
      assert hd(results).score > List.last(results).score

      assert {:ok, displayed} =
               Collection.search(%{collection | score: :similarity}, List.duplicate(0.0, 9),
                 limit: 2
               )

      assert Enum.map(displayed, & &1.id) == ["z-near", "a-far"]
      assert Enum.map(displayed, & &1.score) == [1.0, 1.0]
      assert hd(displayed).distance < List.last(displayed).distance
    end
  end

  test "native workload and search remain consistent during concurrent batched writes" do
    index = Nifs.flat_new_l2()
    assert {:ok, {}} = Nifs.flat_insert_many(index, [{"anchor", [0.0, 0.0]}])

    writer =
      Task.async(fn ->
        for batch <- 1..40 do
          rows = for row <- 1..200, do: {"row-#{row}", [batch / 1, row / 1]}
          assert {:ok, {}} = Nifs.flat_insert_many(index, rows)
        end
      end)

    readers =
      Task.async_stream(
        1..8,
        fn _ ->
          for _ <- 1..50 do
            assert {:ok, {rows, 2}} = Nifs.flat_workload(index)
            assert rows in [1, 201]
            assert {:ok, [{"anchor", distance}]} = Nifs.flat_search(index, [0.0, 0.0], 1)
            assert distance == 0.0
          end
        end,
        max_concurrency: 8,
        timeout: 30_000
      )

    Enum.each(readers, fn result -> assert {:ok, _} = result end)
    Task.await(writer, 30_000)
    assert {:ok, {201, 2}} = Nifs.flat_workload(index)
  end

  defp binary(values), do: for(value <- values, into: <<>>, do: <<value::float-little-32>>)
end
