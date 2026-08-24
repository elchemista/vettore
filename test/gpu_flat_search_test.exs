defmodule VettoreGpuFlatSearchTest do
  use ExUnit.Case, async: false

  alias Vettore.{Collection, Compute, Nifs, Result}
  alias Vettore.Index.Flat

  @application_keys [:gpu, :gpu_fallback, :gpu_min_size]
  @metrics [
    :l2,
    :l2_squared,
    :cosine,
    :inner_product,
    :negative_inner_product,
    :manhattan,
    :chebyshev,
    :hamming,
    :jaccard
  ]

  setup do
    previous = Map.new(@application_keys, &{&1, Application.fetch_env(:vettore, &1)})

    on_exit(fn ->
      Enum.each(previous, fn
        {key, {:ok, value}} -> Application.put_env(:vettore, key, value)
        {key, :error} -> Application.delete_env(:vettore, key)
      end)
    end)

    if System.get_env("VETTORE_REQUIRE_GPU") == "1" do
      assert Compute.gpu_detected?(),
             "VETTORE_REQUIRE_GPU=1 but the resident Flat GPU path has no adapter"
    end

    :ok
  end

  test "Flat accepts compute policy and exposes its real batched workload" do
    assert {:error, :gpu_not_available} =
             Compute.normalize_search_result({:error, "gpu not detected"})

    assert {:error, :gpu_limit_too_large} =
             Compute.normalize_search_result(
               {:error, "gpu flat top-k supports at most 64 results"}
             )

    assert {:error, :gpu_failed} =
             Compute.normalize_search_result({:error, "gpu resident matrix expired"})

    assert {:ok, :unchanged} = Compute.normalize_search_result({:ok, :unchanged})

    assert {:ok, collection} =
             Collection.new(
               dimensions: 3,
               metric: :l2,
               normalize: :none,
               index: :flat,
               index_options: [gpu: false, gpu_min_size: 12, gpu_fallback: :cpu]
             )

    assert :ok =
             Collection.put_many(collection, [
               %{id: "a", vector: [0.0, 0.0, 0.0]},
               %{id: "b", vector: [1.0, 0.0, 0.0]},
               %{id: "c", vector: [0.0, 1.0, 0.0]},
               %{id: "d", vector: [0.0, 0.0, 1.0]}
             ])

    assert {:ok, {4, 3}} = Nifs.flat_workload(collection.index_state)

    assert {:ok, [%Result{id: "a"}, %Result{id: "b"}]} =
             Collection.search(collection, [0.0, 0.0, 0.0], limit: 2)

    assert {:error, :invalid_gpu_option} = Flat.new(:l2, gpu: :sometimes)
    assert {:error, :invalid_gpu_fallback} = Flat.new(:l2, gpu_fallback: :sometimes)
    assert {:error, :invalid_gpu_min_size} = Flat.new(:l2, gpu_min_size: 0)
    assert {:error, :invalid_flat_options} = Flat.new(:l2, gpu: false, gpu: true)
    assert :ok = Collection.close(collection)

    assert {:ok, automatic} =
             Collection.new(
               dimensions: 2,
               metric: :l2,
               normalize: :none,
               index: :flat,
               index_options: [gpu: :auto, gpu_min_size: 10, gpu_fallback: :error]
             )

    assert :ok = Collection.put(automatic, %{id: "small", vector: [1.0, 1.0]})
    assert {:ok, [%Result{id: "small"}]} = Collection.search(automatic, [1.0, 1.0])
    assert {:ok, {0, false}} = Nifs.flat_gpu_cache_info(automatic.index_state)
    assert {:ok, :auto} = Compute.selection(automatic.index_options)
    assert :ok = Collection.close(automatic)
  end

  test "resident Flat search matches SIMD for every metric and rebuilds only after mutation" do
    if Compute.gpu_detected?() do
      dimensions = 8
      embeddings = embeddings(137, dimensions)
      query = [0.25, -1.5, 2.0, 0.75, 1.0, -0.5, 1.25, 0.0]

      for metric <- @metrics do
        assert {:ok, cpu} = collection(metric, dimensions, gpu: false)

        assert {:ok, gpu} =
                 collection(metric, dimensions,
                   gpu: true,
                   gpu_fallback: :error,
                   gpu_min_size: 1
                 )

        assert :ok = Collection.put_many(cpu, embeddings)
        assert :ok = Collection.put_many(gpu, embeddings)
        assert {:ok, {0, false}} = Nifs.flat_gpu_cache_info(gpu.index_state)

        assert {:ok, cpu_results} = Collection.search(cpu, query, limit: 7)
        assert {:ok, gpu_results} = Collection.search(gpu, query, limit: 7)
        assert_same_results(cpu_results, gpu_results, metric)
        assert {:ok, {1, true}} = Nifs.flat_gpu_cache_info(gpu.index_state)

        assert {:ok, repeated} = Collection.search(gpu, query, limit: 7)
        assert_same_results(gpu_results, repeated, metric)
        assert {:ok, {1, true}} = Nifs.flat_gpu_cache_info(gpu.index_state)

        assert :ok = Collection.put(gpu, %{id: "new-row", vector: query})
        assert {:ok, {1, false}} = Nifs.flat_gpu_cache_info(gpu.index_state)
        assert {:ok, _results} = Collection.search(gpu, query, limit: 7)
        assert {:ok, {2, true}} = Nifs.flat_gpu_cache_info(gpu.index_state)

        assert :ok = Collection.delete(gpu, "new-row")
        assert {:ok, {2, false}} = Nifs.flat_gpu_cache_info(gpu.index_state)
        assert {:ok, restored} = Collection.search(gpu, query, limit: 7)
        assert_same_results(cpu_results, restored, metric)
        assert {:ok, {3, true}} = Nifs.flat_gpu_cache_info(gpu.index_state)

        assert :ok = Collection.close(cpu)
        assert :ok = Collection.close(gpu)
      end
    else
      assert true
    end
  end

  test "GPU reduction falls back to SIMD for unusually large result windows" do
    if Compute.gpu_detected?() do
      embeddings = embeddings(80, 2)

      assert {:ok, fallback} =
               collection(:l2, 2, gpu: true, gpu_fallback: :cpu, gpu_min_size: 1)

      assert :ok = Collection.put_many(fallback, embeddings)
      assert {:ok, results} = Collection.search(fallback, [0.0, 0.0], limit: 65)
      assert length(results) == 65
      assert {:ok, {0, false}} = Nifs.flat_gpu_cache_info(fallback.index_state)

      assert {:ok, strict} =
               collection(:l2, 2, gpu: true, gpu_fallback: :error, gpu_min_size: 1)

      assert :ok = Collection.put_many(strict, embeddings)

      assert {:error, :gpu_limit_too_large} =
               Collection.search(strict, [0.0, 0.0], limit: 65)

      assert :ok = Collection.close(fallback)
      assert :ok = Collection.close(strict)
    else
      assert true
    end
  end

  test "transient batched GPU reranking matches the SIMD batch" do
    if Compute.gpu_detected?() do
      vectors =
        embeddings(137, 8)
        |> Enum.map(&{&1.id, &1.vector})

      query = [0.25, -1.5, 2.0, 0.75, 1.0, -0.5, 1.25, 0.0]

      for {metric, code} <- Enum.with_index(@metrics) do
        assert {:ok, expected} = Nifs.vector_top_k(vectors, query, code, 8, 7)
        assert {:ok, actual} = Nifs.gpu_vector_top_k(vectors, query, code, 8, 7)

        assert Enum.map(actual, &elem(&1, 0)) == Enum.map(expected, &elem(&1, 0)),
               "transient GPU ids differ for #{inspect(metric)}"

        Enum.zip(expected, actual)
        |> Enum.each(fn {{_, expected}, {_, actual}} ->
          assert_in_delta actual, expected, max(abs(expected) * 1.0e-4, 1.0e-5)
        end)
      end
    else
      assert true
    end
  end

  test "adaptive exact reranking uses the batched compute policy" do
    if Compute.gpu_detected?() do
      embeddings = embeddings(137, 8)
      query = [0.25, -1.5, 2.0, 0.75, 1.0, -0.5, 1.25, 0.0]
      assert {:ok, cpu} = collection(:l2, 8, gpu: false)

      assert {:ok, gpu} =
               collection(:l2, 8, gpu: true, gpu_fallback: :error, gpu_min_size: 1)

      assert :ok = Collection.put_many(cpu, embeddings)
      assert :ok = Collection.put_many(gpu, embeddings)

      options = [stages: [4], candidates: 64, limit: 7]
      assert {:ok, expected} = Collection.funnel_search(cpu, query, options)
      assert {:ok, actual} = Collection.funnel_search(gpu, query, options)
      assert_same_results(expected, actual, :l2)

      assert :ok = Collection.close(cpu)
      assert :ok = Collection.close(gpu)
    else
      assert true
    end
  end

  test "concurrent first queries build one resident matrix and use independent scratch" do
    if Compute.gpu_detected?() do
      embeddings = embeddings(137, 8)
      query = [0.25, -1.5, 2.0, 0.75, 1.0, -0.5, 1.25, 0.0]

      assert {:ok, gpu} =
               collection(:l2, 8, gpu: true, gpu_fallback: :error, gpu_min_size: 1)

      assert :ok = Collection.put_many(gpu, embeddings)

      searches =
        1..8
        |> Task.async_stream(
          fn _iteration -> Collection.search(gpu, query, limit: 7) end,
          max_concurrency: 8,
          ordered: false,
          timeout: 30_000
        )
        |> Enum.map(fn {:ok, {:ok, results}} -> Enum.map(results, & &1.id) end)

      assert searches |> Enum.uniq() |> length() == 1
      assert {:ok, {1, true}} = Nifs.flat_gpu_cache_info(gpu.index_state)
      assert :ok = Collection.close(gpu)
    else
      assert true
    end
  end

  defp collection(metric, dimensions, compute_options) do
    Collection.new(
      dimensions: dimensions,
      metric: metric,
      normalize: :none,
      score: :raw,
      index: :flat,
      index_options: compute_options
    )
  end

  defp embeddings(count, dimensions) do
    for row <- 0..(count - 1) do
      vector =
        for column <- 0..(dimensions - 1),
            do: coordinate(row, column, dimensions)

      %{id: "row-#{String.pad_leading(Integer.to_string(row), 4, "0")}", vector: vector}
    end
  end

  defp coordinate(row, column, dimensions) when column == dimensions - 1,
    do: if(rem(row + column, 3) == 0, do: 0.0, else: 1.0)

  defp coordinate(row, column, _dimensions),
    do: rem(row * 17 + column * 11, 97) / 19.0 - 2.5

  defp assert_same_results(expected, actual, metric) do
    assert Enum.map(actual, & &1.id) == Enum.map(expected, & &1.id),
           "resident Flat ids differ for #{inspect(metric)}"

    Enum.zip(expected, actual)
    |> Enum.each(fn {expected, actual} ->
      tolerance = max(abs(expected.score) * 1.0e-4, 1.0e-5)
      assert_in_delta actual.score, expected.score, tolerance
    end)
  end
end
