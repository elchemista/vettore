Code.require_file("support/fastembed_fixture.ex", __DIR__)

defmodule VettoreExFastembedIntegrationTest do
  use ExUnit.Case, async: false

  alias Vettore.{Collection, Compute, Distance, Embedding, Nifs, Result, Vector}
  alias Vettore.Test.FastembedFixture

  setup_all do
    {:ok, fixture: FastembedFixture.load!()}
  end

  test "the committed artifact contains real, normalized FastEmbed vectors", %{fixture: fixture} do
    assert fixture.model == "BAAI/bge-small-en-v1.5"
    assert fixture.dimensions == 384
    assert length(fixture.documents) == 30
    assert length(fixture.queries) == 3

    for %{vector: vector} <- fixture.documents ++ fixture.queries do
      assert length(vector) == fixture.dimensions
      assert_in_delta vector_norm(vector), 1.0, 1.0e-5
    end
  end

  test "all search modes retrieve semantic neighbors from real embeddings", %{fixture: fixture} do
    {flat, hnsw} = indexed_collections(fixture)

    on_exit(fn ->
      Vettore.close(flat)
      Vettore.close(hnsw)
    end)

    for query <- fixture.queries do
      assert_category_search(flat, hnsw, fixture, query)
    end
  end

  test "real embeddings match independent f64 metric references in list and binary form", %{
    fixture: fixture
  } do
    assert_kernel_references(fixture)
  end

  test "real embeddings retain CPU GPU parity after writes and concurrent warm searches", %{
    fixture: fixture
  } do
    if Compute.gpu_detected?() do
      for metric <- [:l2, :cosine, :inner_product] do
        {:ok, cpu} =
          Collection.new(
            dimensions: fixture.dimensions,
            metric: metric,
            normalize: :none,
            index_options: [gpu: false]
          )

        {:ok, gpu} =
          Collection.new(
            dimensions: fixture.dimensions,
            metric: metric,
            normalize: :none,
            index_options: [gpu: true, gpu_fallback: :error]
          )

        on_exit(fn ->
          Vettore.close(cpu)
          Vettore.close(gpu)
        end)

        values = embeddings(fixture)
        :ok = Collection.put_many(cpu, values)
        :ok = Collection.put_many(gpu, values)
        assert_real_rank_parity(cpu, gpu, fixture.queries)
        assert {:ok, {1, true}} = Nifs.flat_gpu_cache_info(gpu.index_state)

        replacement = %{id: hd(values).id, vector: hd(fixture.queries).vector}
        :ok = Collection.delete(cpu, replacement.id)
        :ok = Collection.delete(gpu, replacement.id)
        :ok = Collection.put(cpu, replacement)
        :ok = Collection.put(gpu, replacement)
        assert {:ok, {1, false}} = Nifs.flat_gpu_cache_info(gpu.index_state)
        assert_real_rank_parity(cpu, gpu, fixture.queries)
        assert {:ok, {2, true}} = Nifs.flat_gpu_cache_info(gpu.index_state)

        :ok = Collection.delete(cpu, replacement.id)
        :ok = Collection.delete(gpu, replacement.id)
        assert_real_rank_parity(cpu, gpu, fixture.queries)
        assert {:ok, {3, true}} = Nifs.flat_gpu_cache_info(gpu.index_state)

        tasks =
          Task.async_stream(1..12, fn _ -> assert_real_rank_parity(cpu, gpu, fixture.queries) end,
            max_concurrency: 4,
            timeout: 30_000
          )

        Enum.each(tasks, fn result -> assert {:ok, :ok} = result end)
        assert {:ok, {3, true}} = Nifs.flat_gpu_cache_info(gpu.index_state)
      end
    else
      refute System.get_env("VETTORE_REQUIRE_GPU") == "1"
    end
  end

  test "resident GPU Flat matches SIMD on real FastEmbed embeddings", %{fixture: fixture} do
    if Compute.gpu_detected?() do
      {:ok, cpu} =
        build_collection(:real_fastembed_gpu_cpu, fixture.dimensions,
          index: :flat,
          index_options: [gpu: false]
        )

      {:ok, gpu} =
        build_collection(:real_fastembed_gpu_strict, fixture.dimensions,
          index: :flat,
          index_options: [gpu: true, gpu_fallback: :error, gpu_min_size: 1]
        )

      on_exit(fn ->
        Vettore.close(cpu)
        Vettore.close(gpu)
      end)

      values = embeddings(fixture)
      :ok = Collection.put_many(cpu, values)
      :ok = Collection.put_many(gpu, values)
      assert {:ok, {0, false}} = Nifs.flat_gpu_cache_info(gpu.index_state)

      for query <- fixture.queries do
        assert {:ok, cpu_results} = Collection.search(cpu, query.vector, limit: 5)
        assert {:ok, gpu_results} = Collection.search(gpu, query.vector, limit: 5)

        assert Enum.map(gpu_results, & &1.id) == Enum.map(cpu_results, & &1.id)

        for {expected, actual} <- Enum.zip(cpu_results, gpu_results) do
          assert_in_delta actual.score, expected.score, max(abs(expected.score) * 1.0e-4, 1.0e-5)
        end

        assert query.category in (gpu_results
                                  |> Enum.take(3)
                                  |> Enum.map(& &1.metadata.category))
      end

      assert {:ok, {1, true}} = Nifs.flat_gpu_cache_info(gpu.index_state)
    else
      refute System.get_env("VETTORE_REQUIRE_GPU") == "1",
             "VETTORE_REQUIRE_GPU=1 but real-embedding GPU search has no adapter"
    end
  end

  test "snapshot round trips preserve rankings for real embeddings", %{fixture: fixture} do
    path = temporary_path("fastembed-snapshot")
    {:ok, collection} = build_collection(:fastembed_snapshot, fixture.dimensions)
    :ok = Collection.put_many(collection, embeddings(fixture))
    :ok = Collection.snapshot(collection, path)
    {:ok, restored} = Collection.load_snapshot(path)

    on_exit(fn ->
      Vettore.close(collection)
      Vettore.close(restored)
      File.rm(path)
    end)

    for %{vector: query_vector} <- fixture.queries do
      assert {:ok, original} = Collection.search(collection, query_vector, limit: 5)
      assert {:ok, reloaded} = Collection.search(restored, query_vector, limit: 5)
      assert Enum.map(reloaded, & &1.id) == Enum.map(original, & &1.id)
    end
  end

  @tag skip: System.get_env("VETTORE_TEST_EX_FASTEMBED") != "1"
  @tag timeout: 180_000
  test "fresh ex_fastembed inference matches the artifact and searches on the GPU", %{
    fixture: fixture
  } do
    generated = FastembedFixture.generate!()

    assert generated.model == fixture.model
    assert generated.dimensions == fixture.dimensions
    assert_kernel_references(generated)
    {flat, hnsw} = indexed_collections(generated)

    on_exit(fn ->
      Vettore.close(flat)
      Vettore.close(hnsw)
    end)

    for query <- generated.queries, do: assert_category_search(flat, hnsw, generated, query)

    fixture_vectors = Enum.map(fixture.documents ++ fixture.queries, & &1.vector)
    generated_vectors = Enum.map(generated.documents ++ generated.queries, & &1.vector)

    for {expected, actual} <- Enum.zip(fixture_vectors, generated_vectors) do
      assert {:ok, similarity} = Distance.cosine(expected, actual)
      assert similarity > 0.999_9
    end

    if Compute.gpu_detected?() do
      {:ok, gpu} =
        build_collection(:fresh_fastembed_gpu_strict, generated.dimensions,
          index: :flat,
          index_options: [gpu: true, gpu_fallback: :error, gpu_min_size: 1]
        )

      on_exit(fn -> Vettore.close(gpu) end)
      :ok = Collection.put_many(gpu, embeddings(generated))

      for query <- generated.queries do
        assert {:ok, results} = Collection.search(gpu, query.vector, limit: 5)

        assert query.category in (results
                                  |> Enum.take(3)
                                  |> Enum.map(& &1.metadata.category))
      end

      assert {:ok, {1, true}} = Nifs.flat_gpu_cache_info(gpu.index_state)
    else
      refute System.get_env("VETTORE_REQUIRE_GPU") == "1",
             "VETTORE_REQUIRE_GPU=1 but fresh real embeddings cannot reach the GPU"
    end
  end

  defp assert_real_rank_parity(cpu, gpu, queries) do
    for query <- queries do
      assert {:ok, expected} = Collection.search(cpu, query.vector, limit: 10)
      assert {:ok, actual} = Collection.search(gpu, query.vector, limit: 10)
      assert Enum.map(actual, & &1.id) == Enum.map(expected, & &1.id)

      for {left, right} <- Enum.zip(expected, actual),
          do: assert_in_delta(left.score, right.score, 2.0e-5)
    end

    :ok
  end

  defp assert_kernel_references(fixture) do
    gpu? = Compute.gpu_detected?()
    assert gpu? or System.get_env("VETTORE_REQUIRE_GPU") != "1"

    metrics = [
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

    for query <- fixture.queries,
        document <- Enum.take(fixture.documents, 4),
        metric <- metrics do
      left = f32_values(query.vector)
      right = f32_values(document.vector)
      expected = scalar_metric(metric, left, right)
      tolerance = max(abs(expected) * 2.0e-6, 1.0e-7)
      representations = [{left, right}, {f32_binary(left), f32_binary(right)}]

      for {a, b} <- representations do
        assert {:ok, cpu} = Vector.metric(a, b, metric, gpu: false)
        assert_in_delta cpu, expected, tolerance

        assert_gpu_reference(gpu?, a, b, metric, expected, tolerance)
      end
    end
  end

  defp assert_gpu_reference(false, _left, _right, _metric, _expected, _tolerance), do: :ok

  defp assert_gpu_reference(true, left, right, metric, expected, tolerance) do
    assert {:ok, gpu} = Vector.metric(left, right, metric, gpu: true, gpu_fallback: :error)
    assert_in_delta gpu, expected, tolerance
  end

  # BEAM f64 arithmetic is independent of the native implementations. Inputs
  # are rounded to f32 first, matching the NIF interchange contract.
  defp scalar_metric(:l2, left, right), do: :math.sqrt(scalar_metric(:l2_squared, left, right))

  defp scalar_metric(:l2_squared, left, right),
    do: Enum.reduce(Enum.zip(left, right), 0.0, fn {a, b}, sum -> sum + (a - b) * (a - b) end)

  defp scalar_metric(:inner_product, left, right),
    do: Enum.reduce(Enum.zip(left, right), 0.0, fn {a, b}, sum -> sum + a * b end)

  defp scalar_metric(:negative_inner_product, left, right),
    do: -scalar_metric(:inner_product, left, right)

  defp scalar_metric(:cosine, left, right),
    do: scalar_metric(:inner_product, left, right) / (vector_norm(left) * vector_norm(right))

  defp scalar_metric(:manhattan, left, right),
    do: Enum.reduce(Enum.zip(left, right), 0.0, fn {a, b}, sum -> sum + abs(a - b) end)

  defp scalar_metric(:chebyshev, left, right),
    do: Enum.zip(left, right) |> Enum.map(fn {a, b} -> abs(a - b) end) |> Enum.max()

  defp scalar_metric(:hamming, left, right),
    do: Enum.count(Enum.zip(left, right), fn {a, b} -> a != 0.0 != (b != 0.0) end) / 1

  defp scalar_metric(:jaccard, left, right) do
    pairs = Enum.zip(left, right)
    union = Enum.count(pairs, fn {a, b} -> a != 0.0 or b != 0.0 end)
    intersection = Enum.count(pairs, fn {a, b} -> a != 0.0 and b != 0.0 end)
    if union == 0, do: 0.0, else: 1.0 - intersection / union
  end

  defp f32_values(values), do: for(<<value::float-little-32 <- f32_binary(values)>>, do: value)
  defp f32_binary(values), do: for(value <- values, into: <<>>, do: <<value::float-little-32>>)

  defp indexed_collections(fixture) do
    {:ok, flat} = build_collection(:real_fastembed_flat, fixture.dimensions)

    {:ok, hnsw} =
      build_collection(:real_fastembed_hnsw, fixture.dimensions,
        index: :hnsw,
        index_options: [
          m: 8,
          m0: 16,
          ef_construction: 200,
          ef_search: 200,
          max_level: 12
        ]
      )

    values = embeddings(fixture)
    :ok = Collection.put_many(flat, values)
    :ok = Collection.put_many(hnsw, values)
    {flat, hnsw}
  end

  defp build_collection(name, dimensions, options \\ []) do
    Collection.new(
      [name: name, dimensions: dimensions, metric: :cosine, normalize: :l2] ++ options
    )
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

  defp assert_category_search(flat, hnsw, fixture, query) do
    candidates = length(fixture.documents)
    dimensions = fixture.dimensions

    assert {:ok, [%Result{} = exact_top | _] = exact_results} =
             Collection.search(flat, query.vector, limit: 5)

    assert query.category in (exact_results
                              |> Enum.take(3)
                              |> Enum.map(& &1.metadata.category))

    exact_top_id = exact_top.id

    assert {:ok, [%Result{id: ^exact_top_id} | _]} =
             Collection.search(hnsw, query.vector, limit: 5)

    assert {:ok, [%Result{id: ^exact_top_id} | _]} =
             Collection.funnel_search(flat, query.vector,
               stages: [min(128, dimensions), dimensions],
               candidates: candidates,
               limit: 5
             )

    assert {:ok, [%Result{id: ^exact_top_id} | _]} =
             Collection.quantized_search(flat, query.vector,
               candidates: candidates,
               limit: 5
             )

    assert {:ok, [%Result{id: ^exact_top_id} | _]} =
             Collection.multi_vector_search(flat, [query.vector], metric: :cosine, limit: 5)

    assert {:ok, [%Result{id: ^exact_top_id} | _]} =
             Collection.hybrid_search(flat, query.vector,
               generators: [
                 funnel: [stages: [min(128, dimensions), dimensions], candidates: candidates],
                 quantized: [candidates: candidates]
               ],
               rerank: :exact,
               limit: 5
             )

    assert {:ok, [%Result{id: ^exact_top_id} | _]} =
             Collection.hybrid_search(hnsw, query.vector,
               generators: [
                 hnsw: [candidates: candidates],
                 quantized: [candidates: candidates]
               ],
               limit: 5
             )
  end

  defp vector_norm(vector) do
    vector
    |> Enum.reduce(0.0, fn value, sum -> sum + value * value end)
    |> :math.sqrt()
  end

  defp temporary_path(name) do
    unique = System.unique_integer([:positive, :monotonic])
    Path.join(System.tmp_dir!(), "vettore-#{name}-#{unique}.ets")
  end
end
