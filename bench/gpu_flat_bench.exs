defmodule Vettore.GpuFlatBench do
  alias Vettore.{Collection, Compute, Nifs}

  @metrics %{
    "l2" => :l2,
    "l2_squared" => :l2_squared,
    "cosine" => :cosine,
    "inner_product" => :inner_product,
    "negative_inner_product" => :negative_inner_product,
    "manhattan" => :manhattan,
    "chebyshev" => :chebyshev,
    "hamming" => :hamming,
    "jaccard" => :jaccard
  }

  def run do
    unless Compute.gpu_detected?() do
      raise "no GPU adapter detected; configure the platform driver before this benchmark"
    end

    dimensions = positive_env!("VETTORE_BENCH_DIMENSIONS", 384)
    records = positive_env!("VETTORE_BENCH_BATCH", 25_000)
    limit = positive_env!("VETTORE_BENCH_LIMIT", 10)
    time = positive_env!("VETTORE_BENCH_TIME", 5)
    warmup = non_negative_env!("VETTORE_BENCH_WARMUP", 2)
    seed = non_negative_env!("VETTORE_BENCH_SEED", 20_260_825)
    metric = metric_env!()

    if records < limit do
      raise "VETTORE_BENCH_BATCH must be at least VETTORE_BENCH_LIMIT"
    end

    if limit > 64 do
      raise "VETTORE_BENCH_LIMIT must be at most 64 for resident GPU top-k"
    end

    :rand.seed(:exsss, {seed + 1, seed + 2, seed + 3})
    query = random_vector(dimensions)

    embeddings =
      for index <- 1..records do
        %{
          id: "doc-#{String.pad_leading(Integer.to_string(index), 8, "0")}",
          vector: random_vector(dimensions)
        }
      end

    {:ok, cpu} = collection(:gpu_flat_cpu, dimensions, metric, gpu: false)

    {:ok, gpu} =
      collection(:gpu_flat_resident, dimensions, metric,
        gpu: true,
        gpu_fallback: :error,
        gpu_min_size: 1
      )

    try do
      :ok = put_many!(cpu, embeddings)
      :ok = put_many!(gpu, embeddings)

      {cold_microseconds, {:ok, gpu_reference}} =
        :timer.tc(fn -> Collection.search(gpu, query, limit: limit) end)

      {:ok, cpu_reference} = Collection.search(cpu, query, limit: limit)
      validate_top_k!(cpu_reference, gpu_reference)

      {:ok, {builds, resident?}} = Nifs.flat_gpu_cache_info(gpu.index_state)
      adapter = Compute.gpu_info()

      IO.puts("\nAdapter: #{inspect(adapter)}")

      IO.puts(
        "Dataset: #{records} rows x #{dimensions} dimensions, metric=#{metric}, limit=#{limit}"
      )

      IO.puts(
        "First GPU query (matrix upload + search): #{Float.round(cold_microseconds / 1_000, 2)} ms"
      )

      IO.puts("Resident cache: builds=#{builds}, active=#{resident?}\n")

      Benchee.run(
        %{
          "flat/CPU contiguous SIMD" => fn ->
            Collection.search(cpu, query, limit: limit)
          end,
          "flat/GPU resident warm" => fn ->
            Collection.search(gpu, query, limit: limit)
          end
        },
        time: time,
        warmup: warmup,
        memory_time: 0
      )
    after
      Collection.close(cpu)
      Collection.close(gpu)
    end
  end

  defp collection(name, dimensions, metric, index_options) do
    Collection.new(
      name: name,
      dimensions: dimensions,
      metric: metric,
      normalize: :none,
      score: :raw,
      index: :flat,
      index_options: index_options
    )
  end

  defp put_many!(collection, embeddings) do
    case Collection.put_many(collection, embeddings) do
      :ok -> :ok
      {:error, reason} -> raise "benchmark ingestion failed: #{inspect(reason)}"
    end
  end

  defp random_vector(dimensions) do
    for _ <- 1..dimensions, do: :rand.uniform() * 2.0 - 1.0
  end

  defp metric_env! do
    metric = System.get_env("VETTORE_BENCH_METRIC", "cosine")

    case Map.fetch(@metrics, metric) do
      {:ok, metric} ->
        metric

      :error ->
        raise "unsupported VETTORE_BENCH_METRIC=#{inspect(metric)}"
    end
  end

  defp validate_top_k!(cpu, gpu) do
    if length(cpu) != length(gpu) do
      raise "CPU and GPU top-k hit counts differ before benchmark"
    end

    Enum.zip(cpu, gpu)
    |> Enum.each(fn {cpu_hit, gpu_hit} ->
      tolerance = score_tolerance(cpu_hit.score)

      if abs(cpu_hit.score - gpu_hit.score) > tolerance do
        raise "CPU and GPU top-k scores differ before benchmark"
      end
    end)

    boundary_score = List.last(cpu).score

    required_ids =
      cpu
      |> Enum.reject(&scores_tied?(&1.score, boundary_score))
      |> MapSet.new(& &1.id)

    gpu_ids = MapSet.new(gpu, & &1.id)

    unless MapSet.subset?(required_ids, gpu_ids) do
      raise "GPU top-k is missing an unambiguous CPU result before benchmark"
    end
  end

  defp scores_tied?(left, right),
    do: abs(left - right) <= max(score_tolerance(left), score_tolerance(right))

  defp score_tolerance(score), do: max(abs(score) * 1.0e-4, 1.0e-5)

  defp positive_env!(name, default) do
    value = non_negative_env!(name, default)
    if value > 0, do: value, else: raise("#{name} must be positive")
  end

  defp non_negative_env!(name, default) do
    case Integer.parse(System.get_env(name, Integer.to_string(default))) do
      {value, ""} when value >= 0 -> value
      _error -> raise "#{name} must be a non-negative integer"
    end
  end
end

Vettore.GpuFlatBench.run()
