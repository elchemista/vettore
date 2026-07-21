defmodule Vettore.SearchModesBench do
  alias Vettore.{Collection, Embedding, MultiVector}
  alias Vettore.Encoding.Muvera

  @hnsw_metrics ~w(l2 cosine inner_product)a

  def run do
    dimensions = positive_env!("VETTORE_BENCH_DIMENSIONS", 384)
    records = positive_env!("VETTORE_BENCH_BATCH", 1_000)
    limit = positive_env!("VETTORE_BENCH_LIMIT", 10)
    candidates = positive_env!("VETTORE_BENCH_CANDIDATES", min(records, max(limit * 20, limit)))
    time = positive_env!("VETTORE_BENCH_TIME", 3)
    warmup = non_negative_env!("VETTORE_BENCH_WARMUP", 2)
    seed = non_negative_env!("VETTORE_BENCH_SEED", 20_260_721)
    metric = metric_env!()
    stages = stages_env!(dimensions)

    if records < limit do
      raise ArgumentError, "VETTORE_BENCH_BATCH must be at least VETTORE_BENCH_LIMIT"
    end

    if candidates < limit do
      raise ArgumentError, "VETTORE_BENCH_CANDIDATES must be at least VETTORE_BENCH_LIMIT"
    end

    :rand.seed(:exsss, {seed + 1, seed + 2, seed + 3})

    query = random_vector(dimensions)
    query_vectors = [query, perturb(query, 0.05), perturb(query, 0.1)]

    embeddings =
      for index <- 1..records do
        vector = random_vector(dimensions)

        %Embedding{
          id: "doc-#{index}",
          value: "document #{index}",
          vector: vector,
          vectors: [vector, perturb(vector, 0.05), perturb(vector, 0.1)],
          metadata: %{position: index}
        }
      end

    {flat, hnsw} = collections!(dimensions, metric)

    try do
      :ok = put_many!(flat, embeddings)
      :ok = put_many!(hnsw, embeddings)

      vector_scenarios =
        vector_scenarios(flat, hnsw, query, stages, candidates, limit)

      multi_vector_scenarios =
        multi_vector_scenarios(flat, query, query_vectors, candidates, limit)

      algorithm_scenarios = algorithm_scenarios(query_vectors, embeddings, dimensions)

      preflight!(vector_scenarios, "search/flat exact")
      preflight!(multi_vector_scenarios, "multi-vector/flat exact")
      preflight_algorithms!(algorithm_scenarios)

      IO.puts(
        "\nDataset: #{records} records, #{dimensions} dimensions, metric=#{metric}, " <>
          "limit=#{limit}, candidates=#{candidates}, stages=#{inspect(stages)}, seed=#{seed}\n"
      )

      Benchee.run(
        vector_scenarios
        |> Map.merge(multi_vector_scenarios)
        |> Map.merge(algorithm_scenarios),
        time: time,
        warmup: warmup
      )
    after
      :ok = Collection.close(flat)
      :ok = Collection.close(hnsw)
    end
  end

  defp collections!(dimensions, metric) do
    common = [dimensions: dimensions, metric: metric, score: :raw]

    {:ok, flat} = Collection.new([name: :search_modes_flat, index: :flat] ++ common)

    case Collection.new([name: :search_modes_hnsw, index: :hnsw] ++ common) do
      {:ok, hnsw} ->
        {flat, hnsw}

      {:error, reason} ->
        Collection.close(flat)
        raise "could not create HNSW benchmark collection: #{inspect(reason)}"
    end
  end

  defp put_many!(collection, embeddings) do
    case Collection.put_many(collection, embeddings) do
      :ok -> :ok
      {:error, reason} -> raise "benchmark ingestion failed: #{inspect(reason)}"
    end
  end

  defp vector_scenarios(flat, hnsw, query, stages, candidates, limit) do
    %{
      "search/flat exact" => fn -> Collection.search(flat, query, limit: limit) end,
      "search/hnsw ANN" => fn -> Collection.search(hnsw, query, limit: limit) end,
      "funnel/flat" => fn ->
        Collection.funnel_search(flat, query,
          stages: stages,
          candidates: candidates,
          limit: limit
        )
      end,
      "quantized/flat" => fn ->
        Collection.quantized_search(flat, query, candidates: candidates, limit: limit)
      end,
      "hybrid/flat default" => fn ->
        Collection.hybrid_search(flat, query, limit: limit)
      end,
      "hybrid/flat all generators" => fn ->
        Collection.hybrid_search(flat, query,
          generators: [
            search: [candidates: candidates],
            funnel: [stages: stages, candidates: candidates],
            quantized: [candidates: candidates]
          ],
          rerank: :exact,
          limit: limit
        )
      end,
      "hybrid/hnsw default" => fn ->
        Collection.hybrid_search(hnsw, query, limit: limit)
      end,
      "hybrid/hnsw all generators" => fn ->
        Collection.hybrid_search(hnsw, query,
          generators: [
            hnsw: [candidates: candidates],
            funnel: [stages: stages, candidates: candidates],
            quantized: [candidates: candidates]
          ],
          rerank: :exact,
          limit: limit
        )
      end
    }
  end

  defp multi_vector_scenarios(flat, query, query_vectors, candidates, limit) do
    %{
      "multi-vector/flat exact" => fn ->
        Collection.multi_vector_search(flat, query_vectors,
          metric: :inner_product,
          limit: limit
        )
      end,
      "hybrid/flat multi-vector rerank" => fn ->
        Collection.hybrid_search(flat, query,
          generators: [
            search: [candidates: candidates],
            quantized: [candidates: candidates]
          ],
          rerank: {:multi_vector, query_vectors, metric: :inner_product},
          limit: limit
        )
      end
    }
  end

  defp algorithm_scenarios(query_vectors, embeddings, dimensions) do
    document_vectors = hd(embeddings).vectors
    projection_dimensions = min(dimensions, 64)

    muvera_options = [
      num_repetitions: 2,
      num_simhash_projections: 3,
      projection_dimension: projection_dimensions,
      final_projection_dimension: min(max(dimensions, 32), 256),
      seed: 42
    ]

    %{
      "maxsim/direct" => fn ->
        MultiVector.colbert_score(query_vectors, document_vectors, metric: :inner_product)
      end,
      "muvera/query encoding" => fn ->
        Muvera.encode_query(query_vectors, muvera_options)
      end,
      "muvera/document encoding" => fn ->
        Muvera.encode_document(document_vectors, muvera_options)
      end
    }
  end

  defp preflight!(scenarios, reference_name) do
    reference = run_scenario!(Map.fetch!(scenarios, reference_name), reference_name)

    scenarios
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.each(fn {name, scenario} ->
      results = run_scenario!(scenario, name)
      IO.puts("#{String.pad_trailing(name, 36)} overlap@k=#{overlap(reference, results)}")
    end)
  end

  defp preflight_algorithms!(scenarios) do
    scenarios
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.each(fn {name, scenario} ->
      case scenario.() do
        {:ok, result} when is_list(result) ->
          IO.puts("#{String.pad_trailing(name, 36)} output_dimensions=#{length(result)}")

        {:ok, score} when is_number(score) ->
          IO.puts("#{String.pad_trailing(name, 36)} score=#{Float.round(score / 1, 6)}")

        other ->
          raise "#{name} preflight failed: #{inspect(other)}"
      end
    end)
  end

  defp run_scenario!(scenario, name) do
    case scenario.() do
      {:ok, results} when is_list(results) -> results
      other -> raise "#{name} preflight failed: #{inspect(other)}"
    end
  end

  defp overlap(reference, results) do
    reference_ids = MapSet.new(reference, & &1.id)
    result_ids = MapSet.new(results, & &1.id)
    denominator = max(MapSet.size(reference_ids), 1)

    reference_ids
    |> MapSet.intersection(result_ids)
    |> MapSet.size()
    |> Kernel./(denominator)
    |> Float.round(3)
  end

  defp random_vector(dimensions) do
    for _dimension <- 1..dimensions, do: :rand.uniform() * 2.0 - 1.0
  end

  defp perturb(vector, amount) do
    Enum.map(vector, &(&1 + (:rand.uniform() * 2.0 - 1.0) * amount))
  end

  defp metric_env! do
    metric =
      case System.get_env("VETTORE_BENCH_METRIC", "cosine") do
        "l2" -> :l2
        "cosine" -> :cosine
        "inner_product" -> :inner_product
        value -> raise ArgumentError, "unsupported VETTORE_BENCH_METRIC: #{inspect(value)}"
      end

    if metric in @hnsw_metrics,
      do: metric,
      else: raise(ArgumentError, "benchmark metric must be supported by HNSW")
  end

  defp stages_env!(dimensions) do
    case System.get_env("VETTORE_BENCH_STAGES") do
      nil ->
        [max(div(dimensions, 4), 1), max(div(dimensions, 2), 1), dimensions]
        |> Enum.uniq()

      value ->
        stages =
          value
          |> String.split(",", trim: true)
          |> Enum.map(&parse_positive!("VETTORE_BENCH_STAGES", &1))

        if stages != [] and Enum.all?(stages, &(&1 <= dimensions)),
          do: stages,
          else: raise(ArgumentError, "VETTORE_BENCH_STAGES must be within the vector size")
    end
  end

  defp positive_env!(name, default) do
    name
    |> System.get_env(Integer.to_string(default))
    |> then(&parse_positive!(name, &1))
  end

  defp non_negative_env!(name, default) do
    value = System.get_env(name, Integer.to_string(default))

    case Integer.parse(value) do
      {integer, ""} when integer >= 0 -> integer
      _other -> raise ArgumentError, "#{name} must be a non-negative integer"
    end
  end

  defp parse_positive!(name, value) do
    case Integer.parse(value) do
      {integer, ""} when integer > 0 -> integer
      _other -> raise ArgumentError, "#{name} must contain positive integers"
    end
  end
end

Vettore.SearchModesBench.run()
