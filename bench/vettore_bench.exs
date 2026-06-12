alias Vettore.{Collection, Distance, Embedding}

dimensions = String.to_integer(System.get_env("VETTORE_BENCH_DIMENSIONS", "384"))
batch_size = String.to_integer(System.get_env("VETTORE_BENCH_BATCH", "1000"))
time = String.to_integer(System.get_env("VETTORE_BENCH_TIME", "3"))
warmup = String.to_integer(System.get_env("VETTORE_BENCH_WARMUP", "2"))

metrics =
  System.get_env("VETTORE_BENCH_METRICS", "l2,cosine,inner_product")
  |> String.split(",", trim: true)
  |> Enum.map(fn
    "l2" -> :l2
    "l2_squared" -> :l2_squared
    "cosine" -> :cosine
    "inner_product" -> :inner_product
    "negative_inner_product" -> :negative_inner_product
    "manhattan" -> :manhattan
    "chebyshev" -> :chebyshev
    "hamming" -> :hamming
    "jaccard" -> :jaccard
    metric -> raise ArgumentError, "unsupported VETTORE_BENCH_METRICS entry: #{metric}"
  end)

indexes =
  System.get_env("VETTORE_BENCH_INDEXES", "flat,hnsw")
  |> String.split(",", trim: true)
  |> Enum.map(fn
    "flat" -> :flat
    "hnsw" -> :hnsw
    index -> raise ArgumentError, "unsupported VETTORE_BENCH_INDEXES entry: #{index}"
  end)

random_vector = fn ->
  for _ <- 1..dimensions, do: :rand.uniform() * 2.0 - 1.0
end

normalize_for = fn
  :cosine -> :l2
  _metric -> :none
end

hnsw_supported? = fn
  :flat, _metric -> true
  :hnsw, metric -> metric in [:l2, :cosine, :inner_product]
end

query = random_vector.()
candidate = random_vector.()

distance_benchmarks = %{
  "distance_l2_#{dimensions}d" => fn ->
    Distance.l2(query, candidate)
  end,
  "distance_l2_squared_#{dimensions}d" => fn ->
    Distance.l2_squared(query, candidate)
  end,
  "distance_cosine_#{dimensions}d" => fn ->
    Distance.cosine(query, candidate)
  end,
  "distance_cosine_no_normalize_#{dimensions}d" => fn ->
    Distance.cosine(query, candidate, normalize: :none)
  end,
  "distance_inner_product_#{dimensions}d" => fn ->
    Distance.inner_product(query, candidate)
  end,
  "distance_negative_inner_product_#{dimensions}d" => fn ->
    Distance.negative_inner_product(query, candidate)
  end,
  "distance_manhattan_#{dimensions}d" => fn ->
    Distance.manhattan(query, candidate)
  end,
  "distance_chebyshev_#{dimensions}d" => fn ->
    Distance.chebyshev(query, candidate)
  end,
  "distance_hamming_#{dimensions}d" => fn ->
    Distance.hamming(query, candidate)
  end,
  "distance_jaccard_#{dimensions}d" => fn ->
    Distance.jaccard(query, candidate)
  end
}

collection_benchmarks =
  for metric <- metrics,
      index <- indexes,
      hnsw_supported?.(index, metric),
      into: %{} do
    {:ok, collection} =
      Collection.new(
        name: :"#{index}_#{metric}_bench",
        dimensions: dimensions,
        index: index,
        metric: metric,
        normalize: normalize_for.(metric),
        score: :raw
      )

    embeddings =
      for i <- 1..batch_size do
        %Embedding{id: "#{index}-#{metric}-#{i}", vector: random_vector.()}
      end

    :ok = Collection.put_many(collection, embeddings)

    {"collection_#{index}_#{metric}_#{dimensions}d_#{batch_size}",
     fn ->
       Collection.search(collection, query, limit: 10)
     end}
  end

Benchee.run(
  Map.merge(distance_benchmarks, collection_benchmarks),
  time: time,
  warmup: warmup
)
