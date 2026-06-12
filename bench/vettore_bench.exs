alias Vettore.{Collection, Embedding}

dimensions = String.to_integer(System.get_env("VETTORE_BENCH_DIMENSIONS", "384"))
batch_size = String.to_integer(System.get_env("VETTORE_BENCH_BATCH", "1000"))
metrics = [:l2, :cosine, :inner_product]

random_vector = fn ->
  for _ <- 1..dimensions, do: :rand.uniform() * 2.0 - 1.0
end

collections =
  Map.new(metrics, fn metric ->
    {:ok, collection} =
      Collection.new(
        name: :"#{metric}_bench",
        dimensions: dimensions,
        metric: metric,
        normalize: if(metric == :cosine, do: :l2, else: :none),
        score: :raw
      )

    embeddings =
      for i <- 1..batch_size do
        %Embedding{id: "#{metric}-#{i}", vector: random_vector.()}
      end

    :ok = Collection.put_many(collection, embeddings)
    {metric, collection}
  end)

query = random_vector.()

benchmarks =
  Map.new(collections, fn {metric, collection} ->
    {"search_#{metric}_#{dimensions}d_#{batch_size}", fn ->
      Collection.search(collection, query, limit: 10)
    end}
  end)

Benchee.run(benchmarks, time: 3, warmup: 2)
