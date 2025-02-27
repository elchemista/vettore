alias Vettore.Embedding

db = Vettore.new_db()

distances = ["euclidean", "cosine", "dot", "hnsw", "binary"]

collections =
  Enum.map(distances, fn dist ->
    coll_name = "#{dist}_bench_coll"
    {:ok, name} = Vettore.create_collection(db, coll_name, 3, dist)
    name
  end)
# 'collections' now holds each "my_dist_bench_coll"

batch_size = 1000
embeddings_batch =
  for i <- 1..batch_size do
    # define a random 3D vector
    %Embedding{
      id: "emb#{i}",
      vector: [
        Enum.random(0..10) * 1.0,
        Enum.random(0..10) * 1.0,
        Enum.random(0..10) * 1.0
      ],
      metadata: nil
    }
  end

single_embedding = %Embedding{id: "test_one", vector: [1.0, 2.0, 3.0], metadata: %{"tag" => "one"}}

query = [5.0, 5.0, 5.0]
top_k = 10

benchmarks =
  # For each distance-based collection, we define up to four tasks:
  #  - single_insert_<dist>
  #  - batch_insert_<dist>
  #  - similarity_search_<dist>
  Enum.reduce(collections, %{}, fn coll_name, acc ->
    dist = coll_name |> String.replace("_bench_coll", "")

    # single insert
    single_key = "single_insert_#{dist}"
    single_fun = fn ->
      Vettore.insert_embedding(db, coll_name, single_embedding)
    end

    # batch insert
    batch_key = "batch_insert_#{batch_size}_#{dist}"
    batch_fun = fn ->
      # Insert 1000 embeddings in one go
      Vettore.insert_embeddings(db, coll_name, embeddings_batch)
    end

    # similarity search
    search_key = "similarity_search_#{dist}"
    search_fun = fn ->
      # We'll search top_k with the 'query'
      Vettore.similarity_search(db, coll_name, query, limit: top_k)
    end

    acc
    |> Map.put(single_key, single_fun)
    |> Map.put(batch_key, batch_fun)
    |> Map.put(search_key, search_fun)
  end)

Benchee.run(benchmarks, time: 3, warmup: 2)
