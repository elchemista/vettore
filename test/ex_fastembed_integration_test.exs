Code.require_file("support/fastembed_fixture.ex", __DIR__)

defmodule VettoreExFastembedIntegrationTest do
  use ExUnit.Case, async: false

  alias Vettore.{Collection, Distance, Embedding, Result}
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
  test "fresh ex_fastembed inference matches the committed artifact", %{fixture: fixture} do
    generated = FastembedFixture.generate!()

    assert generated.model == fixture.model
    assert generated.dimensions == fixture.dimensions

    fixture_vectors = Enum.map(fixture.documents ++ fixture.queries, & &1.vector)
    generated_vectors = Enum.map(generated.documents ++ generated.queries, & &1.vector)

    for {expected, actual} <- Enum.zip(fixture_vectors, generated_vectors) do
      assert {:ok, similarity} = Distance.cosine(expected, actual)
      assert similarity > 0.999_9
    end
  end

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
