defmodule VettoreExFastembedIntegrationTest do
  use ExUnit.Case, async: false

  @compile {:no_warn_undefined, ExFastembed}
  @moduletag skip: System.get_env("VETTORE_TEST_EX_FASTEMBED") != "1"

  alias Vettore.{Collection, Embedding, Result}

  @model "BAAI/bge-small-en-v1.5"

  @phrases [
    {"cat_1", "A small kitten is sleeping on a warm blanket.", :cats},
    {"cat_2", "The cat is chasing a toy mouse across the floor.", :cats},
    {"cat_3", "A tabby cat curls up beside the sunny window.", :cats},
    {"cat_4", "The kitten drinks milk from a shallow bowl.", :cats},
    {"cat_5", "A quiet house cat watches birds from the sofa.", :cats},
    {"cat_6", "The feline stretches and purrs after a nap.", :cats},
    {"dog_1", "A golden retriever runs after a tennis ball.", :dogs},
    {"dog_2", "The puppy learns to sit and stay in the park.", :dogs},
    {"dog_3", "A loyal dog guards the front porch at night.", :dogs},
    {"dog_4", "The border collie herds sheep across the field.", :dogs},
    {"dog_5", "A wet dog shakes water after swimming in the lake.", :dogs},
    {"dog_6", "The hound follows a scent along the forest trail.", :dogs},
    {"elixir_1", "Elixir processes exchange messages on the BEAM VM.", :elixir},
    {"elixir_2", "Phoenix renders a LiveView page without custom JavaScript.", :elixir},
    {"elixir_3", "Pattern matching makes Elixir function clauses expressive.", :elixir},
    {"elixir_4", "Supervisors restart crashed workers in an OTP application.", :elixir},
    {"elixir_5", "Mix compiles dependencies and runs the test suite.", :elixir},
    {"elixir_6", "ETS tables can store fast in-memory state for Elixir systems.", :elixir},
    {"vector_1", "Approximate nearest neighbor search retrieves similar embeddings.", :vectors},
    {"vector_2", "Cosine similarity compares the direction of two dense vectors.", :vectors},
    {"vector_3", "A vector database indexes embeddings for semantic retrieval.", :vectors},
    {"vector_4", "HNSW graphs trade exact recall for lower search latency.", :vectors},
    {"vector_5", "Binary quantization compresses vectors for cheaper candidate search.",
     :vectors},
    {"vector_6", "Reranking exact vectors improves the final search results.", :vectors},
    {"food_1", "Fresh pasta is tossed with basil pesto and olive oil.", :food},
    {"food_2", "The baker pulls a loaf of sourdough from the oven.", :food},
    {"food_3", "A spicy curry simmers with coconut milk and vegetables.", :food},
    {"food_4", "The chef slices ripe tomatoes for a summer salad.", :food},
    {"food_5", "Dark chocolate melts into a rich dessert sauce.", :food},
    {"food_6", "A bowl of soup warms the table on a cold evening.", :food}
  ]

  test "Vettore searches real ex_fastembed vectors from a small phrase corpus" do
    texts = Enum.map(@phrases, fn {_id, text, _category} -> text end)

    assert {:ok, dimensions} = ExFastembed.load(@model)
    assert {:ok, vectors} = ExFastembed.embed_text(texts)
    assert length(vectors) == length(@phrases)

    assert {:ok, collection} =
             Collection.new(
               name: :real_fastembed_phrases,
               dimensions: dimensions,
               metric: :cosine,
               normalize: :l2
             )

    assert {:ok, hnsw_collection} =
             Collection.new(
               name: :real_fastembed_hnsw,
               dimensions: dimensions,
               metric: :cosine,
               normalize: :l2,
               index: :hnsw,
               index_options: [
                 m: 8,
                 m0: 16,
                 ef_construction: 200,
                 ef_search: 200,
                 max_level: 12
               ]
             )

    on_exit(fn ->
      Vettore.close(collection)
      Vettore.close(hnsw_collection)
    end)

    embeddings =
      @phrases
      |> Enum.zip(vectors)
      |> Enum.map(fn {{id, text, category}, vector} ->
        %Embedding{id: id, value: text, vector: vector, metadata: %{category: category}}
      end)

    assert :ok = Collection.put_many(collection, embeddings)
    assert :ok = Collection.put_many(hnsw_collection, embeddings)

    assert_category_search(
      collection,
      hnsw_collection,
      dimensions,
      "How does OTP restart failed Elixir workers?",
      :elixir
    )

    assert_category_search(
      collection,
      hnsw_collection,
      dimensions,
      "Which document talks about vector similarity search?",
      :vectors
    )

    assert_category_search(
      collection,
      hnsw_collection,
      dimensions,
      "Find text about a kitten or house cat.",
      :cats
    )
  end

  defp assert_category_search(collection, hnsw_collection, dimensions, query, expected_category) do
    assert {:ok, [query_vector]} = ExFastembed.embed_text([query])

    assert {:ok, exact_results} = Collection.search(collection, query_vector, limit: 5)
    assert [%Result{} = exact_top | _] = exact_results

    assert_new_search_matches_exact_top(collection, dimensions, query_vector, exact_top)
    assert_hnsw_matches_exact_top(hnsw_collection, query_vector, exact_top)

    top_categories = Enum.map(exact_results, & &1.metadata.category)

    assert expected_category in Enum.take(top_categories, 3)
  end

  defp assert_hnsw_matches_exact_top(collection, query_vector, exact_top) do
    exact_top_id = exact_top.id

    assert {:ok,
            [
              %Result{
                id: ^exact_top_id,
                value: value,
                metadata: %{category: category}
              }
              | _
            ]} = Collection.search(collection, query_vector, limit: 5)

    assert is_binary(value)
    assert is_atom(category)

    assert {:ok, [%Result{id: ^exact_top_id} | _]} =
             Collection.hybrid_search(collection, query_vector,
               generators: [
                 hnsw: [candidates: length(@phrases)],
                 quantized: [candidates: length(@phrases)]
               ],
               limit: 5
             )
  end

  defp assert_new_search_matches_exact_top(collection, dimensions, query_vector, exact_top) do
    candidates = length(@phrases)
    exact_top_id = exact_top.id

    assert {:ok, [%Result{id: ^exact_top_id} | _]} =
             Collection.funnel_search(collection, query_vector,
               stages: [min(128, dimensions), dimensions],
               candidates: candidates,
               limit: 5
             )

    assert {:ok, [%Result{id: ^exact_top_id} | _]} =
             Collection.quantized_search(collection, query_vector,
               candidates: candidates,
               limit: 5
             )

    assert {:ok, [%Result{id: ^exact_top_id} | _]} =
             Collection.multi_vector_search(collection, [query_vector],
               metric: :cosine,
               limit: 5
             )

    assert {:ok, [%Result{id: ^exact_top_id} | _]} =
             Collection.hybrid_search(collection, query_vector,
               generators: [
                 funnel: [stages: [min(128, dimensions), dimensions], candidates: candidates],
                 quantized: [candidates: candidates]
               ],
               rerank: :exact,
               limit: 5
             )
  end
end
