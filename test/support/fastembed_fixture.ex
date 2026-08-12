defmodule Vettore.Test.FastembedFixture do
  @moduledoc false

  @format_version 1
  @model "BAAI/bge-small-en-v1.5"
  @artifact_path Path.expand("../fixtures/fastembed_bge_small_en_v1_5.etf", __DIR__)

  @documents [
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

  @queries [
    {"How does OTP restart failed Elixir workers?", :elixir},
    {"Which document talks about vector similarity search?", :vectors},
    {"Find text about a kitten or house cat.", :cats}
  ]

  @spec artifact_path() :: Path.t()
  def artifact_path, do: @artifact_path

  @spec load!() :: map()
  def load! do
    @artifact_path
    |> File.read!()
    |> :erlang.binary_to_term([:safe])
    |> validate_fixture!()
  end

  @spec generate!() :: map()
  def generate! do
    texts =
      Enum.map(@documents, fn {_id, text, _category} -> text end) ++
        Enum.map(@queries, fn {text, _category} -> text end)

    {:ok, dimensions} = ExFastembed.load(@model)
    {:ok, vectors} = ExFastembed.embed_text(texts)
    {document_vectors, query_vectors} = Enum.split(vectors, length(@documents))

    %{
      format_version: @format_version,
      model: @model,
      dimensions: dimensions,
      documents: build_documents(document_vectors),
      queries: build_queries(query_vectors)
    }
    |> validate_fixture!()
  end

  @spec write!() :: Path.t()
  def write! do
    fixture = generate!()
    File.mkdir_p!(Path.dirname(@artifact_path))
    File.write!(@artifact_path, :erlang.term_to_binary(fixture, compressed: 9))
    @artifact_path
  end

  defp build_documents(vectors) do
    Enum.zip_with(@documents, vectors, fn {id, text, category}, vector ->
      %{id: id, text: text, category: category, vector: vector}
    end)
  end

  defp build_queries(vectors) do
    Enum.zip_with(@queries, vectors, fn {text, category}, vector ->
      %{text: text, category: category, vector: vector}
    end)
  end

  defp validate_fixture!(
         %{
           format_version: @format_version,
           model: @model,
           dimensions: dimensions,
           documents: documents,
           queries: queries
         } = fixture
       )
       when is_integer(dimensions) and dimensions > 0 and is_list(documents) and
              is_list(queries) do
    expected_count = length(@documents) + length(@queries)

    if length(documents) == length(@documents) and length(queries) == length(@queries) and
         Enum.count(documents ++ queries, &valid_entry?(&1, dimensions)) == expected_count do
      fixture
    else
      raise "invalid FastEmbed fixture contents"
    end
  end

  defp validate_fixture!(_fixture), do: raise("invalid FastEmbed fixture schema")

  defp valid_entry?(%{vector: vector}, dimensions) when is_list(vector) do
    length(vector) == dimensions and Enum.all?(vector, &is_float/1)
  end

  defp valid_entry?(_entry, _dimensions), do: false
end
