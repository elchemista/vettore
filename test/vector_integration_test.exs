defmodule VettoreIntegrationTest do
  use ExUnit.Case, async: true

  alias Vettore.{Collection, Embedding, Encoding.Muvera, MultiVector, Result}

  test "multi-vector chamfer sums best per-query matches" do
    query = [[1.0, 0.0], [0.0, 1.0]]
    doc = [[1.0, 0.0], [1.0, 1.0]]

    assert {:ok, 2.0} = MultiVector.chamfer(query, doc, metric: :inner_product)
  end

  test "colbert_score is an explicit late-interaction alias" do
    query = [[1.0, 0.0], [0.0, 1.0]]
    doc = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]

    assert {:ok, 2.0} = MultiVector.colbert_score(query, doc, metric: :inner_product)
  end

  test "muvera query and document encodings are deterministic and asymmetric" do
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    config = [num_repetitions: 1, num_simhash_projections: 0, seed: 42]

    assert {:ok, query_fde} = Muvera.encode_query(vectors, config)
    assert {:ok, same_query_fde} = Muvera.encode_query(vectors, config)
    assert {:ok, doc_fde} = Muvera.encode_document(vectors, config)

    assert query_fde == same_query_fde
    assert length(query_fde) == 2
    assert length(doc_fde) == 2
    assert query_fde != doc_fde
  end

  test "muvera NIF is loaded and callable directly" do
    assert {:ok, fde} =
             Vettore.Nifs.muvera_encode_query(
               [[1.0, 0.0], [0.0, 1.0]],
               2,
               1,
               0,
               42,
               2,
               nil
             )

    assert fde == [1.0, 1.0]
  end

  test "muvera candidate retrieval recalls exact late-interaction fixtures" do
    query = [[1.0, 0.0], [0.0, 1.0]]

    docs = [
      {"both_axes", [[1.0, 0.0], [0.0, 1.0]]},
      {"x_axis", [[1.0, 0.0], [1.0, 0.0]]},
      {"opposite", [[-1.0, 0.0], [0.0, -1.0]]},
      {"weak", [[0.2, 0.0], [0.0, 0.2]]}
    ]

    exact_top =
      docs
      |> Enum.map(fn {id, vectors} ->
        {:ok, score} = MultiVector.colbert_score(query, vectors, metric: :inner_product)
        {id, score}
      end)
      |> Enum.sort_by(fn {id, score} -> {-score, id} end)
      |> Enum.take(2)
      |> Enum.map(&elem(&1, 0))
      |> MapSet.new()

    config = [
      num_repetitions: 4,
      num_simhash_projections: 1,
      projection_dimension: 2,
      seed: 13
    ]

    assert {:ok, query_fde} = Muvera.encode_query(query, config)

    assert {:ok, collection} =
             Collection.new(dimensions: length(query_fde), metric: :inner_product)

    embeddings =
      Enum.map(docs, fn {id, vectors} ->
        {:ok, vector} = Muvera.encode_document(vectors, config)
        %Embedding{id: id, vector: vector}
      end)

    assert :ok = Collection.put_many(collection, embeddings)

    assert {:ok, results} = Collection.search(collection, query_fde, limit: 3)

    candidate_ids =
      results
      |> Enum.map(fn %Result{id: id} -> id end)
      |> MapSet.new()

    assert MapSet.subset?(exact_top, candidate_ids)
  end
end
