defmodule VettoreIntegrationTest do
  use ExUnit.Case, async: true

  alias Vettore.{Encoding.Muvera, MultiVector}

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
end
