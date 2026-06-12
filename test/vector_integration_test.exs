defmodule VettoreIntegrationTest do
  use ExUnit.Case, async: true

  alias Vettore.{Encoding.Muvera, MultiVector, Postgres}

  test "postgres helpers expose pgvector operators and normalization" do
    assert Postgres.operator(:l2) == "<->"
    assert Postgres.operator(:inner_product) == "<#>"
    assert Postgres.operator(:cosine) == "<=>"
    assert {:ok, [0.6, 0.8]} = Postgres.normalize([3.0, 4.0], :l2)
  end

  test "multi-vector chamfer sums best per-query matches" do
    query = [[1.0, 0.0], [0.0, 1.0]]
    doc = [[1.0, 0.0], [1.0, 1.0]]

    assert {:ok, 2.0} = MultiVector.chamfer(query, doc, metric: :inner_product)
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
end
