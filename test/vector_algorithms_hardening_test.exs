defmodule VettoreAlgorithmsHardeningTest do
  use ExUnit.Case, async: true

  alias Vettore.{Collection, Distance, Embedding, Encoding.Muvera, MultiVector, Nifs, Result}
  alias Vettore.Index.{Flat, HNSW}

  @metrics [
    :l2,
    :l2_squared,
    :cosine,
    :inner_product,
    :negative_inner_product,
    :manhattan,
    :chebyshev,
    :hamming,
    :jaccard
  ]

  describe "flat metric boundaries" do
    test "all supported metrics return stable top-k results" do
      for metric <- @metrics do
        assert {:ok, collection} = Collection.new(dimensions: 2, metric: metric, index: :flat)

        assert :ok =
                 Collection.put_many(collection, [
                   %{id: "b", vector: [0.0, 1.0]},
                   %{id: "a", vector: [1.0, 0.0]},
                   %{id: "c", vector: [1.0, 0.0]}
                 ])

        assert {:ok, [%Result{id: "a"}, %Result{id: "c"}]} =
                 Collection.search(collection, [1.0, 0.0], limit: 2)

        assert :ok = Vettore.close(collection)
      end
    end

    test "flat construction and search reject invalid options" do
      assert {:ok, default_index} = Flat.new(:l2)
      assert is_reference(default_index)
      assert {:error, :invalid_flat_options} = Flat.new(:l2, unknown: true)
      assert {:error, :invalid_flat_options} = Flat.new(:l2, :bad)
      assert {:error, {:unsupported_flat_metric, :unknown}} = Flat.new(:unknown, [])

      assert {:ok, collection} = Collection.new(dimensions: 1, metric: :l2)

      assert {:error, :invalid_search_options} =
               Flat.search(collection, [0.0], unknown: true)

      assert {:error, :invalid_search_options} =
               Flat.search(collection, [0.0], :bad)

      assert {:error, "vector must not be empty"} =
               Flat.put(collection, %Embedding{id: "bad", vector: []})

      assert {:ok, {}} = Nifs.flat_insert(collection.index_state, "phantom", [0.0])
      assert {:ok, []} = Collection.search(collection, [0.0], limit: 1)

      assert :ok = Vettore.close(collection)
    end
  end

  describe "HNSW boundary errors" do
    test "defaults, invalid direct options, and stale native ids are safe" do
      assert HNSW.defaults()[:m] == 16
      assert {:ok, index} = HNSW.new(:l2)
      assert is_reference(index)
      assert {:error, :invalid_hnsw_options} = HNSW.new(:l2, :bad)

      assert {:error, {:unsupported_hnsw_metric, :jaccard}} =
               HNSW.new(:jaccard, [])

      assert {:ok, collection} = Collection.new(dimensions: 1, metric: :l2, index: :hnsw)

      assert {:error, :invalid_search_options} =
               HNSW.search(collection, [0.0], :bad)

      assert {:error, :invalid_limit} = HNSW.search(collection, [0.0], limit: 0)

      assert {:error, "vector must not be empty"} =
               HNSW.put(collection, %Embedding{id: "bad", vector: []})

      assert {:ok, {}} = Nifs.hnsw_insert(collection.index_state, "phantom", [0.0])
      assert {:ok, []} = Collection.search(collection, [0.0], limit: 1)
      assert :ok = Vettore.close(collection)
    end
  end

  describe "batched native helpers" do
    test "vector, binary, and multi-vector top-k helpers validate and order ties" do
      vectors = [{"b", [1.0, 0.0]}, {"a", [1.0, 0.0]}, {"c", [0.0, 1.0]}]

      for metric_code <- 0..8 do
        assert {:ok, [{"a", _raw_a}, {"b", _raw_b}]} =
                 Nifs.vector_top_k(vectors, [1.0, 0.0], metric_code, 2, 2)
      end

      assert {:error, "unknown metric"} = Nifs.vector_top_k(vectors, [1.0, 0.0], 9, 2, 2)

      assert {:error, "invalid prefix dimensions"} =
               Nifs.vector_top_k(vectors, [1.0, 0.0], 0, 0, 2)

      assert {:ok, [{"a", exact_distance}, {"b", 1.0}]} =
               Nifs.binary_top_k([{"b", [1]}, {"a", [3]}], [3], 2, 2)

      assert exact_distance == 0.0

      documents = [
        {"b", [[1.0, 0.0]]},
        {"a", [[1.0, 0.0]]},
        {"c", [[-1.0, 0.0]]}
      ]

      assert {:ok, [{"a", 1.0}, {"b", 1.0}]} =
               Nifs.multi_vector_top_k(documents, [[1.0, 0.0]], 3, 2)

      assert {:ok, 1.0} = Nifs.multi_vector_score([[1.0, 0.0]], [[1.0, 0.0]], 3)

      assert {:error, "unknown metric"} =
               Nifs.multi_vector_score([[1.0]], [[1.0]], 99)
    end
  end

  describe "multi-vector contracts" do
    test "all metrics and aliases produce finite MaxSim scores" do
      query = [[1.0, 0.0], [0.0, 1.0]]
      document = [[1.0, 0.0], [0.0, 1.0]]

      for metric <- @metrics ++ [:dot, :dot_product, :euclidean] do
        assert {:ok, score} = MultiVector.chamfer(query, document, metric: metric)
        assert is_float(score)
        assert score == 2.0
      end
    end

    test "invalid shapes, metrics, values, and options return errors" do
      assert {:error, :dimension_mismatch} =
               MultiVector.chamfer([[1.0, 0.0]], [[1.0]], metric: :inner_product)

      assert {:error, :invalid_multi_vector} =
               MultiVector.chamfer([[1.0, :bad]], [[1.0, 0.0]])

      assert {:error, :invalid_multi_vector} = MultiVector.chamfer(:bad, [])
      assert {:error, :invalid_options} = MultiVector.chamfer([], [], [:bad])
      assert {:error, :invalid_options} = MultiVector.chamfer([], [], unknown: true)

      assert {:error, {:unknown_metric, :unknown}} =
               MultiVector.chamfer([[1.0]], [[1.0]], metric: :unknown)

      assert {:ok, empty_score} = MultiVector.colbert_score([], [], metric: :inner_product)
      assert empty_score == 0.0
    end
  end

  describe "distance edge cases" do
    test "normalization handles extreme f32 values and rejects invalid input" do
      max = 3.402_823_466_385_288_6e38
      assert {:ok, [x, y]} = Distance.normalize([max, max], :l2)
      assert_in_delta x, :math.sqrt(0.5), 1.0e-6
      assert_in_delta y, :math.sqrt(0.5), 1.0e-6
      assert {:ok, [minimum, maximum]} = Distance.normalize([-max, max], :minmax)
      assert minimum == 0.0
      assert maximum == 1.0
      assert {:ok, cosine} = Distance.cosine([max, max], [max, max])
      assert_in_delta cosine, 1.0, 1.0e-6
      assert {:error, :invalid_vector} = Distance.normalize(:bad, :l2)
      assert {:error, :invalid_vector} = Distance.normalize([:bad], :none)
      assert {:error, :metric_overflow} = Distance.inner_product([max], [max])
    end

    test "cosine and packed-vector options are validated" do
      assert {:error, :invalid_options} = Distance.cosine([1.0], [1.0], :bad)
      assert {:error, :invalid_options} = Distance.cosine([1.0], [1.0], unknown: true)
      assert {:error, :invalid_vector} = Distance.cosine([1.0, :bad], [1.0, 0.0])
      assert {:error, :invalid_vector} = Distance.packed_hamming([-1], [0], 1)
      assert {:error, :invalid_vector} = Distance.packed_hamming([], [], 0)
      assert {:error, :invalid_vector} = Distance.packed_jaccard([1], [], 1)
      assert {:error, :invalid_vector} = Distance.packed_jaccard(:bad, [], 1)
    end

    test "result semantics cover similarity, distance, and fallback modes" do
      assert Distance.result_values(:inner_product, 2.0, :similarity) == {2.0, -2.0}
      assert Distance.result_values(:cosine, -1.0, :similarity) == {0.0, 2.0}
      assert Distance.result_values(:jaccard, 1.0, :similarity) == {0.5, 1.0}
      assert Distance.result_values(:unknown, 3.0, :unknown) == {3.0, nil}
    end

    test "MMR exercises every metric and rejects malformed or missing records" do
      initial = [{"a", 0.9}, {"b", 0.8}]
      embeddings = [{"a", [1.0, 0.0]}, {"b", [0.0, 1.0]}]

      for metric <- @metrics do
        assert {:ok, [{"a", 0.9}, {"b", 0.8}]} =
                 Distance.mmr_rerank(initial, embeddings, metric, 0.5, 2)
      end

      assert {:error, :invalid_mmr_args} =
               Distance.mmr_rerank([{"missing", 1.0}], embeddings, :cosine, 0.5, 1)

      assert {:error, :invalid_mmr_args} =
               Distance.mmr_rerank(initial, [{"a", [1.0]}, {"b", [1.0, 2.0]}], :l2, 0.5, 1)

      assert {:error, :invalid_mmr_args} =
               Distance.mmr_rerank([{"a", 1.0}, {"a", 0.5}], embeddings, :l2, 0.5, 1)
    end
  end

  describe "MUVERA validation" do
    test "rejects malformed, unsupported, and unsafe configurations" do
      vectors = [[1.0, 0.0]]

      for {config, reason} <- [
            {[:bad], :invalid_config},
            {[unknown: 1], :invalid_config},
            {[seed: 1, seed: 2], :invalid_config},
            {[dimension: :bad], :invalid_dimension},
            {[dimension: 3], :dimension_mismatch},
            {[num_repetitions: 0], :invalid_repetitions},
            {[num_simhash_projections: :bad], :invalid_simhash_projections},
            {[num_simhash_projections: -1], :invalid_simhash_projections},
            {[num_simhash_projections: 31], :invalid_simhash_projections},
            {[seed: -1], :invalid_seed},
            {[seed: 18_446_744_073_709_551_616], :invalid_seed},
            {[projection_dimension: 0], :invalid_projection_dimension},
            {[final_projection_dimension: 0], :invalid_final_projection_dimension},
            {[num_repetitions: 16_777_217], :encoding_too_large},
            {[final_projection_dimension: 16_777_217], :encoding_too_large}
          ] do
        assert {:error, ^reason} = Muvera.encode_query(vectors, config)
      end

      assert {:error, :empty_vectors} = Muvera.encode_query([])
      assert {:error, :invalid_vectors} = Muvera.encode_query([1.0])
      assert {:error, :dimension_mismatch} = Muvera.encode_query([[1.0], [1.0, 2.0]])
      assert {:error, :invalid_vectors} = Muvera.encode_query([[1.0, :bad]])
      assert {:error, :invalid_vectors} = Muvera.encode_document(:bad)
      assert {:ok, [1.0]} = Muvera.encode_query([[1]])
    end

    test "supports projection and final count-sketch dimensions" do
      assert {:ok, query} =
               Muvera.encode_query([[1.0, 0.0], [0.0, 1.0]],
                 num_repetitions: 2,
                 num_simhash_projections: 1,
                 projection_dimension: 3,
                 final_projection_dimension: 5,
                 seed: 7
               )

      assert {:ok, document} =
               Muvera.encode_document([[1.0, 0.0], [0.0, 1.0]],
                 num_repetitions: 2,
                 num_simhash_projections: 1,
                 projection_dimension: 3,
                 final_projection_dimension: 5,
                 seed: 7
               )

      assert length(query) == 5
      assert length(document) == 5
      assert Enum.all?(query, &is_float/1)
      assert Enum.all?(document, &is_float/1)
    end

    test "reports query accumulation overflow without corrupting document averages" do
      max = 3.402_823_466_385_288_6e38

      assert {:error, :encoding_overflow} = Muvera.encode_query([[max], [max]])
      assert {:ok, [average]} = Muvera.encode_document([[max], [max]])
      assert average == max
    end
  end
end
