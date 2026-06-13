defmodule VettoreDistanceTest do
  use ExUnit.Case, async: true

  alias Vettore.Distance

  describe "named distance functions" do
    test "l2 returns raw euclidean distance" do
      assert {:ok, zero} = Distance.l2([1.0, 2.0], [1.0, 2.0])
      assert zero == 0.0

      assert {:ok, five} = Distance.l2([0.0, 0.0], [3.0, 4.0])
      assert five == 5.0
    end

    test "l2_squared returns squared euclidean distance" do
      assert {:ok, zero} = Distance.l2_squared([1.0, 2.0], [1.0, 2.0])
      assert zero == 0.0

      assert {:ok, 25.0} = Distance.l2_squared([0.0, 0.0], [3.0, 4.0])
    end

    test "cosine normalizes by default" do
      assert {:ok, same} = Distance.cosine([2.0, 0.0], [4.0, 0.0])
      assert {:ok, opposite} = Distance.cosine([1.0, 0.0], [-1.0, 0.0])
      assert {:ok, orthogonal} = Distance.cosine([1.0, 0.0], [0.0, 1.0])

      assert same == 1.0
      assert opposite == -1.0
      assert orthogonal == 0.0
    end

    test "cosine can skip normalization for pre-normalized collection vectors" do
      assert {:ok, raw_dot} = Distance.cosine([2.0, 0.0], [4.0, 0.0], normalize: :none)
      assert raw_dot == 8.0
    end

    test "inner product functions return raw dot semantics" do
      assert {:ok, 32.0} = Distance.inner_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
      assert {:ok, -32.0} = Distance.negative_inner_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    end

    test "manhattan and chebyshev return raw distances" do
      assert {:ok, 7.0} = Distance.manhattan([1.0, 2.0], [4.0, 6.0])
      assert {:ok, 4.0} = Distance.chebyshev([1.0, 2.0], [4.0, 6.0])
    end

    test "hamming and jaccard treat non-zero coordinates as true" do
      assert {:ok, 2.0} = Distance.hamming([1, 0, 1], [0, 0, 0])

      assert {:ok, jaccard} = Distance.jaccard([1, 0, 1], [0, 1, 1])
      assert_in_delta jaccard, 2 / 3, 1.0e-6
    end

    test "compatibility aliases call named native functions" do
      assert {:ok, 5.0} = Distance.euclidean([0.0, 0.0], [3.0, 4.0])
      assert {:ok, 32.0} = Distance.dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    end
  end

  describe "named distance errors" do
    test "dimension mismatch returns an explicit error" do
      assert {:error, :dimension_mismatch} = Distance.l2([1.0], [1.0, 2.0])
      assert {:error, :dimension_mismatch} = Distance.cosine([1.0], [1.0, 2.0])
      assert {:error, :dimension_mismatch} = Distance.inner_product([1.0], [1.0, 2.0])
    end

    test "invalid vectors return an explicit error" do
      assert {:error, :invalid_vector} = Distance.l2([1.0, :bad], [1.0, 2.0])
      assert {:error, :invalid_vector} = Distance.manhattan([1.0], :bad)
    end
  end

  describe "normalize/2" do
    test "none converts numbers to floats without changing scale" do
      assert {:ok, [1.0, 2.0, 3.5]} = Distance.normalize([1, 2, 3.5], :none)
    end

    test "l2 normalization handles normal and zero vectors" do
      assert {:ok, normalized} = Distance.normalize([3.0, 4.0], :l2)
      assert_in_delta Enum.at(normalized, 0), 0.6, 1.0e-6
      assert_in_delta Enum.at(normalized, 1), 0.8, 1.0e-6

      assert {:ok, zeroes} = Distance.normalize([0.0, 0.0], :l2)
      assert zeroes == [0.0, 0.0]
    end

    test "zscore normalization handles normal and constant vectors" do
      assert {:ok, zscores} = Distance.normalize([1.0, 2.0, 3.0], :zscore)
      assert_in_delta Enum.at(zscores, 0), -1.224744871391589, 1.0e-6
      assert Enum.at(zscores, 1) == 0.0
      assert_in_delta Enum.at(zscores, 2), 1.224744871391589, 1.0e-6

      assert {:ok, constant} = Distance.normalize([4.0, 4.0], :zscore)
      assert constant == [0.0, 0.0]
    end

    test "minmax normalization handles normal and constant vectors" do
      assert {:ok, minmax} = Distance.normalize([2.0, 4.0, 6.0], :minmax)
      assert minmax == [0.0, 0.5, 1.0]

      assert {:ok, constant} = Distance.normalize([7.0, 7.0], :minmax)
      assert constant == [0.0, 0.0]
    end

    test "unknown normalization returns explicit error" do
      assert {:error, {:unknown_normalization, :rank}} = Distance.normalize([1.0], :rank)
    end
  end

  describe "score semantics" do
    test "result_values keeps score and distance explicit" do
      assert Distance.result_values(:l2, 5.0, :raw) == {-5.0, 5.0}
      assert Distance.result_values(:l2, 5.0, :similarity) == {1.0 / 6.0, 5.0}
      assert Distance.result_values(:cosine, 0.25, :raw) == {0.25, 0.75}
      assert Distance.result_values(:cosine, 0.25, :similarity) == {0.625, 0.75}
      assert Distance.result_values(:inner_product, 3.0, :raw) == {3.0, -3.0}
    end
  end

  describe "compression and reranking" do
    test "compress_f32_vector packs signs into 64-bit words" do
      assert Distance.compress_f32_vector([1.0, -2.0, 0.0]) == [5]
    end

    test "packed hamming and jaccard work with compressed sign bits" do
      bits1 = Distance.compress_f32_vector([1.0, -2.0, 0.0])
      bits2 = Distance.compress_f32_vector([-1.0, -2.0, 0.0])
      bits3 = Distance.compress_f32_vector([1.0, 2.0, -1.0])

      assert {:ok, 1.0} = Distance.packed_hamming(bits1, bits2, 3)

      assert {:ok, jaccard} = Distance.packed_jaccard(bits1, bits3, 3)
      assert_in_delta jaccard, 2 / 3, 1.0e-6
    end

    test "mmr reranks with named metric dispatch" do
      initial = [{"a", 0.9}, {"b", 0.8}, {"c", 0.1}]
      embeddings = [{"a", [1.0, 0.0]}, {"b", [1.0, 0.0]}, {"c", [0.0, 1.0]}]

      assert {:ok, [{"a", 0.9}, {"c", 0.1}]} =
               Distance.mmr_rerank(initial, embeddings, :cosine, 0.5, 2)
    end

    test "mmr rejects unknown metrics and invalid arguments" do
      assert {:error, {:unknown_metric, "unknown"}} =
               Distance.mmr_rerank([{"a", 0.9}], [{"a", [1.0]}], "unknown", 0.5, 1)

      assert {:error, :invalid_mmr_args} =
               Distance.mmr_rerank([{"a", 0.9}], [{"a", [1.0]}], :l2, 2.0, 1)
    end
  end
end
