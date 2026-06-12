defmodule VettoreDistanceTest do
  use ExUnit.Case, async: true

  alias Vettore.Distance

  describe "compute/4" do
    test "l2 returns raw distance" do
      assert {:ok, zero} = Distance.compute(:l2, [1.0, 2.0], [1.0, 2.0])
      assert zero == 0.0
      assert {:ok, five} = Distance.compute(:l2, [0.0, 0.0], [3.0, 4.0])
      assert five == 5.0
    end

    test "l2_squared returns squared distance" do
      assert {:ok, 25.0} = Distance.compute(:l2_squared, [0.0, 0.0], [3.0, 4.0])
    end

    test "cosine returns raw cosine similarity" do
      assert {:ok, same} = Distance.compute(:cosine, [1.0, 0.0], [1.0, 0.0], normalize: :l2)
      assert {:ok, opposite} = Distance.compute(:cosine, [1.0, 0.0], [-1.0, 0.0], normalize: :l2)
      assert {:ok, orthogonal} = Distance.compute(:cosine, [1.0, 0.0], [0.0, 1.0], normalize: :l2)
      assert same == 1.0
      assert opposite == -1.0
      assert orthogonal == 0.0
    end

    test "inner product and negative inner product" do
      assert {:ok, 32.0} = Distance.compute(:inner_product, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])

      assert {:ok, -32.0} =
               Distance.compute(:negative_inner_product, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    end

    test "additional distance metrics" do
      assert {:ok, 7.0} = Distance.compute(:manhattan, [1.0, 2.0], [4.0, 6.0])
      assert {:ok, 4.0} = Distance.compute(:chebyshev, [1.0, 2.0], [4.0, 6.0])
      assert {:ok, 2} = Distance.compute(:hamming, [1, 0, 1], [0, 0, 0])
      assert {:ok, jaccard} = Distance.compute(:jaccard, [1, 0, 1], [0, 1, 1])
      assert_in_delta jaccard, 2 / 3, 1.0e-6
    end

    test "dimension mismatch returns error" do
      assert {:error, :dimension_mismatch} = Distance.compute(:l2, [1.0], [1.0, 2.0])
    end
  end

  describe "normalize/2" do
    test "l2 normalization handles normal and zero vectors" do
      assert {:ok, [0.6, 0.8]} = Distance.normalize([3.0, 4.0], :l2)
      assert {:ok, zeroes} = Distance.normalize([0.0, 0.0], :l2)
      assert zeroes == [0.0, 0.0]
    end

    test "zscore and minmax normalization" do
      assert {:ok, zscores} = Distance.normalize([1.0, 2.0, 3.0], :zscore)
      assert_in_delta Enum.at(zscores, 0), -1.224744871391589, 1.0e-6
      assert Enum.at(zscores, 1) == 0.0
      assert_in_delta Enum.at(zscores, 2), 1.224744871391589, 1.0e-6

      assert {:ok, minmax} = Distance.normalize([2.0, 4.0, 6.0], :minmax)
      assert minmax == [0.0, 0.5, 1.0]
    end
  end

  describe "compat helpers" do
    test "old named helpers use vNext raw semantics" do
      assert {:ok, 5.0} = Distance.euclidean([0.0, 0.0], [3.0, 4.0])
      assert {:ok, 1.0} = Distance.cosine([1.0, 0.0], [1.0, 0.0])
      assert {:ok, 32.0} = Distance.dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    end

    test "compression and hamming" do
      bits1 = Distance.compress_f32_vector([1.0, -2.0, 0.0])
      bits2 = Distance.compress_f32_vector([-1.0, -2.0, 0.0])
      assert bits1 == [1, 0, 1]
      assert {:ok, 1} = Distance.hamming(bits1, bits2)
    end

    test "mmr rejects unknown metrics" do
      assert {:error, _} = Distance.mmr_rerank([{"a", 0.9}], [{"a", [1.0]}], "unknown", 0.5, 1)
    end
  end
end
