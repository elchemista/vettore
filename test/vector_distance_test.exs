defmodule VettoreDistanceTest do
  use ExUnit.Case, async: true
  alias Vettore.Distance

  @moduletag :distance

  describe "euclidean/2" do
    test "identical vectors → similarity = 1.0" do
      v = [1.0, -2.3, 4.5]
      assert {:ok, sim} = Distance.euclidean(v, v)
      assert_in_delta sim, 1.0, 1.0e-6
    end

    test "nonzero distance is mapped via 1/(1 + d)" do
      # ‖[3,4]‖ = 5 → sim = 1/(1+5) = 1/6
      assert {:ok, sim} = Distance.euclidean([0.0, 0.0], [3.0, 4.0])
      assert_in_delta sim, 1.0 / 6.0, 1.0e-6
    end

    test "dimension mismatch returns error" do
      assert {:error, _} = Distance.euclidean([1.0], [1.0, 2.0])
    end
  end

  describe "cosine/2" do
    test "identical vectors → similarity = 1.0" do
      v = [1.0, 2.0, 3.0]
      assert {:ok, sim} = Distance.cosine(v, v)
      assert_in_delta sim, 1.0, 1.0e-6
    end

    test "orthogonal vectors → similarity = 0.5" do
      # [1,0]⋅[0,1]=0 → (0+1)/2 = 0.5
      assert {:ok, sim} = Distance.cosine([1.0, 0.0], [0.0, 1.0])
      assert_in_delta sim, 0.5, 1.0e-6
    end

    test "dimension mismatch returns error" do
      assert {:error, _} = Distance.cosine([1.0], [1.0, 2.0])
    end
  end

  describe "dot_product/2" do
    test "correct raw dot product" do
      a = [1.0, 2.0, 3.0]
      b = [4.0, 5.0, 6.0]
      # 1*4 + 2*5 + 3*6 = 32
      assert {:ok, 32.0} = Distance.dot_product(a, b)
    end

    test "dimension mismatch returns error" do
      assert {:error, _} = Distance.dot_product([1.0], [1.0, 2.0])
    end
  end

  describe "compress_f32_vector/1 and hamming/2" do
    test "compress then identical vectors → hamming = 0" do
      v = [1.0, -2.0, 3.5, -4.5, 0.0]
      bits = Distance.compress_f32_vector(v)
      assert is_list(bits)
      assert {:ok, 0} = Distance.hamming(bits, bits)
    end

    test "different vectors → hamming > 0" do
      bits1 = Distance.compress_f32_vector([1.0, 2.0, 3.0, 4.0])
      bits2 = Distance.compress_f32_vector([-1.0, 2.0, -3.0, 4.0])
      assert {:ok, d} = Distance.hamming(bits1, bits2)
      assert d > 0
    end

    test "length mismatch returns error" do
      assert {:error, _} = Distance.hamming([0], [0, 1])
    end
  end

  describe "mmr_rerank/5" do
    setup do
      # simple 2D embeddings: a=(1,0), b=(0,1), c=(1,1)
      embeddings = [
        {"a", [1.0, 0.0]},
        {"b", [0.0, 1.0]},
        {"c", [1.0, 1.0]}
      ]

      # initial scores to some query
      initial = [
        {"a", 0.9},
        {"b", 0.8},
        {"c", 0.7}
      ]

      %{embeddings: embeddings, initial: initial}
    end

    test "alpha = 1.0 yields pure relevance order", %{initial: init, embeddings: emb} do
      assert {:ok, out} = Distance.mmr_rerank(init, emb, "dot", 1.0, 2)
      assert Enum.map(out, &elem(&1, 0)) == ["a", "b"]
    end

    test "alpha = 0.0 yields maximal diversity", %{initial: init, embeddings: emb} do
      assert {:ok, out} = Distance.mmr_rerank(init, emb, "dot", 0.0, 2)
      # after picking "a", the least‐similar to "a" is "b" (dot=0 vs c has dot=1)
      assert Enum.map(out, &elem(&1, 0)) == ["a", "b"]
    end

    test "final_k > candidates yields all", %{initial: init, embeddings: emb} do
      assert {:ok, out} = Distance.mmr_rerank(init, emb, "dot", 0.5, 5)
      assert length(out) == 3
    end

    test "invalid distance returns error", %{initial: init, embeddings: emb} do
      assert {:error, _} = Distance.mmr_rerank(init, emb, "unknown", 0.5, 2)
    end
  end
end
