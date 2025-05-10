defmodule VettoreDistanceTest do
  use ExUnit.Case, async: true
  import TestData
  alias Vettore.Distance
  alias Vettore.Embedding

  @moduletag :distance

  setup_all do
    {:ok, embeddings: load_embeddings()}
  end

  describe "euclidean/2" do
    test "identical vectors → similarity = 1.0", %{embeddings: [e | _]} do
      v = e.vector
      assert {:ok, sim} = Distance.euclidean(v, v)
      assert_in_delta sim, 1.0, 1.0e-6
    end

    # keeps the hand-made 3-4-5 example because we need a precise value
    test "non-zero distance is mapped via 1/(1 + d)" do
      # ‖[3,4]‖ = 5 → sim = 1/(1+5) = 1/6
      assert {:ok, sim} = Distance.euclidean([0.0, 0.0], [3.0, 4.0])
      assert_in_delta sim, 1.0 / 6.0, 1.0e-6
    end

    test "dimension mismatch returns error" do
      assert {:error, _} = Distance.euclidean([1.0], [1.0, 2.0])
    end
  end

  describe "cosine/2" do
    test "identical vectors → similarity = 1.0", %{embeddings: [e | _]} do
      v = e.vector
      assert {:ok, sim} = Distance.cosine(v, v)
      assert_in_delta sim, 1.0, 1.0e-6
    end

    # leave orthogonal unit test intact for clarity
    test "orthogonal vectors → similarity = 0.5" do
      assert {:ok, sim} = Distance.cosine([1.0, 0.0], [0.0, 1.0])
      assert_in_delta sim, 0.5, 1.0e-6
    end

    test "dimension mismatch returns error" do
      assert {:error, _} = Distance.cosine([1.0], [1.0, 2.0])
    end
  end

  describe "dot_product/2" do
    test "correct raw dot product using fixture vectors", %{embeddings: [e1, e2 | _]} do
      expected =
        Enum.zip(e1.vector, e2.vector)
        |> Enum.reduce(0.0, fn {a, b}, acc -> acc + a * b end)

      {:ok, dot} = Distance.dot_product(e1.vector, e2.vector)

      # tolerate ±1 e-4 (far larger than the 3 e-8 jitter we saw)
      assert_in_delta dot, expected, 1.0e-4
    end

    test "dimension mismatch returns error" do
      assert {:error, _} = Distance.dot_product([1.0], [1.0, 2.0])
    end
  end

  describe "compress_f32_vector/1 and hamming/2" do
    test "compress then identical vectors → hamming = 0", %{embeddings: [e | _]} do
      bits = Distance.compress_f32_vector(e.vector)
      assert {:ok, 0} = Distance.hamming(bits, bits)
    end

    # keep a deterministic small example for “> 0” check
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
    setup %{embeddings: embeds} do
      # pick the first three fixture embeddings
      [e1, e2, e3 | _] = embeds

      embeddings =
        Enum.map([e1, e2, e3], fn %Embedding{value: id, vector: vec} -> {id, vec} end)

      initial = [
        {elem(List.first(embeddings), 0), 0.9},
        {elem(Enum.at(embeddings, 1), 0), 0.8},
        {elem(Enum.at(embeddings, 2), 0), 0.7}
      ]

      %{embeddings: embeddings, initial: initial}
    end

    test "alpha = 1.0 yields pure relevance order", %{initial: init, embeddings: emb} do
      assert {:ok, out} = Distance.mmr_rerank(init, emb, "dot", 1.0, 2)
      assert Enum.map(out, &elem(&1, 0)) == Enum.map(init, &elem(&1, 0)) |> Enum.take(2)
    end

    test "alpha = 0.0 still returns 2 unique ids", %{initial: init, embeddings: emb} do
      assert {:ok, out} = Distance.mmr_rerank(init, emb, "dot", 0.0, 2)
      ids = Enum.map(out, &elem(&1, 0))
      assert length(ids) == 2
      assert ids == Enum.uniq(ids)
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
