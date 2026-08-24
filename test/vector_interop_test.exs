defmodule VettoreVectorInteropTest do
  use ExUnit.Case, async: false

  alias Vettore.Interop.Nx, as: NxInterop
  alias Vettore.Vector

  describe "representation and validation" do
    test "recognizes lists, f32 binaries, wrappers, and unknown values" do
      binary = f32_binary([1.0, 2.0])
      assert Vector.representation([1, 2]) == :list
      assert Vector.representation(binary) == :f32_binary
      assert Vector.representation(:not_a_vector) == :unknown

      assert {:ok, wrapped} = Vector.new(binary)
      assert Vector.representation(wrapped) == :f32_binary
      assert wrapped.dimensions == 2
      assert Vector.valid?(wrapped)
      assert {:ok, 2} = Vector.dimensions(wrapped)
      assert {:ok, ^binary} = Vector.to_f32_binary(wrapped)
      assert {:ok, copied} = Vector.new(wrapped)
      assert copied == wrapped
    end

    test "a wrapper verifies representation and dimension metadata" do
      assert {:ok, wrapped} = Vector.new([1, 2, 3])
      refute Vector.valid?(%{wrapped | dimensions: 2})
      refute Vector.valid?(%{wrapped | representation: :f32_binary})
      assert {:error, :invalid_vector} = Vector.to_list(%{wrapped | dimensions: 99})
    end

    test "accepts the complete finite f32 domain and rejects other coordinates" do
      assert Vector.valid?([0, -1, 3.402_823_466_385_288_6e38])
      refute Vector.valid?([3.5e38])
      refute Vector.valid?([:bad])
      refute Vector.valid?(f32_bits(0x7FC00000))
      refute Vector.valid?(f32_bits(0x7F800000))
      refute Vector.valid?(<<1, 2, 3>>)
    end

    test "empty flat values have a well-defined zero-dimensional shape" do
      assert {:ok, 0} = Vector.dimensions([])
      assert {:ok, 0} = Vector.dimensions(<<>>)
      assert {:ok, {0}} = Vector.shape([])
      assert {:ok, []} = Vector.to_list(<<>>)
    end

    test "invalid option shapes and targets return tagged errors" do
      assert {:error, :invalid_options} = Vector.new([1.0], [:bad])
      assert {:error, :invalid_options} = Vector.new([1.0], :bad)
      assert {:error, :invalid_options} = Vector.new([1.0], as: :list, as: :list)
      assert {:error, {:unknown_representation, :array}} = Vector.convert([1.0], :array)
      assert {:error, :invalid_vector} = Vector.new(:bad)
    end
  end

  describe "conversion" do
    test "round-trips lists and canonical little-endian f32 binaries" do
      assert {:ok, binary} = Vector.to_f32_binary([1, -2.5, 3.25])
      assert binary == f32_binary([1.0, -2.5, 3.25])
      assert {:ok, [1.0, -2.5, 3.25]} = Vector.to_list(binary)
      assert {:ok, ^binary} = Vector.convert(binary, :same)
      assert {:ok, [1.0, -2.5, 3.25]} = Vector.convert(binary, :list)
    end

    test "new/2 can store a converted representation with dimensions" do
      assert {:ok, vector} = Vector.new([1, 2, 3], as: :f32_binary)
      assert vector.representation == :f32_binary
      assert vector.dimensions == 3
      assert vector.data == f32_binary([1.0, 2.0, 3.0])
      assert {:ok, {3}} = Vector.shape(vector)
    end

    test "wrappers retain explicit multidimensional shape metadata" do
      assert {:ok, vector} = Vector.new([1, 2, 3, 4], shape: {2, 2}, as: :f32_binary)
      assert vector.shape == {2, 2}
      assert {:ok, {2, 2}} = Vector.shape(vector)
      assert {:ok, reshaped} = Vector.reshape(vector, {1, 4})
      assert {:ok, {1, 4}} = Vector.shape(reshaped)
      assert {:error, :invalid_shape} = Vector.reshape(vector, {3, 2})
      assert {:error, :invalid_shape} = Vector.reshape(vector, :flat)
      assert {:error, :invalid_options} = Vector.reshape(vector, {4}, :bad)
      assert {:error, :invalid_shape} = Vector.new([1.0], shape: {-1})
      assert {:error, :invalid_shape} = Vector.new([], shape: {})

      refute Vector.valid?(%{vector | shape: {3, 3}})

      legacy_wrapper = %{vector | shape: nil}
      assert Vector.valid?(legacy_wrapper)
      assert {:ok, {4}} = Vector.shape(legacy_wrapper)
    end

    test "Nx conversion is explicitly unavailable without an Nx dependency" do
      refute NxInterop.available?()
      refute NxInterop.tensor?([1.0])
      assert {:error, :nx_not_available} = Vector.to_nx([1.0])

      fake_tensor = %{__struct__: Module.concat(["Nx", "Tensor"])}
      assert NxInterop.tensor?(fake_tensor)
      assert {:error, :nx_not_available} = Vector.to_list(fake_tensor)
      assert {:error, :nx_not_available} = NxInterop.shape(fake_tensor)
      assert {:error, :nx_not_available} = NxInterop.type(fake_tensor)
      assert {:error, :nx_not_available} = NxInterop.transfer(fake_tensor, :host)
      assert {:error, :invalid_options} = Vector.to_nx([1.0], :bad)
    end
  end

  describe "representation-independent metrics" do
    test "all metrics match the existing list API for binary inputs" do
      left = [1.0, 0.0, 2.0]
      right = [0.0, 3.0, 2.0]
      left_binary = f32_binary(left)
      right_binary = f32_binary(right)

      for metric <- [
            :l2,
            :l2_squared,
            :cosine,
            :inner_product,
            :negative_inner_product,
            :manhattan,
            :chebyshev,
            :hamming,
            :jaccard
          ] do
        assert {:ok, expected} = apply(Vettore.Distance, metric, [left, right])
        assert {:ok, actual} = Vector.metric(left_binary, right_binary, metric)
        assert_in_delta actual, expected, 1.0e-6
      end
    end

    test "mixed and wrapped representations interoperate" do
      assert {:ok, wrapped} = Vector.new([3.0, 4.0], as: :f32_binary)
      assert {:ok, 11.0} = Vector.dot_product(wrapped, [1.0, 2.0])
      assert {:ok, 5.0} = Vector.l2(wrapped, [0.0, 0.0])
      assert {:ok, 25.0} = Vector.l2_squared(wrapped, f32_binary([0.0, 0.0]))
      assert {:ok, -11.0} = Vector.negative_inner_product(wrapped, [1.0, 2.0])
      assert {:ok, 4.0} = Vector.manhattan(wrapped, [1.0, 2.0])
      assert {:ok, 2.0} = Vector.chebyshev(wrapped, [1.0, 2.0])
    end

    test "cosine supports true cosine, raw dot, and transformed inputs" do
      left = f32_binary([2.0, 0.0])
      right = f32_binary([4.0, 0.0])

      assert {:ok, 1.0} = Vector.cosine(left, right)
      assert {:ok, 8.0} = Vector.cosine(left, right, normalize: :none)

      assert {:ok, transformed} =
               Vector.cosine([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], normalize: :zscore)

      assert_in_delta transformed, -3.0, 1.0e-6

      assert {:error, {:unknown_normalization, :unit}} =
               Vector.cosine(left, right, normalize: :unit)

      assert {:error, :invalid_options} = Vector.cosine(left, right, normalize: :l2, extra: true)
      assert {:error, :invalid_options} = Vector.cosine(left, right, :bad)
    end

    test "metric errors remain tagged across binary and list boundaries" do
      assert {:error, :dimension_mismatch} =
               Vector.cosine(f32_binary([1.0]), f32_binary([1.0, 2.0]))

      assert {:error, :invalid_f32_binary} = Vector.l2(<<1, 2, 3>>, <<1, 2, 3>>)
      assert {:error, :invalid_vector} = Vector.l2([:bad], [1.0])
      assert {:error, {:unknown_metric, :angular}} = Vector.metric([1.0], [1.0], :angular)
    end

    test "binary hamming and jaccard retain non-zero semantics" do
      left = f32_binary([1.0, 0.0, -2.0])
      right = f32_binary([0.0, 0.0, 3.0])
      assert {:ok, 1.0} = Vector.hamming(left, right)
      assert {:ok, 0.5} = Vector.jaccard(left, right)
    end
  end

  describe "normalization" do
    test "preserves the input representation by default" do
      binary = f32_binary([3.0, 4.0])
      assert {:ok, normalized} = Vector.normalize(binary)
      assert is_binary(normalized)
      assert {:ok, values} = Vector.to_list(normalized)
      assert_in_delta Enum.at(values, 0), 0.6, 1.0e-6
      assert_in_delta Enum.at(values, 1), 0.8, 1.0e-6

      assert {:ok, normalized_list} = Vector.normalize([3.0, 4.0])
      assert_in_delta Enum.at(normalized_list, 0), 0.6, 1.0e-6
      assert_in_delta Enum.at(normalized_list, 1), 0.8, 1.0e-6
    end

    test "supports every existing normalization and explicit output formats" do
      binary = f32_binary([2.0, 4.0, 6.0])
      assert {:ok, [2.0, 4.0, 6.0]} = Vector.normalize(binary, :none, as: :list)
      assert {:ok, minmax} = Vector.normalize(binary, :minmax, as: :list)
      assert minmax == [0.0, 0.5, 1.0]
      assert {:ok, zscore} = Vector.normalize(binary, :zscore, as: :list)
      assert_in_delta Enum.sum(zscore), 0.0, 1.0e-6
      assert {:ok, normalized_binary} = Vector.normalize([3.0, 4.0], :l2, as: :f32_binary)
      assert is_binary(normalized_binary)
    end

    test "rejects unknown methods, representations, and malformed options" do
      assert {:error, {:unknown_normalization, :rank}} = Vector.normalize([1.0], :rank)

      assert {:error, {:unknown_representation, :array}} =
               Vector.normalize([1.0], :l2, as: :array)

      assert {:error, :invalid_options} = Vector.normalize([1.0], :l2, [:bad])
      assert {:error, :invalid_options} = Vector.normalize([1.0], :l2, :bad)
    end
  end

  describe "pooling" do
    test "mean_pool combines list, binary, and wrapped rows" do
      assert {:ok, wrapped} = Vector.new([7.0, 8.0, 9.0], as: :f32_binary)

      assert {:ok, [4.0, 5.0, 6.0]} =
               Vector.mean_pool([
                 [1.0, 2.0, 3.0],
                 f32_binary([4.0, 5.0, 6.0]),
                 wrapped
               ])

      assert {:ok, binary} = Vector.mean_pool([[1.0, 3.0], [3.0, 5.0]], as: :f32_binary)
      assert binary == f32_binary([2.0, 4.0])
    end

    test "mean_pool_f32 selects matrix rows and counts duplicate token ids" do
      matrix =
        f32_binary([
          1.0,
          2.0,
          3.0,
          4.0,
          5.0,
          6.0,
          7.0,
          8.0,
          9.0
        ])

      assert {:ok, pooled_binary} = Vector.mean_pool_f32(matrix, 3, [0, 2])
      assert pooled_binary == f32_binary([4.0, 5.0, 6.0])

      assert {:ok, [5.0, 6.0, 7.0]} = Vector.mean_pool_f32(matrix, 3, [1, 1, 2], as: :list)
    end

    test "pooling validates rows, matrix shape, indices, dimensions, and options" do
      matrix = f32_binary([1.0, 2.0, 3.0, 4.0])

      assert {:error, :empty_selection} = Vector.mean_pool([])
      assert {:error, :dimension_mismatch} = Vector.mean_pool([[1.0], [1.0, 2.0]])
      assert {:error, :dimension_mismatch} = Vector.mean_pool([[]])
      assert {:error, :invalid_vector} = Vector.mean_pool([[1.0], [:bad]])
      assert {:error, :invalid_options} = Vector.mean_pool([[1.0]], [:bad])
      assert {:error, :invalid_options} = Vector.mean_pool(:bad, :bad)
      assert {:error, :empty_selection} = Vector.mean_pool_f32(matrix, 2, [], as: :list)
      assert {:error, :invalid_dimensions} = Vector.mean_pool_f32(matrix, 0, [0])
      assert {:error, :invalid_dimensions} = Vector.mean_pool_f32(matrix, -1, [0])
      assert {:error, :invalid_row_index} = Vector.mean_pool_f32(matrix, 2, [-1])
      assert {:error, :invalid_row_index} = Vector.mean_pool_f32(matrix, 2, [2])
      assert {:error, :matrix_shape_mismatch} = Vector.mean_pool_f32(<<1, 2, 3>>, 2, [0])
      assert {:error, :invalid_options} = Vector.mean_pool_f32(matrix, 2, [0], extra: true)
      assert {:error, :invalid_arguments} = Vector.mean_pool_f32(:not_binary, 2, [0])
    end
  end

  describe "matrix interchange" do
    test "stack creates validated list and binary matrices" do
      assert {:ok, matrix} = Vector.stack([[1, 2], f32_binary([3.0, 4.0])])
      assert matrix == f32_binary([1.0, 2.0, 3.0, 4.0])
      assert {:ok, {2, 2}} = Vector.matrix_shape_f32(matrix, 2)
      assert {:ok, {2, 2}} = Vector.validate_matrix_f32(matrix, 2)
      assert Vector.valid_matrix_f32?(matrix, 2)

      assert {:ok, [[1.0, 2.0], [3.0, 4.0]]} =
               Vector.stack([[1, 2], [3, 4]], as: :list)

      assert {:error, :empty_selection} = Vector.stack([])
      assert {:error, :invalid_options} = Vector.stack([[1.0]], backend: :gpu)
      assert {:error, :invalid_options} = Vector.stack(:bad)

      assert {:error, {:unknown_representation, :matrix}} =
               Vector.stack([[1.0]], as: :matrix)
    end

    test "matrix validation separates structural shape from finite-value validation" do
      nan_matrix = f32_bits(0x7FC00000)
      assert {:ok, {1, 1}} = Vector.matrix_shape_f32(nan_matrix, 1)
      assert {:error, :invalid_vector} = Vector.validate_matrix_f32(nan_matrix, 1)
      refute Vector.valid_matrix_f32?(nan_matrix, 1)
      assert {:ok, {0, 3}} = Vector.matrix_shape_f32(<<>>, 3)
      assert {:error, :matrix_shape_mismatch} = Vector.matrix_shape_f32(<<1, 2, 3>>, 2)
      assert {:error, :invalid_dimensions} = Vector.matrix_shape_f32(<<>>, 0)
      assert {:error, :invalid_arguments} = Vector.matrix_shape_f32(:bad, 2)
    end

    test "take_rows_f32 preserves order, duplicates, and output shape" do
      matrix = f32_binary([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

      assert {:ok, selected} = Vector.take_rows_f32(matrix, 2, [2, 0, 2])
      assert selected == f32_binary([5.0, 6.0, 1.0, 2.0, 5.0, 6.0])
      assert {:ok, ^selected} = Vector.take_rows_f32(matrix, 2, [2, 0, 2], as: :same)

      assert {:ok, [[3.0, 4.0], [1.0, 2.0]]} =
               Vector.take_rows_f32(matrix, 2, [1, 0], as: :list)

      assert {:ok, <<>>} = Vector.take_rows_f32(matrix, 2, [])
      assert {:error, :invalid_row_index} = Vector.take_rows_f32(matrix, 2, [3])
      assert {:error, :invalid_row_index} = Vector.take_rows_f32(matrix, 2, [-1])
      assert {:error, :invalid_options} = Vector.take_rows_f32(matrix, 2, [0], backend: :gpu)

      assert {:error, {:unknown_representation, :array}} =
               Vector.take_rows_f32(matrix, 2, [0], as: :array)

      assert {:error, :invalid_arguments} = Vector.take_rows_f32(:bad, 2, [0])

      nan_matrix = matrix <> f32_bits(0x7FC00000) <> f32_binary([7.0])
      assert {:error, :invalid_vector} = Vector.take_rows_f32(nan_matrix, 2, [3], as: :list)
      assert {:error, :invalid_vector} = Vector.take_rows_f32(nan_matrix, 2, [3])
    end
  end

  describe "native fallback containment" do
    test "raised and thrown NIF failures use the supplied fallback" do
      previous = Application.get_env(:vettore, :native_f32, :missing)
      Application.put_env(:vettore, :native_f32, true)

      on_exit(fn ->
        if previous == :missing,
          do: Application.delete_env(:vettore, :native_f32),
          else: Application.put_env(:vettore, :native_f32, previous)
      end)

      assert :fallback == Vector.run_native(fn -> raise "native failure" end, fn -> :fallback end)
      assert :fallback == Vector.run_native(fn -> throw(:native_failure) end, fn -> :fallback end)
    end
  end

  defp f32_binary(values) do
    for value <- values, into: <<>>, do: <<value::float-little-32>>
  end

  defp f32_bits(bits), do: <<bits::unsigned-little-32>>
end
