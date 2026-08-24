defmodule VettoreVectorNativeFallbackTest do
  use ExUnit.Case, async: false

  alias Vettore.Vector

  setup do
    previous = Application.get_env(:vettore, :native_f32)
    Application.put_env(:vettore, :native_f32, false)

    on_exit(fn ->
      if is_nil(previous) do
        Application.delete_env(:vettore, :native_f32)
      else
        Application.put_env(:vettore, :native_f32, previous)
      end
    end)

    :ok
  end

  test "fallback decodes, normalizes, and scores f32 binaries" do
    left = f32_binary([3.0, 4.0])
    right = f32_binary([6.0, 8.0])

    assert {:ok, [3.0, 4.0]} = Vector.to_list(left)
    assert {:ok, normalized} = Vector.normalize(left, :l2, as: :list)
    assert_in_delta Enum.at(normalized, 0), 0.6, 1.0e-6
    assert_in_delta Enum.at(normalized, 1), 0.8, 1.0e-6
    assert {:ok, 50.0} = Vector.dot_product(left, right)
    assert {:ok, 1.0} = Vector.cosine(left, right)
    assert {:error, :dimension_mismatch} = Vector.l2(left, f32_binary([1.0]))
    assert {:error, :invalid_f32_binary} = Vector.to_list(<<1, 2, 3>>)
  end

  test "fallback mean-pools selected rows with the native contract" do
    matrix = f32_binary([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    assert {:ok, [3.0, 4.0]} = Vector.mean_pool_f32(matrix, 2, [0, 2], as: :list)
    assert {:ok, [3.0, 4.0]} = Vector.mean_pool_f32(matrix, 2, [1, 1], as: :list)
    assert {:error, :empty_selection} = Vector.mean_pool_f32(matrix, 2, [], as: :list)
    assert {:error, :matrix_shape_mismatch} = Vector.mean_pool_f32(<<>>, 2, [0])
    assert {:error, :matrix_shape_mismatch} = Vector.mean_pool_f32(<<1, 2, 3>>, 2, [0])
    assert {:error, :invalid_row_index} = Vector.mean_pool_f32(matrix, 2, [3])

    nan_matrix = f32_bits(0x7FC00000) <> f32_binary([1.0])
    assert {:error, :invalid_vector} = Vector.mean_pool_f32(nan_matrix, 2, [0], as: :list)
  end

  defp f32_binary(values) do
    for value <- values, into: <<>>, do: <<value::float-little-32>>
  end

  defp f32_bits(bits), do: <<bits::unsigned-little-32>>
end
