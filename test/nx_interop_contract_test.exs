defmodule VettoreNxInteropContractTest do
  use ExUnit.Case, async: false

  alias Vettore.Interop.Nx, as: NxInterop
  alias Vettore.Vector

  setup_all do
    created_stub? = not NxInterop.available?()

    if created_stub? do
      create_nx_stub()

      on_exit(fn ->
        :code.purge(Module.concat(["Nx"]))
        :code.delete(Module.concat(["Nx"]))
        :code.purge(Module.concat(["Nx", "Tensor"]))
        :code.delete(Module.concat(["Nx", "Tensor"]))
      end)
    end

    :ok
  end

  test "runtime adapter round-trips tensors without a Vettore Nx dependency" do
    assert NxInterop.available?()
    assert {:ok, tensor} = Vector.to_nx([1, 2.5, -3])
    assert NxInterop.tensor?(tensor)
    assert {:ok, {3}} = Vector.shape(tensor)
    assert {:ok, 3} = Vector.dimensions(tensor)
    assert {:ok, [1.0, 2.5, -3.0]} = Vector.to_list(tensor)
    assert {:ok, binary} = Vector.to_f32_binary(tensor)
    assert binary == f32_binary([1.0, 2.5, -3.0])
    assert {:error, :invalid_vector} = NxInterop.from_list(:bad)
    assert {:error, :invalid_vector} = NxInterop.to_list(:bad)
  end

  test "normalization and metrics can return and consume the host tensor type" do
    assert {:ok, tensor} = Vector.to_nx([3.0, 4.0])
    assert {:ok, normalized} = Vector.normalize(tensor)
    assert NxInterop.tensor?(normalized)
    assert {:ok, values} = Vector.to_list(normalized)
    assert_in_delta Enum.at(values, 0), 0.6, 1.0e-6
    assert_in_delta Enum.at(values, 1), 0.8, 1.0e-6
    assert {:ok, 1.0} = Vector.cosine(tensor, f32_binary([6.0, 8.0]))

    matrix = f32_binary([1.0, 2.0, 3.0, 4.0])
    assert {:ok, pooled} = Vector.mean_pool_f32(matrix, 2, [0, 1], as: :nx)
    assert {:ok, [2.0, 3.0]} = Vector.to_list(pooled)
  end

  test "runtime adapter contains host conversion failures" do
    type = Module.concat(["Nx", "Tensor"])
    invalid = struct(type, data: :raise, shape: :raise)

    assert {:error, :invalid_vector} = NxInterop.shape(invalid)
    assert {:error, :invalid_vector} = NxInterop.to_list(invalid)
    assert {:error, :invalid_vector} = NxInterop.from_list([:raise])
  end

  defp create_nx_stub do
    tensor_module = Module.concat(["Nx", "Tensor"])
    nx_module = Module.concat(["Nx"])

    Module.create(
      tensor_module,
      quote do
        defstruct [:data, :shape]
      end,
      Macro.Env.location(__ENV__)
    )

    Module.create(
      nx_module,
      quote do
        def tensor([:raise], _opts), do: raise("host tensor failure")

        def tensor(values, _opts) do
          struct(unquote(tensor_module), data: values, shape: {length(values)})
        end

        def to_flat_list(%{__struct__: unquote(tensor_module), data: :raise}),
          do: raise("host flatten failure")

        def to_flat_list(%{__struct__: unquote(tensor_module), data: values}), do: values

        def shape(%{__struct__: unquote(tensor_module), shape: :raise}),
          do: raise("host shape failure")

        def shape(%{__struct__: unquote(tensor_module), shape: shape}), do: shape
      end,
      Macro.Env.location(__ENV__)
    )
  end

  defp f32_binary(values) do
    for value <- values, into: <<>>, do: <<value::float-little-32>>
  end
end
