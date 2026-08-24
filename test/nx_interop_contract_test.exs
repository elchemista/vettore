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
    assert {:ok, :f32} = NxInterop.type(tensor)
    assert {:ok, 3} = Vector.dimensions(tensor)
    assert {:ok, [1.0, 2.5, -3.0]} = Vector.to_list(tensor)
    assert {:ok, binary} = Vector.to_f32_binary(tensor)
    assert binary == f32_binary([1.0, 2.5, -3.0])
    assert {:error, :invalid_vector} = NxInterop.from_list(:bad)
    assert {:error, :invalid_vector} = NxInterop.to_list(:bad)
    assert {:ok, direct} = NxInterop.from_list([1.0, 2.0])
    assert {:ok, {2}} = NxInterop.shape(direct)
  end

  test "shape, backend creation, transfer, and explicit conversion stay runtime-only" do
    backend = {Module.concat(["EXLA", "Backend"]), client: :cuda}

    assert {:ok, tensor} =
             Vector.to_nx([1.0, 2.0, 3.0, 4.0], shape: {2, 2}, backend: backend)

    assert {:ok, {2, 2}} = Vector.shape(tensor)
    assert tensor.backend == backend
    assert {:ok, transferred} = NxInterop.transfer(tensor, :host)
    assert transferred.backend == :host
    assert {:ok, [1.0, 2.0, 3.0, 4.0]} = Vector.from_nx(transferred, :list)
    assert {:ok, binary} = Vector.from_nx(transferred)
    assert binary == f32_binary([1.0, 2.0, 3.0, 4.0])

    assert {:ok, wrapped} = Vector.new(tensor, as: :f32_binary)
    assert wrapped.shape == {2, 2}
    assert {:ok, round_trip} = Vector.to_nx(wrapped)
    assert {:ok, {2, 2}} = Vector.shape(round_trip)

    assert {:ok, reshaped} = Vector.reshape(wrapped, {4, 1})
    assert {:ok, {4, 1}} = Vector.shape(reshaped)
    assert {:ok, reshaped_tensor} = Vector.to_nx(reshaped)
    assert {:ok, {4, 1}} = Vector.shape(reshaped_tensor)

    assert {:ok, nx_wrapper} = Vector.new(tensor)
    assert Vector.valid?(nx_wrapper)
    refute Vector.valid?(%{nx_wrapper | shape: {4}})

    assert {:ok, converted_wrapper} = Vector.new([1.0, 2.0], as: :nx)
    assert {:ok, {2}} = Vector.shape(converted_wrapper)

    assert {:ok, stacked} = Vector.stack([[1.0, 2.0], [3.0, 4.0]], as: :nx, backend: backend)
    assert {:ok, {2, 2}} = Vector.shape(stacked)
    assert stacked.backend == backend

    matrix = f32_binary([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert {:ok, selected} = Vector.take_rows_f32(matrix, 2, [2, 0], as: :nx)
    assert {:ok, {2, 2}} = Vector.shape(selected)
    assert {:ok, [5.0, 6.0, 1.0, 2.0]} = Vector.to_list(selected)
  end

  test "Nx option and shape validation contains malformed host requests" do
    assert {:error, :invalid_shape} = NxInterop.from_list([1.0, 2.0], shape: {3})
    assert {:error, :invalid_shape} = NxInterop.from_list([1.0], shape: {-1})
    assert {:error, :invalid_shape} = NxInterop.from_list([], shape: {})
    assert {:error, :invalid_shape} = NxInterop.from_list([1.0], shape: :flat)
    assert {:error, :invalid_options} = NxInterop.from_list([1.0], unknown: true)
    assert {:error, :invalid_options} = NxInterop.from_list([1.0], :bad)
    assert {:error, :invalid_vector} = NxInterop.transfer(:bad, :host)
    assert {:error, :invalid_vector} = NxInterop.type(:bad)
    assert {:error, :invalid_vector} = Vector.from_nx(:bad)
    assert {:error, {:unknown_representation, :array}} = Vector.from_nx(%{}, :array)
    assert {:error, :invalid_shape} = Vector.to_nx([1.0, 2.0], shape: {1, 1})
    assert {:error, :invalid_options} = Vector.to_nx([1.0], backend: :cpu, backend: :gpu)
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
    invalid_type = struct(type, data: [], shape: {0}, type: :raise)
    transfer_failure = struct(type, data: [], shape: {0}, type: :f32)

    assert {:error, :invalid_vector} = NxInterop.shape(invalid)
    assert {:error, :invalid_vector} = NxInterop.to_list(invalid)
    assert {:error, :invalid_vector} = NxInterop.from_list([:raise])
    assert {:error, :invalid_vector} = NxInterop.type(invalid_type)
    assert {:error, :invalid_vector} = NxInterop.transfer(transfer_failure, :raise)
  end

  # Dynamic host-contract clauses intentionally live together so the temporary
  # Nx module mirrors one complete external API surface.
  # credo:disable-for-next-line Credo.Check.Refactor.CyclomaticComplexity
  defp create_nx_stub do
    tensor_module = Module.concat(["Nx", "Tensor"])
    nx_module = Module.concat(["Nx"])

    Module.create(
      tensor_module,
      quote do
        defstruct [:data, :shape, :type, :backend]
      end,
      Macro.Env.location(__ENV__)
    )

    Module.create(
      nx_module,
      quote do
        def tensor([:raise], _opts), do: raise("host tensor failure")

        def tensor(values, opts) do
          struct(unquote(tensor_module),
            data: values,
            shape: {length(values)},
            type: Keyword.fetch!(opts, :type),
            backend: Keyword.get(opts, :backend, :default)
          )
        end

        def reshape(%{__struct__: unquote(tensor_module)} = tensor, shape),
          do: %{tensor | shape: shape}

        def to_flat_list(%{__struct__: unquote(tensor_module), data: :raise}),
          do: raise("host flatten failure")

        def to_flat_list(%{__struct__: unquote(tensor_module), data: values}), do: values

        def shape(%{__struct__: unquote(tensor_module), shape: :raise}),
          do: raise("host shape failure")

        def shape(%{__struct__: unquote(tensor_module), shape: shape}), do: shape

        def type(%{__struct__: unquote(tensor_module), type: :raise}),
          do: raise("host type failure")

        def type(%{__struct__: unquote(tensor_module), type: type}), do: type

        def backend_transfer(%{__struct__: unquote(tensor_module)}, :raise),
          do: raise("host transfer failure")

        def backend_transfer(%{__struct__: unquote(tensor_module)} = tensor, backend),
          do: %{tensor | backend: backend}
      end,
      Macro.Env.location(__ENV__)
    )
  end

  defp f32_binary(values) do
    for value <- values, into: <<>>, do: <<value::float-little-32>>
  end
end
