defmodule Vettore.Interop.Nx do
  @moduledoc """
  Optional interoperability with `Nx.Tensor` values.

  Vettore does not use Nx for its vector kernels. This module deliberately
  resolves Nx at runtime so applications that only use lists or little-endian
  f32 binaries do not need to install or start Nx.

  Add `{:nx, "~> 0.11"}` to the host application's dependencies when tensor
  conversion is required.
  """

  @nx Module.concat(["Nx"])
  @nx_tensor Module.concat(["Nx", "Tensor"])

  @doc "Returns whether a compatible Nx module is available in the host application."
  @spec available?() :: boolean()
  def available? do
    Code.ensure_loaded?(@nx) and
      function_exported?(@nx, :tensor, 2) and
      function_exported?(@nx, :to_flat_list, 1)
  end

  @doc "Returns whether the value is an Nx tensor without requiring Nx at compile time."
  @spec tensor?(term()) :: boolean()
  def tensor?(%{__struct__: struct}), do: struct == @nx_tensor
  def tensor?(_value), do: false

  @doc "Returns an Nx tensor's original shape."
  @spec shape(term()) :: {:ok, tuple()} | {:error, :invalid_vector | :nx_not_available}
  def shape(tensor) do
    with :ok <- ensure_tensor(tensor) do
      # Dynamic dispatch is what keeps Nx out of Vettore's dependency graph.
      # credo:disable-for-next-line Credo.Check.Refactor.Apply
      {:ok, apply(@nx, :shape, [tensor])}
    end
  rescue
    _error -> {:error, :invalid_vector}
  end

  @doc "Returns an Nx tensor's element type."
  @spec type(term()) :: {:ok, term()} | {:error, :invalid_vector | :nx_not_available}
  def type(tensor) do
    with :ok <- ensure_tensor(tensor),
         true <- function_exported?(@nx, :type, 1) do
      # credo:disable-for-next-line Credo.Check.Refactor.Apply
      {:ok, apply(@nx, :type, [tensor])}
    else
      false -> {:error, :nx_not_available}
      {:error, reason} -> {:error, reason}
    end
  rescue
    _error -> {:error, :invalid_vector}
  end

  @doc "Flattens an Nx tensor into an ordinary Elixir list."
  @spec to_list(term()) :: {:ok, [number()]} | {:error, :invalid_vector | :nx_not_available}
  def to_list(tensor) do
    with :ok <- ensure_tensor(tensor) do
      # credo:disable-for-next-line Credo.Check.Refactor.Apply
      {:ok, apply(@nx, :to_flat_list, [tensor])}
    end
  rescue
    _error -> {:error, :invalid_vector}
  end

  @doc "Creates an f32 Nx tensor, optionally with a shape and host-provided backend."
  @spec from_list([number()], keyword()) ::
          {:ok, term()}
          | {:error, :invalid_options | :invalid_shape | :invalid_vector | :nx_not_available}
  def from_list(vector, opts \\ [])

  def from_list(vector, opts) when is_list(vector) and is_list(opts) do
    with :ok <- validate_options(opts),
         :ok <- validate_shape(Keyword.get(opts, :shape, {length(vector)}), length(vector)),
         true <- available?(),
         {:ok, tensor} <- create_tensor(vector, opts) do
      reshape_tensor(tensor, Keyword.get(opts, :shape))
    else
      false -> {:error, :nx_not_available}
      {:error, reason} -> {:error, reason}
    end
  rescue
    _error -> {:error, :invalid_vector}
  end

  def from_list(_vector, opts) when not is_list(opts), do: {:error, :invalid_options}
  def from_list(_vector, _opts), do: {:error, :invalid_vector}

  @doc "Transfers an Nx tensor to a host-provided backend such as EXLA."
  @spec transfer(term(), term()) :: {:ok, term()} | {:error, :invalid_vector | :nx_not_available}
  def transfer(tensor, backend) do
    with :ok <- ensure_tensor(tensor),
         true <- function_exported?(@nx, :backend_transfer, 2) do
      # credo:disable-for-next-line Credo.Check.Refactor.Apply
      {:ok, apply(@nx, :backend_transfer, [tensor, backend])}
    else
      false -> {:error, :nx_not_available}
      {:error, reason} -> {:error, reason}
    end
  rescue
    _error -> {:error, :invalid_vector}
  end

  defp ensure_tensor(tensor) do
    cond do
      not tensor?(tensor) -> {:error, :invalid_vector}
      not available?() -> {:error, :nx_not_available}
      true -> :ok
    end
  end

  defp create_tensor(vector, opts) do
    tensor_opts =
      [type: :f32]
      |> maybe_put_backend(Keyword.fetch(opts, :backend))

    # credo:disable-for-next-line Credo.Check.Refactor.Apply
    {:ok, apply(@nx, :tensor, [vector, tensor_opts])}
  end

  defp reshape_tensor(tensor, nil), do: {:ok, tensor}

  defp reshape_tensor(tensor, shape) do
    if function_exported?(@nx, :reshape, 2) do
      # credo:disable-for-next-line Credo.Check.Refactor.Apply
      {:ok, apply(@nx, :reshape, [tensor, shape])}
    else
      {:error, :nx_not_available}
    end
  end

  defp maybe_put_backend(opts, {:ok, backend}), do: Keyword.put(opts, :backend, backend)
  defp maybe_put_backend(opts, :error), do: opts

  defp validate_options(opts) do
    if Keyword.keyword?(opts) do
      keys = Keyword.keys(opts)

      if keys == Enum.uniq(keys) and Enum.all?(keys, &(&1 in [:backend, :shape])),
        do: :ok,
        else: {:error, :invalid_options}
    else
      {:error, :invalid_options}
    end
  end

  defp validate_shape(shape, dimensions) when is_tuple(shape) do
    entries = Tuple.to_list(shape)

    if Enum.all?(entries, &(is_integer(&1) and &1 >= 0)) and shape_size(entries) == dimensions,
      do: :ok,
      else: {:error, :invalid_shape}
  end

  defp validate_shape(_shape, _dimensions), do: {:error, :invalid_shape}

  defp shape_size([]), do: 1
  defp shape_size(entries), do: Enum.product(entries)
end
