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

  @doc "Creates an f32 Nx tensor from a flat vector."
  @spec from_list([number()]) :: {:ok, term()} | {:error, :invalid_vector | :nx_not_available}
  def from_list(vector) when is_list(vector) do
    if available?() do
      # credo:disable-for-next-line Credo.Check.Refactor.Apply
      {:ok, apply(@nx, :tensor, [vector, [type: :f32]])}
    else
      {:error, :nx_not_available}
    end
  rescue
    _error -> {:error, :invalid_vector}
  end

  def from_list(_vector), do: {:error, :invalid_vector}

  defp ensure_tensor(tensor) do
    cond do
      not tensor?(tensor) -> {:error, :invalid_vector}
      not available?() -> {:error, :nx_not_available}
      true -> :ok
    end
  end
end
