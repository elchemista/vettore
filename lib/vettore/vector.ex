defmodule Vettore.Vector do
  @moduledoc """
  Representation-independent dense vector operations.

  The same API accepts ordinary numeric lists, row-major little-endian f32
  binaries, `%Vettore.Vector{}` wrappers, and—when the host application has Nx
  installed—`Nx.Tensor` values. Native kernels operate directly on f32
  binaries, while Nx remains an optional interchange format rather than a
  runtime requirement.

  Little-endian f32 binaries are suitable for persistence and model matrices:

      iex> {:ok, binary} = Vettore.Vector.to_f32_binary([3.0, 4.0])
      iex> Vettore.Vector.dimensions(binary)
      {:ok, 2}
      iex> {:ok, normalized} = Vettore.Vector.normalize(binary, :l2, as: :list)
      iex> Enum.map(normalized, &Float.round(&1, 1))
      [0.6, 0.8]

  The wrapper struct is useful at application boundaries where representation
  and dimensions should travel together:

      iex> {:ok, vector} = Vettore.Vector.new([1, 2, 3], as: :f32_binary)
      iex> {vector.representation, vector.dimensions}
      {:f32_binary, 3}
      iex> Vettore.Vector.to_list(vector)
      {:ok, [1.0, 2.0, 3.0]}
  """

  alias Vettore.{Distance, Nifs}
  alias Vettore.Interop.Nx, as: NxInterop

  @enforce_keys [:data, :dimensions, :representation]
  defstruct [:data, :dimensions, :representation]

  @type representation :: :list | :f32_binary | :nx
  @type target_representation :: representation() | :same
  @type raw_vector :: [number()] | binary() | term()
  @type t :: %__MODULE__{
          data: raw_vector(),
          dimensions: non_neg_integer(),
          representation: representation()
        }
  @type vector :: raw_vector() | t()
  @type normalization :: :none | :l2 | :zscore | :minmax
  @type metric :: Distance.metric()

  @f32_max 3.402_823_466_385_288_6e38
  @usize_max 18_446_744_073_709_551_615
  @metric_codes %{
    l2: 0,
    l2_squared: 1,
    cosine: 2,
    inner_product: 3,
    negative_inner_product: 4,
    manhattan: 5,
    chebyshev: 6,
    hamming: 7,
    jaccard: 8
  }
  @normalization_codes %{none: 0, l2: 1, zscore: 2, minmax: 3}
  @native_errors %{
    "dimension mismatch" => :dimension_mismatch,
    "empty row selection" => :empty_selection,
    "invalid dimensions" => :invalid_dimensions,
    "invalid f32 binary" => :invalid_f32_binary,
    "invalid row index" => :invalid_row_index,
    "matrix shape mismatch" => :matrix_shape_mismatch,
    "metric overflow" => :metric_overflow,
    "vector contains a non-finite value" => :invalid_vector
  }

  @doc "Builds a dimensioned vector wrapper in the requested representation."
  @spec new(vector(), keyword()) :: {:ok, t()} | {:error, term()}
  def new(vector, opts \\ [])

  def new(vector, opts) when is_list(opts) do
    with :ok <- validate_options(opts, [:as]),
         target <- Keyword.get(opts, :as, :same),
         {:ok, target} <- resolve_explicit_target(target, vector),
         {:ok, data} <- convert(vector, target),
         {:ok, dimensions} <- dimensions(data) do
      {:ok, %__MODULE__{data: data, dimensions: dimensions, representation: target}}
    end
  end

  def new(_vector, _opts), do: {:error, :invalid_options}

  @doc "Returns the storage representation used by a supported vector value."
  @spec representation(term()) :: representation() | :unknown
  def representation(%__MODULE__{representation: representation}), do: representation
  def representation(vector) when is_list(vector), do: :list
  def representation(vector) when is_binary(vector), do: :f32_binary

  def representation(vector) do
    if NxInterop.tensor?(vector), do: :nx, else: :unknown
  end

  @doc "Checks the vector representation, finite f32 values, and wrapper metadata."
  @spec valid?(term()) :: boolean()
  def valid?(vector), do: match?({:ok, _vector}, to_list(vector))

  @doc "Returns the flattened coordinate count."
  @spec dimensions(vector()) :: {:ok, non_neg_integer()} | {:error, term()}
  def dimensions(%__MODULE__{} = vector) do
    with :ok <- validate_wrapper(vector), do: {:ok, vector.dimensions}
  end

  def dimensions(vector) when is_list(vector) do
    with {:ok, vector} <- validate_list(vector), do: {:ok, length(vector)}
  end

  def dimensions(vector) when is_binary(vector) do
    with {:ok, vector} <- decode_binary(vector), do: {:ok, length(vector)}
  end

  def dimensions(vector) do
    with {:ok, vector} <- NxInterop.to_list(vector),
         {:ok, vector} <- validate_list(vector) do
      {:ok, length(vector)}
    end
  end

  @doc "Returns the original tensor shape, or a one-dimensional shape for flat values."
  @spec shape(vector()) :: {:ok, tuple()} | {:error, term()}
  def shape(%__MODULE__{} = vector) do
    with :ok <- validate_wrapper(vector) do
      shape(vector.data)
    end
  end

  def shape(vector) when is_list(vector) or is_binary(vector) do
    with {:ok, dimensions} <- dimensions(vector), do: {:ok, {dimensions}}
  end

  def shape(vector), do: NxInterop.shape(vector)

  @doc "Converts any supported vector to a validated flat float list."
  @spec to_list(vector()) :: {:ok, [float()]} | {:error, term()}
  def to_list(%__MODULE__{} = vector) do
    with :ok <- validate_wrapper(vector), do: to_list(vector.data)
  end

  def to_list(vector) when is_list(vector), do: validate_list(vector)
  def to_list(vector) when is_binary(vector), do: decode_binary(vector)

  def to_list(vector) do
    with {:ok, vector} <- NxInterop.to_list(vector), do: validate_list(vector)
  end

  @doc "Converts any supported vector to a validated little-endian f32 binary."
  @spec to_f32_binary(vector()) :: {:ok, binary()} | {:error, term()}
  def to_f32_binary(%__MODULE__{} = vector) do
    with :ok <- validate_wrapper(vector), do: to_f32_binary(vector.data)
  end

  def to_f32_binary(vector) when is_binary(vector) do
    with {:ok, _vector} <- decode_binary(vector), do: {:ok, vector}
  end

  def to_f32_binary(vector) do
    with {:ok, vector} <- to_list(vector) do
      {:ok, encode_binary(vector)}
    end
  end

  @doc "Converts a vector to an Nx tensor when Nx is installed by the host application."
  @spec to_nx(vector()) :: {:ok, term()} | {:error, term()}
  def to_nx(vector) do
    with {:ok, vector} <- to_list(vector), do: NxInterop.from_list(vector)
  end

  @doc "Converts between list, f32-binary, and optional Nx representations."
  @spec convert(vector(), target_representation()) :: {:ok, raw_vector()} | {:error, term()}
  def convert(vector, :same) do
    with {:ok, target} <- resolve_explicit_target(:same, vector), do: convert(vector, target)
  end

  def convert(vector, :list), do: to_list(vector)
  def convert(vector, :f32_binary), do: to_f32_binary(vector)
  def convert(vector, :nx), do: to_nx(vector)
  def convert(_vector, target), do: {:error, {:unknown_representation, target}}

  @doc "Normalizes a vector while allowing the result representation to be selected."
  @spec normalize(vector(), normalization(), keyword()) :: {:ok, raw_vector()} | {:error, term()}
  def normalize(vector, method \\ :l2, opts \\ [])

  def normalize(vector, method, opts) when is_list(opts) do
    with :ok <- validate_options(opts, [:as]),
         {:ok, target} <- output_target(vector, opts),
         {:ok, normalized} <- normalize_to_list(vector, method) do
      convert(normalized, target)
    end
  end

  def normalize(_vector, _method, _opts), do: {:error, :invalid_options}

  @doc "Computes one named metric over any pair of supported representations."
  @spec metric(vector(), vector(), metric()) :: {:ok, float()} | {:error, term()}
  def metric(left, right, metric) when is_map_key(@metric_codes, metric) do
    case {unwrap(left), unwrap(right)} do
      {{:ok, left}, {:ok, right}} when is_binary(left) and is_binary(right) ->
        binary_metric(left, right, metric)

      _other ->
        with {:ok, left} <- to_list(left),
             {:ok, right} <- to_list(right) do
          apply(Distance, metric, [left, right])
        end
    end
  end

  def metric(_left, _right, metric), do: {:error, {:unknown_metric, metric}}

  @doc "L2/Euclidean distance over any supported representation."
  @spec l2(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def l2(left, right), do: metric(left, right, :l2)

  @doc "Squared L2 distance over any supported representation."
  @spec l2_squared(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def l2_squared(left, right), do: metric(left, right, :l2_squared)

  @doc "Inner/dot product over any supported representation."
  @spec inner_product(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def inner_product(left, right), do: metric(left, right, :inner_product)

  @doc "Compatibility alias for `inner_product/2`."
  @spec dot_product(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def dot_product(left, right), do: inner_product(left, right)

  @doc "Negative inner product over any supported representation."
  @spec negative_inner_product(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def negative_inner_product(left, right), do: metric(left, right, :negative_inner_product)

  @doc "Manhattan/L1 distance over any supported representation."
  @spec manhattan(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def manhattan(left, right), do: metric(left, right, :manhattan)

  @doc "Chebyshev/L-infinity distance over any supported representation."
  @spec chebyshev(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def chebyshev(left, right), do: metric(left, right, :chebyshev)

  @doc "Hamming distance using Vettore's non-zero coordinate semantics."
  @spec hamming(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def hamming(left, right), do: metric(left, right, :hamming)

  @doc "Jaccard distance using Vettore's non-zero coordinate semantics."
  @spec jaccard(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def jaccard(left, right), do: metric(left, right, :jaccard)

  @doc "Cosine similarity, with the same normalization options as `Vettore.Distance.cosine/3`."
  @spec cosine(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def cosine(left, right, opts \\ [])

  def cosine(left, right, opts) when is_list(opts) do
    with :ok <- validate_options(opts, [:normalize]) do
      cosine_with_normalization(left, right, Keyword.get(opts, :normalize, :l2))
    end
  end

  def cosine(_left, _right, _opts), do: {:error, :invalid_options}

  @doc "Mean-pools a non-empty list of equally sized vectors."
  @spec mean_pool([vector()], keyword()) :: {:ok, raw_vector()} | {:error, term()}
  def mean_pool(vectors, opts \\ [])

  def mean_pool(vectors, opts) when is_list(vectors) and is_list(opts) do
    with :ok <- validate_options(opts, [:as]),
         {:ok, rows, dimensions} <- validate_rows(vectors),
         matrix = rows |> List.flatten() |> encode_binary(),
         target = Keyword.get(opts, :as, :list),
         {:ok, target} <- resolve_explicit_target(target, rows),
         {:ok, pooled} <-
           native_mean_pool(matrix, dimensions, Enum.to_list(0..(length(rows) - 1))) do
      convert(pooled, target)
    end
  end

  def mean_pool(_vectors, _opts), do: {:error, :invalid_options}

  @doc """
  Mean-pools selected rows from a row-major little-endian f32 matrix.

  The result defaults to `:f32_binary`; pass `as: :list` or `as: :nx` when a
  different interchange representation is needed. Repeated indices are
  counted repeatedly, matching token-sequence pooling semantics.
  """
  @spec mean_pool_f32(binary(), pos_integer(), [non_neg_integer()], keyword()) ::
          {:ok, raw_vector()} | {:error, term()}
  def mean_pool_f32(matrix, dimensions, row_indices, opts \\ [])

  def mean_pool_f32(matrix, dimensions, row_indices, opts)
      when is_binary(matrix) and is_integer(dimensions) and is_list(row_indices) and
             is_list(opts) do
    with :ok <- validate_options(opts, [:as]),
         :ok <- validate_matrix_dimensions(dimensions),
         :ok <- validate_indices(row_indices),
         target = Keyword.get(opts, :as, :f32_binary),
         {:ok, target} <- resolve_explicit_target(target, matrix),
         {:ok, pooled} <- native_mean_pool(matrix, dimensions, row_indices) do
      convert(pooled, target)
    end
  end

  def mean_pool_f32(_matrix, _dimensions, _row_indices, _opts),
    do: {:error, :invalid_arguments}

  defp validate_wrapper(%__MODULE__{} = vector) do
    with true <- vector.representation in [:list, :f32_binary, :nx],
         true <- representation(vector.data) == vector.representation,
         {:ok, dimensions} <- dimensions(vector.data),
         true <- dimensions == vector.dimensions do
      :ok
    else
      _error -> {:error, :invalid_vector}
    end
  end

  defp validate_list(vector) do
    if Enum.all?(vector, &valid_coordinate?/1) do
      {:ok, Enum.map(vector, &(&1 / 1))}
    else
      {:error, :invalid_vector}
    end
  end

  defp valid_coordinate?(value) when is_integer(value), do: abs(value) <= @f32_max

  defp valid_coordinate?(value) when is_float(value) do
    value >= -@f32_max and value <= @f32_max
  end

  defp valid_coordinate?(_value), do: false

  defp decode_binary(binary) do
    run_native(
      fn ->
        binary
        |> Nifs.decode_f32_binary()
        |> normalize_native_error()
      end,
      fn -> decode_binary_fallback(binary) end
    )
  end

  defp encode_binary(vector) do
    for value <- vector, into: <<>>, do: <<value::float-little-32>>
  end

  defp normalize_to_list(vector, method) when is_map_key(@normalization_codes, method) do
    case unwrap(vector) do
      {:ok, binary} when is_binary(binary) ->
        normalize_binary(binary, method)

      _other ->
        with {:ok, vector} <- to_list(vector), do: Distance.normalize(vector, method)
    end
  end

  defp normalize_to_list(_vector, method), do: {:error, {:unknown_normalization, method}}

  defp cosine_with_normalization(left, right, :l2), do: metric(left, right, :cosine)
  defp cosine_with_normalization(left, right, :none), do: metric(left, right, :inner_product)

  defp cosine_with_normalization(left, right, method) when method in [:zscore, :minmax] do
    with {:ok, left} <- normalize(left, method, as: :list),
         {:ok, right} <- normalize(right, method, as: :list) do
      Distance.cosine(left, right, normalize: :none)
    end
  end

  defp cosine_with_normalization(_left, _right, method),
    do: {:error, {:unknown_normalization, method}}

  defp validate_rows([]), do: {:error, :empty_selection}

  defp validate_rows(vectors) do
    with {:ok, rows} <- collect_rows(vectors),
         dimensions <- rows |> hd() |> length(),
         true <- dimensions > 0,
         true <- Enum.all?(rows, &(length(&1) == dimensions)) do
      {:ok, rows, dimensions}
    else
      false -> {:error, :dimension_mismatch}
      {:error, reason} -> {:error, reason}
    end
  end

  defp collect_rows(vectors) do
    Enum.reduce_while(vectors, {:ok, []}, fn vector, {:ok, rows} ->
      case to_list(vector) do
        {:ok, row} -> {:cont, {:ok, [row | rows]}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, rows} -> {:ok, Enum.reverse(rows)}
      error -> error
    end
  end

  defp native_mean_pool(matrix, dimensions, row_indices) do
    run_native(
      fn ->
        matrix
        |> Nifs.mean_pool_f32(dimensions, row_indices)
        |> normalize_native_error()
      end,
      fn -> mean_pool_fallback(matrix, dimensions, row_indices) end
    )
  end

  defp binary_metric(left, right, metric) do
    run_native(
      fn ->
        left
        |> Nifs.metric_f32_binary(right, Map.fetch!(@metric_codes, metric))
        |> normalize_native_error()
      end,
      fn -> binary_metric_fallback(left, right, metric) end
    )
  end

  defp binary_metric_fallback(left, right, metric) do
    with {:ok, left} <- decode_binary_fallback(left),
         {:ok, right} <- decode_binary_fallback(right) do
      apply(Distance, metric, [left, right])
    end
  end

  defp normalize_binary(binary, method) do
    run_native(
      fn ->
        binary
        |> Nifs.normalize_f32_binary(Map.fetch!(@normalization_codes, method))
        |> normalize_native_error()
      end,
      fn -> normalize_binary_fallback(binary, method) end
    )
  end

  defp normalize_binary_fallback(binary, method) do
    with {:ok, vector} <- decode_binary_fallback(binary), do: Distance.normalize(vector, method)
  end

  defp decode_binary_fallback(binary) when rem(byte_size(binary), 4) == 0 do
    decode_binary_fallback(binary, [])
  end

  defp decode_binary_fallback(_binary), do: {:error, :invalid_f32_binary}

  defp decode_binary_fallback(<<>>, acc), do: {:ok, Enum.reverse(acc)}

  defp decode_binary_fallback(<<bits::little-unsigned-32, rest::binary>>, acc) do
    if Bitwise.band(bits, 0x7F800000) == 0x7F800000 do
      {:error, :invalid_vector}
    else
      <<value::float-little-32>> = <<bits::little-unsigned-32>>
      decode_binary_fallback(rest, [value | acc])
    end
  end

  defp mean_pool_fallback(matrix, dimensions, row_indices) do
    row_bytes = dimensions * 4

    cond do
      row_indices == [] ->
        {:error, :empty_selection}

      row_bytes <= 0 or matrix == <<>> or rem(byte_size(matrix), row_bytes) != 0 ->
        {:error, :matrix_shape_mismatch}

      true ->
        mean_pool_valid_matrix(matrix, row_bytes, row_indices)
    end
  end

  defp mean_pool_valid_matrix(matrix, row_bytes, row_indices) do
    row_count = div(byte_size(matrix), row_bytes)

    if Enum.all?(row_indices, &(&1 < row_count)) do
      row_indices
      |> Enum.map(fn row ->
        matrix
        |> binary_part(row * row_bytes, row_bytes)
        |> decode_binary_fallback()
      end)
      |> mean_decoded_rows()
    else
      {:error, :invalid_row_index}
    end
  end

  defp mean_decoded_rows(rows) do
    case Enum.all?(rows, &match?({:ok, _row}, &1)) do
      true ->
        values = Enum.map(rows, fn {:ok, row} -> row end)
        divisor = length(values)

        mean =
          values
          |> Enum.zip()
          |> Enum.map(fn coordinates ->
            coordinates
            |> Tuple.to_list()
            |> Enum.sum()
            |> Kernel./(divisor)
          end)

        validate_list(mean)

      false ->
        {:error, :invalid_vector}
    end
  end

  defp validate_indices(row_indices) do
    if Enum.all?(row_indices, &(is_integer(&1) and &1 >= 0 and &1 <= @usize_max)) do
      :ok
    else
      {:error, :invalid_row_index}
    end
  end

  defp native_f32_enabled? do
    Application.get_env(:vettore, :native_f32, true)
  end

  defp run_native(native, fallback) do
    if native_f32_enabled?() do
      try do
        native.()
      rescue
        _exception -> fallback.()
      catch
        _kind, _reason -> fallback.()
      end
    else
      fallback.()
    end
  end

  defp validate_matrix_dimensions(dimensions)
       when is_integer(dimensions) and dimensions > 0 and dimensions <= @usize_max,
       do: :ok

  defp validate_matrix_dimensions(_dimensions), do: {:error, :invalid_dimensions}

  defp output_target(vector, opts) do
    opts
    |> Keyword.get(:as, :same)
    |> resolve_explicit_target(vector)
  end

  defp resolve_explicit_target(:same, vector) do
    case default_representation(vector) do
      :unknown -> {:error, :invalid_vector}
      representation -> {:ok, representation}
    end
  end

  defp resolve_explicit_target(target, _vector) when target in [:list, :f32_binary, :nx],
    do: {:ok, target}

  defp resolve_explicit_target(target, _vector),
    do: {:error, {:unknown_representation, target}}

  defp default_representation(%__MODULE__{representation: representation}), do: representation
  defp default_representation(vector), do: representation(vector)

  defp unwrap(%__MODULE__{} = vector) do
    with :ok <- validate_wrapper(vector), do: {:ok, vector.data}
  end

  defp unwrap(vector), do: {:ok, vector}

  defp validate_options(opts, allowed) do
    if Keyword.keyword?(opts) do
      keys = Keyword.keys(opts)

      if keys == Enum.uniq(keys) and Enum.all?(keys, &(&1 in allowed)) do
        :ok
      else
        {:error, :invalid_options}
      end
    else
      {:error, :invalid_options}
    end
  end

  defp normalize_native_error({:error, reason}) when is_binary(reason) do
    {:error, Map.get(@native_errors, reason, reason)}
  end

  defp normalize_native_error(result), do: result
end
