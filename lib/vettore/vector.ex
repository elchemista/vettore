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

  alias Vettore.{Compute, Distance, Nifs}
  alias Vettore.Interop.Nx, as: NxInterop

  @enforce_keys [:data, :dimensions, :representation]
  defstruct [:data, :dimensions, :representation, :shape]

  @type representation :: :list | :f32_binary | :nx
  @type target_representation :: representation() | :same
  @type raw_vector :: [number()] | binary() | term()
  @type t :: %__MODULE__{
          data: raw_vector(),
          dimensions: non_neg_integer(),
          representation: representation(),
          shape: tuple() | nil
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
    with :ok <- validate_options(opts, [:as, :shape]),
         target <- Keyword.get(opts, :as, :same),
         {:ok, target} <- resolve_explicit_target(target, vector),
         {:ok, dimensions} <- dimensions(vector),
         {:ok, source_shape} <- shape(vector),
         selected_shape = Keyword.get(opts, :shape, source_shape),
         :ok <- validate_shape(selected_shape, dimensions),
         {:ok, data} <- convert_with_shape(vector, target, selected_shape) do
      {:ok,
       %__MODULE__{
         data: data,
         dimensions: dimensions,
         representation: target,
         shape: selected_shape
       }}
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
      if is_tuple(vector.shape), do: {:ok, vector.shape}, else: shape(vector.data)
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

  @doc "Converts a vector to an Nx tensor, preserving shape and accepting a host backend."
  @spec to_nx(vector(), keyword()) :: {:ok, term()} | {:error, term()}
  def to_nx(vector, opts \\ [])

  def to_nx(vector, opts) when is_list(opts) do
    with :ok <- validate_options(opts, [:backend, :shape]),
         {:ok, values} <- to_list(vector),
         {:ok, source_shape} <- shape(vector),
         selected_shape = Keyword.get(opts, :shape, source_shape),
         :ok <- validate_shape(selected_shape, length(values)) do
      NxInterop.from_list(
        values,
        opts
        |> Keyword.put(:shape, selected_shape)
      )
    end
  end

  def to_nx(_vector, _opts), do: {:error, :invalid_options}

  @doc "Converts an Nx tensor into a validated Vettore representation."
  @spec from_nx(term(), :list | :f32_binary) :: {:ok, raw_vector()} | {:error, term()}
  def from_nx(tensor, target \\ :f32_binary)

  def from_nx(tensor, target) when target in [:list, :f32_binary] do
    if NxInterop.tensor?(tensor), do: convert(tensor, target), else: {:error, :invalid_vector}
  end

  def from_nx(_tensor, target), do: {:error, {:unknown_representation, target}}

  @doc "Returns a wrapper with new shape metadata, without changing coordinate order."
  @spec reshape(vector(), tuple(), keyword()) :: {:ok, t()} | {:error, term()}
  def reshape(vector, shape, opts \\ [])

  def reshape(vector, shape, opts) when is_list(opts) do
    new(vector, Keyword.put(opts, :shape, shape))
  end

  def reshape(_vector, _shape, _opts), do: {:error, :invalid_options}

  @doc "Converts between list, f32-binary, and optional Nx representations."
  @spec convert(vector(), target_representation()) :: {:ok, raw_vector()} | {:error, term()}
  def convert(vector, :same) do
    with {:ok, target} <- resolve_explicit_target(:same, vector), do: convert(vector, target)
  end

  def convert(vector, :list), do: to_list(vector)
  def convert(vector, :f32_binary), do: to_f32_binary(vector)
  def convert(vector, :nx), do: to_nx(vector)
  def convert(_vector, target), do: {:error, {:unknown_representation, target}}

  @doc "Returns the row/column shape of a row-major little-endian f32 matrix."
  @spec matrix_shape_f32(binary(), pos_integer()) ::
          {:ok, {non_neg_integer(), pos_integer()}} | {:error, term()}
  def matrix_shape_f32(matrix, dimensions)
      when is_binary(matrix) and is_integer(dimensions) do
    with :ok <- validate_matrix_dimensions(dimensions),
         row_bytes = dimensions * 4,
         true <- rem(byte_size(matrix), row_bytes) == 0 do
      {:ok, {div(byte_size(matrix), row_bytes), dimensions}}
    else
      false -> {:error, :matrix_shape_mismatch}
      {:error, reason} -> {:error, reason}
    end
  end

  def matrix_shape_f32(_matrix, _dimensions), do: {:error, :invalid_arguments}

  @doc "Validates a complete f32 matrix and returns its shape."
  @spec validate_matrix_f32(binary(), pos_integer()) ::
          {:ok, {non_neg_integer(), pos_integer()}} | {:error, term()}
  def validate_matrix_f32(matrix, dimensions) do
    with {:ok, shape} <- matrix_shape_f32(matrix, dimensions),
         {:ok, _values} <- decode_binary(matrix) do
      {:ok, shape}
    end
  end

  @doc "Returns whether a complete row-major f32 matrix is structurally valid and finite."
  @spec valid_matrix_f32?(term(), term()) :: boolean()
  def valid_matrix_f32?(matrix, dimensions),
    do: match?({:ok, _shape}, validate_matrix_f32(matrix, dimensions))

  @doc "Stacks equally sized vectors into a row-major matrix representation."
  @spec stack([vector()], keyword()) :: {:ok, term()} | {:error, term()}
  def stack(vectors, opts \\ [])

  def stack(vectors, opts) when is_list(vectors) and is_list(opts) do
    with :ok <- validate_options(opts, [:as, :backend]),
         {:ok, rows, dimensions} <- validate_rows(vectors) do
      render_matrix(rows, dimensions, opts)
    end
  end

  def stack(_vectors, _opts), do: {:error, :invalid_options}

  @doc "Selects rows from a row-major f32 matrix without mean pooling them."
  @spec take_rows_f32(binary(), pos_integer(), [non_neg_integer()], keyword()) ::
          {:ok, term()} | {:error, term()}
  def take_rows_f32(matrix, dimensions, row_indices, opts \\ [])

  def take_rows_f32(matrix, dimensions, row_indices, opts)
      when is_binary(matrix) and is_integer(dimensions) and is_list(row_indices) and
             is_list(opts) do
    with :ok <- validate_options(opts, [:as, :backend]),
         {:ok, {row_count, ^dimensions}} <- matrix_shape_f32(matrix, dimensions),
         :ok <- validate_indices(row_indices),
         true <- Enum.all?(row_indices, &(&1 < row_count)),
         {:ok, rows} <- copy_matrix_rows(matrix, dimensions, row_indices) do
      render_matrix(rows, dimensions, opts)
    else
      false -> {:error, :invalid_row_index}
      {:error, reason} -> {:error, reason}
    end
  end

  def take_rows_f32(_matrix, _dimensions, _row_indices, _opts),
    do: {:error, :invalid_arguments}

  @doc "Normalizes a vector while allowing the result representation to be selected."
  @spec normalize(vector(), normalization(), keyword()) :: {:ok, raw_vector()} | {:error, term()}
  def normalize(vector, method \\ :l2, opts \\ [])

  def normalize(vector, method, opts) when is_list(opts) do
    with :ok <- validate_options(opts, [:as, :gpu, :gpu_fallback, :gpu_min_size]),
         :ok <- validate_normalization(method),
         {:ok, target} <- output_target(vector, opts),
         {:ok, dimensions} <- dimensions(vector),
         {:ok, normalized} <-
           Compute.run(
             compute_options(opts),
             dimensions,
             fn -> normalize_to_list(vector, method) end,
             fn -> gpu_normalize(vector, method) end
           ) do
      convert(normalized, target)
    end
  end

  def normalize(_vector, _method, _opts), do: {:error, :invalid_options}

  @doc "Computes one named metric over any pair of supported representations."
  @spec metric(vector(), vector(), metric(), keyword()) :: {:ok, float()} | {:error, term()}
  def metric(left, right, metric, opts \\ [])

  def metric(left, right, metric, opts)
      when is_map_key(@metric_codes, metric) and is_list(opts) do
    with :ok <- validate_options(opts, [:gpu, :gpu_fallback, :gpu_min_size]),
         {:ok, left_dimensions} <- dimensions(left),
         {:ok, right_dimensions} <- dimensions(right),
         true <- left_dimensions == right_dimensions do
      Compute.run(
        compute_options(opts),
        left_dimensions,
        fn -> cpu_metric(left, right, metric) end,
        fn -> gpu_metric(left, right, metric) end
      )
    else
      false -> {:error, :dimension_mismatch}
      {:error, reason} -> {:error, reason}
    end
  end

  def metric(_left, _right, _metric, opts) when not is_list(opts),
    do: {:error, :invalid_options}

  def metric(_left, _right, metric, _opts), do: {:error, {:unknown_metric, metric}}

  defp cpu_metric(left, right, metric) do
    case {unwrap(left), unwrap(right)} do
      {{:ok, left}, {:ok, right}} when is_binary(left) and is_binary(right) ->
        binary_metric(left, right, metric)

      _other ->
        with {:ok, left} <- to_list(left),
             {:ok, right} <- to_list(right) do
          apply(Distance, metric, [left, right, [gpu: false]])
        end
    end
  end

  @doc "L2/Euclidean distance over any supported representation."
  @spec l2(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def l2(left, right), do: metric(left, right, :l2)

  @doc "L2/Euclidean distance with per-call compute options."
  @spec l2(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def l2(left, right, opts), do: metric(left, right, :l2, opts)

  @doc "Squared L2 distance over any supported representation."
  @spec l2_squared(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def l2_squared(left, right), do: metric(left, right, :l2_squared)

  @doc "Squared L2 distance with per-call compute options."
  @spec l2_squared(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def l2_squared(left, right, opts), do: metric(left, right, :l2_squared, opts)

  @doc "Inner/dot product over any supported representation."
  @spec inner_product(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def inner_product(left, right), do: metric(left, right, :inner_product)

  @doc "Inner/dot product with per-call compute options."
  @spec inner_product(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def inner_product(left, right, opts), do: metric(left, right, :inner_product, opts)

  @doc "Compatibility alias for `inner_product/2`."
  @spec dot_product(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def dot_product(left, right), do: inner_product(left, right)

  @doc "Compatibility alias for `inner_product/3`."
  @spec dot_product(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def dot_product(left, right, opts), do: inner_product(left, right, opts)

  @doc "Negative inner product over any supported representation."
  @spec negative_inner_product(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def negative_inner_product(left, right), do: metric(left, right, :negative_inner_product)

  @doc "Negative inner product with per-call compute options."
  @spec negative_inner_product(vector(), vector(), keyword()) ::
          {:ok, float()} | {:error, term()}
  def negative_inner_product(left, right, opts),
    do: metric(left, right, :negative_inner_product, opts)

  @doc "Manhattan/L1 distance over any supported representation."
  @spec manhattan(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def manhattan(left, right), do: metric(left, right, :manhattan)

  @doc "Manhattan/L1 distance with per-call compute options."
  @spec manhattan(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def manhattan(left, right, opts), do: metric(left, right, :manhattan, opts)

  @doc "Chebyshev/L-infinity distance over any supported representation."
  @spec chebyshev(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def chebyshev(left, right), do: metric(left, right, :chebyshev)

  @doc "Chebyshev/L-infinity distance with per-call compute options."
  @spec chebyshev(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def chebyshev(left, right, opts), do: metric(left, right, :chebyshev, opts)

  @doc "Hamming distance using Vettore's non-zero coordinate semantics."
  @spec hamming(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def hamming(left, right), do: metric(left, right, :hamming)

  @doc "Hamming distance with per-call compute options."
  @spec hamming(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def hamming(left, right, opts), do: metric(left, right, :hamming, opts)

  @doc "Jaccard distance using Vettore's non-zero coordinate semantics."
  @spec jaccard(vector(), vector()) :: {:ok, float()} | {:error, term()}
  def jaccard(left, right), do: metric(left, right, :jaccard)

  @doc "Jaccard distance with per-call compute options."
  @spec jaccard(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def jaccard(left, right, opts), do: metric(left, right, :jaccard, opts)

  @doc "Cosine similarity, with the same normalization options as `Vettore.Distance.cosine/3`."
  @spec cosine(vector(), vector(), keyword()) :: {:ok, float()} | {:error, term()}
  def cosine(left, right, opts \\ [])

  def cosine(left, right, opts) when is_list(opts) do
    with :ok <-
           validate_options(opts, [
             :normalize,
             :gpu,
             :gpu_fallback,
             :gpu_min_size
           ]) do
      cosine_with_normalization(
        left,
        right,
        Keyword.get(opts, :normalize, :l2),
        compute_options(opts)
      )
    end
  end

  def cosine(_left, _right, _opts), do: {:error, :invalid_options}

  @doc "Mean-pools a non-empty list of equally sized vectors."
  @spec mean_pool([vector()], keyword()) :: {:ok, raw_vector()} | {:error, term()}
  def mean_pool(vectors, opts \\ [])

  def mean_pool(vectors, opts) when is_list(vectors) and is_list(opts) do
    with :ok <- validate_options(opts, [:as, :gpu, :gpu_fallback, :gpu_min_size]),
         {:ok, rows, dimensions} <- validate_rows(vectors),
         matrix = rows |> List.flatten() |> encode_binary(),
         target = Keyword.get(opts, :as, :list),
         {:ok, target} <- resolve_explicit_target(target, rows),
         {:ok, pooled} <-
           compute_mean_pool(
             matrix,
             dimensions,
             Enum.to_list(0..(length(rows) - 1)),
             dimensions * length(rows),
             opts
           ) do
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
    with :ok <- validate_options(opts, [:as, :gpu, :gpu_fallback, :gpu_min_size]),
         :ok <- validate_matrix_dimensions(dimensions),
         :ok <- validate_indices(row_indices),
         :ok <- validate_non_empty_selection(row_indices),
         target = Keyword.get(opts, :as, :f32_binary),
         {:ok, target} <- resolve_explicit_target(target, matrix),
         {:ok, pooled} <-
           compute_mean_pool(
             matrix,
             dimensions,
             row_indices,
             dimensions * length(row_indices),
             opts
           ) do
      convert(pooled, target)
    end
  end

  def mean_pool_f32(_matrix, _dimensions, _row_indices, _opts),
    do: {:error, :invalid_arguments}

  defp validate_wrapper(%__MODULE__{} = vector) do
    with true <- vector.representation in [:list, :f32_binary, :nx],
         true <- representation(vector.data) == vector.representation,
         {:ok, dimensions} <- dimensions(vector.data),
         true <- dimensions == vector.dimensions,
         :ok <- validate_optional_shape(vector.shape, dimensions),
         :ok <- validate_nx_wrapper_shape(vector) do
      :ok
    else
      _error -> {:error, :invalid_vector}
    end
  end

  defp validate_optional_shape(nil, _dimensions), do: :ok
  defp validate_optional_shape(shape, dimensions), do: validate_shape(shape, dimensions)

  defp validate_nx_wrapper_shape(%__MODULE__{representation: :nx, shape: shape, data: data})
       when is_tuple(shape) do
    case NxInterop.shape(data) do
      {:ok, ^shape} -> :ok
      _other -> {:error, :invalid_vector}
    end
  end

  defp validate_nx_wrapper_shape(%__MODULE__{}), do: :ok

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
        with {:ok, vector} <- to_list(vector),
             do: Distance.normalize(vector, method, gpu: false)
    end
  end

  defp cosine_with_normalization(left, right, :l2, opts),
    do: metric(left, right, :cosine, opts)

  defp cosine_with_normalization(left, right, :none, opts),
    do: metric(left, right, :inner_product, opts)

  defp cosine_with_normalization(left, right, method, opts)
       when method in [:zscore, :minmax] do
    normalization_opts = Keyword.put(opts, :as, :list)

    with {:ok, left} <- normalize(left, method, normalization_opts),
         {:ok, right} <- normalize(right, method, normalization_opts) do
      metric(left, right, :inner_product, opts)
    end
  end

  defp cosine_with_normalization(_left, _right, method, _opts),
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

  defp render_matrix(rows, dimensions, opts) do
    target = Keyword.get(opts, :as, :f32_binary)
    backend = Keyword.fetch(opts, :backend)

    cond do
      backend != :error and target != :nx ->
        {:error, :invalid_options}

      target == :list ->
        {:ok, rows}

      target == :f32_binary ->
        {:ok, rows |> List.flatten() |> encode_binary()}

      target == :nx ->
        nx_opts =
          [shape: {length(rows), dimensions}]
          |> maybe_put_option(:backend, backend)

        rows
        |> List.flatten()
        |> NxInterop.from_list(nx_opts)

      true ->
        {:error, {:unknown_representation, target}}
    end
  end

  defp copy_matrix_rows(matrix, dimensions, row_indices) do
    row_bytes = dimensions * 4

    Enum.reduce_while(row_indices, {:ok, []}, fn row_index, {:ok, rows} ->
      row = binary_part(matrix, row_index * row_bytes, row_bytes)

      case decode_binary(row) do
        {:ok, values} -> {:cont, {:ok, [values | rows]}}
        {:error, reason} -> {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:ok, rows} -> {:ok, Enum.reverse(rows)}
      {:error, reason} -> {:error, reason}
    end
  end

  defp maybe_put_option(opts, key, {:ok, value}), do: Keyword.put(opts, key, value)
  defp maybe_put_option(opts, _key, :error), do: opts

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

  defp compute_mean_pool(matrix, dimensions, row_indices, workload_size, opts) do
    Compute.run(
      compute_options(opts),
      workload_size,
      fn -> native_mean_pool(matrix, dimensions, row_indices) end,
      fn ->
        matrix
        |> Nifs.gpu_mean_pool_f32(dimensions, row_indices)
        |> normalize_native_error()
      end
    )
  end

  defp gpu_metric(left, right, metric) do
    with {:ok, left} <- to_list(left),
         {:ok, right} <- to_list(right) do
      left
      |> Nifs.gpu_metric(right, Map.fetch!(@metric_codes, metric))
      |> normalize_native_error()
    end
  end

  defp gpu_normalize(vector, method) do
    with {:ok, vector} <- to_list(vector) do
      vector
      |> Nifs.gpu_normalize(Map.fetch!(@normalization_codes, method))
      |> normalize_native_error()
    end
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
      apply(Distance, metric, [left, right, [gpu: false]])
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
    with {:ok, vector} <- decode_binary_fallback(binary),
         do: Distance.normalize(vector, method, gpu: false)
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

    if row_bytes <= 0 or matrix == <<>> or rem(byte_size(matrix), row_bytes) != 0,
      do: {:error, :matrix_shape_mismatch},
      else: mean_pool_valid_matrix(matrix, row_bytes, row_indices)
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

  defp validate_non_empty_selection([]), do: {:error, :empty_selection}
  defp validate_non_empty_selection(_row_indices), do: :ok

  defp native_f32_enabled? do
    Application.get_env(:vettore, :native_f32, true)
  end

  @doc false
  @spec run_native((-> term()), (-> term())) :: term()
  def run_native(native, fallback) do
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

  defp validate_shape(shape, dimensions) when is_tuple(shape) do
    entries = Tuple.to_list(shape)

    if Enum.all?(entries, &(is_integer(&1) and &1 >= 0)) and shape_size(entries) == dimensions,
      do: :ok,
      else: {:error, :invalid_shape}
  end

  defp validate_shape(_shape, _dimensions), do: {:error, :invalid_shape}

  defp shape_size([]), do: 1
  defp shape_size(entries), do: Enum.product(entries)

  defp validate_normalization(method) when is_map_key(@normalization_codes, method), do: :ok

  defp validate_normalization(method), do: {:error, {:unknown_normalization, method}}

  defp output_target(vector, opts) do
    opts
    |> Keyword.get(:as, :same)
    |> resolve_explicit_target(vector)
  end

  defp convert_with_shape(vector, :nx, shape), do: to_nx(vector, shape: shape)
  defp convert_with_shape(vector, target, _shape), do: convert(vector, target)

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

  defp compute_options(opts),
    do: Keyword.take(opts, [:gpu, :gpu_fallback, :gpu_min_size])

  defp normalize_native_error({:error, reason}) when is_binary(reason) do
    {:error, Map.get(@native_errors, reason, reason)}
  end

  defp normalize_native_error(result), do: result
end
