defmodule Vettore.Encoding.Muvera do
  @moduledoc """
  MUVERA-style fixed dimensional encodings for multi-vector retrieval.

  This implements the public FDE boundary: query encodings sum vectors in each
  partition, document encodings average vectors in each partition, and both use
  the same deterministic partition configuration.
  """

  alias Vettore.Nifs

  @max_output_dimensions 16_777_216
  @u64_max 18_446_744_073_709_551_615
  @f32_max 3.402_823_466_385_288_6e38
  @config_keys ~w(dimension num_repetitions num_simhash_projections seed projection_dimension final_projection_dimension)a

  @type vector :: [number()]
  @type config :: keyword()
  @type normalized_config :: %{
          dimension: pos_integer(),
          num_repetitions: pos_integer(),
          num_simhash_projections: non_neg_integer(),
          seed: non_neg_integer(),
          projection_dimension: pos_integer(),
          final_projection_dimension: pos_integer() | nil
        }

  @doc """
  Encodes query vectors by summing projected vectors in each partition.
  """
  @spec encode_query([vector()], config()) :: {:ok, [float()]} | {:error, term()}
  def encode_query(vectors, config \\ []), do: encode(vectors, config, :sum)

  @doc """
  Encodes document vectors by averaging projected vectors in each partition.
  """
  @spec encode_document([vector()], config()) :: {:ok, [float()]} | {:error, term()}
  def encode_document(vectors, config \\ []), do: encode(vectors, config, :average)

  @spec encode([vector()] | term(), config() | term(), :sum | :average) ::
          {:ok, [float()]} | {:error, term()}
  defp encode(vectors, config, mode) when is_list(vectors) and is_list(config) do
    with :ok <- validate_keyword(config),
         {:ok, vectors, dimension} <- prepare_vectors(vectors),
         {:ok, config} <- normalize_config(config, dimension) do
      native_encode(vectors, config, mode)
      |> normalize_native_error()
    end
  end

  defp encode(_vectors, _config, _mode), do: {:error, :invalid_vectors}

  @spec native_encode([vector()], normalized_config(), :sum | :average) ::
          {:ok, [float()]} | {:error, String.t()}
  defp native_encode(vectors, config, :sum) do
    Nifs.muvera_encode_query(
      vectors,
      config.dimension,
      config.num_repetitions,
      config.num_simhash_projections,
      config.seed,
      config.projection_dimension,
      config.final_projection_dimension
    )
  end

  defp native_encode(vectors, config, :average) do
    Nifs.muvera_encode_document(
      vectors,
      config.dimension,
      config.num_repetitions,
      config.num_simhash_projections,
      config.seed,
      config.projection_dimension,
      config.final_projection_dimension
    )
  end

  @spec normalize_native_error(term()) :: term()
  defp normalize_native_error({:error, "encoding overflow"}), do: {:error, :encoding_overflow}
  defp normalize_native_error(other), do: other

  @spec normalize_config(config(), pos_integer()) :: {:ok, normalized_config()} | {:error, atom()}
  defp normalize_config(config, dimension) do
    config = Map.new(config)

    normalized = %{
      dimension: Map.get(config, :dimension, dimension),
      num_repetitions: Map.get(config, :num_repetitions, 1),
      num_simhash_projections: Map.get(config, :num_simhash_projections, 0),
      seed: Map.get(config, :seed, 1),
      projection_dimension: Map.get(config, :projection_dimension, dimension),
      final_projection_dimension: Map.get(config, :final_projection_dimension)
    }

    with :ok <- validate_dimension(normalized.dimension, dimension),
         :ok <- validate_repetitions(normalized.num_repetitions),
         :ok <- validate_simhash_projections(normalized.num_simhash_projections),
         :ok <- validate_seed(normalized.seed),
         :ok <- validate_projection_dimension(normalized.projection_dimension),
         :ok <- validate_final_dimension(normalized.final_projection_dimension),
         :ok <- validate_encoding_size(normalized) do
      {:ok, normalized}
    end
  end

  @spec validate_dimension(term(), pos_integer()) ::
          :ok | {:error, :dimension_mismatch | :invalid_dimension}
  defp validate_dimension(value, expected) when is_integer(value) do
    if value == expected, do: :ok, else: {:error, :dimension_mismatch}
  end

  defp validate_dimension(_value, _expected), do: {:error, :invalid_dimension}

  @spec validate_repetitions(term()) :: :ok | {:error, :invalid_repetitions}
  defp validate_repetitions(value) do
    if positive_integer?(value), do: :ok, else: {:error, :invalid_repetitions}
  end

  @spec validate_simhash_projections(term()) ::
          :ok | {:error, :invalid_simhash_projections}
  defp validate_simhash_projections(value)
       when is_integer(value) and value >= 0 and value < 31,
       do: :ok

  defp validate_simhash_projections(_value), do: {:error, :invalid_simhash_projections}

  @spec validate_seed(term()) :: :ok | {:error, :invalid_seed}
  defp validate_seed(value) when is_integer(value) and value >= 0 and value <= @u64_max, do: :ok
  defp validate_seed(_value), do: {:error, :invalid_seed}

  @spec validate_projection_dimension(term()) ::
          :ok | {:error, :invalid_projection_dimension}
  defp validate_projection_dimension(value) do
    if positive_integer?(value), do: :ok, else: {:error, :invalid_projection_dimension}
  end

  @spec validate_final_dimension(term()) ::
          :ok | {:error, :invalid_final_projection_dimension}
  defp validate_final_dimension(value) do
    if valid_final_dimension?(value),
      do: :ok,
      else: {:error, :invalid_final_projection_dimension}
  end

  @spec validate_encoding_size(normalized_config()) :: :ok | {:error, :encoding_too_large}
  defp validate_encoding_size(config) do
    if encoding_size(config) <= @max_output_dimensions,
      do: :ok,
      else: {:error, :encoding_too_large}
  end

  @spec prepare_vectors(term()) ::
          {:ok, [[float()]], pos_integer()}
          | {:error, :empty_vectors | :dimension_mismatch | :invalid_vectors}
  defp prepare_vectors([]), do: {:error, :empty_vectors}

  defp prepare_vectors([first | _] = vectors) when is_list(first) and first != [] do
    dimension = length(first)

    cond do
      not Enum.all?(vectors, &(is_list(&1) and length(&1) == dimension)) ->
        {:error, :dimension_mismatch}

      not Enum.all?(vectors, &Enum.all?(&1, fn value -> finite_f32?(value) end)) ->
        {:error, :invalid_vectors}

      true ->
        {:ok, Enum.map(vectors, fn vector -> Enum.map(vector, &(&1 / 1)) end), dimension}
    end
  end

  defp prepare_vectors(_vectors), do: {:error, :invalid_vectors}

  @spec validate_keyword(term()) :: :ok | {:error, :invalid_config}
  defp validate_keyword(config) do
    keys = if Keyword.keyword?(config), do: Keyword.keys(config), else: []

    if Keyword.keyword?(config) and Enum.all?(keys, &(&1 in @config_keys)) and
         length(keys) == MapSet.size(MapSet.new(keys)),
       do: :ok,
       else: {:error, :invalid_config}
  end

  @spec encoding_size(normalized_config()) :: non_neg_integer()
  defp encoding_size(config) do
    full =
      config.num_repetitions *
        Bitwise.bsl(1, config.num_simhash_projections) * config.projection_dimension

    max(full, config.final_projection_dimension || full)
  end

  @spec positive_integer?(term()) :: boolean()
  defp positive_integer?(value), do: is_integer(value) and value > 0

  @spec valid_final_dimension?(term()) :: boolean()
  defp valid_final_dimension?(nil), do: true
  defp valid_final_dimension?(value), do: positive_integer?(value)

  @spec finite_f32?(term()) :: boolean()
  defp finite_f32?(value) when is_integer(value),
    do: value >= -@f32_max and value <= @f32_max

  defp finite_f32?(value) when is_float(value),
    do: value >= -@f32_max and value <= @f32_max

  defp finite_f32?(_value), do: false
end
