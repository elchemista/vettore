defmodule Vettore.Encoding.Muvera do
  @moduledoc """
  MUVERA-style fixed dimensional encodings for multi-vector retrieval.

  This implements the public FDE boundary: query encodings sum vectors in each
  partition, document encodings average vectors in each partition, and both use
  the same deterministic partition configuration.
  """

  alias Vettore.Nifs

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
    with {:ok, dimension} <- dimension(vectors),
         {:ok, config} <- normalize_config(config, dimension) do
      native_encode(vectors, config, mode)
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

    cond do
      normalized.dimension != dimension -> {:error, :dimension_mismatch}
      normalized.num_repetitions <= 0 -> {:error, :invalid_repetitions}
      normalized.num_simhash_projections < 0 -> {:error, :invalid_simhash_projections}
      normalized.num_simhash_projections >= 31 -> {:error, :invalid_simhash_projections}
      normalized.projection_dimension <= 0 -> {:error, :invalid_projection_dimension}
      true -> {:ok, normalized}
    end
  end

  @spec dimension(term()) ::
          {:ok, pos_integer()} | {:error, :empty_vectors | :dimension_mismatch | :invalid_vectors}
  defp dimension([]), do: {:error, :empty_vectors}

  defp dimension([first | _] = vectors) when is_list(first) do
    dimension = length(first)

    if Enum.all?(vectors, &(is_list(&1) and length(&1) == dimension)) do
      {:ok, dimension}
    else
      {:error, :dimension_mismatch}
    end
  end

  defp dimension(_vectors), do: {:error, :invalid_vectors}
end
