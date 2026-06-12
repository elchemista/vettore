defmodule Vettore.Encoding.Muvera do
  @moduledoc """
  MUVERA-style fixed dimensional encodings for multi-vector retrieval.

  This implements the public FDE boundary: query encodings sum vectors in each
  partition, document encodings average vectors in each partition, and both use
  the same deterministic partition configuration.
  """

  import Bitwise

  @doc """
  Encodes query vectors by summing projected vectors in each partition.
  """
  def encode_query(vectors, config \\ []), do: encode(vectors, config, :sum)

  @doc """
  Encodes document vectors by averaging projected vectors in each partition.
  """
  def encode_document(vectors, config \\ []), do: encode(vectors, config, :average)

  defp encode(vectors, config, mode) when is_list(vectors) and is_list(config) do
    with {:ok, dimension} <- dimension(vectors),
         {:ok, config} <- normalize_config(config, dimension) do
      repetitions = config.num_repetitions
      partitions = pow2(config.num_simhash_projections)
      projection_dimension = config.projection_dimension
      output_size = repetitions * partitions * projection_dimension

      {encoded, counts} =
        Enum.reduce(0..(repetitions - 1), {List.duplicate(0.0, output_size), %{}}, fn rep,
                                                                                      {out,
                                                                                       counts} ->
          Enum.reduce(vectors, {out, counts}, fn vector, {out, counts} ->
            partition = partition_index(vector, config, rep)
            projected = project(vector, config, rep)
            base = rep * partitions * projection_dimension + partition * projection_dimension
            out = add_at(out, base, projected)
            counts = Map.update(counts, {rep, partition}, 1, &(&1 + 1))
            {out, counts}
          end)
        end)

      encoded =
        if mode == :average do
          average_partitions(encoded, counts, repetitions, partitions, projection_dimension)
        else
          encoded
        end

      final_project(encoded, config)
    end
  end

  defp encode(_vectors, _config, _mode), do: {:error, :invalid_vectors}

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

  defp dimension([]), do: {:error, :empty_vectors}

  defp dimension([first | _] = vectors) when is_list(first) do
    dimension = length(first)

    if Enum.all?(vectors, &(is_list(&1) and length(&1) == dimension)) do
      {:ok, dimension}
    else
      {:error, :dimension_mismatch}
    end
  end

  defp partition_index(_vector, %{num_simhash_projections: 0}, _rep), do: 0

  defp partition_index(vector, config, rep) do
    0..(config.num_simhash_projections - 1)
    |> Enum.reduce(0, fn projection, acc ->
      dot =
        vector
        |> Enum.with_index()
        |> Enum.reduce(0.0, fn {value, dim}, sum ->
          sum + value * random_weight(config.seed, rep, projection, dim)
        end)

      (acc <<< 1) + if(dot >= 0.0, do: 1, else: 0)
    end)
  end

  defp project(vector, %{projection_dimension: projection_dimension, dimension: dimension}, _rep)
       when projection_dimension == dimension do
    vector
  end

  defp project(vector, config, rep) do
    0..(config.projection_dimension - 1)
    |> Enum.map(fn projection ->
      vector
      |> Enum.with_index()
      |> Enum.reduce(0.0, fn {value, dim}, sum ->
        sum + value * random_sign(config.seed + 17, rep, projection, dim)
      end)
    end)
  end

  defp final_project(vector, %{final_projection_dimension: nil}), do: {:ok, vector}

  defp final_project(vector, %{final_projection_dimension: dimension, seed: seed})
       when is_integer(dimension) and dimension > 0 do
    out = List.duplicate(0.0, dimension)

    vector
    |> Enum.with_index()
    |> Enum.reduce(out, fn {value, index}, acc ->
      slot = rem(abs(:erlang.phash2({seed, :final, index})), dimension)
      sign = random_sign(seed, :final, index, slot)
      List.update_at(acc, slot, &(&1 + sign * value))
    end)
    |> then(&{:ok, &1})
  end

  defp average_partitions(encoded, counts, repetitions, partitions, projection_dimension) do
    Enum.reduce(0..(repetitions - 1), encoded, fn rep, out ->
      Enum.reduce(0..(partitions - 1), out, fn partition, out ->
        case Map.get(counts, {rep, partition}, 0) do
          0 ->
            out

          count ->
            base = rep * partitions * projection_dimension + partition * projection_dimension

            Enum.reduce(0..(projection_dimension - 1), out, fn offset, out ->
              List.update_at(out, base + offset, &(&1 / count))
            end)
        end
      end)
    end)
  end

  defp add_at(out, base, values) do
    values
    |> Enum.with_index()
    |> Enum.reduce(out, fn {value, offset}, acc ->
      List.update_at(acc, base + offset, &(&1 + value))
    end)
  end

  defp random_weight(seed, rep, projection, dim) do
    (:erlang.phash2({seed, rep, projection, dim}, 2_000_001) - 1_000_000) / 1_000_000
  end

  defp random_sign(seed, rep, projection, dim) do
    if rem(:erlang.phash2({seed, rep, projection, dim}), 2) == 0, do: 1.0, else: -1.0
  end

  defp pow2(n), do: round(:math.pow(2, n))
end
