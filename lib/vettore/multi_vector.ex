defmodule Vettore.MultiVector do
  @moduledoc """
  Multi-vector scoring helpers.
  """

  alias Vettore.Distance

  @doc """
  Computes Chamfer/MaxSim similarity.

  For each query vector, this finds the best matching document vector and sums
  those best scores.
  """
  def chamfer(query_vectors, document_vectors, opts \\ [])

  def chamfer(query_vectors, document_vectors, opts)
      when is_list(query_vectors) and is_list(document_vectors) do
    metric = Keyword.get(opts, :metric, :inner_product)

    cond do
      query_vectors == [] ->
        {:ok, 0.0}

      document_vectors == [] ->
        {:ok, 0.0}

      true ->
        query_vectors
        |> Enum.reduce_while(0.0, fn query_vector, acc ->
          case best_similarity(query_vector, document_vectors, metric) do
            {:ok, score} -> {:cont, acc + score}
            {:error, reason} -> {:halt, {:error, reason}}
          end
        end)
        |> case do
          {:error, reason} -> {:error, reason}
          score -> {:ok, score}
        end
    end
  end

  def chamfer(_query_vectors, _document_vectors, _opts), do: {:error, :invalid_multi_vector}

  defp best_similarity(query_vector, document_vectors, metric) do
    document_vectors
    |> Enum.reduce_while(nil, fn document_vector, best ->
      case Distance.compute(metric, query_vector, document_vector,
             normalize: normalize_for(metric)
           ) do
        {:ok, value} ->
          similarity = as_similarity(metric, value)
          {:cont, if(best == nil, do: similarity, else: max(best, similarity))}

        {:error, reason} ->
          {:halt, {:error, reason}}
      end
    end)
    |> case do
      {:error, reason} -> {:error, reason}
      nil -> {:ok, 0.0}
      best -> {:ok, best}
    end
  end

  defp as_similarity(metric, value) when metric in [:cosine, :inner_product, :dot], do: value
  defp as_similarity(_metric, distance), do: 1.0 / (1.0 + distance)

  defp normalize_for(:cosine), do: :l2
  defp normalize_for(_metric), do: :none
end
