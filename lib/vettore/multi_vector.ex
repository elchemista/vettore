defmodule Vettore.MultiVector do
  @moduledoc """
  Multi-vector scoring helpers.
  """

  alias Vettore.{Distance, Nifs}

  @type vector :: [number()]
  @type metric :: Distance.metric() | atom()

  @doc """
  Computes Chamfer/MaxSim similarity.

  For each query vector, this finds the best matching document vector and sums
  those best scores.

  ## Examples

      iex> Vettore.MultiVector.chamfer(
      ...>   [[1.0, 0.0], [0.0, 1.0]],
      ...>   [[1.0, 0.0], [1.0, 1.0]],
      ...>   metric: :inner_product
      ...> )
      {:ok, 2.0}
  """
  @spec chamfer([vector()], [vector()], keyword()) ::
          {:ok, float()} | {:error, term()}
  def chamfer(query_vectors, document_vectors, opts \\ [])

  def chamfer(query_vectors, document_vectors, opts)
      when is_list(query_vectors) and is_list(document_vectors) and is_list(opts) do
    if Keyword.keyword?(opts) and Keyword.keys(opts) in [[], [:metric]] do
      metric = normalize_metric(Keyword.get(opts, :metric, :inner_product))

      with {:ok, metric_code} <- metric_code(metric),
           {:ok, query_vectors} <- prepare_vectors(query_vectors),
           {:ok, document_vectors} <- prepare_vectors(document_vectors) do
        Nifs.multi_vector_score(query_vectors, document_vectors, metric_code)
        |> normalize_native_error()
      end
    else
      {:error, :invalid_options}
    end
  end

  def chamfer(_query_vectors, _document_vectors, _opts), do: {:error, :invalid_multi_vector}

  @doc """
  Computes a ColBERT-style late interaction score.

  This is an explicit alias for Chamfer/MaxSim scoring: each query vector takes
  its best match from the document vectors, then the best-match scores are
  summed.

  ## Examples

      iex> Vettore.MultiVector.colbert_score(
      ...>   [[1.0, 0.0], [0.0, 1.0]],
      ...>   [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
      ...>   metric: :inner_product
      ...> )
      {:ok, 2.0}
  """
  @spec colbert_score([vector()], [vector()], keyword()) ::
          {:ok, float()} | {:error, term()}
  def colbert_score(query_vectors, document_vectors, opts \\ []) do
    chamfer(query_vectors, document_vectors, opts)
  end

  @spec prepare_vectors(term()) :: {:ok, [[float()]]} | {:error, :invalid_multi_vector}
  defp prepare_vectors([]), do: {:ok, []}

  defp prepare_vectors([first | _] = vectors) when is_list(first) and first != [] do
    dimensions = length(first)

    if Enum.all?(vectors, fn vector ->
         is_list(vector) and length(vector) == dimensions and Enum.all?(vector, &finite_f32?/1)
       end) do
      {:ok, Enum.map(vectors, fn vector -> Enum.map(vector, &(&1 / 1)) end)}
    else
      {:error, :invalid_multi_vector}
    end
  end

  defp prepare_vectors(_vectors), do: {:error, :invalid_multi_vector}

  @spec metric_code(metric()) :: {:ok, 0..8} | {:error, {:unknown_metric, term()}}
  defp metric_code(:l2), do: {:ok, 0}
  defp metric_code(:l2_squared), do: {:ok, 1}
  defp metric_code(:cosine), do: {:ok, 2}
  defp metric_code(:inner_product), do: {:ok, 3}
  defp metric_code(:negative_inner_product), do: {:ok, 4}
  defp metric_code(:manhattan), do: {:ok, 5}
  defp metric_code(:chebyshev), do: {:ok, 6}
  defp metric_code(:hamming), do: {:ok, 7}
  defp metric_code(:jaccard), do: {:ok, 8}
  defp metric_code(metric), do: {:error, {:unknown_metric, metric}}

  @spec normalize_metric(term()) :: term()
  defp normalize_metric(:dot), do: :inner_product
  defp normalize_metric(:dot_product), do: :inner_product
  defp normalize_metric(:euclidean), do: :l2
  defp normalize_metric(metric), do: metric

  @spec normalize_native_error(term()) :: term()
  defp normalize_native_error({:error, "dimension mismatch"}), do: {:error, :dimension_mismatch}

  defp normalize_native_error({:error, "vector contains a non-finite value"}),
    do: {:error, :invalid_multi_vector}

  defp normalize_native_error({:error, "score overflow"}), do: {:error, :score_overflow}

  defp normalize_native_error(other), do: other

  @spec finite_f32?(term()) :: boolean()
  defp finite_f32?(value) when is_integer(value),
    do: value >= -3.402_823_466_385_288_6e38 and value <= 3.402_823_466_385_288_6e38

  defp finite_f32?(value) when is_float(value),
    do: value >= -3.402_823_466_385_288_6e38 and value <= 3.402_823_466_385_288_6e38

  defp finite_f32?(_value), do: false
end
