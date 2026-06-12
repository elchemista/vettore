defmodule Vettore.Postgres do
  @moduledoc """
  PostgreSQL/pgvector readiness helpers without adding a database dependency.
  """

  alias Vettore.Distance

  @doc """
  Returns the pgvector operator for a metric.
  """
  def operator(:l2), do: "<->"
  def operator(:euclidean), do: "<->"
  def operator(:inner_product), do: "<#>"
  def operator(:dot), do: "<#>"
  def operator(:cosine), do: "<=>"
  def operator(:l1), do: "<+>"
  def operator(:manhattan), do: "<+>"
  def operator(:hamming), do: "<~>"
  def operator(:jaccard), do: "<%>"
  def operator(metric), do: {:error, {:unknown_pgvector_metric, metric}}

  @doc """
  Applies Vettore normalization before a vector is sent to PostgreSQL.
  """
  def normalize(vector, method), do: Distance.normalize(vector, method)
end
