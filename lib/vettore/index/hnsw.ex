defmodule Vettore.Index.HNSW do
  @moduledoc """
  HNSW index boundary.

  The current native library does not expose a standalone HNSW resource that can
  reference ETS-owned ids, so this module preserves the architecture boundary
  and falls back to exact flat search until that native accelerator is added.
  """

  @behaviour Vettore.Index

  alias Vettore.Index.Flat

  @impl true
  def search(collection, query, opts), do: Flat.search(collection, query, opts)
end
