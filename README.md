# Vettore

Vettore is an ETS-native vector toolkit for Elixir. It provides collection
storage, exact similarity search, distance helpers, normalization utilities,
PostgreSQL/pgvector interoperability helpers, and multi-vector retrieval
building blocks.

Vettore vNext makes ETS the canonical store. Native/Rust acceleration can be
added behind index or distance boundaries, but the public architecture no
longer depends on a Rust-owned database resource.

## Installation

```elixir
def deps do
  [
    {:vettore, "~> 0.3.0"}
  ]
end
```

## Collections

```elixir
{:ok, collection} =
  Vettore.Collection.new(
    name: :documents,
    dimensions: 3,
    store: :ets,
    index: :flat,
    metric: :cosine,
    normalize: :l2,
    score: :raw
  )

:ok =
  Vettore.Collection.put_many(collection, [
    %Vettore.Embedding{id: "a", vector: [1.0, 0.0, 0.0], metadata: %{"kind" => "axis"}},
    %Vettore.Embedding{id: "b", vector: [0.0, 1.0, 0.0]}
  ])

{:ok, results} = Vettore.Collection.search(collection, [1.0, 0.0, 0.0], limit: 2)
```

Search returns `%Vettore.Result{}` structs with explicit semantics:

```elixir
%Vettore.Result{
  id: "a",
  score: 1.0,
  distance: 0.0,
  metric: :cosine,
  metadata: %{"kind" => "axis"}
}
```

## Distances And Normalization

```elixir
Vettore.Distance.compute(:l2, [0.0, 0.0], [3.0, 4.0])
# {:ok, 5.0}

Vettore.Distance.compute(:cosine, [1.0, 0.0], [0.0, 1.0], normalize: :l2)
# {:ok, 0.0}

Vettore.Distance.normalize([3.0, 4.0], :l2)
# {:ok, [0.6, 0.8]}
```

Supported metrics:

- `:l2`
- `:l2_squared`
- `:cosine`
- `:inner_product`
- `:negative_inner_product`
- `:manhattan`
- `:chebyshev`
- `:hamming`
- `:jaccard`

Supported normalization modes:

- `:none`
- `:l2`
- `:zscore`
- `:minmax`

## PostgreSQL Helpers

Vettore does not depend on PostgreSQL or Ecto, but it exposes pgvector operator
helpers so applications can share the same metric vocabulary:

```elixir
Vettore.Postgres.operator(:l2)            # "<->"
Vettore.Postgres.operator(:inner_product) # "<#>"
Vettore.Postgres.operator(:cosine)        # "<=>"
Vettore.Postgres.normalize(vector, :l2)
```

## Multi-Vector Retrieval

```elixir
Vettore.MultiVector.chamfer(query_vectors, document_vectors, metric: :inner_product)

Vettore.Encoding.Muvera.encode_query(vectors, num_repetitions: 1, num_simhash_projections: 4)
Vettore.Encoding.Muvera.encode_document(vectors, num_repetitions: 1, num_simhash_projections: 4)
```

The MUVERA module implements the fixed-dimensional encoding boundary: query
FDEs sum vectors in partitions and document FDEs average vectors in partitions.
Use FDEs for first-pass candidate retrieval, then rerank candidates with exact
Chamfer/MaxSim similarity.

## Compatibility

The old `Vettore.new/0`, `create_collection/5`, `insert/3`, `batch/3`, and
`similarity_search/4` functions remain as a small compatibility layer backed by
ETS collections. New code should prefer `Vettore.Collection`.

## Tests

```bash
mix test
```
