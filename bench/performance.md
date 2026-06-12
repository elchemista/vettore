## Vettore vNext Benchmarks

Run:

```bash
mix run bench/vettore_bench.exs
```

Defaults:

- dimensions: `384`
- records per collection: `1000`
- metrics: `:l2`, `:cosine`, `:inner_product`
- index: exact ETS flat scan

Override dimensions or batch size:

```bash
VETTORE_BENCH_DIMENSIONS=768 VETTORE_BENCH_BATCH=10000 mix run bench/vettore_bench.exs
```

Recommended benchmark matrix:

- `384D`: `1_000`, `10_000`, `100_000` records
- `768D`: `1_000`, `10_000`, `100_000` records

The benchmark currently measures the ETS flat index. Once a standalone native
HNSW index is exposed behind `Vettore.Index.HNSW`, add equivalent `index: :hnsw`
runs and compare recall/latency against exact flat search.
