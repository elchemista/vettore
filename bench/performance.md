## Vettore 0.3.2 Benchmarks

Run:

```bash
mix run bench/vettore_bench.exs
```

Defaults:

- dimensions: `384`
- records per collection: `1000`
- metrics: `:l2`, `:cosine`, `:inner_product`
- indexes: native exact flat and HNSW

Override dimensions, batch size, indexes, or timing:

```bash
VETTORE_BENCH_DIMENSIONS=768 VETTORE_BENCH_BATCH=10000 mix run bench/vettore_bench.exs

VETTORE_BENCH_INDEXES=flat \
VETTORE_BENCH_TIME=5 \
VETTORE_BENCH_WARMUP=3 \
mix run bench/vettore_bench.exs
```

Recommended benchmark matrix:

- `384D`: `1_000`, `10_000`, `100_000` records
- `768D`: `1_000`, `10_000`, `100_000` records

The benchmark creates both index types from the same random-vector generator.
HNSW runs are emitted only for its supported metrics (`:l2`, `:cosine`, and
`:inner_product`). Use flat results as the exact latency baseline; measure HNSW
recall separately against flat top-k results for representative application
embeddings rather than relying on random vectors alone.

## Every search mode

Run the deterministic search-mode matrix:

```bash
mix run bench/search_modes_bench.exs
```

It benchmarks and preflights:

- exact flat and approximate HNSW search;
- Matryoshka funnel and binary-quantized candidate search;
- default and explicit hybrid pipelines on flat and HNSW indexes;
- exact multi-vector search and hybrid MaxSim reranking;
- direct MaxSim plus MUVERA query and document encodings.

Before timing, the script runs every scenario and prints `overlap@k` against
the corresponding exact flat or exact multi-vector baseline. All modes use the
same seeded records, so repeated runs on one machine are comparable. Available
settings are `VETTORE_BENCH_DIMENSIONS`, `VETTORE_BENCH_BATCH`,
`VETTORE_BENCH_LIMIT`, `VETTORE_BENCH_CANDIDATES`, `VETTORE_BENCH_STAGES`,
`VETTORE_BENCH_METRIC`, `VETTORE_BENCH_SEED`, `VETTORE_BENCH_TIME`, and
`VETTORE_BENCH_WARMUP`.

A small smoke run is useful after algorithm changes:

```bash
VETTORE_BENCH_DIMENSIONS=16 \
VETTORE_BENCH_BATCH=64 \
VETTORE_BENCH_LIMIT=5 \
VETTORE_BENCH_CANDIDATES=32 \
VETTORE_BENCH_TIME=1 \
VETTORE_BENCH_WARMUP=0 \
mix run bench/search_modes_bench.exs
```

## ETS owner write overhead

Collections use a supervised process as their ETS owner. Searches and reads go
directly and concurrently to protected ETS (`read_concurrency: true`); they do
not cross the owner process. Writes do cross it so callers cannot mutate
canonical records without updating the native index. Measure that write
boundary separately with:

```bash
VETTORE_BENCH_BATCH=10000 \
VETTORE_BENCH_DIMENSIONS=16 \
mix run bench/ets_owner_bench.exs
```

The script compares direct public ETS writes, one owner call per record, and a
single `put_many` owner call. Use `put_many/2` for ingestion: it amortizes the
message and performs one atomic ETS batch insertion.

Measure the direct protected-ETS read path independently with:

```bash
VETTORE_BENCH_READS=1000000 \
VETTORE_BENCH_READ_WORKERS=8 \
mix run bench/ets_read_bench.exs
```

This compares raw ETS lookup, `Vettore.Store.ETS.get/2`, and parallel callers.
Results depend on OTP, scheduler count, CPU topology, and the record payload, so
compare changes on the same host rather than treating one throughput number as
a portable guarantee.
