## Vettore Benchmarks

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

## L2 duplicates and underflow recovery

```bash
cargo bench --manifest-path native/vettore/Cargo.toml --bench l2_duplicates --locked
```

This harness uses standard-library timing to measure the production L2 kernel on identical,
nearly identical, and distinct inputs at 384, 768, and 769 dimensions. Inputs
use separate allocations and `black_box`; vector generation, NIF conversion,
ranking, and ETS hydration are outside the timed loop. Each result is the median
of seven rounds of 200,000 calls after warmup, with min/max also reported.

On 2026-10-18, Rust 1.93.1 release builds on an Intel Core Ultra 5 125H, pinned
to CPU 0 with `taskset -c 0`, produced these medians (nanoseconds per call):

| Dimensions | Identical, f64 recovery | Identical, byte-equality shortcut | Near duplicate, before / after | Distinct, before / after |
| --- | ---: | ---: | ---: | ---: |
| 384 | 606.8 | 283.6 | 268.7 / 265.3 | 262.3 / 258.3 |
| 768 | 1202.9 | 551.4 | 529.7 / 517.2 | 534.3 / 519.0 |
| 769 | 1203.8 | 553.7 | 530.6 / 524.6 | 529.4 / 522.2 |

The baseline is the same release source with only the identical-input shortcut
removed. Identical inputs avoid a second arithmetic pass, while distinct inputs
whose squares underflow still recover in f64. The comparison itself still reads
the inputs. These are local kernel measurements, not end-to-end collection
speedups; rerun on the deployment CPU and representative duplicate frequencies.

## Orthogonal dot products

```bash
cargo bench --manifest-path native/vettore/Cargo.toml --bench dot_orthogonal --locked
```

This compares the production dot kernel with its own raw f32 reduction, using
one-hot orthogonal vectors, disjoint sparse vectors, dense cancellation, normal
nonzero products, and tiny representable products. The raw reduction is only a
timing baseline: the tiny-product case verifies that it loses a representable
score which the production kernel recovers. IP and NegIP share this dot kernel.

On the same CPU/toolchain described above, pinned to CPU 0, medians of seven
rounds of 100,000 calls were (nanoseconds per call, raw / recovered):

| Dimensions | One-hot orthogonal | Sparse orthogonal | Dense cancellation | Dense nonzero |
| --- | ---: | ---: | ---: | ---: |
| 384 | 184.8 / 441.7 | 186.1 / 450.4 | 189.1 / 449.2 | 186.4 / 192.0 |
| 768 | 376.4 / 884.3 | 378.9 / 888.0 | 386.6 / 909.7 | 390.6 / 383.6 |
| 769 | 375.5 / 892.8 | 380.5 / 883.7 | 375.2 / 887.3 | 384.6 / 378.6 |
| 1536 | 765.7 / 1753.7 | 754.1 / 1790.8 | 758.2 / 1788.5 | 751.8 / 757.3 |

Exact zero results still trigger the f64 recovery pass and cost about 2.3–2.4
times the raw reduction here. This remains a performance tradeoff for sparse
workloads: an f32 zero alone cannot distinguish orthogonality from lost tiny
products. These measurements do not imply an end-to-end search slowdown of the
same magnitude. Normal nonzero products stay on the SIMD path.

## Resident GPU Flat versus contiguous SIMD Flat

Run the dedicated exact-search benchmark on a machine with a supported adapter:

```bash
VETTORE_BENCH_BATCH=25000 \
VETTORE_BENCH_DIMENSIONS=384 \
VETTORE_BENCH_LIMIT=10 \
mix run bench/gpu_flat_bench.exs
```

The script builds identical CPU and GPU collections, verifies rank-by-rank score
parity and every unambiguous CPU id before timing, and permits id permutations
only inside an f32-tied top-k boundary. It reports the first GPU query separately
(resident-matrix upload plus search), then compares warm resident queries
against the contiguous SIMD scan. The useful break-even point depends on
adapter, driver, dimensions, collection size, query concurrency, and mutation
frequency. Measure warm and cold numbers separately: a workload that rebuilds
the matrix after every write does not have the same economics as a read-heavy
collection loaded with one `put_many/2`.

Available settings are `VETTORE_BENCH_DIMENSIONS`, `VETTORE_BENCH_BATCH`,
`VETTORE_BENCH_LIMIT`, `VETTORE_BENCH_METRIC`, `VETTORE_BENCH_SEED`,
`VETTORE_BENCH_TIME`, and `VETTORE_BENCH_WARMUP`.

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
