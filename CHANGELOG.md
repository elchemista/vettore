# Changelog

All notable changes to Vettore are documented here. The project follows
Semantic Versioning.

## [Unreleased]

## [0.3.5] - 2026-08-25

### Added

- Added `Vettore.Vector` as a validated interchange layer for numeric lists,
  little-endian f32 binaries, dimensioned vector wrappers, and host-provided Nx
  tensors.
- Added native metrics, normalization, and row-selective mean pooling directly
  over little-endian f32 binaries, including Model2Vec-style token-table
  pooling without materializing the full matrix on the BEAM.
- Added the zero-dependency `Vettore.Interop.Nx` runtime adapter. Vettore does
  not declare Nx as a dependency; applications that already use Nx can opt
  into tensor conversion without changing Vettore's core runtime.
- Added native wgpu compute kernels for every dense metric, L2/z-score/min-max
  normalization, and row-selective mean pooling. CPU SIMD remains the default;
  `gpu: true` and `gpu: :auto` can be configured globally or per call with an
  explicit CPU/error fallback policy.
- Added `Vettore.gpu_detected?/0`, `Vettore.gpu_info/0`, and
  `Vettore.Compute.info/0` for hardware detection and runtime diagnostics.
- Added shape-preserving wrappers and Nx conversion, backend transfer/type
  helpers, plus matrix `stack/2`, `take_rows_f32/4`, shape, and full-validation
  APIs.
- Added an exact GPU Flat search path with a generation-aware resident embedding
  matrix, one-query-by-all-rows scoring shaders for every metric, device-side
  two-stage top-k reduction, and readback of only the final ids and scores.
- Added batched GPU scoring for exact adaptive/hybrid reranks and a dedicated
  cold-upload versus warm-query Flat benchmark.

### Changed

- Replaced the Flat index's fragmented vector map with a contiguous row-major
  matrix plus id-to-row map. CPU exact scans retain portable SIMD kernels with
  improved locality, while mutations atomically invalidate the optional GPU
  snapshot.
- Extended every `Vettore.Distance` dense metric and normalization helper with
  per-call compute options while preserving existing arities.
- Nx remains runtime-only and absent from both Mix and Cargo dependencies; GPU
  execution is implemented entirely in Rust and does not use Nx.
- Vectorized the f64 accumulation used by stable cosine scoring and exceptional
  overflow recovery, so numerical safeguards no longer force those hot loops
  back to scalar iteration.
- Flat index options now accept `:gpu`, `:gpu_min_size`, and `:gpu_fallback`.
  Automatic selection uses the real `rows * dimensions` search workload.
- GPU Flat queries reuse bounded pools of query, score, top-k, uniform, bind
  group, and staging resources. Concurrent warm searches do not hold the cache
  construction lock while dispatching.
- GPU calls run concurrently without a process-wide execution mutex. Runtime
  failures invalidate the cached device, initialization failures use a bounded
  retry window, and readback waits default to a configurable 10-second timeout.

### Fixed

- Stabilized GPU metrics across the finite f32 range by scaling operands before
  shader reduction and rescaling checked results on the host. This fixes cosine
  overflow/underflow, cancelling extreme dot products, and large L1/L2 inputs.
- Replaced one-pass f32 normalization statistics with two-pass f64 host
  preparation, fixing tiny L2 vectors and catastrophic z-score cancellation.
- Guarded dispatch counts and storage bindings against the selected device
  limits before submitting work, preventing wgpu validation panics on oversized
  vectors and matrices.
- Made `gpu: :auto` use CPU when no adapter is available even when the forced-GPU
  fallback policy is `:error`, and normalized runtime failures to stable atoms.
- Added direct GPU NIF paths for little-endian f32 binaries, structural binary
  dimension checks, GPU-backed CI with lavapipe, Vector/Compute doctests, and an
  explicit Rust 1.91 MSRV check.
- Aligned transient CPU and GPU top-k overflow semantics with Flat search: only
  the unrepresentable row is skipped, while valid candidates are retained.
- Added explicit per-score validity in resident shaders so overflow cannot be
  mistaken for a zero-valued candidate, including at the finite f32 boundary.
- Made hardware-dependent parity tests and benchmark preflights aware of f32
  reduction-order ties. Scores and unambiguous ranks remain checked, while ids
  may permute inside a numerically tied boundary group.
- Made the resident GPU benchmark parse metrics through a closed lookup table,
  fixing its CI preflight in a fresh BEAM where the requested metric atom did
  not already exist.
- Added strict resident-GPU parity coverage over committed and freshly inferred
  384-dimensional `BAAI/bge-small-en-v1.5` document/query embeddings, including
  semantic retrieval and resident-cache reuse assertions.
- Rebuilt resident buffers after every effective mutation and cached
  deterministic build failures by index generation, preventing repeated full
  snapshots for an unchanged unsupported matrix.

### Performance

- GPU devices and compiled pipelines are initialized lazily and reused. CPU
  calls skip GPU detection entirely, and GPU mean pooling uploads only selected
  rows rather than the complete model matrix.
- Flat snapshot sorting, device upload, dispatch, and readback now happen
  outside the index read lock; writers are blocked only while the immutable host
  snapshot is copied.
- Replaced the single-thread-per-chunk resident top-k pass with a 16-lane local
  reduction followed by the compact final merge.
- Parallelized GPU mean pooling across 256 lanes per output column and scaled
  each selected column before accumulation, reducing drift and avoiding
  representable means overflowing during intermediate f32 sums.
- Cached failed GPU runtime initialization for ten seconds so hosts without an
  adapter do not enumerate and request a device on every fallback call.

## [0.3.4] - 2026-08-23

### Fixed

- Updated the local Rust toolchain pin to the crate's Rust 1.91 MSRV.
- Corrected ExDoc's landing page and source links for versioned releases.
- Replaced panic-prone HNSW graph lookups with recoverable errors and made
  flat/HNSW searches skip individual rows whose score overflows.
- Serialized ETS snapshots with writes so snapshots represent one consistent
  point in time.
- Aligned collection metric aliases with the compatibility constructor and
  made empty-id reads and deletes return `{:error, :invalid_id}` consistently.

### Changed

- Changed the `Vettore.new/1` default from `score: :raw` to
  `score: :similarity`, matching the compatibility API. This changes
  `Result.score` values for callers that omitted the option; pass
  `score: :raw` to preserve the previous scale. Existing snapshots retain the
  score mode stored in their configuration.
- Collections and compatibility databases are now reclaimed automatically when
  the process that created them exits. Long-lived resources must be created by
  a long-lived owner rather than handed off from a short-lived task or request
  process.

### Security and reliability

- Capped implicit adaptive-search candidate counts at 1,000,000 and reject
  adaptive searches whose result limit exceeds that bound. Explicit candidate
  counts above the bound, previously accepted, now return
  `{:error, :invalid_candidates}`.
- Added Hex and RustSec dependency audits to CI.

## [0.3.3] - 2026-08-12

### Fixed

- Rejected non-UTF-8 ids before ETS writes and Rustler decoding, preventing a
  failed insert from leaving the canonical store and native index inconsistent.
- Made cosine collection ranking compute true cosine even when normalization is
  explicitly disabled or uses a non-L2 transform.
- Kept quantized Hamming and Jaccard candidate selection consistent with their
  non-zero truth semantics.
- Serialized ETS-backed store/index mutations so concurrent puts and deletes of
  the same id cannot leave phantom native entries.
- Corrected compatibility insert and batch return ids when an empty `id` falls
  back to `value`.

### Changed

- Centralized index lifecycle, mutation, and search boundaries to remove cyclic
  module dependencies and keep flat and HNSW behavior aligned.
- Added reverse HNSW edges, diversified pruning, local reconnection after
  deletes, and fresh search bounds to improve recall and deletion cost.
- MMR now runs in one native batch instead of one NIF transition per pair.
- Hybrid search reuses one ETS snapshot across adaptive generators, multi-stage
  funnel search narrows candidates progressively, and `close/1` clears native
  index memory immediately.

### Tests and release engineering

- Raised enforced Elixir line coverage to 99%; the full suite reaches 99.36%
  with 175 passing checks.
- Added a committed embedding fixture generated by the real `ex_fastembed`
  backend, plus deterministic offline integration tests and an opt-in fixture
  regeneration script.
- Updated Elixir and Rust dependency locks, including `wide` 1.6.1.

## [0.3.2] - 2026-07-21

### Fixed

- Fixed HNSW reciprocal-link pruning that left almost every newly inserted node
  unreachable and caused severe recall loss.
- Hydrated HNSW and flat results from canonical ETS records so `value` and
  `metadata` are preserved, while stale native ids are ignored safely.
- Normalized representative vectors derived from multi-vector records before
  cosine indexing.
- Corrected `:negative_inner_product` score conversion and MMR similarity
  semantics, including malformed or missing MMR ids.
- Replaced raising input paths with tagged errors for collection options,
  vectors, adaptive search, multi-vector scoring, and MUVERA configuration.
- Validated custom-store records before adaptive NIF calls so malformed ids,
  duplicate rows, dimensions, and out-of-f32 values return tagged errors rather
  than raising during Rustler decoding.
- Made snapshot index overrides persistent when a loaded collection is
  snapshotted again; added schema, record, checksum, and corruption validation.
- Loaded legacy public-table snapshots under protected ownership instead of
  preserving unsafe table permissions.
- Prevented collection-table leaks when index construction, snapshot restore,
  duplicate compatibility creation, or database deletion fails.
- Made compatibility-database shutdown atomic with concurrent collection
  creation and removed the unsafe timeout on large owner-mediated batches.
- Deleted drained ETS tables before acknowledging database shutdown so callers
  consistently observe `{:error, :closed}` immediately after `close/1`.
- Added overflow-safe L2, cosine, z-score, and min-max normalization for extreme
  finite f32 values.
- Recovered valid large L2 and cancelling dot-product results with f64 fallback,
  while rejecting genuinely unrepresentable squared distances and MaxSim sums.
- Propagated metric failures through MMR and late-interaction reranking instead
  of silently substituting misleading zero scores.
- Bounded MUVERA seed decoding and rejected non-finite encoding accumulation
  instead of returning corrupted vectors.
- Kept Rust panics unwindable in release NIFs so Rustler can contain them
  instead of aborting the BEAM VM.

### Changed

- ETS tables are now owned by supervised temporary workers instead of the
  process that creates a collection. Tables survive caller exit and are
  protected against out-of-band writes.
- Added idempotent `Vettore.close/1` for deterministic collection and
  compatibility-database cleanup.
- Collection construction and search reject unknown, duplicated, or malformed
  options instead of silently ignoring them.
- Snapshot files are written through a temporary file and include ETS object
  count and MD5 integrity metadata.
- Local Rust builds are now explicitly enabled with `VETTORE_BUILD=1`; normal
  dependency builds use the published precompiled NIFs.

### Performance

- Protected ETS tables keep concurrent-read optimization enabled while avoiding
  unused concurrent-writer bookkeeping behind their single supervised owner.
- Direct ETS reads no longer perform redundant owner/table liveness probes;
  closed-table races are handled on the actual operation instead.
- Default cosine helper calls now use one overflow-safe native kernel instead
  of two normalization calls followed by a third metric call.
- Exact flat search computes each metric once and retains top-k results with a
  bounded heap instead of sorting every record.
- HNSW batches inserts under one native write lock.
- Funnel, quantized, hybrid, and multi-vector paths batch candidate scoring into
  dirty CPU NIF calls, avoiding one NIF transition per record or vector pair.

### Tests and release engineering

- Added 50 Rust unit tests covering every metric, scalar/SIMD differential
  checks, all top-k limits, HNSW graph invariants and recall, packed-bit word
  boundaries, multi-vector scoring, and MUVERA safety. Algorithm modules exceed
  99% Rust line coverage.
- Expanded the Elixir suite to 161 passing checks, including 60 doctests,
  fault-injected failure paths, concurrent readers and writers, and 98%+ line
  coverage (excluding unmeasurable NIF fallback stubs).
- Added real `BAAI/bge-small-en-v1.5` exact/HNSW/hybrid integration coverage.
- Added deterministic latency and overlap benchmarks for every search mode,
  direct MaxSim, MUVERA encodings, and the ETS read/write ownership boundary.
- Added pull-request CI for enforced Elixir and Rust algorithm coverage, Credo,
  Dialyzer, docs, Hex package contents, Rust formatting, tests, and Clippy.
- Added explicit Cargo feature forwarding and CI checks for Rustler NIF 2.15 and
  2.16 precompiled artifact builds.
- Updated Rustler to 0.38, RustlerPrecompiled to 0.9, and `wide` to 1.5. Local
  native builds now require Rust 1.91 or newer.
- Isolated the real `ex_fastembed` integration dependency from published Hex
  metadata so its older Rustler constraint cannot block application updates.
- Made the complete precompiled-NIF matrix manually runnable from a release
  branch without creating or publishing a GitHub Release.
- The tag release workflow now validates version/tag parity and publishes a
  generated Rustler checksum file with the native archives.

## [0.3.1]

- Previous release.
