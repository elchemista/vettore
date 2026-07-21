# Changelog

All notable changes to Vettore are documented here. The project follows
Semantic Versioning.

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
