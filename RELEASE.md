# Releasing Vettore 0.3.2

The Elixir project, Rust crate, lockfile, README, and changelog are prepared for
`0.3.2`. The checked-in checksum file still describes `0.3.1` artifacts until
the new cross-platform archives exist; never publish the Hex package with that
old checksum map.

## 1. Verify the source release

From a clean checkout with Rust installed:

```bash
VETTORE_BUILD=1 mix deps.get --locked
VETTORE_BUILD=1 mix compile --warnings-as-errors
VETTORE_BUILD=1 mix format --check-formatted
VETTORE_BUILD=1 mix test --cover --warnings-as-errors
VETTORE_BUILD=1 \
VETTORE_BENCH_DIMENSIONS=16 \
VETTORE_BENCH_BATCH=64 \
VETTORE_BENCH_LIMIT=5 \
VETTORE_BENCH_CANDIDATES=32 \
VETTORE_BENCH_TIME=1 \
VETTORE_BENCH_WARMUP=0 \
mix run bench/search_modes_bench.exs
VETTORE_BUILD=1 mix credo --strict
VETTORE_BUILD=1 mix dialyzer
cargo fmt --manifest-path native/vettore/Cargo.toml --all --check
cargo test --manifest-path native/vettore/Cargo.toml --locked
cargo llvm-cov --manifest-path native/vettore/Cargo.toml --all-features --ignore-filename-regex 'src/nifs\.rs' --summary-only --fail-under-lines 98
cargo clippy --manifest-path native/vettore/Cargo.toml --all-targets --all-features --locked -- -D warnings
VETTORE_BUILD=1 mix docs
VETTORE_BUILD=1 mix hex.build
```

Confirm CI is green and `git diff --check` has no output.

The Rust coverage gate excludes only `src/nifs.rs`: Rustler macro-generated NIF
entry functions are outside Cargo unit-test instrumentation. Those public
boundaries are exercised through the BEAM by the Elixir integration suite;
algorithm modules remain subject to the 98% Rust line threshold.

The benchmark smoke run must preflight every search mode successfully and print
overlap against the exact vector and multi-vector baselines before timing.

## 2. Build and publish native archives

Merge the release commit, then create and push the matching tag:

```bash
git tag -s v0.3.2 -m "Vettore 0.3.2"
git push origin v0.3.2
```

The `Build precompiled NIFs` workflow rejects a tag that differs from the
version in `mix.exs`. It builds every configured NIF/target pair, creates the
GitHub release, and attaches both the archives and
`checksum-Elixir.Vettore.Nifs.exs`.

## 3. Install and verify the generated checksums

After the release workflow succeeds:

```bash
gh release download v0.3.2 \
  --pattern 'checksum-Elixir.Vettore.Nifs.exs' \
  --clobber
git diff -- checksum-Elixir.Vettore.Nifs.exs
```

Every checksum key must contain `v0.3.2`. Commit the generated checksum file.
Then verify a clean precompiled build without `VETTORE_BUILD`:

```bash
MIX_ENV=prod mix clean
MIX_ENV=prod mix compile --force
mix test
mix hex.build
mix hex.publish --dry-run
```

The production compile must download and load a `0.3.2` archive successfully.
Inspect the Hex package listing and confirm it contains the new checksum,
`CHANGELOG.md`, `RELEASE.md`, Elixir sources, and Rust sources.

## 4. Publish and smoke-test

```bash
mix hex.publish
```

In a fresh temporary Mix project, depend on `{:vettore, "~> 0.3.2"}` without
Rust installed. Create flat and HNSW collections, insert records with metadata,
search them, snapshot/reload them, and call `Vettore.close/1`. Finally, mark the
changelog entry with the release date.
