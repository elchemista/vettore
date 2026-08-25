# Releasing Vettore

Until the release workflow finishes, the checked-in checksum map describes the
currently published native artifacts. Native-code changes require rebuilding
the complete target matrix and replacing that map before a new package is
published.

## 1. Verify the source release

From a clean checkout with Rust 1.91 or newer installed:

```bash
VETTORE_BUILD=1 mix deps.get --locked
MIX_ENV=test VETTORE_TEST_EX_FASTEMBED=1 mix deps.get --locked
VETTORE_BUILD=1 mix compile --warnings-as-errors
VETTORE_BUILD=1 mix format --check-formatted
VETTORE_BUILD=1 \
VETTORE_TEST_EX_FASTEMBED=1 \
VETTORE_GPU_ALLOW_SOFTWARE=1 \
VETTORE_REQUIRE_GPU=1 \
mix test --cover --warnings-as-errors
VETTORE_BUILD=1 \
VETTORE_BENCH_DIMENSIONS=16 \
VETTORE_BENCH_BATCH=64 \
VETTORE_BENCH_LIMIT=5 \
VETTORE_BENCH_CANDIDATES=32 \
VETTORE_BENCH_TIME=1 \
VETTORE_BENCH_WARMUP=0 \
mix run bench/search_modes_bench.exs
VETTORE_BUILD=1 \
VETTORE_GPU_ALLOW_SOFTWARE=1 \
VETTORE_BENCH_DIMENSIONS=16 \
VETTORE_BENCH_BATCH=64 \
VETTORE_BENCH_LIMIT=5 \
VETTORE_BENCH_TIME=1 \
VETTORE_BENCH_WARMUP=0 \
mix run bench/gpu_flat_bench.exs
VETTORE_BUILD=1 mix credo --strict
VETTORE_BUILD=1 mix dialyzer
mix hex.audit
cargo fmt --manifest-path native/vettore/Cargo.toml --all --check
VETTORE_GPU_ALLOW_SOFTWARE=1 VETTORE_REQUIRE_GPU=1 \
cargo test --manifest-path native/vettore/Cargo.toml --locked
cargo check --manifest-path native/vettore/Cargo.toml --locked --no-default-features --features nif_version_2_15
cargo check --manifest-path native/vettore/Cargo.toml --locked --no-default-features --features nif_version_2_16
VETTORE_GPU_ALLOW_SOFTWARE=1 VETTORE_REQUIRE_GPU=1 \
cargo llvm-cov --manifest-path native/vettore/Cargo.toml --all-features --ignore-filename-regex 'src/(nifs|gpu)\.rs' --summary-only --fail-under-lines 98
cargo clippy --manifest-path native/vettore/Cargo.toml --all-targets --all-features --locked -- -D warnings
VETTORE_BUILD=1 mix docs
VETTORE_BUILD=1 mix hex.build
```

Confirm CI is green and `git diff --check` has no output.

The Rust coverage gate excludes the external runtime boundaries in
`src/nifs.rs` and `src/gpu.rs`: Rustler entry functions and hardware/driver I/O
contain environment-specific failure branches. Those boundaries are exercised
through the BEAM and an available hardware or software wgpu adapter. Pure GPU
validation/reduction logic lives in `src/gpu_math.rs` and remains subject to the
98% Rust line threshold with every other algorithm module.

The benchmark smoke runs must preflight every search mode and validate GPU Flat
scores plus all unambiguous ids before timing. Ids inside an f32-tied top-k
boundary may differ across adapters and reduction orders.

Before creating the tag, open **Actions → Build precompiled NIFs → Run
workflow**, select the release branch, and run it manually. A manual run builds
and uploads the complete NIF/target artifact matrix for inspection but skips the
GitHub Release publishing job. Only a tag matching the version in `mix.exs`
publishes assets.

## 2. Build and publish native archives

Merge the release commit, then create and push the matching tag:

```bash
release_version="$(mix run --no-start -e 'IO.write(Mix.Project.config()[:version])')"
git tag -s "v${release_version}" -m "Vettore ${release_version}"
git push origin "v${release_version}"
```

The `Build precompiled NIFs` workflow rejects a tag that differs from the
version in `mix.exs`. It builds every configured NIF/target pair, creates the
GitHub release, and attaches both the archives and
`checksum-Elixir.Vettore.Nifs.exs`.

## 3. Install and verify the generated checksums

After the release workflow succeeds:

```bash
release_version="$(mix run --no-start -e 'IO.write(Mix.Project.config()[:version])')"
gh release download "v${release_version}" \
  --pattern 'checksum-Elixir.Vettore.Nifs.exs' \
  --clobber
git diff -- checksum-Elixir.Vettore.Nifs.exs
```

Every checksum key must contain the current release version. Commit the
generated checksum file. Verify that mechanically before building the package:

```bash
release_version="$(mix run --no-start -e 'IO.write(Mix.Project.config()[:version])')"
test "$(grep -c '=>' checksum-Elixir.Vettore.Nifs.exs)" -gt 0
test "$(grep -vc -- "-v${release_version}-" checksum-Elixir.Vettore.Nifs.exs)" -eq 2
```

The two non-matching lines are the opening and closing map delimiters. Then
verify a clean precompiled build without `VETTORE_BUILD`:

```bash
MIX_ENV=prod mix clean
MIX_ENV=prod mix compile --force
mix test
mix hex.build
mix hex.publish --dry-run
```

The production compile must download and load the current release archive
successfully.
Inspect the Hex package listing and confirm it contains the new checksum,
`CHANGELOG.md`, `RELEASE.md`, Elixir sources, and Rust sources.

## 4. Publish and smoke-test

```bash
mix hex.publish
```

In a fresh temporary Mix project, depend on `{:vettore, "~> 0.3.5"}` without
Rust installed. Create flat and HNSW collections, insert records with metadata,
search them, snapshot/reload them, and call `Vettore.close/1`. Finally, mark the
changelog entry with the release date if it was not already finalized in the
release commit.
