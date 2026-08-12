# FastEmbed test artifact

`fastembed_bge_small_en_v1_5.etf` contains document and query vectors produced
by `BAAI/bge-small-en-v1.5` through `ex_fastembed`. The normal test suite reads
this deterministic artifact and does not access the network or model cache.

Regenerate it after an intentional model or `ex_fastembed` update with:

```bash
MIX_ENV=test VETTORE_BUILD=1 \
  mix run test/support/generate_fastembed_fixture.exs
```

The CI integration test also generates fresh vectors and compares them with
the committed artifact.
