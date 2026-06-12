# Vettore Native

This crate contains only native acceleration code for Vettore.

Canonical collection storage lives in Elixir/ETS. The Rust crate must not own
database state or collection records.

Current native surface:

- distance and similarity kernels
- vector normalization
- sign-bit compression
- native HNSW index resource
- MUVERA/FDE query encoding
- MUVERA/FDE document encoding

Removed from the native crate:

- Rust-owned vector database
- Rust-owned collection storage
- old similarity search wrappers over Rust collections
- old MMR wrappers over Rust collections
