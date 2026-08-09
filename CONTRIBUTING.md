# Contributing to OmniToken

## Before you start

Read [`tokenizer-project-plan.md`](tokenizer-project-plan.md), especially:
- **§0 Honesty baseline** — the benchmark rules that gate every performance claim.
- **§4.8 Roofline sanity check** — the physical ceilings that validate every throughput number.
- **§8 Module structure** — the dependency direction; don't invert it.

## Workflow

```
cargo test --workspace    # must pass before any PR
```

- Each non-trivial code path needs at least one `#[test]` that fails if the logic breaks.
- Benchmark claims: single-thread before multi-thread; in-memory and disk-streaming separately.
  Never compare published gigatoken numbers from different hardware to your local run.
- Mark deliberate simplifications with `// ponytail:` comments naming the ceiling and upgrade path.
- New crates need a clear justification: does an existing crate already cover this need?

## Phase ownership

| Phase | Pick up when |
|-------|-------------|
| 2 — WordPiece/Unigram | Phase 1 benchmarks confirm direction |
| 3 — MPHF cache | Offline corpus profiling data exists; hit-rate sweep done |
| 4 — SIMD tuning | Phase 3 cache baseline measured; SWAR vs AVX2 vs AVX-512/VNNI benchmarked per-op |
| 5 — Full validation | Phase 4 hardware numbers locked |
| 6 — Launch | All prior phases done; numbers checked against roofline |

## Build config caveat

`target-cpu=native` + `lto=fat` is known to hang on some toolchain versions
(rust-lang/rust #49766). If blocked, swap to `x86-64-v3` in `.cargo/config.toml`.
