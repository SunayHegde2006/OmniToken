<p align="center">
  <img src="assets/banner.svg" alt="OmniToken Banner" width="100%">
</p>

<p align="center">
  <b>Universal, research-grounded tokenizer engine for BPE, WordPiece, and Unigram vocabularies — targeting consumer hardware (AMD Ryzen 5 7600, DDR5-5600, Gen3 NVMe).</b>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT licensed"></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/rust-1.85%2B-orange.svg" alt="Rust"></a>
  <a href="#empirical-benchmark-results"><img src="https://img.shields.io/badge/throughput-1.77%20GB%2Fs-success.svg" alt="Throughput"></a>
  <a href="https://github.com/SunayHegde2006/OmniToken"><img src="https://img.shields.io/badge/version-v0.1.0-green.svg" alt="Version"></a>
</p>

---

## Overview

OmniToken is a high-performance, universal tokenization engine written in Rust. It ingests every major tokenizer vocabulary format into one universal intermediate representation (`VocabIr`) and encodes with a unified automaton that executes **BPE**, **WordPiece**, and **Unigram** in the same trie-walk loop.

- **Primary Competitor:** [gigatoken](https://github.com/marcelroed/gigatoken) — BPE engine benchmarked on a 144-core server.
- **Our Wedge:** Universal format support (BPE, WordPiece, Unigram, tiktoken, SentencePiece, GGUF) + inference-time low latency + SWAR/AVX2 pretokenization + hybrid hot-tier cache grounded in PtrHash literature.

---

## Key Features

- ⚡ **1.77+ GB/s Multi-Core Throughput**: Scaled across 6 physical cores / 12 SMT threads on consumer DDR5 hardware.
- 🎯 **Universal Vocab IR (`vocab-ir`)**: Ingest HuggingFace `tokenizers.json` (BPE/WordPiece/Unigram), tiktoken `.tiktoken` files, SentencePiece JSON, and GGUF metadata.
- 🔄 **Unified Automaton (`walker`)**: One trie walker handles BPE priority queues ($O(N \log M)$ per Zouhar et al.), WordPiece LinMaxMatch ($O(N)$ per Song et al.), and Unigram Viterbi DP.
- 🚀 **SWAR / SIMD Pretokenizer (`pretokenizer`)**: 256-entry byte-class table with 8-byte u64 branchless SWAR dispatch and AVX2 vectorization (`x86-64-v3`).
- 💎 **Hybrid Hot-Tier Cache (`hot-cache`)**: Lock-free RCU double-buffered static tier, XXH3-64 fingerprint verification, 64-byte padded `CountMinSketch`, and SwissTable fallback.
- 📊 **Roofline-Grounded Benchmarking (`bench-harness`)**: Automated L3-resident vs DRAM-resident throughput validation against physical hardware bandwidth limits.

---

## Command Line Interface & Flags

<details>
<summary><strong>⚙️ <code>omnitoken</code> — CLI Options & Subcommands</strong></summary>

**`omnitoken encode`** — Tokenize input text from stdin or string:
```bash
omnitoken encode --vocab <path> [OPTIONS]
```

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--vocab` | `-v` | path | **required** | Path to `tokenizers.json` or `.tiktoken` file. |
| `--input` | `-i` | string | `stdin` | Direct input text string to tokenize. |

</details>

<details>
<summary><strong>⚙️ <code>bench</code> — Benchmarking & Roofline Harness</strong></summary>

**`bench`** — Measure throughput and cross-check roofline physical ceilings:
```bash
bench --vocab <path> [OPTIONS]
```

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--vocab` | `-v` | path | **required** | Path to HuggingFace `tokenizers.json`. |
| `--corpus` | `-c` | path | `synthetic` | Optional path to text corpus file. |
| `--threads` | `-t` | int | `1` | Number of Rayon threads (1 = single-thread baseline). |
| `--bytes` | `-b` | int | `16777216` | Bytes of synthetic corpus to generate if no file provided. |
| `--parity` | `-p` | flag | `off` | Run token-by-token parity check against vocabulary table. |
| `--mmap` | | flag | `off` | Enable memory-mapped file reader for disk streaming. |

</details>

---

## System Architecture

### Workspace Directory Layout

```text
crates/
├── vocab-ir/        # Universal IR loader (HF tokenizers.json, tiktoken, SPM, GGUF)
├── trie-builder/    # Offline Aho-Corasick trie + failure links & continuation prefixes
├── pretokenizer/    # SWAR / AVX2 256-entry byte classifier & UTF-8 chunk splitter
├── walker/          # Unified automaton: O(N log M) BPE, MaxMatch WordPiece, Viterbi Unigram
├── hot-cache/       # Hybrid MPHF static tier + CountMin sketch + SwissTable overflow
├── omnitoken/       # Unified CLI binary + mimalloc allocator configuration
└── bench-harness/   # Reproducible roofline-checked benchmark harness
```

### Dependency Architecture

```text
vocab-ir ────────► trie-builder ──────► walker ◄───── pretokenizer
                     │                     ▲
                     └─────────────────────┼───────── hot-cache
                                           │
                                     omnitoken / bench-harness
```

---


## Empirical Benchmark Results

> **Hardware Environment:** AMD Ryzen 5 7600 (6 Cores / 12 SMT Threads @ 5.1 GHz), Dual-Channel DDR5-5600, Ubuntu Linux 24.04 LTS (WSL2).  
> **Compiler:** `rustc 1.85+`, `--release` (`lto = "fat"`, `codegen-units = 1`, `target-cpu = x86-64-v3`).

```text
Throughput vs. Thread Count (DRAM-Resident 64MiB Input)
========================================================================
 1 Thread  [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0.295 GB/s
 2 Threads [██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  0.499 GB/s
 4 Threads [████████████████████████████░░░░░░░░░░░░░░░░░░░]  1.045 GB/s
 6 Threads [████████████████████████████████████████░░░░░░░]  1.399 GB/s
12 Threads [███████████████████████████████████████████████]  1.771 GB/s
========================================================================
```

### 1. L3-Resident Regime (4.0 MiB Input ≤ 32MB L3 Cache)

| Threads | Input Size | Wall Time (s) | Throughput (GB/s) | Throughput (MiB/s) | Physical Ceiling | Validation |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 4.0 MiB | 0.0150s | **0.279 GB/s** | 266 MiB/s | L3 Bandwidth | ✓ Plausible |
| **2** | 4.0 MiB | 0.0084s | **0.498 GB/s** | 475 MiB/s | L3 Bandwidth | ✓ Plausible |
| **4** | 4.0 MiB | 0.0043s | **0.983 GB/s** | 937 MiB/s | L3 Bandwidth | ✓ Plausible |
| **6** | 4.0 MiB | 0.0034s | **1.242 GB/s** | 1184 MiB/s | L3 Bandwidth | ✓ Plausible |
| **12** | 4.0 MiB | 0.0042s | **0.993 GB/s** | 947 MiB/s | L3 Bandwidth | ✓ Saturation Point |

### 2. DRAM-Resident Regime (64.0 MiB Input > 32MB L3 Cache)

| Threads | Input Size | Wall Time (s) | Throughput (GB/s) | Throughput (MiB/s) | Physical Ceiling | Validation |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 64.0 MiB | 0.2275s | **0.295 GB/s** | 281 MiB/s | ≈63–80 GB/s (DDR5-5600) | ✓ Plausible |
| **2** | 64.0 MiB | 0.1344s | **0.499 GB/s** | 476 MiB/s | ≈63–80 GB/s (DDR5-5600) | ✓ Plausible |
| **4** | 64.0 MiB | 0.0642s | **1.045 GB/s** | 997 MiB/s | ≈63–80 GB/s (DDR5-5600) | ✓ Plausible |
| **6** | 64.0 MiB | 0.0480s | **1.399 GB/s** | 1334 MiB/s | ≈63–80 GB/s (DDR5-5600) | ✓ Plausible |
| **12** | 64.0 MiB | 0.0379s | **1.771 GB/s** | 1689 MiB/s | ≈63–80 GB/s (DDR5-5600) | ✓ Plausible |

---

## Roofline Sanity Matrix

| Resource | Hardware Spec | Sustained Physical Ceiling | OmniToken Status |
|----------|---------------|---------------------------|------------------|
| **L3 Cache** | 32 MB shared (Zen 4 CCD) | ~50-cycle latency (~8–9 ns) | Verified L3 resident at 4.0 MiB |
| **DRAM** | Dual-Channel DDR5-5600 | ≈63–80 GB/s sustained | Verified DRAM resident at 64.0 MiB |
| **NVMe** | PCIe Gen3 x4 | ≈3.5 GB/s sequential read | In-memory processing path |
| **CPU FPU** | 6C / 12T Zen 4 | 256-bit AVX2 vector execution | SWAR + AVX2 (`x86-64-v3`) active |

---

## Quick Start & Usage

### 1. Download Standard Vocab

```bash
pip install tokenizers
python3 -c "from tokenizers import Tokenizer; Tokenizer.from_pretrained('gpt2').save('gpt2.json')"
```

### 2. Build Release Binaries

```bash
cargo build --release
```

### 3. Run OmniToken CLI

```bash
echo "the quick brown fox jumps over the lazy dog" | ./target/release/omnitoken encode --vocab gpt2.json
```

### 4. Run Benchmark Harness

```bash
# Single-thread baseline benchmark
./target/release/bench --vocab gpt2.json --threads 1 --bytes 16777216

# Multi-thread 6-core scaling benchmark
./target/release/bench --vocab gpt2.json --threads 6 --bytes 67108864

# Parity verification mode
./target/release/bench --vocab gpt2.json --parity
```

---

## Roadmap & Implementation Status

| Phase | Status | Deliverable | Description |
|-------|--------|-------------|-------------|
| **0** | ✅ **Done** | Literature Synthesis | Ground-truth memory latency & bandwidth tables (Zen 4 CCD, DDR5-5600, NVMe). |
| **1** | ✅ **Done** | BPE Prototype & Harness | Workspace initialization, zero-copy BPE encoder, and roofline benchmark harness. |
| **2** | ✅ **Done** | Unified Automaton Engine | Full support for BPE priority queues, WordPiece MaxMatch trie walk, and Unigram Viterbi DP. |
| **3** | ✅ **Done** | Hybrid Hot Cache | Double-buffered static MPHF tier, 64-byte aligned CountMin sketch, and SwissTable overflow. |
| **4** | ✅ **Done** | SIMD Pretokenizer & Hardware | SWAR byte-classification, `x86-64-v3` AVX2 vectorization, LTO/PGO build config. |
| **5** | ✅ **Done** | Validation & Verification | Automated test suites (`cargo test --workspace`), real corpus benchmarks, and ceiling sanity checks. |
| **6** | ✅ **Done** | Launch & Documentation | Production CLI, SVG visual branding, and verified hardware roofline benchmarks. |

---

## Verification & Tests

Run all unit and integration tests across workspace crates:

```bash
cargo test --workspace
```

---

## Citation & Licensing

Cite this repository if used in tokenization performance research:

```bibtex
@software{omnitoken2026,
  author  = {Hegde, Sunay},
  title   = {{OmniToken}: Universal High-Performance Tokenizer Engine for Consumer Hardware},
  year    = {2026},
  url     = {https://github.com/SunayHegde2006/OmniToken}
}
```

Licensed under the [MIT License](LICENSE).
