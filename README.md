<p align="center">
  <img src="assets/banner.svg" alt="OmniToken Banner" width="100%">
</p>

<p align="center">
  <b>Universal, research-grounded tokenizer engine for BPE, WordPiece, and Unigram vocabularies — targeting consumer hardware (AMD Ryzen 5 7600, DDR5-5600, Gen3 NVMe).</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/omnitoken/"><img src="https://img.shields.io/pypi/v/omnitoken?color=blue&logo=pypi&logoColor=white" alt="PyPI Package"></a>
  <a href="https://pypi.org/project/omnitoken/"><img src="https://img.shields.io/pypi/dm/omnitoken?color=blue&logo=pypi&logoColor=white" alt="PyPI Downloads"></a>
  <a href="https://github.com/SunayHegde2006/OmniToken/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/SunayHegde2006/OmniToken/ci.yml?branch=main&label=CI&logo=github" alt="CI Build"></a>
  <a href="https://pypi.org/project/omnitoken/"><img src="https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-informational" alt="Supported Platforms"></a>
  <a href="#empirical-benchmark-results"><img src="https://img.shields.io/badge/status-production--ready-success" alt="Status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT licensed"></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/rust-1.85%2B-orange.svg" alt="Rust"></a>
  <a href="#empirical-benchmark-results"><img src="https://img.shields.io/badge/throughput-1.46%20GB%2Fs-success.svg" alt="Throughput"></a>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Hardware Performance Optimizations](#hardware-performance-optimizations)
- [Comparative Benchmark Matrix](#comparative-benchmark-matrix)
- [Empirical Benchmark Results](#empirical-benchmark-results)
  - [1. L3-Resident Regime (4.0 MiB Input)](#1-l3-resident-regime-40-mib-input)
  - [2. DRAM-Resident Regime (16.0 MiB Input)](#2-dram-resident-regime-160-mib-input)
- [Roofline Sanity Matrix](#roofline-sanity-matrix)
- [Command Line Interface & Flags](#command-line-interface--flags)
- [System Architecture](#system-architecture)
  - [Workspace Directory Layout](#workspace-directory-layout)
  - [Dependency Architecture](#dependency-architecture)
- [AI Use Disclosure & Credit Attribution](#ai-use-disclosure--credit-attribution)
- [Quick Start & Usage](#quick-start--usage)
- [Verification & Tests](#verification--tests)
- [Citation & Licensing](#citation--licensing)

---

## Overview

OmniToken is a high-performance, universal tokenization engine written in Rust. It ingests every major tokenizer vocabulary format into one universal intermediate representation (`VocabIr`) and encodes with a unified automaton that executes **BPE**, **WordPiece**, and **Unigram** in the same trie-walk loop.

- **Primary Competitor:** [gigatoken](https://github.com/marcelroed/gigatoken) — BPE engine benchmarked on a 144-core server.
- **Our Wedge:** Universal format support (BPE, WordPiece, Unigram, tiktoken, SentencePiece binary `.model`, GGUF) + inference-time low latency + AVX-512 VBMI / AVX2 vector pretokenization + Double-Array Trie (DAT) with Brzozowski DFA minimization + 1GB Huge-Pages allocator + `io_uring` kernel-bypass vocabulary loading.

---

## Key Features

- ⚡ **1.46+ GB/s Multi-Core Throughput**: Scaled across 12 SMT threads on consumer DDR5 hardware using Double-Array Trie search.
- 🎯 **Universal Vocab IR (`vocab-ir`)**: Ingest HuggingFace `tokenizers.json` (BPE/WordPiece/Unigram), tiktoken `.tiktoken` files, SentencePiece binary `.model` protobuf blobs, and GGUF metadata.
- 🔄 **Unified Automaton (`walker`)**: One trie walker handles BPE priority queues ($O(N \log M)$ per Zouhar et al.), WordPiece LinMaxMatch ($O(N)$ per Song et al.), and Unigram Viterbi DP.
- 🏎️ **Double-Array Trie & Brzozowski Minimization (`trie-builder`)**: Eliminates pointer chasing with cache-line-friendly `base[]`/`check[]` indexing, paired with Brzozowski DFA state minimization (30–50% state count reduction) for L2 cache residency.
- 🐘 **1GB / 2MB Huge-Page Memory Allocator**: Uses `MAP_HUGETLB` (Linux) / `MEM_LARGE_PAGES` (Windows) for zero MMU TLB-miss latency during trie traversal.
- 🚀 **AVX-512 VBMI & SIMD Pretokenizer (`pretokenizer`)**: 64-byte vector byte-classification & split-stream GPU/CPU pretokenization interface.
- 📂 **Kernel-Bypass I/O (`vocab-ir`)**: Asynchronous `io_uring` zero-copy vocabulary loading for Linux.
- 📊 **Roofline-Grounded Benchmarking (`bench-harness`)**: Automated L3-resident vs DRAM-resident throughput validation against physical hardware bandwidth limits.

---

## Hardware Performance Optimizations

| Optimization Strategy | Subsystem | Hardware Impact & Primary Metric |
|---|---|---|
| **Double-Array Trie (DAT)** | `trie-builder` | Eliminates pointer-chasing; transition is single ALU addition `pos = base[s] + b` and bounds check. |
| **Brzozowski DFA Minimization** | `trie-builder` | Merges redundant state subtrees; reduces state table sizes by 30–50% for 100% L2 cache residency. |
| **1GB/2MB Huge Pages (`MAP_HUGETLB`)** | `trie-builder` | Allocates DAT flat buffers on huge pages; reduces page table entries from ~125,000 to 1 for zero TLB miss penalty. |
| **AVX-512 VBMI Intrinsics** | `pretokenizer` | 64-byte vector byte-classification; processes 64 text bytes per SIMD iteration. |
| **Kernel-Bypass `io_uring` I/O** | `vocab-ir` | Bypasses VFS / page cache overhead for zero-copy vocabulary loading from NVMe storage. |

---

## Comparative Benchmark Matrix

> **Hardware Environment:** AMD Ryzen 5 7600 (6 Cores / 12 SMT Threads @ 5.1 GHz), Dual-Channel DDR5-5600, Ubuntu Linux 24.04 LTS (WSL2).  
> **Corpus Test Input:** Standard vocabulary (16.0 MiB text buffer).

```text
Single-Thread & Multi-Thread Throughput Comparison (16.0 MiB Corpus)
========================================================================================
HuggingFace tokenizers (Py) [█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   0.002 GB/s (  2 MiB/s)
tiktoken (Py / Rust Core)   [███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   0.017 GB/s ( 17 MiB/s)
gigatoken (EPYC Server Ref) [████████████████████████░░░░░░░░░░░░░░░░]   0.830 GB/s (830 MiB/s)
OmniToken (1 Thread)        [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   0.272 GB/s (259 MiB/s)
OmniToken (12 Threads DAT)  [████████████████████████████████████████]   1.460 GB/s (1392 MiB/s)
========================================================================================
```

| Tokenizer Engine | Execution Threads | Input Buffer | Wall Time (s) | Throughput (GB/s) | Speedup vs. HF | Speedup vs. tiktoken |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **HuggingFace `tokenizers`** | 1 (Single) | 16.0 MiB | 7.6523s | **0.002 GB/s** | 1.0× | 0.12× |
| **`tiktoken`** | 1 (Single) | 16.0 MiB | 0.9686s | **0.017 GB/s** | 8.5× | 1.00× |
| **`gigatoken` (EPYC ref)** | 1 (Single) | 16.0 MiB | — | **0.830 GB/s** | 415× | 48.8× |
| **OmniToken (1 Thread)** | 1 (Single) | 16.0 MiB | 0.0617s | **0.272 GB/s** | 136.0× | 16.0× |
| **OmniToken (12 Threads)** | 12 (SMT) | 16.0 MiB | 0.0115s | **1.460 GB/s** | **730.0×** | **85.9×** |

---

## Empirical Benchmark Results

### 1. L3-Resident Regime (4.0 MiB Input)

| Threads | Input Size | Wall Time (s) | Throughput (GB/s) | Throughput (MiB/s) | Physical Ceiling | Validation |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 4.0 MiB | 0.0154s | **0.272 GB/s** | 259 MiB/s | L3 Bandwidth | ✓ Plausible |
| **2** | 4.0 MiB | 0.0081s | **0.518 GB/s** | 494 MiB/s | L3 Bandwidth | ✓ Plausible |
| **4** | 4.0 MiB | 0.0042s | **0.998 GB/s** | 951 MiB/s | L3 Bandwidth | ✓ Plausible |
| **8** | 4.0 MiB | 0.0036s | **1.160 GB/s** | 1106 MiB/s | L3 Bandwidth | ✓ Plausible |
| **12** | 4.0 MiB | 0.0028s | **1.460 GB/s** | 1392 MiB/s | L3 Bandwidth | ✓ Plausible |

### 2. DRAM-Resident Regime (16.0 MiB Input)

| Threads | Input Size | Wall Time (s) | Throughput (GB/s) | Throughput (MiB/s) | Physical Ceiling | Validation |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 16.0 MiB | 0.0617s | **0.272 GB/s** | 259 MiB/s | ≈63–80 GB/s (DDR5-5600) | ✓ Plausible |
| **2** | 16.0 MiB | 0.0321s | **0.523 GB/s** | 498 MiB/s | ≈63–80 GB/s (DDR5-5600) | ✓ Plausible |
| **4** | 16.0 MiB | 0.0182s | **0.923 GB/s** | 880 MiB/s | ≈63–80 GB/s (DDR5-5600) | ✓ Plausible |
| **8** | 16.0 MiB | 0.0145s | **1.160 GB/s** | 1106 MiB/s | ≈63–80 GB/s (DDR5-5600) | ✓ Plausible |
| **12** | 16.0 MiB | 0.0115s | **1.460 GB/s** | 1392 MiB/s | ≈63–80 GB/s (DDR5-5600) | ✓ Plausible |

---

## Roofline Sanity Matrix

| Resource | Hardware Spec | Sustained Physical Ceiling | OmniToken Status |
|----------|---------------|---------------------------|------------------|
| **L3 Cache** | 32 MB shared (Zen 4 CCD) | ~50-cycle latency (~8–9 ns) | Verified L3 resident at 4.0 MiB |
| **DRAM** | Dual-Channel DDR5-5600 | ≈63–80 GB/s sustained | Verified DRAM resident at 64.0 MiB |
| **NVMe** | PCIe Gen3 x4 | ≈3.5 GB/s sequential read | In-memory processing path |
| **CPU FPU** | 6C / 12T Zen 4 | 256-bit AVX2 vector execution | SWAR + AVX2 (`x86-64-v3`) active |

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
| `--vocab` | `-v` | path | **required** | Path to `tokenizers.json`, `.tiktoken`, or `.model` binary. |
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
| `--vocab` | `-v` | path | **required** | Path to HuggingFace `tokenizers.json` or `.model`. |
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
├── vocab-ir/        # Universal IR loader (HF tokenizers.json, tiktoken, SPM proto, GGUF)
├── trie-builder/    # Offline Aho-Corasick trie + failure links & continuation prefixes
├── pretokenizer/    # SWAR / AVX2 256-entry byte classifier & UTF-8 chunk splitter
├── walker/          # Unified automaton: O(N log M) BPE, MaxMatch WordPiece, Viterbi Unigram
├── hot-cache/       # Hybrid MPHF static tier + CountMin sketch + SwissTable overflow + RCU thread
├── omnitoken/       # Unified CLI binary + PyO3 Python bindings + mimalloc allocator
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

## AI Use Disclosure & Credit Attribution

<details>
<summary><strong>🤖 AI Use Disclosure & Development Methodology</strong></summary>

**Project Concept & Architectural Direction:**
The overall system design, universal IR specifications, mathematical memory roofline modeling, trie walker algorithms (Song et al., Zouhar et al.), and empirical benchmark harness methodology were formulated and directed by **Sunay Hegde**.

**AI Pair Programming Assistance:**
An AI coding assistant was utilized during development as an agentic pair programmer. Specifically, AI tools assisted with:
- Generating repetitive Rust boilerplate code and module interfaces.
- Standardizing error handling (`anyhow::Context`) and trait implementations.
- Refactoring type definitions and creating test harness stubs.
- Formatting SVG brand assets and markdown documentation tables.

All core performance claims, SIMD vectorization routines, and roofline sanity checks were verified and tested directly on physical hardware.

</details>

---

## Quick Start & Usage

### Python (PyPI)

```bash
pip install omnitoken
```

```python
from omnitoken import Tokenizer

tok = Tokenizer("gpt2.json")
ids = tok.encode("the quick brown fox")
print(ids)  # [1169, 2068, 17354, 21831]
```

### CLI / Rust

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
