<p align="center">
  <img src="assets/banner.svg" alt="OmniToken Banner" width="100%">
</p>

<p align="center">
  <b>Universal, hardware-adaptive tokenization runtime for BPE, WordPiece, and Unigram — with zero-copy I/O HAL, GPU-native BlockBPE linked-list scan, SoA Double-Array Trie, stable C-ABI, and DLPack tensor handoff.</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/omnitoken/"><img src="https://img.shields.io/badge/pypi-v1.0.0-blue?logo=pypi&logoColor=white" alt="PyPI Package"></a>
  <a href="https://pypi.org/project/omnitoken/"><img src="https://img.shields.io/pypi/dm/omnitoken?color=blue&logo=pypi&logoColor=white" alt="PyPI Downloads"></a>
  <a href="https://github.com/SunayHegde2006/OmniToken/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/SunayHegde2006/OmniToken/ci.yml?branch=main&label=CI&logo=github" alt="CI Build"></a>
  <a href="https://pypi.org/project/omnitoken/"><img src="https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-informational" alt="Supported Platforms"></a>
  <a href="#empirical-benchmark-results"><img src="https://img.shields.io/badge/status-v1.0.0%20production--ready-success" alt="Status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT licensed"></a>
  <a href="https://www.rust-lang.org"><img src="https://img.shields.io/badge/rust-1.85%2B-orange.svg" alt="Rust"></a>
  <a href="#empirical-benchmark-results"><img src="https://img.shields.io/badge/throughput-3.02%20GB%2Fs-success.svg" alt="Throughput"></a>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [OmniToken v1.0.0 Architecture & Roadmap](#omnitoken-v100-architecture--roadmap)
  - [1. Universal I/O HAL & `.otk` Binary Blob Format](#1-universal-io-hal--otk-binary-blob-format)
  - [2. Structure of Arrays (SoA) DAT Layout & Branchless Transition](#2-structure-of-arrays-soa-dat-layout--branchless-transition)
  - [3. Vectorized & GPU Pretokenizer Shader Interface](#3-vectorized--gpu-pretokenizer-shader-interface)
  - [4. GPU BlockBPE Linked-List Scan Algorithm & Streaming Encoder](#4-gpu-blockbpe-linked-list-scan-algorithm--streaming-encoder)
  - [5. ShortWordDict $O(1)$ Lookup & Space-Symbol Normalization](#5-shortworddict-o1-lookup--space-symbol-normalization)
  - [6. Adaptive Hybrid Dispatcher & Dynamic Chunk Sizing](#6-adaptive-hybrid-dispatcher--dynamic-chunk-sizing)
  - [7. Stable C-ABI & Zero-Copy DLPack Tensor Handoff](#7-stable-c-abi--zero-copy-dlpack-tensor-handoff)
- [Hardware Performance Optimizations](#hardware-performance-optimizations)
- [Comparative Benchmark Matrix](#comparative-benchmark-matrix)
- [Empirical Benchmark Results](#empirical-benchmark-results)
  - [1. Single-Thread vs Multi-Thread Scaling](#1-single-thread-vs-multi-thread-scaling)
  - [2. Time-To-First-Token (TTFT) & Latency](#2-time-to-first-token-ttft--latency)
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

OmniToken (v1.0.0) is a hardware-adaptive, universal tokenization engine written in Rust. It ingests every major tokenizer vocabulary format into one universal intermediate representation (`VocabIr`) and executes **BPE**, **WordPiece**, and **Unigram** tokenization using high-performance Double-Array Trie (DAT) automatons and GPU-native parallel algorithms.

- **Primary Competitor:** [gigatoken](https://github.com/marcelroed/gigatoken) — BPE engine benchmarked on a 144-core server.
- **Our Wedge:** Universal format support (BPE, WordPiece, Unigram, tiktoken, SentencePiece binary `.model`, GGUF, `.otk`) + zero-copy kernel bypass (`io_uring`/DirectStorage) + AVX-512 VBMI / ARM NEON / WASM pretokenization + Double-Array Trie (DAT) with SoA layout + 1GB Huge-Pages + GPU BlockBPE linked-list scan + stable C-ABI & DLPack zero-copy tensor handoff.

---

## Key Features

- ⚡ **3.017 GB/s Multi-Core Throughput**: Scaled across 12 SMT threads on consumer DDR5 hardware using `ShortWordDict` $O(1)$ length-bucketed lookups and Double-Array Trie search.
- 📦 **`.otk` Zero-Copy Binary Blob Format**: Offline compilation of tokenizers into checksum-verified, memory-mapped `.otk` binary blobs.
- 🎯 **Universal Vocab IR (`vocab-ir`)**: Ingest HuggingFace `tokenizers.json` (BPE/WordPiece/Unigram), tiktoken `.tiktoken` files, SentencePiece binary `.model` protobuf blobs, GGUF metadata, and `.otk` blobs.
- 🔄 **Unified Automaton (`walker`)**: Priority queue BPE ($O(N \log M)$ per Zouhar et al.), BlockBPE warp-synchronous scan, WordPiece LinMaxMatch ($O(N)$ per Song et al.), and Unigram Viterbi DP.
- 🏎️ **Structure of Arrays (SoA) DAT (`trie-builder`)**: Padded 128-byte aligned `base_soa[]` and `check_soa[]` memory layouts eliminating SIMD/GPU cache line splitting.
- 🐘 **1GB / 2MB Huge-Page Memory Allocator**: Uses `MAP_HUGETLB` (Linux) / `MEM_LARGE_PAGES` (Windows) for zero MMU TLB-miss latency during trie traversal.
- 🚀 **AVX-512, ARM NEON & GPU Pretokenizer (`pretokenizer`)**: 64-byte vector byte-classification with 256-entry GPU shared memory classifier table.
- 🔌 **Universal I/O HAL (`vocab-ir`)**: Asynchronous `io_uring` (Linux), `F_NOCACHE` (macOS), and `DirectStorage` (Windows) kernel-bypass vocabulary loading.
- 🔀 **Adaptive Workload Dispatcher (`dispatcher`)**: Payload size ($B < 4\text{KB}$) analytical cost model automatically routing small prompts to single-thread JIT paths and large batches to parallel BlockBPE SIMD/GPU paths.
- 🌐 **Stable C-ABI & DLPack Zero-Copy Handoff (`c_api`)**: Export flat token arrays directly to PyTorch, vLLM, and JAX tensors without host memory copying.

---

## OmniToken v2.0 Architecture & Roadmap

### 1. Universal I/O HAL & `.otk` Binary Blob Format
OmniToken v0.3.0 introduces the **Universal Direct I/O HAL** and **`.otk` (OmniToken Compiled)** binary blob format. Instead of parsing heavy HuggingFace `tokenizers.json` files at startup, `omnitoken compile` compiles the vocabulary into a zero-copy `.otk` binary format:
- **Magic Header:** `\x7fOTK\x01\x00\x00\x00`
- **Zero-Copy Loading:** Uses OS-native asynchronous kernel bypass (`io_uring` on Linux, `F_NOCACHE` on macOS, `DirectStorage` on Windows).
- **Startup Overhead:** Reduces vocabulary initialization time from ~35ms down to **254 µs**.

### 2. Structure of Arrays (SoA) DAT Layout
Standard Double-Array Tries store interleaving state nodes that create uncoalesced memory access patterns on modern GPU warps and SIMD units. OmniToken's `DatSoaLayout` reorganizes the `base[]` and `check[]` buffers into structure-of-arrays representation aligned to 128-byte (32 × `i32`) boundaries:
```text
State Index:   [0] [1] [2] ... [31] | [32] ... [63]
base_soa[]:    [ 1,  4, -1, ...,  0 ] | [ 12, ..., 0 ]
check_soa[]:   [ 0,  1, -1, ...,  0 ] | [  4, ..., 0 ]
```

### 3. Vectorized & GPU Pretokenizer Shader Interface
The pretokenization pipeline dynamically dispatches byte classification to hardware vector units:
- **x86_64:** AVX-512 VBMI / AVX-512BW / AVX2 SWAR paths.
- **aarch64:** ARM NEON `vtbl` 128-bit vector table classification.
- **GPU Shader Interface:** `GpuPretokenizerTable` exposing a 256-entry lookup table suitable for mapping into 100KB GPU Shared Memory per SM.

### 4. GPU BlockBPE Linked-List Scan Algorithm
OmniToken implements the **BlockBPE Same-Pair Merge Algorithm (Mode B)** for parallel sequence tokenization:
1. **Subgroup Min-Rank Scan:** Simulates warp-synchronous reduction to find the global minimum rank merge pair $R_{\min}$ across text subword sequences.
2. **Left-to-Right Overlap Resolution:** Resolves conflicting merges deterministically using parallel prefix scan semantics.
3. **Doubly-Linked List Compaction:** Updates sequence pointers (`prev`, `next`) in $O(1)$ without $O(N)$ array shifts.
4. **Conformance Gate:** `verify_bpe_conformance()` enforces 100% bit-exact output parity between CPU PriorityQueue BPE and GPU BlockBPE.

### 5. ShortWordDict $O(1)$ Lookup & Space-Symbol Normalization
- **Length-Bucketed Lookups:** `ShortWordDict` provides $O(1)$ key-to-token lookups for subword strings $\le 8$ bytes using Fibonacci integer hashing.
- **GPT-2 Space Normalization:** Maps byte-encoder space symbols (`Ġ` $\rightarrow$ `' '`) during dictionary construction and dispatches preceding-space candidate lookups (`" word"`), pushing fast-path whole-word hit rate to 100%.

### 6. Adaptive Hybrid Dispatcher
The `AdaptiveDispatcher` computes execution cost equations:
- $T_{\text{cpu}}(B) = C_{\text{cpu}}(B) + M_{\text{memcpy}}$
- $T_{\text{gpu}}(B) = L_{\text{kernel}} + \frac{B}{P} + C_{\text{gpu}}(B)$
When prompt length $B < 4\text{KB}$, kernel launch latency $L_{\text{kernel}}$ dominates, so the dispatcher routes to the single-threaded CPU path. For large batches or files ($B \ge 4\text{KB}$), it routes to parallel BlockBPE SIMD workers.

### 7. Stable C-ABI & Zero-Copy DLPack Tensor Handoff
OmniToken provides a stable `extern "C"` ABI and PyTorch/vLLM zero-copy tensor handoff:
```c
// C-ABI Functions
int32_t omni_tokenizer_create(const char* vocab_path, OmniTokenizerHandle** out_handle);
void omni_tokenizer_free(OmniTokenizerHandle* handle);
int32_t omni_encode_batch(OmniTokenizerHandle* handle, const char** text_ptrs, size_t batch_size, uint32_t** out_tokens, size_t** out_lengths);
void omni_free_batch(uint32_t* tokens_ptr, size_t* lengths_ptr);
int32_t omni_get_last_error(char* buf, size_t max_len);
```
- **DLPack Support:** `DLManagedTensor` exports token ID vectors directly into PyTorch C++ tensors without CPU-to-GPU memory copies.

---

## Hardware Performance Optimizations

| Optimization Strategy | Subsystem | Hardware Impact & Primary Metric |
|---|---|---|
| **Universal `.otk` Format** | `vocab-ir` | Reduces startup latency to **254 µs** via zero-copy mmap & kernel bypass. |
| **Structure of Arrays (SoA)** | `trie-builder` | Padded 128-byte alignment for 100% SIMD/GPU cache line memory coalescing. |
| **BlockBPE Doubly-Linked List** | `walker` | Parallel same-pair BPE merge algorithm eliminating $O(N)$ vector shifts. |
| **1GB/2MB Huge Pages (`MAP_HUGETLB`)** | `trie-builder` | Allocates DAT flat buffers on huge pages; reduces page table entries from ~125,000 to 1 for zero TLB miss penalty. |
| **Adaptive Hybrid Dispatcher** | `dispatcher` | Payload analytical model routing small prompts ($B < 4\text{KB}$) to low-latency JIT path. |
| **Stable C-ABI & DLPack** | `c_api` | Zero-copy tensor handoff for direct framework integration with vLLM / PyTorch. |

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
OmniToken (1 Thread)        [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   0.449 GB/s (428 MiB/s)
OmniToken (6 Threads)       [████████████████████████████████░░░░░░░░]   1.975 GB/s (1883 MiB/s)
OmniToken (12 Threads DAT)  [████████████████████████████████████████]   3.017 GB/s (2877 MiB/s)
========================================================================================
```

| Tokenizer Engine | Execution Threads | Input Buffer | Wall Time (s) | Throughput (GB/s) | Speedup vs. HF | Speedup vs. tiktoken |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **HuggingFace `tokenizers`** | 1 (Single) | 16.0 MiB | 7.6523s | **0.002 GB/s** | 1.0× | 0.12× |
| **`tiktoken`** | 1 (Single) | 16.0 MiB | 0.9686s | **0.017 GB/s** | 8.5× | 1.00× |
| **`gigatoken` (EPYC ref)** | 1 (Single) | 16.0 MiB | — | **0.830 GB/s** | 415× | 48.8× |
| **OmniToken (1 Thread)** | 1 (Single) | 16.0 MiB | 0.0373s | **0.449 GB/s** | 224.5× | 26.4× |
| **OmniToken (6 Threads)** | 6 (Cores) | 16.0 MiB | 0.0085s | **1.975 GB/s** | 987.5× | 116.2× |
| **OmniToken (12 Threads)** | 12 (SMT) | 16.0 MiB | 0.0055s | **3.017 GB/s** | **1508.5×** | **177.5×** |

---

## Empirical Benchmark Results

### 1. Single-Thread vs Multi-Thread Scaling

| Threads | Engine | Chunking | Wall Time (s) | Throughput (GB/s) | Throughput (MiB/s) | Token Rate | Validation |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | `fast` | Auto | 0.0373s | **0.449 GB/s** | 428 MiB/s | 91.89 Mtok/s | ✓ Plausible |
| **6** | `fast` | Auto | 0.0085s | **1.975 GB/s** | 1883 MiB/s | 403.88 Mtok/s | ✓ Plausible |
| **12** | `fast` | Auto | 0.0058s | **2.869 GB/s** | 2736 MiB/s | 586.91 Mtok/s | ✓ Plausible |
| **12** | `fast` | 256 KiB | 0.0055s | **3.017 GB/s** | 2877 MiB/s | 617.17 Mtok/s | ✓ Plausible |

### 2. Time-To-First-Token (TTFT) & Latency

| Benchmark Metric | Measured Result | Target Budget | Status |
|---|---|---|---|
| **Vocab Load (`.otk` mmap)** | **254 µs** | < 1,000 µs | PASS |
| **Trie Build Latency** | **48 µs** | < 500 µs | PASS |
| **Single-Prompt TTFT** | **17 µs** | < 50 µs | PASS |
| **BPE Bit-Exact Conformance** | **100% Match** | 100% | PASS |

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

**`omnitoken encode`** — Tokenize input text from stdin:
```bash
omnitoken encode --vocab <path.json or model.otk>
```

**`omnitoken compile`** — Compile HuggingFace `tokenizers.json` to zero-copy `.otk` binary blob:
```bash
omnitoken compile --vocab gpt2.json --out gpt2.otk
```

</details>

<details>
<summary><strong>⚙️ <code>bench</code> — Benchmarking & Roofline Harness</strong></summary>

**`bench`** — Measure throughput and cross-check roofline physical ceilings:
```bash
bench --vocab <path.json or model.otk> [OPTIONS]
```

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--vocab` | `-v` | path | **required** | Path to HuggingFace `tokenizers.json` or `.otk` model. |
| `--corpus` | `-c` | path | `synthetic` | Optional path to text corpus file. |
| `--threads` | `-t` | int | `12` | Number of Rayon threads. |
| `--engine` | `-e` | string | `fast` | Benchmark engine target (`fast`, `pre`, `word-lookup`, `bpe-only`). |
| `--chunk-kb` | | int | `0` (auto) | Custom chunk size in KiB for thread worker decomposition. |
| `--stats` | | flag | `off` | Print fast-path whole-word hit rate coverage statistics. |
| `--bytes` | `-b` | int | `16777216` | Bytes of synthetic corpus to generate if no file provided. |
| `--warmup` | | int | `3` | Number of warmup iterations executed before timing. |
| `--iterations` | | int | `5` | Number of timed iterations for median timing. |
| `--no-flatten` | | flag | `off` | Output raw worker chunk vectors without flattening copy overhead. |
| `--parity` | `-p` | flag | `off` | Run token-by-token parity & differential fuzzing check. |
| `--mmap` | | flag | `off` | Enable memory-mapped file reader for disk streaming. |

</details>

---

## System Architecture

### Workspace Directory Layout

```text
crates/
├── omnitoken-core/  # Portable core primitives (TokenId, OmniError, TokenSink, Capabilities)
├── vocab-ir/        # Universal IR loader (HF tokenizers.json, tiktoken, SPM proto, GGUF, .otk)
├── trie-builder/    # Offline DAT builder + Brzozowski DFA minimization + SoA 128-byte layout + branchless DAT transition
├── pretokenizer/    # SWAR / AVX2 / ARM NEON byte classifier & GpuPretokenizer shader table
├── walker/          # PriorityQueue BPE, BlockBPE linked-list scan, WordPiece, Unigram DP, ShortWordDict, StreamingEncoder
├── omnitoken/       # AdaptiveDispatcher + Dynamic Chunking + Stable C-ABI + DLPack zero-copy handoff + PyO3
└── bench-harness/   # Multi-scenario benchmark suite, TTFT, microbench engines & differential fuzzing harness
```

### Dependency Architecture

```text
omnitoken-core
   ▲
   ├───────► vocab-ir ──────► trie-builder ──────► walker ◄───── pretokenizer
   │           │                  │                     ▲
   │           │                  └─────────────────────┤
   │           │                                        │
   └───────────┴────────────────────────────────── omnitoken / bench-harness
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
from omnitoken import PyTokenizer

tok = PyTokenizer("gpt2.otk")
ids = tok.encode("the quick brown fox")
print(ids)  # [1169, 2068, 17354, 21831]
```

### CLI / Rust

### 1. Build Release Binaries

```bash
cargo build --release
```

### 2. Compile Model to Zero-Copy `.otk` Format

```bash
./target/release/omnitoken compile --vocab gpt2.json --out gpt2.otk
```

### 3. Run OmniToken CLI

```bash
echo "the quick brown fox jumps over the lazy dog" | ./target/release/omnitoken encode --vocab gpt2.otk
```

### 4. Run Benchmark & Differential Fuzzing Suite

```bash
# Multi-thread 12-thread scaling benchmark on compiled .otk
./target/release/bench --vocab gpt2.otk --threads 12 --bytes 67108864

# Differential fuzzing parity check
./target/release/bench --vocab gpt2.otk --parity
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
  title   = {{OmniToken}: Universal Hardware-Adaptive Tokenization Runtime for Consumer Hardware},
  year    = {2026},
  url     = {https://github.com/SunayHegde2006/OmniToken}
}
```

Licensed under the [MIT License](LICENSE).
