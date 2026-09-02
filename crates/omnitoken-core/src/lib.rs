//! `omnitoken-core` — portable primitives shared across all OmniToken crates.
//!
//! This crate has **no external dependencies** and is `#[no_std]`-friendly
//! (the `std` feature is on by default for convenience).
//!
//! # What lives here
//! - [`TokenId`] / [`State`] — fundamental numeric aliases.
//! - [`OmniError`] — structured error type (allocates nothing in the hot path).
//! - [`TokenSink`] — output abstraction (push tokens anywhere: Vec, slice, callback).
//! - [`Capabilities`] — runtime hardware detection (threads, SIMD, cache size).

// ─── Types ────────────────────────────────────────────────────────────────────

/// Opaque token identifier.  u32 matches HuggingFace / tiktoken convention and
/// fits every production vocabulary (GPT-4 vocab < 200 000).
pub type TokenId = u32;

/// DAT / automaton state index.  Same width as TokenId for uniform array indexing.
pub type State = u32;

// ─── Error ───────────────────────────────────────────────────────────────────

/// Structured error type.  Variants carry no heap-allocated strings so that
/// checking `is_err()` in the hot path has zero allocation cost.
/// Format via `Display` only on cold paths (logging / user messages).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OmniError {
    /// Underlying OS / file I/O failure (code only — message lives in the caller).
    Io,
    /// Binary blob header, magic, or TLV framing is malformed.
    InvalidFormat,
    /// Cache or compiled blob was written by a newer version.
    UnsupportedVersion(u32),
    /// Input bytes are not valid UTF-8 at the declared boundary.
    InvalidUtf8,
    /// Vocabulary table is inconsistent (e.g. merge references unknown token).
    InvalidVocabulary,
    /// Caller-supplied output buffer is too small for the result.
    BufferTooSmall,
    /// Unexpected internal invariant violation.
    Internal,
}

impl core::fmt::Display for OmniError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io                    => write!(f, "I/O error"),
            Self::InvalidFormat         => write!(f, "invalid format"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported version: {v}"),
            Self::InvalidUtf8           => write!(f, "invalid UTF-8"),
            Self::InvalidVocabulary     => write!(f, "invalid vocabulary"),
            Self::BufferTooSmall        => write!(f, "output buffer too small"),
            Self::Internal              => write!(f, "internal error"),
        }
    }
}

// std::error::Error blanket impl (only when std is available)
impl std::error::Error for OmniError {}

// ─── TokenSink ────────────────────────────────────────────────────────────────

/// Output sink for encoded token IDs.
///
/// Decouple the encoder from the output destination: push to a `Vec`, write
/// into a pre-allocated slice, fire a callback, or hand off to Python/C.
///
/// The default `extend` impl calls `push` in a loop. Override for bulk copies.
pub trait TokenSink {
    fn push(&mut self, token: TokenId);

    #[inline]
    fn extend(&mut self, tokens: &[TokenId]) {
        for &t in tokens { self.push(t); }
    }

    /// Hint: reserve capacity for `additional` more tokens.  May be ignored.
    #[inline]
    fn reserve(&mut self, _additional: usize) {}
}

/// Sink that appends to a `Vec<TokenId>`.
pub struct VecSink(pub Vec<TokenId>);

impl VecSink {
    #[inline] pub fn new() -> Self { Self(Vec::new()) }
    #[inline] pub fn with_capacity(n: usize) -> Self { Self(Vec::with_capacity(n)) }
    #[inline] pub fn into_vec(self) -> Vec<TokenId> { self.0 }
}

impl Default for VecSink {
    fn default() -> Self { Self::new() }
}

impl TokenSink for VecSink {
    #[inline] fn push(&mut self, token: TokenId) { self.0.push(token); }
    #[inline] fn extend(&mut self, tokens: &[TokenId]) { self.0.extend_from_slice(tokens); }
    #[inline] fn reserve(&mut self, n: usize) { self.0.reserve(n); }
}

// ─── Capabilities ─────────────────────────────────────────────────────────────

/// Runtime hardware capabilities snapshot.
///
/// Detect once at startup; pass `&Capabilities` to the dispatcher and kernel
/// selector.  Never detect inside a hot loop.
#[derive(Debug, Clone)]
pub struct Capabilities {
    /// Number of logical CPU threads available (capped to `usize::MAX`).
    pub thread_count: usize,
    /// True if the CPU supports AVX2 (256-bit SIMD, x86_64 only).
    pub avx2: bool,
    /// True if the CPU supports AVX-512 VBMI (512-bit with byte shuffle).
    pub avx512vbmi: bool,
    /// True if the CPU supports AVX-512 BW (512-bit with byte/word ops).
    pub avx512bw: bool,
    /// True if compiled for aarch64 (NEON is always available on aarch64).
    pub neon: bool,
    /// True if compiled for wasm32 with SIMD feature.
    pub wasm_simd: bool,
    /// Estimated L3 cache size in bytes (0 = unknown).
    pub l3_cache_bytes: usize,
}

impl Capabilities {
    /// Detect capabilities from the current runtime environment.
    pub fn detect() -> Self {
        let thread_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        #[cfg(target_arch = "x86_64")]
        let (avx2, avx512vbmi, avx512bw) = (
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("avx512vbmi"),
            std::is_x86_feature_detected!("avx512bw"),
        );
        #[cfg(not(target_arch = "x86_64"))]
        let (avx2, avx512vbmi, avx512bw) = (false, false, false);

        #[cfg(target_arch = "aarch64")]
        let neon = true;
        #[cfg(not(target_arch = "aarch64"))]
        let neon = false;

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        let wasm_simd = true;
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        let wasm_simd = false;

        // Hardware CPUID L3 probe for x86_64 with sysfs fallback.
        let l3_cache_bytes = Self::probe_l3_cache();

        Self { thread_count, avx2, avx512vbmi, avx512bw, neon, wasm_simd, l3_cache_bytes }
    }

    /// Best available SIMD width in bytes (for chunk sizing heuristics).
    #[inline]
    pub fn simd_width(&self) -> usize {
        if self.avx512vbmi || self.avx512bw { 64 }
        else if self.avx2 { 32 }
        else if self.neon || self.wasm_simd { 16 }
        else { 1 }
    }

    /// Returns true if any parallel execution is sensible.
    #[inline]
    pub fn parallel(&self) -> bool { self.thread_count > 1 }

    fn probe_l3_cache() -> usize {
        // 1. Hardware CPUID probe for x86_64 (Intel & AMD)
        #[cfg(target_arch = "x86_64")]
        {
            let max_leaf = std::arch::x86_64::__cpuid(0).eax;
            if max_leaf >= 4 {
                // Subleaf 3 = Cache Level 3
                let cpuid4 = std::arch::x86_64::__cpuid_count(4, 3);
                let cache_type = cpuid4.eax & 0x1F;
                if cache_type != 0 {
                    let ways = ((cpuid4.ebx >> 22) & 0x3FF) + 1;
                    let partitions = ((cpuid4.ebx >> 12) & 0x3FF) + 1;
                    let line_size = (cpuid4.ebx & 0xFFF) + 1;
                    let sets = cpuid4.ecx + 1;
                    let size = (ways * partitions * line_size * sets) as usize;
                    if size > 0 { return size; }
                }
            }
            let ext_max = std::arch::x86_64::__cpuid(0x8000_0000).eax;
            if ext_max >= 0x8000_0006 {
                let ext6 = std::arch::x86_64::__cpuid(0x8000_0006);
                let size_512k = ((ext6.ecx >> 18) as usize) * 512 * 1024;
                if size_512k > 0 { return size_512k; }
            }
        }

        // 2. Linux sysfs fallback
        #[cfg(target_os = "linux")]
        {
            if let Ok(s) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index3/size") {
                let s = s.trim();
                if let Some(n) = s.strip_suffix('K').and_then(|n| n.parse::<usize>().ok()) {
                    return n * 1024;
                }
                if let Some(n) = s.strip_suffix('M').and_then(|n| n.parse::<usize>().ok()) {
                    return n * 1024 * 1024;
                }
            }
        }

        0
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_sink_collects() {
        let mut s = VecSink::new();
        s.push(1); s.push(2); s.extend(&[3, 4]);
        assert_eq!(s.into_vec(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn capabilities_detect_does_not_panic() {
        let caps = Capabilities::detect();
        assert!(caps.thread_count >= 1);
        let w = caps.simd_width();
        assert!(w >= 1 && w.is_power_of_two());
    }

    #[test]
    fn omni_error_display() {
        assert_eq!(OmniError::InvalidFormat.to_string(), "invalid format");
        assert_eq!(OmniError::UnsupportedVersion(3).to_string(), "unsupported version: 3");
    }
}
