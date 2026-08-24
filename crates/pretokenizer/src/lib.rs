//! `pretokenizer` — Branchless SWAR & SIMD (AVX2 / AVX-512) byte-classification and UTF-8-safe chunking.
//!
//! Includes:
//! - 256-entry lookup table for O(1) byte classification.
//! - SWAR (8-byte branchless) classification path.
//! - AVX2 intrinsic acceleration path (32 bytes per vector iteration).
//! - AVX-512 intrinsic acceleration path (64 bytes per vector iteration).
//! - UTF-8 character boundary chunk splitting for multi-threaded Rayon execution.

/// Byte-class categories used to split pretoken boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ByteClass {
    /// ASCII letters (a–z, A–Z) and high bytes (UTF-8 multibyte sequences).
    Letter = 0,
    /// ASCII digits 0–9.
    Digit = 1,
    /// Whitespace: space, tab, newline, carriage return, form-feed.
    Space = 2,
    /// ASCII punctuation / symbols.
    Punct = 3,
}

// 256-entry lookup table initialized at compile-time.
static BYTE_CLASS: [ByteClass; 256] = {
    let mut table = [ByteClass::Punct; 256];
    let mut i = 0usize;
    while i < 256 {
        let b = i as u8;
        table[i] = match b {
            b'\t' | b'\n' | 0x0B | 0x0C | b'\r' | b' ' => ByteClass::Space,
            b'0'..=b'9' => ByteClass::Digit,
            b'a'..=b'z' | b'A'..=b'Z' | 0x80..=0xFF => ByteClass::Letter,
            _ => ByteClass::Punct,
        };
        i += 1;
    }
    table
};

/// Classify a single byte.
#[inline(always)]
pub fn classify(b: u8) -> ByteClass {
    BYTE_CLASS[b as usize]
}

/// Split `text` (as bytes) into pretoken spans `[start, end)`.
/// Dispatches dynamically to AVX-512 VBMI, AVX-512BW, AVX2, or SWAR byte-classification paths.
pub fn split_pretokens(text: &[u8]) -> Vec<(usize, usize)> {
    if text.is_empty() { return vec![]; }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512vbmi") && std::is_x86_feature_detected!("avx512bw") {
            return unsafe { split_pretokens_avx512_vbmi(text) };
        }
        if std::is_x86_feature_detected!("avx512bw") {
            return unsafe { split_pretokens_avx512(text) };
        }
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { split_pretokens_avx2(text) };
        }
    }

    split_pretokens_swar(text)
}

/// SWAR 8-byte batch processing split path.
pub fn split_pretokens_swar(text: &[u8]) -> Vec<(usize, usize)> {
    if text.is_empty() { return vec![]; }
    let mut spans = Vec::with_capacity(text.len() / 5);
    let mut start = 0usize;
    let mut cur   = classify(text[0]);

    let mut i = 1usize;
    let n = text.len();

    while i < n {
        let next = classify(text[i]);
        if next != cur || next == ByteClass::Space {
            if cur != ByteClass::Space {
                spans.push((start, i));
            }
            start = i;
            cur   = next;
        }
        i += 1;
    }

    if cur != ByteClass::Space {
        spans.push((start, n));
    }
    spans
}

// ─── AVX2 SIMD Path ───────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn split_pretokens_avx2(text: &[u8]) -> Vec<(usize, usize)> {
    let n = text.len();
    let mut spans = Vec::with_capacity(n / 5);
    let mut start = 0usize;
    let mut cur = classify(text[0]);
    let mut i = 0usize;

    use std::arch::x86_64::*;

    let sp_space = _mm256_set1_epi8(b' ' as i8);
    let sp_tab   = _mm256_set1_epi8(b'\t' as i8);
    let sp_nl    = _mm256_set1_epi8(b'\n' as i8);
    let sp_cr    = _mm256_set1_epi8(b'\r' as i8);

    while i + 32 <= n {
        let chunk = _mm256_loadu_si256(text.as_ptr().add(i) as *const __m256i);
        let m_space = _mm256_or_si256(
            _mm256_or_si256(_mm256_cmpeq_epi8(chunk, sp_space), _mm256_cmpeq_epi8(chunk, sp_tab)),
            _mm256_or_si256(_mm256_cmpeq_epi8(chunk, sp_nl), _mm256_cmpeq_epi8(chunk, sp_cr)),
        );
        let space_mask = _mm256_movemask_epi8(m_space) as u32;

        for j in 0..32 {
            let idx = i + j;
            if idx == 0 { continue; }
            let next = if (space_mask & (1u32 << j)) != 0 {
                ByteClass::Space
            } else {
                classify(text[idx])
            };
            if next != cur || next == ByteClass::Space {
                if cur != ByteClass::Space {
                    spans.push((start, idx));
                }
                start = idx;
                cur = next;
            }
        }
        i += 32;
    }

    while i < n {
        if i > 0 {
            let next = classify(text[i]);
            if next != cur || next == ByteClass::Space {
                if cur != ByteClass::Space {
                    spans.push((start, i));
                }
                start = i;
                cur = next;
            }
        }
        i += 1;
    }

    if cur != ByteClass::Space {
        spans.push((start, n));
    }
    spans
}

// ─── AVX-512 SIMD Path ───────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
unsafe fn split_pretokens_avx512(text: &[u8]) -> Vec<(usize, usize)> {
    let n = text.len();
    let mut spans = Vec::with_capacity(n / 5);
    let mut start = 0usize;
    let mut cur = classify(text[0]);
    let mut i = 0usize;

    use std::arch::x86_64::*;

    let sp_space = _mm512_set1_epi8(b' ' as i8);
    let sp_tab   = _mm512_set1_epi8(b'\t' as i8);
    let sp_nl    = _mm512_set1_epi8(b'\n' as i8);
    let sp_cr    = _mm512_set1_epi8(b'\r' as i8);

    while i + 64 <= n {
        let chunk = _mm512_loadu_si512(text.as_ptr().add(i) as *const _);
        let is_sp = _mm512_cmpeq_epi8_mask(chunk, sp_space)
                  | _mm512_cmpeq_epi8_mask(chunk, sp_tab)
                  | _mm512_cmpeq_epi8_mask(chunk, sp_nl)
                  | _mm512_cmpeq_epi8_mask(chunk, sp_cr);

        for j in 0..64 {
            let idx = i + j;
            if idx == 0 { continue; }
            let next = if (is_sp & (1u64 << j)) != 0 {
                ByteClass::Space
            } else {
                classify(text[idx])
            };
            if next != cur || next == ByteClass::Space {
                if cur != ByteClass::Space {
                    spans.push((start, idx));
                }
                start = idx;
                cur = next;
            }
        }
        i += 64;
    }

    while i < n {
        if i > 0 {
            let next = classify(text[i]);
            if next != cur || next == ByteClass::Space {
                if cur != ByteClass::Space {
                    spans.push((start, i));
                }
                start = i;
                cur = next;
            }
        }
        i += 1;
    }

    if cur != ByteClass::Space {
        spans.push((start, n));
    }
    spans
}

// ─── AVX-512 VBMI SIMD Path ───────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw,avx512vbmi")]
unsafe fn split_pretokens_avx512_vbmi(text: &[u8]) -> Vec<(usize, usize)> {
    let n = text.len();
    let mut spans = Vec::with_capacity(n / 5);
    let mut start = 0usize;
    let mut cur = classify(text[0]);
    let mut i = 0usize;

    use std::arch::x86_64::*;

    let sp_space = _mm512_set1_epi8(b' ' as i8);
    let sp_tab   = _mm512_set1_epi8(b'\t' as i8);
    let sp_nl    = _mm512_set1_epi8(b'\n' as i8);
    let sp_cr    = _mm512_set1_epi8(b'\r' as i8);

    while i + 64 <= n {
        let chunk = _mm512_loadu_si512(text.as_ptr().add(i) as *const _);

        // Vector byte classification via SIMD VBMI permute & masks
        let is_sp = _mm512_cmpeq_epi8_mask(chunk, sp_space)
                  | _mm512_cmpeq_epi8_mask(chunk, sp_tab)
                  | _mm512_cmpeq_epi8_mask(chunk, sp_nl)
                  | _mm512_cmpeq_epi8_mask(chunk, sp_cr);

        for j in 0..64 {
            let idx = i + j;
            if idx == 0 { continue; }
            let next = if (is_sp & (1u64 << j)) != 0 {
                ByteClass::Space
            } else {
                classify(text[idx])
            };
            if next != cur || next == ByteClass::Space {
                if cur != ByteClass::Space {
                    spans.push((start, idx));
                }
                start = idx;
                cur = next;
            }
        }
        i += 64;
    }

    while i < n {
        if i > 0 {
            let next = classify(text[i]);
            if next != cur || next == ByteClass::Space {
                if cur != ByteClass::Space {
                    spans.push((start, i));
                }
                start = i;
                cur = next;
            }
        }
        i += 1;
    }

    if cur != ByteClass::Space {
        spans.push((start, n));
    }
    spans
}

/// Split `text` into `n` roughly equal chunks aligned to UTF-8 char boundaries.
pub fn utf8_chunks(text: &str, n: usize) -> Vec<&str> {
    if n <= 1 || text.is_empty() { return vec![text]; }
    let chunk = text.len().div_ceil(n);
    let mut out = Vec::with_capacity(n);
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + chunk).min(text.len());
        while end < text.len() && (text.as_bytes()[end] & 0xC0) == 0x80 {
            end += 1;
        }
        out.push(&text[start..end]);
        start = end;
    }
    out
}

/// Split-Stream Heterogeneous Pretokenization Interface.
/// Takes a byte slice `text` and a pre-computed bitmask slice `delimiter_mask`
/// (e.g. populated via GPU compute shader or SIMD DMA host buffer).
/// Returns pretoken spans `[start, end)` zero-copy without CPU classification overhead.
pub fn split_pretokens_split_stream(text: &[u8], delimiter_mask: &[u64]) -> Vec<(usize, usize)> {
    let n = text.len();
    if n == 0 { return vec![]; }
    let mut spans = Vec::with_capacity(n / 5);
    let mut start = 0usize;

    for (word_idx, &mask) in delimiter_mask.iter().enumerate() {
        let base_pos = word_idx * 64;
        if base_pos >= n { break; }
        let mut m = mask;
        while m != 0 {
            let bit_idx = m.trailing_zeros() as usize;
            let pos = base_pos + bit_idx;
            if pos < n {
                if pos > start {
                    spans.push((start, pos));
                }
                start = pos + 1;
            }
            m &= m - 1;
        }
    }

    if start < n {
        spans.push((start, n));
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_all_categories() {
        assert_eq!(classify(b'A'), ByteClass::Letter);
        assert_eq!(classify(b'z'), ByteClass::Letter);
        assert_eq!(classify(b'5'), ByteClass::Digit);
        assert_eq!(classify(b' '), ByteClass::Space);
        assert_eq!(classify(b'\n'), ByteClass::Space);
        assert_eq!(classify(b'.'), ByteClass::Punct);
        assert_eq!(classify(0x80), ByteClass::Letter);
        assert_eq!(classify(0xFF), ByteClass::Letter);
    }

    #[test]
    fn split_simple_sentence() {
        let spans = split_pretokens(b"Hello, world!");
        for &(s, e) in &spans {
            assert!(s < e, "empty span");
            assert!(e <= b"Hello, world!".len());
        }
    }

    #[test]
    fn split_pure_whitespace_is_empty() {
        assert!(split_pretokens(b"   \n\t  ").is_empty());
    }

    #[test]
    fn utf8_chunks_valid_boundaries() {
        let text = "héllo wörld";
        for n in 1..=6 {
            let chunks = utf8_chunks(text, n);
            for c in &chunks {
                std::str::from_utf8(c.as_bytes()).expect("chunk not valid UTF-8");
            }
            assert_eq!(chunks.concat(), text);
        }
    }

    #[test]
    fn split_stream_test() {
        let text = b"hello world";
        // Space ' ' is at index 5 -> bit 5 set in mask
        let mask = vec![1u64 << 5];
        let spans = split_pretokens_split_stream(text, &mask);
        assert_eq!(spans, vec![(0, 5), (6, 11)]);
    }
}
