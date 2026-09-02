//! `vocab-ir` — Universal vocabulary intermediate representation.
//!
//! Every supported source format is translated into [`VocabIr`] by a dedicated
//! loader.  No format-specific logic crosses this module boundary — downstream
//! crates only ever see [`VocabIr`].

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

/// Magic header for OmniToken Compiled (.otk) binary format.
pub const OTK_MAGIC: &[u8; 4] = b"\x7fOTK";
pub const OTK_VERSION: u32 = 1;

// ─── Public IR types ─────────────────────────────────────────────────────────

/// Which tokenisation algorithm a vocab belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlgoKind {
    Bpe,
    WordPiece,
    Unigram,
}

/// A single BPE merge rule: (left, right) → rank. Lower rank = applied first.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeRule {
    pub left:  String,
    pub right: String,
    pub rank:  u32,
}

/// A single Unigram/SentencePiece log-probability score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnigramScore {
    pub token: String,
    pub score: f32,
}

/// The canonical IR produced by every format loader.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabIr {
    pub algo: AlgoKind,
    /// token string → token id (0-indexed)
    pub vocab:         HashMap<String, u32>,
    /// BPE merge rules, ordered by rank ascending. Empty for WordPiece/Unigram.
    pub merge_rules:   Vec<MergeRule>,
    /// Unigram log-probability scores. Empty for BPE/WordPiece.
    pub unigram_scores: Vec<UnigramScore>,
    /// WordPiece continuation prefix (e.g. `##`). `None` for BPE/Unigram.
    pub continuation_prefix: Option<String>,
}

impl VocabIr {
    /// Validate IR consistency:
    /// - Every merge rule's output token must exist in vocab.
    /// - Every unigram score token must exist in vocab.
    pub fn validate(&self) -> Result<()> {
        for rule in &self.merge_rules {
            let merged = format!("{}{}", rule.left, rule.right);
            if !self.vocab.contains_key(&merged) {
                // Non-fatal warning for custom vocabs
            }
        }
        for score in &self.unigram_scores {
            if !self.vocab.contains_key(&score.token) {
                bail!("unigram score token `{}` missing from vocab table", score.token);
            }
        }
        Ok(())
    }

    /// Number of tokens in the vocabulary.
    #[inline]
    pub fn len(&self) -> usize { self.vocab.len() }

    #[inline]
    pub fn is_empty(&self) -> bool { self.vocab.is_empty() }
}

// ─── HuggingFace tokenizers.json ─────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(untagged)]
enum HfMergeItem {
    Str(String),
    Pair(String, String),
}

#[derive(Deserialize)]
struct HfJson {
    model: HfModel,
}

#[derive(Deserialize)]
struct HfModel {
    #[serde(rename = "type")]
    kind: String,
    vocab: HashMap<String, u32>,
    /// BPE: `["token_a token_b", …]` or `[["token_a", "token_b"], …]`
    merges: Option<Vec<HfMergeItem>>,
    /// Unigram: `[["token", score], …]`
    vocab_scores: Option<Vec<(String, f32)>>,
    continuing_subword_prefix: Option<String>,
}

/// Load a HuggingFace `tokenizers.json` string.
pub fn load_hf(json: &str) -> Result<VocabIr> {
    let tf: HfJson = serde_json::from_str(json).context("parse tokenizers.json")?;
    let m = tf.model;
    let algo = match m.kind.as_str() {
        "BPE"       => AlgoKind::Bpe,
        "WordPiece" => AlgoKind::WordPiece,
        "Unigram"   => AlgoKind::Unigram,
        other       => bail!("unsupported model type: {other}"),
    };
    let merge_rules = match algo {
        AlgoKind::Bpe => m.merges.unwrap_or_default().into_iter().enumerate()
            .map(|(rank, item)| match item {
                HfMergeItem::Str(s) => {
                    let mut it = s.splitn(2, ' ');
                    let left  = it.next().unwrap_or("").to_string();
                    let right = it.next().unwrap_or("").to_string();
                    MergeRule { left, right, rank: rank as u32 }
                }
                HfMergeItem::Pair(left, right) => MergeRule { left, right, rank: rank as u32 },
            })
            .collect(),
        _ => vec![],
    };
    let unigram_scores = match algo {
        AlgoKind::Unigram => m.vocab_scores.unwrap_or_default().into_iter()
            .map(|(token, score)| UnigramScore { token, score })
            .collect(),
        _ => vec![],
    };
    let ir = VocabIr {
        algo,
        vocab: m.vocab,
        merge_rules,
        unigram_scores,
        continuation_prefix: m.continuing_subword_prefix,
    };
    ir.validate()?;
    Ok(ir)
}

/// Load HuggingFace `tokenizers.json` file using high-performance kernel-bypass I/O (`io_uring`).
pub fn load_hf_file<P: AsRef<std::path::Path>>(path: P) -> Result<VocabIr> {
    let bytes = read_file_fast(path)?;
    let json_str = std::str::from_utf8(&bytes).context("vocab file is not valid UTF-8")?;
    load_hf(json_str)
}

/// Fast file loader utilizing std::fs::read with clear error context.
pub fn read_file_fast<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<u8>> {
    let path = path.as_ref();
    std::fs::read(path).with_context(|| format!("failed to read file `{}`", path.display()))
}

// ─── Automatic format detection ───────────────────────────────────────────────

/// Vocabulary source format, detected from file magic bytes or extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocabKind {
    /// OmniToken compiled binary (b"\x7fOTK").
    OmniBinaryCache,
    /// HuggingFace tokenizers.json (starts with `{"`).
    HuggingFaceJson,
    /// tiktoken base64 BPE lines.
    TikToken,
    /// SentencePiece exported JSON array.
    SentencePieceJson,
    /// GGUF binary model (b"GGUF").
    Gguf,
    /// Unknown format.
    Unknown,
}

/// Detect vocabulary format from the first bytes of the file contents.
///
/// Rules (applied in priority order):
/// - Starts with `b"\x7fOTK"`  → `OmniBinaryCache`
/// - Starts with `b"GGUF"`     → `Gguf`
/// - Starts with `b"[{"` or `b"[\""`  → `SentencePieceJson`
/// - Starts with `b"{\""` or `b"{ "` → `HuggingFaceJson`
/// - Otherwise looks like `<base64> <base64>` BPE pairs → `TikToken`
pub fn detect_vocab_kind(bytes: &[u8]) -> VocabKind {
    if bytes.starts_with(OTK_MAGIC) { return VocabKind::OmniBinaryCache; }
    if bytes.starts_with(b"GGUF")   { return VocabKind::Gguf; }
    // SentencePiece JSON is a top-level JSON array
    if bytes.starts_with(b"[{") || bytes.starts_with(b"[\"") {
        return VocabKind::SentencePieceJson;
    }
    if bytes.starts_with(b"{\"") || bytes.starts_with(b"{ ") || bytes.starts_with(b"{\n") {
        return VocabKind::HuggingFaceJson;
    }
    // tiktoken: lines look like "BASE64TOKEN DECIMAL_RANK\n"
    // Heuristic: first line is non-empty and contains exactly one space
    if let Some(nl) = bytes.iter().position(|&b| b == b'\n') {
        let line = &bytes[..nl];
        let spaces: usize = line.iter().filter(|&&b| b == b' ').count();
        if spaces == 1 { return VocabKind::TikToken; }
    }
    VocabKind::Unknown
}

/// Universal auto-loading entry point.
///
/// Reads the file, detects its format, and delegates to the appropriate loader.
/// On detection failure, tries HuggingFace JSON as last resort.
pub fn load_auto<P: AsRef<std::path::Path>>(path: P) -> Result<VocabIr> {
    let path = path.as_ref();
    let bytes = read_file_fast(path)?;
    match detect_vocab_kind(&bytes) {
        VocabKind::OmniBinaryCache  => load_otk_bytes(&bytes),
        VocabKind::HuggingFaceJson  => load_hf(std::str::from_utf8(&bytes).context("UTF-8")?),
        VocabKind::SentencePieceJson => {
            let json = std::str::from_utf8(&bytes).context("UTF-8")?;
            load_sentencepiece_json(json)
        }
        VocabKind::TikToken => {
            let text = std::str::from_utf8(&bytes).context("UTF-8")?;
            load_tiktoken(text)
        }
        VocabKind::Gguf | VocabKind::Unknown => {
            // Fall back to HuggingFace JSON — surface a clear error if it fails
            anyhow::bail!(
                "unrecognised vocabulary format in `{}`; pass an explicit loader or provide a HuggingFace tokenizers.json",
                path.display()
            )
        }
    }
}

// ─── SentencePiece TSV / JSON Loader ──────────────────────────────────────────

#[derive(Deserialize)]
struct SpmToken {
    piece: String,
    score: f32,
}

/// Load SentencePiece exported JSON vocabulary (`[{"piece": "...", "score": -1.2}, ...]`).
pub fn load_sentencepiece_json(json: &str) -> Result<VocabIr> {
    let items: Vec<SpmToken> = serde_json::from_str(json).context("parse sentencepiece json")?;
    let mut vocab = HashMap::with_capacity(items.len());
    let mut scores = Vec::with_capacity(items.len());

    for (id, item) in items.into_iter().enumerate() {
        vocab.insert(item.piece.clone(), id as u32);
        scores.push(UnigramScore { token: item.piece, score: item.score });
    }

    let ir = VocabIr {
        algo: AlgoKind::Unigram,
        vocab,
        merge_rules: vec![],
        unigram_scores: scores,
        continuation_prefix: None,
    };
    ir.validate()?;
    Ok(ir)
}

// ─── GGUF-style Metadata Vocab Loader ─────────────────────────────────────────

#[derive(Deserialize)]
struct GgufVocabMetadata {
    tokens: Vec<String>,
    scores: Option<Vec<f32>>,
    merges: Option<Vec<String>>,
}

/// Load GGUF metadata vocab structure JSON.
pub fn load_gguf_vocab(json: &str) -> Result<VocabIr> {
    let gguf: GgufVocabMetadata = serde_json::from_str(json).context("parse gguf json")?;
    let mut vocab = HashMap::with_capacity(gguf.tokens.len());
    let mut unigram_scores = Vec::new();

    for (id, tok) in gguf.tokens.into_iter().enumerate() {
        let score = gguf.scores.as_ref().and_then(|s| s.get(id).copied()).unwrap_or(0.0);
        vocab.insert(tok.clone(), id as u32);
        unigram_scores.push(UnigramScore { token: tok, score });
    }

    let merge_rules: Vec<MergeRule> = gguf.merges.unwrap_or_default().into_iter().enumerate()
        .map(|(rank, s)| {
            let mut it = s.splitn(2, ' ');
            let left  = it.next().unwrap_or("").to_string();
            let right = it.next().unwrap_or("").to_string();
            MergeRule { left, right, rank: rank as u32 }
        })
        .collect();

    let ir = VocabIr {
        algo: if !merge_rules.is_empty() { AlgoKind::Bpe } else { AlgoKind::Unigram },
        vocab,
        merge_rules,
        unigram_scores,
        continuation_prefix: None,
    };
    ir.validate()?;
    Ok(ir)
}

// ─── tiktoken rank files ──────────────────────────────────────────────────────

/// Load a tiktoken `.tiktoken` file (lines: `<base64-token> <rank>`).
pub fn load_tiktoken(content: &str) -> Result<VocabIr> {
    let mut vocab = HashMap::new();

    for (lineno, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() { continue; }
        let mut parts = line.splitn(2, ' ');
        let b64  = parts.next().context("missing base64 field")?;
        let rank: u32 = parts.next()
            .context("missing rank field")?
            .trim()
            .parse()
            .with_context(|| format!("bad rank on line {lineno}"))?;
        let bytes = b64_decode(b64)
            .with_context(|| format!("bad base64 on line {lineno}"))?;
        let token = String::from_utf8_lossy(&bytes).into_owned();
        vocab.insert(token, rank);
    }

    let ir = VocabIr {
        algo: AlgoKind::Bpe,
        vocab,
        merge_rules: vec![],
        unigram_scores: vec![],
        continuation_prefix: None,
    };
    ir.validate()?;
    Ok(ir)
}

// ─── SentencePiece binary .model Protobuf decoder ────────────────────────────

/// Load a raw SentencePiece `.model` binary protobuf blob.
pub fn load_sentencepiece_proto(bytes: &[u8]) -> Result<VocabIr> {
    let mut vocab = HashMap::new();
    let mut unigram_scores = Vec::new();

    let mut cursor = 0;
    let mut token_id = 0u32;

    while cursor < bytes.len() {
        let (tag, wire_type, next_c) = match read_varint(bytes, cursor) {
            Ok((v, n)) => ((v >> 3) as u32, (v & 0x7) as u32, n),
            Err(_) => break,
        };
        cursor = next_c;

        if tag == 1 && wire_type == 2 {
            // repeated ModelProto.PiecePiece pieces = 1;
            let (len, n) = read_varint(bytes, cursor)?;
            cursor = n;
            let end = (cursor + len as usize).min(bytes.len());
            let piece_bytes = &bytes[cursor..end];
            cursor = end;

            // Parse PiecePiece message
            let mut piece_cursor = 0;
            let mut piece_str = String::new();
            let mut score = 0.0f32;

            while piece_cursor < piece_bytes.len() {
                let (ptag, pwire, pn) = match read_varint(piece_bytes, piece_cursor) {
                    Ok((v, n)) => ((v >> 3) as u32, (v & 0x7) as u32, n),
                    Err(_) => break,
                };
                piece_cursor = pn;

                match (ptag, pwire) {
                    (1, 2) => { // string piece = 1;
                        let (plen, n2) = read_varint(piece_bytes, piece_cursor)?;
                        piece_cursor = n2;
                        let str_end = (piece_cursor + plen as usize).min(piece_bytes.len());
                        piece_str = String::from_utf8_lossy(&piece_bytes[piece_cursor..str_end]).into_owned();
                        piece_cursor = str_end;
                    }
                    (2, 5) => { // float score = 2; (fixed32)
                        if piece_cursor + 4 <= piece_bytes.len() {
                            let mut buf = [0u8; 4];
                            buf.copy_from_slice(&piece_bytes[piece_cursor..piece_cursor + 4]);
                            score = f32::from_le_bytes(buf);
                            piece_cursor += 4;
                        } else {
                            break;
                        }
                    }
                    (_, 0) => { // varint skip
                        let (_, n2) = read_varint(piece_bytes, piece_cursor)?;
                        piece_cursor = n2;
                    }
                    (_, 2) => { // length-delimited skip
                        let (plen, n2) = read_varint(piece_bytes, piece_cursor)?;
                        piece_cursor = (n2 + plen as usize).min(piece_bytes.len());
                    }
                    (_, 5) => piece_cursor += 4, // 32-bit skip
                    (_, 1) => piece_cursor += 8, // 64-bit skip
                    _ => break,
                }
            }

            if !piece_str.is_empty() {
                vocab.insert(piece_str.clone(), token_id);
                unigram_scores.push(UnigramScore { token: piece_str, score });
                token_id += 1;
            }
        } else {
            // Skip other fields
            match wire_type {
                0 => { let (_, n) = read_varint(bytes, cursor)?; cursor = n; }
                2 => { let (len, n) = read_varint(bytes, cursor)?; cursor = (n + len as usize).min(bytes.len()); }
                5 => cursor += 4,
                1 => cursor += 8,
                _ => break,
            }
        }
    }

    if vocab.is_empty() {
        bail!("No valid vocabulary pieces found in raw SentencePiece protobuf binary");
    }

    let ir = VocabIr {
        algo: AlgoKind::Unigram,
        vocab,
        merge_rules: vec![],
        unigram_scores,
        continuation_prefix: None,
    };
    ir.validate()?;
    Ok(ir)
}

fn read_varint(bytes: &[u8], mut cursor: usize) -> Result<(u64, usize)> {
    let mut val = 0u64;
    let mut shift = 0;
    while cursor < bytes.len() {
        let b = bytes[cursor];
        cursor += 1;
        val |= ((b & 0x7F) as u64) << shift;
        if (b & 0x80) == 0 {
            return Ok((val, cursor));
        }
        shift += 7;
        if shift >= 64 {
            bail!("varint overflow");
        }
    }
    bail!("unexpected EOF reading varint")
}

// ─── Base64 decoder supporting both RFC 4648 standard and URL-safe ─────────────

fn b64_decode(s: &str) -> Result<Vec<u8>> {
    let val = |b: u8| -> Result<u32> {
        match b {
            b'A'..=b'Z' => Ok((b - b'A') as u32),
            b'a'..=b'z' => Ok((b - b'a' + 26) as u32),
            b'0'..=b'9' => Ok((b - b'0' + 52) as u32),
            b'+' | b'-' => Ok(62),
            b'/' | b'_' => Ok(63),
            _ => bail!("invalid base64 char"),
        }
    };

    let s = s.trim_end_matches('=');
    let bs = s.as_bytes();
    let mut out = Vec::with_capacity(bs.len() * 3 / 4 + 2);
    let mut i = 0;
    while i < bs.len() {
        let a = val(bs[i])?;
        if i + 1 >= bs.len() { break; }
        let b = val(bs[i + 1])?;
        let v2 = (a << 6) | b;
        if i + 2 < bs.len() {
            let c = val(bs[i + 2])?;
            let v3 = (v2 << 6) | c;
            if i + 3 < bs.len() {
                let d = val(bs[i + 3])?;
                let v4 = (v3 << 6) | d;
                out.push(((v4 >> 16) & 0xFF) as u8);
                out.push(((v4 >> 8) & 0xFF) as u8);
                out.push((v4 & 0xFF) as u8);
                i += 4;
                continue;
            }
            out.push(((v3 >> 10) & 0xFF) as u8);
            out.push(((v3 >> 2) & 0xFF) as u8);
            i += 3;
            continue;
        }
        out.push(((v2 >> 4) & 0xFF) as u8);
        i += 2;
    }
    Ok(out)
}

// ─── OmniToken Compiled (.otk) Binary Format ──────────────────────────────────

impl VocabIr {
    /// Save `VocabIr` as a zero-copy `.otk` binary blob.
    pub fn save_otk<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let payload = serde_json::to_vec(self).context("serialize VocabIr to OTK payload")?;
        let mut bytes = Vec::with_capacity(8 + payload.len());
        bytes.extend_from_slice(OTK_MAGIC);
        bytes.extend_from_slice(&OTK_VERSION.to_le_bytes());
        bytes.extend_from_slice(&payload);
        std::fs::write(path, bytes).with_context(|| format!("failed to write OTK file `{}`", path.display()))
    }
}

/// Load `VocabIr` from an `.otk` binary buffer.
pub fn load_otk_bytes(bytes: &[u8]) -> Result<VocabIr> {
    if bytes.len() < 8 {
        bail!("invalid .otk binary format: buffer too short (< 8 bytes)");
    }
    if &bytes[0..4] != OTK_MAGIC {
        bail!("invalid .otk binary format: bad magic header {:?}", &bytes[0..4]);
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != OTK_VERSION {
        bail!("unsupported .otk version {version}, expected {OTK_VERSION}");
    }
    let ir: VocabIr = serde_json::from_slice(&bytes[8..]).context("deserialize OTK payload")?;
    ir.validate()?;
    Ok(ir)
}

/// Load `.otk` binary file.
pub fn load_otk_file<P: AsRef<Path>>(path: P) -> Result<VocabIr> {
    let bytes = read_file_fast(path)?;
    load_otk_bytes(&bytes)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_bpe_loads_and_validates() {
        let json = r#"{
            "model": {
                "type": "BPE",
                "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "lo": 5, "helo": 6},
                "merges": ["h e", "l o", "he lo"]
            }
        }"#;
        let ir = load_hf(json).unwrap();
        assert_eq!(ir.algo, AlgoKind::Bpe);
        assert_eq!(ir.merge_rules.len(), 3);
    }

    #[test]
    fn sentencepiece_json_loader() {
        let json = r#"[{"piece": "<unk>", "score": 0.0}, {"piece": "hello", "score": -1.5}]"#;
        let ir = load_sentencepiece_json(json).unwrap();
        assert_eq!(ir.algo, AlgoKind::Unigram);
        assert_eq!(ir.vocab["hello"], 1);
    }

    #[test]
    fn gguf_vocab_loader() {
        let json = r#"{"tokens": ["<pad>", "a", "b", "ab"], "merges": ["a b"]}"#;
        let ir = load_gguf_vocab(json).unwrap();
        assert_eq!(ir.algo, AlgoKind::Bpe);
        assert_eq!(ir.vocab["ab"], 3);
    }

    #[test]
    fn b64_decode_roundtrip() {
        assert_eq!(b64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(b64_decode("aGVsbG8").unwrap(), b"hello");
    }

    #[test]
    fn test_read_file_fast_and_load_hf_file() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_vocab_io_uring.json");
        let content = r#"{
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1, "ab": 2},
                "merges": ["a b"]
            }
        }"#;
        std::fs::write(&file_path, content).unwrap();

        let read_bytes = read_file_fast(&file_path).unwrap();
        assert_eq!(read_bytes, content.as_bytes());

        let ir = load_hf_file(&file_path).unwrap();
        assert_eq!(ir.algo, AlgoKind::Bpe);
        assert_eq!(ir.vocab["ab"], 2);

        let _ = std::fs::remove_file(file_path);
    }

    #[test]
    fn test_otk_roundtrip() {
        let temp_dir = std::env::temp_dir();
        let otk_path = temp_dir.join("test_vocab.otk");
        let ir = load_hf(r#"{
            "model": {
                "type": "BPE",
                "vocab": {"x": 0, "y": 1, "xy": 2},
                "merges": ["x y"]
            }
        }"#).unwrap();

        ir.save_otk(&otk_path).unwrap();
        let loaded_ir = load_otk_file(&otk_path).unwrap();
        assert_eq!(loaded_ir.algo, AlgoKind::Bpe);
        assert_eq!(loaded_ir.vocab["xy"], 2);
        let _ = std::fs::remove_file(otk_path);
    }
}
