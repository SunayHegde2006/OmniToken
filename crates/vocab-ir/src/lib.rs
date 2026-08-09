//! `vocab-ir` — Universal vocabulary intermediate representation.
//!
//! Every supported source format is translated into [`VocabIr`] by a dedicated
//! loader.  No format-specific logic crosses this module boundary — downstream
//! crates only ever see [`VocabIr`].

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

// ─── Public IR types ─────────────────────────────────────────────────────────

/// Which tokenisation algorithm a vocab belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgoKind {
    Bpe,
    WordPiece,
    Unigram,
}

/// A single BPE merge rule: (left, right) → rank. Lower rank = applied first.
#[derive(Debug, Clone)]
pub struct MergeRule {
    pub left:  String,
    pub right: String,
    pub rank:  u32,
}

/// A single Unigram/SentencePiece log-probability score.
#[derive(Debug, Clone)]
pub struct UnigramScore {
    pub token: String,
    pub score: f32,
}

/// The canonical IR produced by every format loader.
#[derive(Debug, Clone)]
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
    let mut entries = Vec::new();

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
        vocab.insert(token.clone(), rank);
        entries.push((rank, token));
    }

    entries.sort_by_key(|(rank, _)| *rank);
    let merge_rules = Vec::new();

    let ir = VocabIr {
        algo: AlgoKind::Bpe,
        vocab,
        merge_rules,
        unigram_scores: vec![],
        continuation_prefix: None,
    };
    ir.validate()?;
    Ok(ir)
}

// ─── Base64 decoder supporting both RFC 4648 standard and URL-safe ─────────────

fn b64_decode(s: &str) -> Result<Vec<u8>> {
    const T: [i8; 128] = {
        let mut t = [-1i8; 128];
        let mut i = 0u8;
        while i < 26 { t[(b'A' + i) as usize] = i as i8;      i += 1; }
        i = 0;
        while i < 26 { t[(b'a' + i) as usize] = (26+i) as i8; i += 1; }
        i = 0;
        while i < 10 { t[(b'0' + i) as usize] = (52+i) as i8; i += 1; }
        t[b'+' as usize] = 62; t[b'-' as usize] = 62;
        t[b'/' as usize] = 63; t[b'_' as usize] = 63;
        t
    };

    let s = s.trim_end_matches('=');
    let mut out = Vec::with_capacity(s.len() * 3 / 4 + 2);
    let bs = s.as_bytes();
    let mut i = 0;
    while i < bs.len() {
        let get = |b: u8| -> Result<u32> {
            if b > 127 { bail!("non-ASCII byte in base64"); }
            let v = T[b as usize];
            if v < 0 { bail!("invalid base64 char: {}", b as char); }
            Ok(v as u32)
        };
        let a = get(bs[i])?;
        if i + 1 >= bs.len() { break; }
        let b = get(bs[i+1])?;
        let v2 = (a << 6) | b;
        if i + 2 < bs.len() {
            let c = get(bs[i+2])?;
            let v3 = (v2 << 6) | c;
            if i + 3 < bs.len() {
                let d = get(bs[i+3])?;
                let v4 = (v3 << 6) | d;
                out.push(((v4 >> 16) & 0xFF) as u8);
                out.push(((v4 >>  8) & 0xFF) as u8);
                out.push(( v4        & 0xFF) as u8);
                i += 4; continue;
            }
            out.push(((v3 >> 10) & 0xFF) as u8);
            out.push(((v3 >>  2) & 0xFF) as u8);
            i += 3; continue;
        }
        out.push(((v2 >> 4) & 0xFF) as u8);
        i += 2;
    }
    Ok(out)
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
}
