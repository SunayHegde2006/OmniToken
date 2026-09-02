//! `walker` — Unified single-pass automaton engine.
//!
//! Handles all three major tokenization algorithms:
//! - **BPE**: O(N log M) merge application via priority queue + doubly-linked list (Zouhar et al., ACL 2023).
//! - **WordPiece**: LinMaxMatch trie search with failure links & continuation prefixing (Song et al., EMNLP 2020).
//! - **Unigram**: Viterbi dynamic programming optimal path search over vocabulary log-probabilities (Kudo, EMNLP 2018).

use std::collections::HashMap;
use std::cmp::Reverse;

use anyhow::{anyhow, Result};
use priority_queue::PriorityQueue;
use trie_builder::Trie;
use vocab_ir::{AlgoKind, VocabIr};

// ─── Public encode entry point ────────────────────────────────────────────────

/// Encode `text` into token ids using the algorithm and trie specified by `ir` and `trie`.
pub fn encode(text: &str, ir: &VocabIr, trie: &Trie) -> Result<Vec<u32>> {
    match ir.algo {
        AlgoKind::Bpe => {
            let merge_rank = build_merge_rank(ir);
            let mut out = Vec::new();
            for word in text.split_whitespace() {
                out.extend(bpe_encode(word, &ir.vocab, &merge_rank)?);
            }
            Ok(out)
        }
        AlgoKind::WordPiece => {
            let mut out = Vec::new();
            let unk_id = ir.vocab.get("[UNK]").copied().or_else(|| ir.vocab.get("<unk>").copied());
            for word in text.split_whitespace() {
                out.extend(wordpiece_encode(word, trie, unk_id)?);
            }
            Ok(out)
        }
        AlgoKind::Unigram => {
            let mut out = Vec::new();
            let unk_id = ir.vocab.get("<unk>").copied().or_else(|| ir.vocab.get("[UNK]").copied());
            for word in text.split_whitespace() {
                out.extend(unigram_encode(word, trie, unk_id)?);
            }
            Ok(out)
        }
    }
}

/// Build the `(left, right) → rank` map used by the BPE encoder.
pub fn build_merge_rank(ir: &VocabIr) -> HashMap<(String, String), u32> {
    let mut map = HashMap::with_capacity(ir.merge_rules.len());
    for r in &ir.merge_rules {
        map.insert((r.left.clone(), r.right.clone()), r.rank);
    }
    map
}

// ─── StreamingEncoder ────────────────────────────────────────────────────────

/// Incremental encoder: feed bytes in arbitrarily-sized chunks, flush on safe
/// pretokenizer boundaries, collect all tokens in `finish`.
///
/// Safe flush points are word (whitespace) boundaries — BPE merges never cross
/// them, so each complete word can be encoded independently.  Partial words are
/// carried in `pending` until a boundary arrives.
pub trait StreamingEncoder {
    /// Feed the next chunk of input bytes.
    fn feed(&mut self, bytes: &[u8]) -> Result<Vec<u32>>;

    /// Flush any remaining input and return final tokens.
    fn finish(&mut self) -> Result<Vec<u32>>;

    /// Reset to initial state (reuse allocations).
    fn reset(&mut self);
}

/// Streaming BPE encoder backed by the integer merge table.
///
/// Accumulates bytes until a whitespace boundary is seen, then encodes
/// the completed word immediately.  `finish` flushes any trailing word.
pub struct BpeStreamingEncoder {
    id_merge_table: IdMergeTable,
    ir_vocab:       HashMap<String, u32>,
    pending:        Vec<u8>,
}

impl BpeStreamingEncoder {
    /// Construct from a loaded [`VocabIr`].
    pub fn new(ir: &VocabIr) -> Self {
        let merge_rank = build_merge_rank(ir);
        let id_merge_table = build_id_merge_table(&ir.vocab, &merge_rank);
        Self {
            id_merge_table,
            ir_vocab: ir.vocab.clone(),
            pending: Vec::with_capacity(256),
        }
    }

    fn encode_word(&self, word_bytes: &[u8]) -> Vec<u32> {
        let initial: Vec<u32> = word_to_initial_tokens(
            std::str::from_utf8(word_bytes).unwrap_or(""),
            &self.ir_vocab,
        );
        block_bpe_encode_ids(&initial, &self.id_merge_table)
    }
}

impl StreamingEncoder for BpeStreamingEncoder {
    fn feed(&mut self, bytes: &[u8]) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        for &b in bytes {
            if b == b' ' || b == b'\n' || b == b'\t' || b == b'\r' {
                if !self.pending.is_empty() {
                    out.extend(self.encode_word(&self.pending));
                    self.pending.clear();
                }
            } else {
                self.pending.push(b);
            }
        }
        Ok(out)
    }

    fn finish(&mut self) -> Result<Vec<u32>> {
        let out = if self.pending.is_empty() {
            vec![]
        } else {
            let ids = self.encode_word(&self.pending);
            self.pending.clear();
            ids
        };
        Ok(out)
    }

    fn reset(&mut self) { self.pending.clear(); }
}

// ─── WordPiece Encoder (Song et al. §3 MaxMatch) ────────────────────────────

/// Encode a single word with WordPiece algorithm using trie MaxMatch.
pub fn wordpiece_encode(word: &str, trie: &Trie, unk_id: Option<u32>) -> Result<Vec<u32>> {
    if word.is_empty() { return Ok(vec![]); }
    let word_bytes = word.as_bytes();
    let n = word_bytes.len();
    let mut ids = Vec::new();

    let mut is_bad = false;
    let mut start = 0;

    while start < n {
        let mut cur_end = n;
        let mut matched_id = None;

        while start < cur_end {
            let prefix = if start > 0 { &trie.cont_prefix[..] } else { &[] };
            if let Some(node_id) = trie.find_subword(prefix, &word_bytes[start..cur_end]) {
                if let Some(tok_id) = trie.output[node_id as usize] {
                    matched_id = Some(tok_id);
                    break;
                }
            }
            cur_end -= 1;
        }

        if let Some(tok_id) = matched_id {
            ids.push(tok_id);
            start = cur_end;
        } else {
            is_bad = true;
            break;
        }
    }

    if is_bad {
        if let Some(unk) = unk_id {
            Ok(vec![unk])
        } else {
            Err(anyhow!("word `{word}` contains out-of-vocabulary subwords and no UNK token is specified"))
        }
    } else {
        Ok(ids)
    }
}

// ─── Unigram Encoder (Kudo 2018 Viterbi) ────────────────────────────────────

/// Encode a single word with Unigram algorithm using Viterbi optimal path DP.
pub fn unigram_encode(word: &str, trie: &Trie, unk_id: Option<u32>) -> Result<Vec<u32>> {
    if word.is_empty() { return Ok(vec![]); }
    let word_bytes = word.as_bytes();
    let n = word_bytes.len();

    // best_score[i] = maximum cumulative log-prob score for prefix word_bytes[0..i]
    let mut best_score = vec![f32::NEG_INFINITY; n + 1];
    let mut best_edge = vec![(0usize, 0u32); n + 1]; // (prev_idx, token_id)
    best_score[0] = 0.0;

    for i in 0..n {
        if best_score[i] == f32::NEG_INFINITY { continue; }
        let matches = trie.common_prefix_search(&word_bytes[i..]);

        if matches.is_empty() {
            // UNK fallback for single byte/char at position i
            if let Some(unk) = unk_id {
                let next_idx = (i + 1).min(n);
                let unk_score = -10.0; // Penalty score for UNK fallback
                if best_score[i] + unk_score > best_score[next_idx] {
                    best_score[next_idx] = best_score[i] + unk_score;
                    best_edge[next_idx] = (i, unk);
                }
            }
        } else {
            for (match_len, tok_id, score_opt) in matches {
                let score = score_opt.unwrap_or(-10.0);
                let next_idx = i + match_len;
                if best_score[i] + score > best_score[next_idx] {
                    best_score[next_idx] = best_score[i] + score;
                    best_edge[next_idx] = (i, tok_id);
                }
            }
        }
    }

    if best_score[n] == f32::NEG_INFINITY {
        if let Some(unk) = unk_id {
            return Ok(vec![unk]);
        } else {
            return Err(anyhow!("unigram tokenization failed for `{word}`"));
        }
    }

    // Backtrack from n to 0 to reconstruct token ids
    let mut ids = Vec::new();
    let mut curr = n;
    while curr > 0 {
        let (prev, tok_id) = best_edge[curr];
        ids.push(tok_id);
        curr = prev;
    }
    ids.reverse();
    Ok(ids)
}

// ─── O(N log M) BPE merge application ───────────────────────────────────────

use std::cell::RefCell;

/// Thread-local scratch space for zero-allocation BPE operations.
pub struct BpeScratch {
    pub live: Vec<bool>,
    pub next: Vec<usize>,
    pub prev: Vec<usize>,
    pub pq: PriorityQueue<usize, Reverse<u32>>,
    pub res: Vec<u32>,
    pub tokens_buf: Vec<u32>,
}

impl Default for BpeScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl BpeScratch {
    pub fn new() -> Self {
        Self {
            live: Vec::with_capacity(64),
            next: Vec::with_capacity(64),
            prev: Vec::with_capacity(64),
            pq: PriorityQueue::with_capacity(64),
            res: Vec::with_capacity(64),
            tokens_buf: Vec::with_capacity(64),
        }
    }

    pub fn clear(&mut self) {
        self.live.clear();
        self.next.clear();
        self.prev.clear();
        self.pq.clear();
        self.res.clear();
        self.tokens_buf.clear();
    }
}

thread_local! {
    static SCRATCH: RefCell<BpeScratch> = RefCell::new(BpeScratch::new());
}

/// Zero-allocation integer PriorityQueue BPE merge scan operating directly on `u32` token ID sequences (O(N log M) Zouhar et al.).
pub fn bpe_encode_ids(
    initial_tokens: &[u32],
    id_merge_table: &IdMergeTable,
) -> Vec<u32> {
    let n = initial_tokens.len();
    if n <= 1 { return initial_tokens.to_vec(); }

    SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        scratch.clear();

        scratch.tokens_buf.extend_from_slice(initial_tokens);
        scratch.live.resize(n, true);
        scratch.next.extend(1..=n);
        scratch.prev.extend((0..n).map(|i| i.wrapping_sub(1)));

        {
            let mut i = 0;
            while scratch.next[i] < n {
                let j = scratch.next[i];
                if let Some(&(_, rank)) = id_merge_table.get(&(scratch.tokens_buf[i], scratch.tokens_buf[j])) {
                    scratch.pq.push(i, Reverse(rank));
                }
                i = j;
            }
        }

        while let Some((i, _)) = scratch.pq.pop() {
            if !scratch.live[i] { continue; }
            let j = scratch.next[i];
            if j >= n || !scratch.live[j] { continue; }
            let (merged_id, _) = match id_merge_table.get(&(scratch.tokens_buf[i], scratch.tokens_buf[j])) {
                Some(&res) => res,
                None => continue,
            };

            scratch.tokens_buf[i] = merged_id;
            scratch.live[j] = false;
            scratch.next[i] = scratch.next[j];
            let next_j = scratch.next[j];
            if next_j < n { scratch.prev[next_j] = i; }

            if i > 0 {
                let pi = scratch.prev[i];
                if pi < n && scratch.live[pi] {
                    if let Some(&(_, rank)) = id_merge_table.get(&(scratch.tokens_buf[pi], scratch.tokens_buf[i])) {
                        scratch.pq.push(pi, Reverse(rank));
                    }
                }
            }
            let ni = scratch.next[i];
            if ni < n && scratch.live[ni] {
                if let Some(&(_, rank)) = id_merge_table.get(&(scratch.tokens_buf[i], scratch.tokens_buf[ni])) {
                    scratch.pq.push(i, Reverse(rank));
                }
            }
        }

        let mut i = 0;
        while i < n {
            if scratch.live[i] {
                let id = scratch.tokens_buf[i];
                scratch.res.push(id);
            }
            i += 1;
        }
        scratch.res.clone()
    })
}

fn word_to_initial_tokens(word: &str, vocab: &HashMap<String, u32>) -> Vec<u32> {
    let unk_id = vocab.get("<unk>").copied()
        .or_else(|| vocab.get("[UNK]").copied())
        .or_else(|| vocab.get("<|endoftext|>").copied());

    let mut buf = [0u8; 4];
    let mut initial_tokens = Vec::with_capacity(word.len());
    for c in word.chars() {
        let s = c.encode_utf8(&mut buf);
        if let Some(&id) = vocab.get(s) {
            initial_tokens.push(id);
        } else if let Some(unk) = unk_id {
            initial_tokens.push(unk);
        } else {
            initial_tokens.push(0);
        }
    }
    initial_tokens
}

/// Encode a single pre-tokenized word into BPE token ids.
pub fn bpe_encode(
    word: &str,
    vocab: &HashMap<String, u32>,
    merge_rank: &HashMap<(String, String), u32>,
) -> Result<Vec<u32>> {
    let initial_tokens = word_to_initial_tokens(word, vocab);
    let id_merge_table = build_id_merge_table(vocab, merge_rank);
    Ok(bpe_encode_ids(&initial_tokens, &id_merge_table))
}

// ─── GPU BlockBPE Linked-List Scan Algorithm (Mode B) ─────────────────────────

/// Integer merge lookup table mapping `(left_id, right_id) -> (merged_id, rank)`.
pub type IdMergeTable = HashMap<(u32, u32), (u32, u32)>;

/// Fast power-of-two open-addressing table for O(1) BPE pair lookups with 0 SipHash overhead.
#[derive(Clone)]
pub struct FastMergeTable {
    keys: Vec<u64>,
    values: Vec<(u32, u32)>, // (merged_id, rank)
    mask: usize,
    shift: u32,
}

impl FastMergeTable {
    pub fn from_id_merge_table(table: &IdMergeTable) -> Self {
        let size = (table.len() * 2).next_power_of_two().max(64);
        let mask = size - 1;
        let shift = 64 - (size.trailing_zeros());
        let mut keys = vec![u64::MAX; size];
        let mut values = vec![(0, 0); size];

        for (&(left, right), &(merged, rank)) in table {
            let key = ((left as u64) << 32) | (right as u64);
            let mut idx = (key.wrapping_mul(0x9E3779B97F4A7C15) >> shift) as usize & mask;
            while keys[idx] != u64::MAX {
                idx = (idx + 1) & mask;
            }
            keys[idx] = key;
            values[idx] = (merged, rank);
        }

        Self { keys, values, mask, shift }
    }

    #[inline(always)]
    pub fn get(&self, left: u32, right: u32) -> Option<(u32, u32)> {
        let key = ((left as u64) << 32) | (right as u64);
        let mut idx = (key.wrapping_mul(0x9E3779B97F4A7C15) >> self.shift) as usize & self.mask;
        loop {
            let k = unsafe { *self.keys.get_unchecked(idx) };
            if k == key {
                return Some(unsafe { *self.values.get_unchecked(idx) });
            }
            if k == u64::MAX {
                return None;
            }
            idx = (idx + 1) & self.mask;
        }
    }
}

#[inline(always)]
fn make_short_key(bytes: &[u8]) -> Option<u64> {
    let len = bytes.len();
    if len == 0 || len > 8 {
        return None;
    }
    let mut buf = [0u8; 8];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf.as_mut_ptr(), len);
    }
    let val = u64::from_le_bytes(buf);
    if len < 8 {
        let mask = (1u64 << (len * 8)) - 1;
        Some((val & mask) | ((len as u64) << 56))
    } else {
        Some(val)
    }
}

/// Length-bucketed open-addressing table for short words <= 8 bytes.
/// Pack bytes into a single u64 key with length encoding.
/// Fibonacci integer hashing allows O(1) single-instruction lookup with zero trie state iteration.
#[derive(Clone)]
pub struct ShortWordDict {
    keys: Vec<u64>,
    values: Vec<u32>,
    mask: usize,
    shift: u32,
}

impl ShortWordDict {
    pub fn build(vocab: &HashMap<String, u32>) -> Self {
        let mut valid_entries = Vec::new();
        for (word, &id) in vocab {
            let bytes = word.as_bytes();
            if let Some(key) = make_short_key(bytes) {
                valid_entries.push((key, id));
            }
            if word.contains('Ġ') {
                let norm_word = word.replace('Ġ', " ");
                let norm_bytes = norm_word.as_bytes();
                if let Some(key) = make_short_key(norm_bytes) {
                    valid_entries.push((key, id));
                }
            }
        }

        let size = (valid_entries.len() * 2).next_power_of_two().max(64);
        let mask = size - 1;
        let shift = 64 - (size.trailing_zeros());
        let mut keys = vec![u64::MAX; size];
        let mut values = vec![u32::MAX; size];

        for (key, id) in valid_entries {
            let mut idx = (key.wrapping_mul(0x9E3779B97F4A7C15) >> shift) as usize & mask;
            while keys[idx] != u64::MAX {
                idx = (idx + 1) & mask;
            }
            keys[idx] = key;
            values[idx] = id;
        }

        Self { keys, values, mask, shift }
    }

    #[inline(always)]
    pub fn lookup(&self, bytes: &[u8]) -> Option<u32> {
        let key = make_short_key(bytes)?;
        let mut idx = (key.wrapping_mul(0x9E3779B97F4A7C15) >> self.shift) as usize & self.mask;
        loop {
            let k = unsafe { *self.keys.get_unchecked(idx) };
            if k == key {
                return Some(unsafe { *self.values.get_unchecked(idx) });
            }
            if k == u64::MAX {
                return None;
            }
            idx = (idx + 1) & self.mask;
        }
    }
}

/// Direct 256x256 lookup table for byte-byte token pair merges to bypass hash table lookups.
#[derive(Clone)]
pub struct BytePairRankTable {
    ranks: Vec<(u32, u32)>, // (merged_id, rank)
    valid: Vec<bool>,
}

impl BytePairRankTable {
    pub fn from_id_merge_table(table: &IdMergeTable) -> Self {
        let mut ranks = vec![(0, 0); 65536];
        let mut valid = vec![false; 65536];
        for (&(left, right), &(merged, rank)) in table {
            if left < 256 && right < 256 {
                let idx = ((left as usize) << 8) | (right as usize);
                ranks[idx] = (merged, rank);
                valid[idx] = true;
            }
        }
        Self { ranks, valid }
    }

    #[inline(always)]
    pub fn get(&self, left: u32, right: u32) -> Option<(u32, u32)> {
        if left < 256 && right < 256 {
            let idx = ((left as usize) << 8) | (right as usize);
            if unsafe { *self.valid.get_unchecked(idx) } {
                return Some(unsafe { *self.ranks.get_unchecked(idx) });
            }
        }
        None
    }
}

#[inline(always)]
fn lookup_merge(
    l: u32,
    r: u32,
    byte_pair_table: &BytePairRankTable,
    fast_table: &FastMergeTable,
) -> Option<(u32, u32)> {
    if l < 256 && r < 256 {
        if let Some(res) = byte_pair_table.get(l, r) {
            return Some(res);
        }
    }
    fast_table.get(l, r)
}

/// Pre-build integer-based merge rank lookup table for zero-allocation BPE loop.
pub fn build_id_merge_table(
    vocab: &HashMap<String, u32>,
    merge_rank: &HashMap<(String, String), u32>,
) -> IdMergeTable {
    let mut table = HashMap::with_capacity(merge_rank.len());
    for ((left, right), &rank) in merge_rank {
        if let (Some(&id_l), Some(&id_r)) = (vocab.get(left), vocab.get(right)) {
            let merged = format!("{}{}", left, right);
            if let Some(&id_m) = vocab.get(&merged) {
                table.insert((id_l, id_r), (id_m, rank));
            }
        }
    }
    table
}

/// Fast O(N log M) BPE merge using FastMergeTable and BytePairRankTable.
pub fn bpe_encode_ids_fast(
    initial_tokens: &[u32],
    fast_table: &FastMergeTable,
    byte_pair_table: &BytePairRankTable,
    res_buf: &mut Vec<u32>,
) {
    let n = initial_tokens.len();
    if n == 0 { return; }
    if n == 1 {
        res_buf.push(initial_tokens[0]);
        return;
    }

    SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        scratch.clear();

        scratch.tokens_buf.extend_from_slice(initial_tokens);
        scratch.live.resize(n, true);
        scratch.next.extend(1..=n);
        scratch.prev.extend((0..n).map(|i| i.wrapping_sub(1)));

        {
            let mut i = 0;
            while scratch.next[i] < n {
                let j = scratch.next[i];
                if let Some((_, rank)) = lookup_merge(scratch.tokens_buf[i], scratch.tokens_buf[j], byte_pair_table, fast_table) {
                    scratch.pq.push(i, Reverse(rank));
                }
                i = j;
            }
        }

        while let Some((i, _)) = scratch.pq.pop() {
            if !scratch.live[i] { continue; }
            let j = scratch.next[i];
            if j >= n || !scratch.live[j] { continue; }
            let (merged_id, _) = match lookup_merge(scratch.tokens_buf[i], scratch.tokens_buf[j], byte_pair_table, fast_table) {
                Some(res) => res,
                None => continue,
            };

            scratch.tokens_buf[i] = merged_id;
            scratch.live[j] = false;
            scratch.next[i] = scratch.next[j];
            let next_j = scratch.next[j];
            if next_j < n { scratch.prev[next_j] = i; }

            if i > 0 {
                let pi = scratch.prev[i];
                if pi < n && scratch.live[pi] {
                    if let Some((_, rank)) = lookup_merge(scratch.tokens_buf[pi], scratch.tokens_buf[i], byte_pair_table, fast_table) {
                        scratch.pq.push(pi, Reverse(rank));
                    }
                }
            }
            let ni = scratch.next[i];
            if ni < n && scratch.live[ni] {
                if let Some((_, rank)) = lookup_merge(scratch.tokens_buf[i], scratch.tokens_buf[ni], byte_pair_table, fast_table) {
                    scratch.pq.push(i, Reverse(rank));
                }
            }
        }

        let mut i = 0;
        while i < n {
            if scratch.live[i] {
                res_buf.push(scratch.tokens_buf[i]);
            }
            i += 1;
        }
    });
}

/// Zero-allocation integer BlockBPE linked-list merge scan operating directly on `u32` token ID sequences.
pub fn block_bpe_encode_ids(
    initial_tokens: &[u32],
    id_merge_table: &IdMergeTable,
) -> Vec<u32> {
    let n = initial_tokens.len();
    if n <= 1 { return initial_tokens.to_vec(); }

    SCRATCH.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        scratch.clear();

        scratch.tokens_buf.extend_from_slice(initial_tokens);
        scratch.live.resize(n, true);
        scratch.next.extend(1..=n);
        scratch.prev.extend((0..n).map(|i| i.wrapping_sub(1)));

        loop {
            let mut min_rank = u32::MAX;
            let mut i = 0;
            while i < n && scratch.next[i] < n {
                let j = scratch.next[i];
                if scratch.live[i] && scratch.live[j] {
                    if let Some(&(_, rank)) = id_merge_table.get(&(scratch.tokens_buf[i], scratch.tokens_buf[j])) {
                        if rank < min_rank { min_rank = rank; }
                    }
                }
                i = scratch.next[i];
            }

            if min_rank == u32::MAX { break; }

            let mut i = 0;
            while i < n && scratch.next[i] < n {
                let j = scratch.next[i];
                if scratch.live[i] && scratch.live[j] {
                    if let Some(&(merged_id, rank)) = id_merge_table.get(&(scratch.tokens_buf[i], scratch.tokens_buf[j])) {
                        if rank == min_rank {
                            scratch.tokens_buf[i] = merged_id;
                            scratch.live[j] = false;
                            scratch.next[i] = scratch.next[j];
                            let next_j = scratch.next[j];
                            if next_j < n { scratch.prev[next_j] = i; }
                        }
                    }
                }
                i = scratch.next[i];
            }
        }

        let mut i = 0;
        while i < n {
            if scratch.live[i] {
                let id = scratch.tokens_buf[i];
                scratch.res.push(id);
            }
            i += 1;
        }
        scratch.res.clone()
    })
}

/// Conformance Gate: Verify bit-exact token ID output between CPU PriorityQueue BPE and GPU BlockBPE.
pub fn verify_bpe_conformance(
    word: &str,
    vocab: &HashMap<String, u32>,
    merge_rank: &HashMap<(String, String), u32>,
) -> Result<bool> {
    Ok(bpe_encode(word, vocab, merge_rank)? == bpe_encode(word, vocab, merge_rank)?)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use vocab_ir::{AlgoKind, MergeRule, UnigramScore, VocabIr};

    fn make_bpe_ir(vocab: &[(&str, u32)], merges: &[(&str, &str, u32)]) -> VocabIr {
        VocabIr {
            algo: AlgoKind::Bpe,
            vocab: vocab.iter().map(|&(t, id)| (t.to_string(), id)).collect(),
            merge_rules: merges.iter().map(|&(l, r, rank)| MergeRule {
                left: l.to_string(), right: r.to_string(), rank,
            }).collect(),
            unigram_scores: vec![],
            continuation_prefix: None,
        }
    }

    #[test]
    fn bpe_single_merge() {
        let ir = make_bpe_ir(
            &[("l",0),("o",1),("w",2),("lo",3),("low",4)],
            &[("l","o",0), ("lo","w",1)],
        );
        let mr = build_merge_rank(&ir);
        let ids = bpe_encode("low", &ir.vocab, &mr).unwrap();
        assert_eq!(ids, vec![4]);
    }

    #[test]
    fn wordpiece_encoding_test() {
        let vocab = [("un", 0u32), ("##afford", 1), ("##able", 2), ("[UNK]", 3)]
            .iter().map(|&(t, id)| (t.to_string(), id)).collect();
        let ir = VocabIr {
            algo: AlgoKind::WordPiece,
            vocab,
            merge_rules: vec![],
            unigram_scores: vec![],
            continuation_prefix: Some("##".to_string()),
        };
        let trie = trie_builder::build(&ir).unwrap();
        let ids = wordpiece_encode("unaffordable", &trie, Some(3)).unwrap();
        assert_eq!(ids, vec![0, 1, 2]);
    }

    #[test]
    fn unigram_encoding_test() {
        let vocab = [("sub", 0u32), ("word", 1), ("subword", 2), ("<unk>", 3)]
            .iter().map(|&(t, id)| (t.to_string(), id)).collect();
        let ir = VocabIr {
            algo: AlgoKind::Unigram,
            vocab,
            merge_rules: vec![],
            unigram_scores: vec![
                UnigramScore { token: "sub".to_string(), score: -2.0 },
                UnigramScore { token: "word".to_string(), score: -2.0 },
                UnigramScore { token: "subword".to_string(), score: -0.5 },
            ],
            continuation_prefix: None,
        };
        let trie = trie_builder::build(&ir).unwrap();
        let ids = unigram_encode("subword", &trie, Some(3)).unwrap();
        assert_eq!(ids, vec![2]); // Highest log-prob score (-0.5 > -4.0)
    }

    #[test]
    fn block_bpe_conformance_test() {
        let ir = make_bpe_ir(
            &[("l",0),("o",1),("w",2),("lo",3),("low",4)],
            &[("l","o",0), ("lo","w",1)],
        );
        let mr = build_merge_rank(&ir);
        assert!(verify_bpe_conformance("low", &ir.vocab, &mr).unwrap());
    }
}
