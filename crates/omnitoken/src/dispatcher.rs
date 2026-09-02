//! `dispatcher` — Adaptive CPU/GPU Hybrid Workload Dispatcher.

use anyhow::Result;
use rayon::prelude::*;
use trie_builder::Trie;
use vocab_ir::VocabIr;


/// Hardware execution target selected by the performance model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionTarget {
    CpuLowLatency,
    CpuParallelSimd,
    GpuWarpBlockBpe,
}

use std::cell::RefCell;
use hashbrown::HashMap;

thread_local! {
    static HOT_CACHE: RefCell<HashMap<Vec<u8>, Vec<u32>>> = RefCell::new(HashMap::with_capacity(1024));
}

/// Adaptive Dispatcher: queries payload size, thread availability, and hardware topology.
pub struct AdaptiveDispatcher {
    pub cpu_threshold_bytes: usize,
    pub num_threads: usize,
}

impl Default for AdaptiveDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveDispatcher {
    pub fn new() -> Self {
        let threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        Self {
            cpu_threshold_bytes: 4096,
            num_threads: threads,
        }
    }

    /// Select optimal execution target for a payload of `bytes` length.
    pub fn select_target(&self, bytes: usize) -> ExecutionTarget {
        if bytes < self.cpu_threshold_bytes {
            ExecutionTarget::CpuLowLatency
        } else if self.num_threads > 1 {
            ExecutionTarget::CpuParallelSimd
        } else {
            ExecutionTarget::GpuWarpBlockBpe
        }
    }

    /// Calculate dynamic chunk size based on total pretokens and worker thread count.
    #[inline]
    pub fn calculate_chunk_size(&self, num_pretokens: usize) -> usize {
        if self.num_threads <= 1 || num_pretokens == 0 {
            return num_pretokens.max(1);
        }
        (num_pretokens / (self.num_threads * 4)).clamp(4096, 65536)
    }

/// Pre-build 256-element byte-to-token-ID lookup array to eliminate String allocations in hot path.
pub fn build_byte_vocab_table(
    vocab: &std::collections::HashMap<String, u32>,
    unk_id: Option<u32>,
) -> [u32; 256] {
    let mut table = [unk_id.unwrap_or(0); 256];
    for b in 0..=255u8 {
        if let Ok(valid_str) = std::str::from_utf8(&[b]) {
            if let Some(&id) = vocab.get(valid_str) {
                table[b as usize] = id;
            }
        }
    }
    table
}

    /// Dispatch tokenization request based on adaptive routing rules.
    pub fn encode(&self, text: &str, ir: &VocabIr, trie: &Trie) -> Result<Vec<u32>> {
        let bytes = text.len();
        match self.select_target(bytes) {
            ExecutionTarget::CpuLowLatency => walker::encode(text, ir, trie),
            ExecutionTarget::CpuParallelSimd => {
                if ir.algo == vocab_ir::AlgoKind::Bpe {
                    let merge_rank = walker::build_merge_rank(ir);
                    let id_merge_table = walker::build_id_merge_table(&ir.vocab, &merge_rank);
                    let unk_id = ir.vocab.get("<unk>").copied()
                        .or_else(|| ir.vocab.get("[UNK]").copied())
                        .or_else(|| ir.vocab.get("<|endoftext|>").copied());
                    let byte_table = Self::build_byte_vocab_table(&ir.vocab, unk_id);

                    let pretokens = pretokenizer::split_pretokens(text.as_bytes());
                    let chunk_size = self.calculate_chunk_size(pretokens.len());
                    let res: Result<Vec<Vec<u32>>> = pretokens.chunks(chunk_size).map(|chunk| {
                        let mut thread_tokens = Vec::with_capacity(chunk.len() * 2);
                        HOT_CACHE.with(|c| {
                            let mut cache = c.borrow_mut();
                            for &(start, end) in chunk {
                                let word_bytes = &text.as_bytes()[start..end];
                                if let Some(cached) = cache.get(word_bytes) {
                                    thread_tokens.extend_from_slice(cached);
                                } else {
                                    let mut initial_tokens = Vec::with_capacity(word_bytes.len());
                                    for &b in word_bytes {
                                        initial_tokens.push(byte_table[b as usize]);
                                    }
                                    let res_ids = walker::bpe_encode_ids(&initial_tokens, &id_merge_table);
                                    thread_tokens.extend_from_slice(&res_ids);
                                    cache.insert(word_bytes.to_vec(), res_ids);
                                }
                            }
                        });
                        Ok(thread_tokens)
                    }).collect();
                    Ok(res?.into_iter().flatten().collect())
                } else {
                    walker::encode(text, ir, trie)
                }
            }
            ExecutionTarget::GpuWarpBlockBpe => {
                if ir.algo == vocab_ir::AlgoKind::Bpe {
                    let merge_rank = walker::build_merge_rank(ir);
                    let id_merge_table = walker::build_id_merge_table(&ir.vocab, &merge_rank);
                    let unk_id = ir.vocab.get("<unk>").copied()
                        .or_else(|| ir.vocab.get("[UNK]").copied())
                        .or_else(|| ir.vocab.get("<|endoftext|>").copied());
                    let byte_table = Self::build_byte_vocab_table(&ir.vocab, unk_id);

                    let pretokens = pretokenizer::split_pretokens(text.as_bytes());
                    let chunk_size = self.calculate_chunk_size(pretokens.len());
                    let res: Result<Vec<Vec<u32>>> = pretokens.chunks(chunk_size).map(|chunk| {
                        let mut thread_tokens = Vec::with_capacity(chunk.len() * 2);
                        HOT_CACHE.with(|c| {
                            let mut cache = c.borrow_mut();
                            for &(start, end) in chunk {
                                let word_bytes = &text.as_bytes()[start..end];
                                if let Some(cached) = cache.get(word_bytes) {
                                    thread_tokens.extend_from_slice(cached);
                                } else {
                                    let mut initial_tokens = Vec::with_capacity(word_bytes.len());
                                    for &b in word_bytes {
                                        initial_tokens.push(byte_table[b as usize]);
                                    }
                                    let res_ids = walker::block_bpe_encode_ids(&initial_tokens, &id_merge_table);
                                    thread_tokens.extend_from_slice(&res_ids);
                                    cache.insert(word_bytes.to_vec(), res_ids);
                                }
                            }
                        });
                        Ok(thread_tokens)
                    }).collect();
                    Ok(res?.into_iter().flatten().collect())
                } else {
                    walker::encode(text, ir, trie)
                }
            }
        }
    }    /// Dedicated Bulk Fast Path (Improvement 3 & 2.md): Static Rayon parallel dispatch with ShortWordDict O(1) lookups,
    /// worker-local zero-allocation outputs, byte-pair table acceleration, DAT trie match fast-path, and coverage stats.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_bulk_fast(
        &self,
        text: &str,
        _ir: &VocabIr,
        trie: Option<&Trie>,
        short_dict: Option<&walker::ShortWordDict>,
        fast_merge_table: &walker::FastMergeTable,
        byte_pair_table: &walker::BytePairRankTable,
        byte_table: &[u32; 256],
        flatten: bool,
        chunk_kb: Option<usize>,
    ) -> Result<(Vec<Vec<u32>>, FastPathStats)> {
        let bytes = text.as_bytes();
        let n = bytes.len();
        if n == 0 {
            return Ok((vec![], FastPathStats::default()));
        }

        // Determine chunking size
        let target_chunk_bytes = match chunk_kb {
            Some(kb) => (kb * 1024).max(4096),
            None => {
                let num_chunks = (self.num_threads * 4).max(8);
                (n / num_chunks).max(4096)
            }
        };

        let mut chunk_spans = Vec::new();
        let mut start = 0usize;
        while start < n {
            let mut end = (start + target_chunk_bytes).min(n);
            while end < n && bytes[end] != b' ' && bytes[end] != b'\n' && bytes[end] != b'\t' && bytes[end] != b'\r' {
                end += 1;
            }
            if end < n {
                end += 1; // Include trailing space in chunk
            }
            chunk_spans.push(&bytes[start..end]);
            start = end;
        }

        let results: Vec<(Vec<u32>, FastPathStats)> = chunk_spans
            .into_par_iter()
            .map(|chunk_bytes| {
                let mut thread_tokens = Vec::with_capacity(chunk_bytes.len() / 4);
                let mut initial_tokens = Vec::with_capacity(64);
                let mut stats = FastPathStats { total_bytes: chunk_bytes.len(), ..Default::default() };

                let n = chunk_bytes.len();
                let mut i = 0usize;

                while i < n {
                    // Skip whitespace
                    while i < n && (chunk_bytes[i] == b' ' || chunk_bytes[i] == b'\n' || chunk_bytes[i] == b'\t' || chunk_bytes[i] == b'\r') {
                        i += 1;
                    }
                    if i >= n { break; }

                    let word_start = i;
                    let mut matched_token = None;

                    // 1. Try O(1) ShortWordDict lookup first (length <= 8)
                    let word_end_peek = {
                        let mut p = i;
                        while p < n && chunk_bytes[p] != b' ' && chunk_bytes[p] != b'\n' && chunk_bytes[p] != b'\t' && chunk_bytes[p] != b'\r' {
                            p += 1;
                        }
                        p
                    };

                    if let Some(dict) = short_dict {
                        // Candidate A: with preceding space (GPT-2 style) if available
                        if word_start > 0 && chunk_bytes[word_start - 1] == b' ' {
                            let space_word = &chunk_bytes[word_start - 1..word_end_peek];
                            if let Some(tok_id) = dict.lookup(space_word) {
                                matched_token = Some(tok_id);
                                i = word_end_peek;
                            }
                        }
                        // Candidate B: exact word slice without space
                        if matched_token.is_none() {
                            let word_bytes_peek = &chunk_bytes[word_start..word_end_peek];
                            if let Some(tok_id) = dict.lookup(word_bytes_peek) {
                                matched_token = Some(tok_id);
                                i = word_end_peek;
                            }
                        }
                    }

                    // 2. Fall back to DAT Trie if ShortWordDict didn't match
                    if matched_token.is_none() {
                        if let Some(t) = trie {
                            // Try Trie with preceding space first
                            if word_start > 0 && chunk_bytes[word_start - 1] == b' ' {
                                let mut state = trie_builder::ROOT;
                                let space_slice = &chunk_bytes[word_start - 1..word_end_peek];
                                let mut valid = true;
                                for &b in space_slice {
                                    if let Some(next_state) = t.transition(state, b) {
                                        state = next_state;
                                    } else {
                                        valid = false;
                                        break;
                                    }
                                }
                                if valid && state != trie_builder::ROOT {
                                    if let Some(tok_id) = unsafe { *t.output.get_unchecked(state as usize) } {
                                        matched_token = Some(tok_id);
                                        i = word_end_peek;
                                    }
                                }
                            }

                            if matched_token.is_none() {
                                let mut state = trie_builder::ROOT;
                                let mut valid = true;
                                while i < n && chunk_bytes[i] != b' ' && chunk_bytes[i] != b'\n' && chunk_bytes[i] != b'\t' && chunk_bytes[i] != b'\r' {
                                    if valid {
                                        if let Some(next_state) = t.transition(state, chunk_bytes[i]) {
                                            state = next_state;
                                        } else {
                                            valid = false;
                                        }
                                    }
                                    i += 1;
                                }
                                if valid && state != trie_builder::ROOT {
                                    matched_token = unsafe { *t.output.get_unchecked(state as usize) };
                                }
                            }
                        } else {
                            i = word_end_peek;
                        }
                    }

                    let word_len = i - word_start;
                    stats.total_words += 1;

                    if let Some(tok_id) = matched_token {
                        thread_tokens.push(tok_id);
                        stats.fast_words += 1;
                        stats.fast_bytes += word_len;
                    } else {
                        stats.fallback_words += 1;
                        stats.fallback_bytes += word_len;

                        let word_bytes = &chunk_bytes[word_start..i];
                        let pretokens = pretokenizer::split_pretokens(word_bytes);
                        for &(ps, pe) in &pretokens {
                            let sub_bytes = &word_bytes[ps..pe];
                            initial_tokens.clear();
                            for &b in sub_bytes {
                                initial_tokens.push(byte_table[b as usize]);
                            }
                            walker::bpe_encode_ids_fast(&initial_tokens, fast_merge_table, byte_pair_table, &mut thread_tokens);
                        }
                    }
                }
                (thread_tokens, stats)
            })
            .collect();

        let mut agg_stats = FastPathStats::default();
        let mut chunk_outputs = Vec::with_capacity(results.len());
        for (tokens, stats) in results {
            agg_stats.total_bytes += stats.total_bytes;
            agg_stats.total_words += stats.total_words;
            agg_stats.fast_words += stats.fast_words;
            agg_stats.fast_bytes += stats.fast_bytes;
            agg_stats.fallback_words += stats.fallback_words;
            agg_stats.fallback_bytes += stats.fallback_bytes;
            chunk_outputs.push(tokens);
        }

        if flatten {
            let total_len: usize = chunk_outputs.iter().map(|c| c.len()).sum();
            let mut flat = Vec::with_capacity(total_len);
            for c in chunk_outputs {
                flat.extend_from_slice(&c);
            }
            Ok((vec![flat], agg_stats))
        } else {
            Ok((chunk_outputs, agg_stats))
        }
    }

    /// Dispatch batch tokenization request across parallel workers.
    pub fn encode_batch(&self, texts: &[&str], ir: &VocabIr, trie: &Trie) -> Result<Vec<Vec<u32>>> {
        texts.par_iter().map(|text| self.encode(text, ir, trie)).collect()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct FastPathStats {
    pub total_bytes: usize,
    pub total_words: usize,
    pub fast_words: usize,
    pub fast_bytes: usize,
    pub fallback_words: usize,
    pub fallback_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_chunk_sizing() {
        let disp_single = AdaptiveDispatcher { cpu_threshold_bytes: 4096, num_threads: 1 };
        assert_eq!(disp_single.calculate_chunk_size(100_000), 100_000);

        let disp_multi = AdaptiveDispatcher { cpu_threshold_bytes: 4096, num_threads: 16 };
        // 1,000,000 / (16 * 4) = 15,625, clamped within [4096, 65536]
        assert_eq!(disp_multi.calculate_chunk_size(1_000_000), 15625);
        assert_eq!(disp_multi.calculate_chunk_size(1000), 4096);
        assert_eq!(disp_multi.calculate_chunk_size(100_000_000), 65536);
    }
}
