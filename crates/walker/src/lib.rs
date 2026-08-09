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
            let mut sub = Vec::new();
            if start > 0 {
                sub.extend_from_slice(&trie.cont_prefix);
            }
            sub.extend_from_slice(&word_bytes[start..cur_end]);

            if let Some(node_id) = trie.find_node(&sub) {
                if let Some(tok_id) = trie.nodes[node_id as usize].output {
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

/// Encode a single pre-tokenized word into BPE token ids.
pub fn bpe_encode(
    word: &str,
    vocab: &HashMap<String, u32>,
    merge_rank: &HashMap<(String, String), u32>,
) -> Result<Vec<u32>> {
    let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
    let n = syms.len();

    if n == 0 { return Ok(vec![]); }
    if n == 1 {
        return Ok(vec![*vocab.get(&syms[0])
            .ok_or_else(|| anyhow!("unknown token: {:?}", syms[0]))?]);
    }

    let mut live = vec![true; n];
    let mut next: Vec<usize> = (1..=n).collect();
    let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();

    let mut pq: PriorityQueue<usize, Reverse<u32>> = PriorityQueue::new();
    {
        let mut i = 0;
        while next[i] < n {
            let j = next[i];
            if let Some(&rank) = merge_rank.get(&(syms[i].clone(), syms[j].clone())) {
                pq.push(i, Reverse(rank));
            }
            i = j;
        }
    }

    while let Some((i, _)) = pq.pop() {
        if !live[i] { continue; }
        let j = next[i];
        if j >= n || !live[j] { continue; }
        if merge_rank.get(&(syms[i].clone(), syms[j].clone())).is_none() { continue; }

        let merged = format!("{}{}", syms[i], syms[j]);
        syms[i] = merged;
        live[j] = false;
        next[i] = next[j];
        if next[j] < n { prev[next[j]] = i; }

        if i > 0 {
            let pi = prev[i];
            if pi < n && live[pi] {
                if let Some(&rank) = merge_rank.get(&(syms[pi].clone(), syms[i].clone())) {
                    pq.push(pi, Reverse(rank));
                }
            }
        }
        let ni = next[i];
        if ni < n && live[ni] {
            if let Some(&rank) = merge_rank.get(&(syms[i].clone(), syms[ni].clone())) {
                pq.push(i, Reverse(rank));
            }
        }
    }

    let mut ids = Vec::new();
    let mut i = 0;
    while i < n {
        if live[i] {
            let id = *vocab.get(&syms[i])
                .ok_or_else(|| anyhow!("merged token not in vocab: {:?}", syms[i]))?;
            ids.push(id);
        }
        i += 1;
    }
    Ok(ids)
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
}
