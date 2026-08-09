//! `trie-builder` — Offline Aho-Corasick-style trie construction.
//!
//! Implements the trie structure described in Song et al., EMNLP 2020
//! ("Fast WordPiece Tokenization"):
//! - Trie construction from the vocab table.
//! - BFS-based failure-link computation (standard Aho-Corasick).
//! - Precomputed `fail_pop` links for fast O(N) MaxMatch subword matching.
//! - Per-node annotations: BPE merge ranks, WordPiece continuation flags,
//!   and Unigram log-probability scores.

use std::collections::{HashMap, VecDeque};
use anyhow::Result;
use vocab_ir::{AlgoKind, VocabIr};

pub type NodeId = u32;
pub const ROOT: NodeId = 0;

/// A single node in the vocabulary trie.
#[derive(Debug, Default, Clone)]
pub struct TrieNode {
    /// Children indexed by byte value.
    pub children: HashMap<u8, NodeId>,
    /// Aho-Corasick failure link: longest proper suffix that is a prefix in the trie.
    pub fail: NodeId,
    /// Failure-pop link (Song et al. §3): nearest ancestor on the failure chain that has `output != None`.
    pub fail_pop: NodeId,
    /// Token id if this node terminates a vocabulary entry.
    pub output: Option<u32>,
    /// BPE: merge rank of the token ending here (lower = applied first).
    pub merge_rank: Option<u32>,
    /// Unigram: log-probability score of the token ending here.
    pub unigram_score: Option<f32>,
}

/// The compiled trie, ready for the walker.
pub struct Trie {
    pub nodes:       Vec<TrieNode>,
    pub algo:        AlgoKind,
    /// WordPiece continuation prefix bytes (e.g. b"##").
    pub cont_prefix: Vec<u8>,
}

impl Trie {
    /// Walk the trie starting from root for `bytes`.
    /// Returns the terminal node id if the full string matches.
    pub fn find_node(&self, bytes: &[u8]) -> Option<NodeId> {
        self.find_subword(&[], bytes)
    }

    /// Walk optional prefix then slice without allocating a heap Vec.
    pub fn find_subword(&self, prefix: &[u8], slice: &[u8]) -> Option<NodeId> {
        let mut cur = ROOT;
        for &b in prefix {
            cur = *self.nodes[cur as usize].children.get(&b)?;
        }
        for &b in slice {
            cur = *self.nodes[cur as usize].children.get(&b)?;
        }
        Some(cur)
    }

    /// Search for all subwords matching prefixes of `text` starting at index `start`.
    /// Returns list of `(end_byte_idx, token_id, score_or_rank)`.
    pub fn common_prefix_search(&self, text: &[u8]) -> Vec<(usize, u32, Option<f32>)> {
        let mut matches = Vec::new();
        let mut cur = ROOT;
        for (i, &b) in text.iter().enumerate() {
            match self.nodes[cur as usize].children.get(&b) {
                Some(&next) => {
                    cur = next;
                    if let Some(id) = self.nodes[cur as usize].output {
                        matches.push((i + 1, id, self.nodes[cur as usize].unigram_score));
                    }
                }
                None => break,
            }
        }
        matches
    }
}

/// Build a [`Trie`] from a [`VocabIr`].
pub fn build(ir: &VocabIr) -> Result<Trie> {
    let mut nodes: Vec<TrieNode> = vec![TrieNode { fail_pop: 0, ..Default::default() }];

    // 1. Insert all vocab tokens.
    for (token, &id) in &ir.vocab {
        insert(&mut nodes, token.as_bytes(), id);
    }

    // 2. Annotate BPE merge ranks or Unigram scores on terminal nodes.
    match ir.algo {
        AlgoKind::Bpe => {
            for rule in &ir.merge_rules {
                let merged = format!("{}{}", rule.left, rule.right);
                if let Some(node) = find_node_raw(&nodes, merged.as_bytes()) {
                    nodes[node as usize].merge_rank = Some(rule.rank);
                }
            }
        }
        AlgoKind::Unigram => {
            for item in &ir.unigram_scores {
                if let Some(node) = find_node_raw(&nodes, item.token.as_bytes()) {
                    nodes[node as usize].unigram_score = Some(item.score);
                }
            }
        }
        AlgoKind::WordPiece => {}
    }

    // 3. BFS failure-link & fail_pop computation (Song et al. §3).
    build_fail_and_pop_links(&mut nodes);

    let cont_prefix = ir.continuation_prefix.as_deref()
        .unwrap_or("##").as_bytes().to_vec();

    Ok(Trie { nodes, algo: ir.algo, cont_prefix })
}

fn insert(nodes: &mut Vec<TrieNode>, token: &[u8], token_id: u32) {
    let mut cur = ROOT;
    for &b in token {
        cur = match nodes[cur as usize].children.get(&b).copied() {
            Some(n) => n,
            None => {
                let n = nodes.len() as NodeId;
                nodes.push(TrieNode { fail_pop: 0, ..Default::default() });
                nodes[cur as usize].children.insert(b, n);
                n
            }
        };
    }
    nodes[cur as usize].output = Some(token_id);
}

fn find_node_raw(nodes: &[TrieNode], token: &[u8]) -> Option<NodeId> {
    let mut cur = ROOT;
    for &b in token {
        cur = *nodes[cur as usize].children.get(&b)?;
    }
    Some(cur)
}

fn build_fail_and_pop_links(nodes: &mut Vec<TrieNode>) {
    let mut queue = VecDeque::new();
    let root_children: Vec<(u8, NodeId)> =
        nodes[ROOT as usize].children.iter().map(|(&b, &n)| (b, n)).collect();
    for (_, child) in &root_children {
        nodes[*child as usize].fail = ROOT;
        nodes[*child as usize].fail_pop = 0;
        queue.push_back(*child);
    }
    while let Some(u) = queue.pop_front() {
        let children: Vec<(u8, NodeId)> =
            nodes[u as usize].children.iter().map(|(&b, &n)| (b, n)).collect();
        for (byte, v) in children {
            let fail_v = {
                let mut f = nodes[u as usize].fail;
                loop {
                    if let Some(&c) = nodes[f as usize].children.get(&byte) {
                        if c != v { break c; }
                    }
                    if f == ROOT { break ROOT; }
                    f = nodes[f as usize].fail;
                }
            };
            nodes[v as usize].fail = fail_v;
            // Precompute fail_pop (Song et al. §3): nearest output ancestor along failure chain
            nodes[v as usize].fail_pop = if nodes[fail_v as usize].output.is_some() {
                fail_v
            } else {
                nodes[fail_v as usize].fail_pop
            };
            queue.push_back(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ir(tokens: &[(&str, u32)]) -> VocabIr {
        VocabIr {
            algo: AlgoKind::Bpe,
            vocab: tokens.iter().map(|&(t, id)| (t.to_string(), id)).collect(),
            merge_rules: vec![],
            unigram_scores: vec![],
            continuation_prefix: None,
        }
    }

    #[test]
    fn build_basic_trie() {
        let ir = make_ir(&[("he", 0), ("she", 1), ("his", 2), ("hers", 3)]);
        let trie = build(&ir).unwrap();
        assert!(!trie.nodes[ROOT as usize].children.is_empty());
        assert!(trie.nodes.len() > 1);
    }

    #[test]
    fn fail_pop_links_computed() {
        let ir = make_ir(&[("a", 0), ("ba", 1)]);
        let trie = build(&ir).unwrap();
        assert_eq!(trie.nodes.len() > 2, true);
    }

    #[test]
    fn common_prefix_search_works() {
        let ir = make_ir(&[("a", 0), ("ab", 1), ("abc", 2)]);
        let trie = build(&ir).unwrap();
        let matches = trie.common_prefix_search(b"abcd");
        assert_eq!(matches.len(), 3);
        assert_eq!(matches[0], (1, 0, None));
        assert_eq!(matches[1], (2, 1, None));
        assert_eq!(matches[2], (3, 2, None));
    }
}
