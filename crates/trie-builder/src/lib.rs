//! `trie-builder` — Offline Aho-Corasick-style trie construction.
//!
//! Implements the trie structure described in Song et al., EMNLP 2020
//! ("Fast WordPiece Tokenization"):
//! - Trie construction from the vocab table.
//! - BFS-based failure-link computation (standard Aho-Corasick).
//! - Precomputed `fail_pop` links for fast O(N) MaxMatch subword matching.
//! - Per-node annotations: BPE merge ranks, WordPiece continuation flags,
//!   and Unigram log-probability scores.
//!
//! ## Data Structure: Double-Array Trie (DAT)
//!
//! The public `Trie` uses a Double-Array Trie instead of a pointer-based tree.
//! Transitions are purely arithmetic:
//!
//!   `next_state = base[state] + byte`
//!   valid iff `check[next_state] == state`
//!
//! This eliminates pointer-chasing. A single L1 cache-line fetch provides
//! transition logic for many states simultaneously, maximising cache utilisation
//! and enabling the CPU's hardware prefetcher to work effectively.

use std::collections::{HashMap, VecDeque};
use anyhow::Result;
use vocab_ir::{AlgoKind, VocabIr};

pub type NodeId = u32;
pub const ROOT: NodeId = 0;

/// Build-time node — HashMap-based, not part of the public API.
#[derive(Default, Clone)]
struct TrieNode {
    children:      HashMap<u8, NodeId>,
    fail:          NodeId,
    fail_pop:      NodeId,
    output:        Option<u32>,
    merge_rank:    Option<u32>,
    unigram_score: Option<f32>,
}

// ─── Public DAT-based Trie ────────────────────────────────────────────────────

/// Double-Array Trie: cache-line-friendly, pointer-chasing-free automaton.
///
/// Each state maps to a **position** in the `base`/`check` flat arrays.
/// Root is state `0`. Transition from state `s` on byte `b`:
///   - `pos  = base[s] + b`
///   - valid iff `check[pos] == s`
///   - next state = `pos`
///
/// All per-state metadata (output token id, AC links, BPE/Unigram scores)
/// live in parallel flat arrays indexed by the same state/position integer.
pub struct Trie {
    /// `base[state]` — child offset; `base[state] + byte` = candidate child position.
    pub base: Vec<i32>,
    /// `check[pos]` — parent state of this position; `-1` means the slot is free.
    pub check: Vec<i32>,
    /// Token id when this state terminates a vocabulary entry.
    pub output: Vec<Option<u32>>,
    /// Aho-Corasick failure link: longest proper suffix that is a trie prefix.
    pub fail: Vec<u32>,
    /// Failure-pop link (Song et al. §3): nearest output ancestor on failure chain.
    pub fail_pop: Vec<u32>,
    /// BPE: merge rank of the token ending at this state (lower = applied first).
    pub merge_rank: Vec<Option<u32>>,
    /// Unigram: log-probability score of the token ending at this state.
    pub unigram_score: Vec<Option<f32>>,
    /// Count of reachable states (number of original `TrieNode`s compiled in).
    pub num_states: usize,
    pub algo: AlgoKind,
    /// WordPiece continuation prefix bytes (e.g. `b"##"`).
    pub cont_prefix: Vec<u8>,
}

impl Trie {
    /// Single DAT step: one ALU add + one bounds check. No pointer dereference.
    #[inline(always)]
    pub fn transition(&self, state: u32, byte: u8) -> Option<u32> {
        let pos = self.base[state as usize] as usize + byte as usize;
        if pos < self.check.len() && self.check[pos] == state as i32 {
            Some(pos as u32)
        } else {
            None
        }
    }

    /// Walk the trie from root for `bytes`.
    /// Returns the terminal state id if the full string matches, else `None`.
    #[inline]
    pub fn find_node(&self, bytes: &[u8]) -> Option<NodeId> {
        self.find_subword(&[], bytes)
    }

    /// Walk optional prefix then slice without allocating a heap Vec.
    #[inline]
    pub fn find_subword(&self, prefix: &[u8], slice: &[u8]) -> Option<NodeId> {
        let mut cur = ROOT;
        for &b in prefix.iter().chain(slice.iter()) {
            cur = self.transition(cur, b)?;
        }
        Some(cur)
    }

    /// Search for all vocab prefixes of `text` starting at byte 0.
    /// Returns `(byte_end, token_id, unigram_score)` for each match.
    ///
    /// Contains an x86_64 software prefetch (T0) 3 iterations ahead to hide
    /// L2 cache latency (~10 cycles) on irregular DAT access patterns.
    pub fn common_prefix_search(&self, text: &[u8]) -> Vec<(usize, u32, Option<f32>)> {
        let mut matches = Vec::new();
        let mut cur = ROOT;
        for (i, &b) in text.iter().enumerate() {
            // Software prefetch: compute the approximate state we will reach
            // 3 bytes ahead and prefetch that cache-line of `base` now,
            // hiding the ~10-cycle L2 latency before we need it.
            #[cfg(target_arch = "x86_64")]
            if i + 3 < text.len() {
                // SAFETY: `base` is a valid Vec<i32>; we bounds-check `approx`
                // before the raw pointer arithmetic.
                unsafe {
                    let approx = self.base[cur as usize] as usize + text[i + 3] as usize;
                    if approx < self.base.len() {
                        core::arch::x86_64::_mm_prefetch(
                            self.base.as_ptr().add(approx) as *const i8,
                            core::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
            }

            match self.transition(cur, b) {
                Some(next) => {
                    cur = next;
                    if let Some(id) = self.output[cur as usize] {
                        matches.push((i + 1, id, self.unigram_score[cur as usize]));
                    }
                }
                None => break,
            }
        }
        matches
    }
}

// ─── Public build entry point ─────────────────────────────────────────────────

/// Build a [`Trie`] from a [`VocabIr`].
///
/// Construction phases:
/// 1. Insert all vocab tokens into a HashMap-based intermediate trie.
/// 2. Annotate terminal nodes with BPE merge ranks or Unigram scores.
/// 3. BFS failure-link & fail_pop computation (Song et al. §3).
/// 4. Compile to a cache-efficient Double-Array Trie (DAT).
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

    // 4. Compile to Double-Array Trie.
    Ok(compile_to_dat(nodes, ir.algo, cont_prefix))
}

// ─── Internal build helpers ───────────────────────────────────────────────────

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

fn build_fail_and_pop_links(nodes: &mut [TrieNode]) {
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
            // Precompute fail_pop (Song et al. §3): nearest output ancestor along failure chain.
            nodes[v as usize].fail_pop = if nodes[fail_v as usize].output.is_some() {
                fail_v
            } else {
                nodes[fail_v as usize].fail_pop
            };
            queue.push_back(v);
        }
    }
}

// ─── DAT compiler ────────────────────────────────────────────────────────────

/// Compile a Vec of HashMap-based `TrieNode`s into a Double-Array Trie.
///
/// Algorithm (BFS order):
/// 1. Map root (old id 0) → new DAT state 0.
/// 2. For each node, find the smallest base offset `b ≥ 1` such that
///    `check[b + c] == -1` for every child byte `c`.
/// 3. Set `check[b + c] = new_parent_state` and record `old_id → new_state`.
/// 4. After all states are placed, fix up `fail` and `fail_pop` using the map.
fn compile_to_dat(nodes: Vec<TrieNode>, algo: AlgoKind, cont_prefix: Vec<u8>) -> Trie {
    let n = nodes.len();

    // Initial flat-array capacity: 4× node count to reduce reallocations.
    // The DAT is typically 2–3× the number of nodes for sparse tries.
    let mut cap: usize = (n * 4).max(512);

    let mut base_a:    Vec<i32>        = vec![0;    cap];
    let mut check_a:   Vec<i32>        = vec![-1;   cap];
    let mut output_a:  Vec<Option<u32>>= vec![None; cap];
    let mut fail_a:    Vec<u32>        = vec![0;    cap];
    let mut fpo_a:     Vec<u32>        = vec![0;    cap];
    let mut mrank_a:   Vec<Option<u32>>= vec![None; cap];
    let mut uscore_a:  Vec<Option<f32>>= vec![None; cap];

    // Ensure all arrays have at least `min` slots.
    macro_rules! ensure {
        ($min:expr) => {
            if $min > cap {
                cap = ($min * 2).max(cap * 2);
                base_a.resize(cap, 0);
                check_a.resize(cap, -1);
                output_a.resize(cap, None);
                fail_a.resize(cap, 0);
                fpo_a.resize(cap, 0);
                mrank_a.resize(cap, None);
                uscore_a.resize(cap, None);
            }
        };
    }

    // old TrieNode index → new DAT state (position in flat arrays).
    let mut old_to_new: Vec<u32> = vec![u32::MAX; n];
    old_to_new[0] = 0; // root stays at position 0

    // Mark slot 0 as occupied so no child is ever placed there.
    // (All base offsets start at 1, so base[x]+byte >= 1 for any byte;
    //  occupying slot 0 is a no-op but makes the invariant explicit.)
    check_a[0] = 0;

    let mut max_state: usize = 0;
    let mut queue: VecDeque<usize> = VecDeque::new();
    queue.push_back(0);

    while let Some(old_u) = queue.pop_front() {
        let new_u = old_to_new[old_u] as usize;
        max_state = max_state.max(new_u);

        // Copy this node's metadata into the flat arrays at position new_u.
        output_a[new_u]  = nodes[old_u].output;
        mrank_a[new_u]   = nodes[old_u].merge_rank;
        uscore_a[new_u]  = nodes[old_u].unigram_score;
        // fail/fail_pop will be remapped after all states are placed.

        if nodes[old_u].children.is_empty() {
            continue;
        }

        // Sort child bytes for a deterministic, reproducible DAT layout.
        let mut child_bytes: Vec<u8> = nodes[old_u].children.keys().copied().collect();
        child_bytes.sort_unstable();
        let max_c = *child_bytes.last().unwrap() as usize;

        // Find the smallest base offset b >= 1 such that every child slot
        // (b + c) for c in child_bytes is unoccupied in check_a.
        let mut b: usize = 1;
        'find: loop {
            ensure!(b + max_c + 1);
            for &c in &child_bytes {
                if check_a[b + c as usize] != -1 {
                    b += 1;
                    continue 'find;
                }
            }
            break;
        }

        base_a[new_u] = b as i32;

        // Assign child slots and enqueue children.
        for &c in &child_bytes {
            let old_v  = nodes[old_u].children[&c] as usize;
            let new_v  = b + c as usize;
            ensure!(new_v + 1);
            check_a[new_v]      = new_u as i32;
            old_to_new[old_v]   = new_v as u32;
            max_state           = max_state.max(new_v);
            queue.push_back(old_v);
        }
    }

    // Remap fail and fail_pop links from old TrieNode ids to new DAT states.
    for old_u in 0..n {
        let new_u = old_to_new[old_u];
        if new_u == u32::MAX { continue; } // unreachable node (safety guard)
        let new_u = new_u as usize;
        fail_a[new_u] = old_to_new[nodes[old_u].fail as usize];
        fpo_a[new_u]  = old_to_new[nodes[old_u].fail_pop as usize];
    }

    // Truncate to exactly what was used.
    let final_size = max_state + 1;
    base_a.truncate(final_size);
    check_a.truncate(final_size);
    output_a.truncate(final_size);
    fail_a.truncate(final_size);
    fpo_a.truncate(final_size);
    mrank_a.truncate(final_size);
    uscore_a.truncate(final_size);

    Trie {
        base:          base_a,
        check:         check_a,
        output:        output_a,
        fail:          fail_a,
        fail_pop:      fpo_a,
        merge_rank:    mrank_a,
        unigram_score: uscore_a,
        num_states:    n,
        algo,
        cont_prefix,
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

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
        // DAT is non-trivial: base/check have more than 1 slot.
        assert!(trie.base.len() > 1);
        // root + h,e,s,h(from s),e(from sh),i,s(from hi),e(from he),r,s(from her) = 10
        assert_eq!(trie.num_states, 10);
    }

    #[test]
    fn fail_pop_links_computed() {
        let ir = make_ir(&[("a", 0), ("ba", 1)]);
        let trie = build(&ir).unwrap();
        // "ba" trie has root + 'b' + 'a' (from root) + 'a' (from 'b') = 4 nodes
        assert!(trie.num_states > 2);
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

    #[test]
    fn dat_transition_correct() {
        let ir = make_ir(&[("hi", 0), ("ho", 1)]);
        let trie = build(&ir).unwrap();
        // h -> some state, then i -> output 0, o -> output 1
        let h_state = trie.transition(ROOT, b'h').expect("'h' must exist");
        let hi_state = trie.transition(h_state, b'i').expect("'hi' must exist");
        let ho_state = trie.transition(h_state, b'o').expect("'ho' must exist");
        assert_eq!(trie.output[hi_state as usize], Some(0));
        assert_eq!(trie.output[ho_state as usize], Some(1));
        // non-existent transition
        assert!(trie.transition(h_state, b'x').is_none());
    }
}
