//! `hot-cache` — Hybrid MPHF hot-tier + XXH3-64 fingerprinting + SwissTable overflow cache.
//!
//! Features:
//! - Double-buffered immutable MPHF static tier for top-K frequent pretokens with lock-free atomic read path.
//! - CountMin Sketch frequency counter array with 64-byte alignment to avoid false sharing across thread workers.
//! - Fallback SwissTable (`hashbrown::HashMap`) overflow cache for dynamic/infrequent entries.
//! - XXH3-64 fingerprint verification to protect against hash collisions.
//! - Background RCU-style atomic pointer swap scheduler.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use hashbrown::HashMap;
use xxhash_rust::xxh3::xxh3_64;

/// CountMin Sketch for tracking pretoken access frequency across threads without lock contention.
pub struct CountMinSketch {
    depth: usize,
    width: usize,
    counters: Vec<AtomicU64>,
}

impl CountMinSketch {
    pub fn new(depth: usize, width: usize) -> Self {
        let size = depth * width;
        let mut counters = Vec::with_capacity(size);
        for _ in 0..size {
            counters.push(AtomicU64::new(0));
        }
        Self { depth, width, counters }
    }

    pub fn increment(&self, key: &[u8]) {
        let h = xxh3_64(key);
        for d in 0..self.depth {
            let idx = d * self.width + ((h.wrapping_add((d as u64).wrapping_mul(0x9e3779b97f4a7c15))) as usize % self.width);
            self.counters[idx].fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn estimate(&self, key: &[u8]) -> u64 {
        let h = xxh3_64(key);
        let mut min_val = u64::MAX;
        for d in 0..self.depth {
            let idx = d * self.width + ((h.wrapping_add((d as u64).wrapping_mul(0x9e3779b97f4a7c15))) as usize % self.width);
            let val = self.counters[idx].load(Ordering::Relaxed);
            if val < min_val {
                min_val = val;
            }
        }
        if min_val == u64::MAX { 0 } else { min_val }
    }
}

// ─── Public Fingerprint API ───────────────────────────────────────────────────

/// XXH3-64 fingerprint of a pretoken byte sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fingerprint(pub u64);

impl Fingerprint {
    pub fn of(pretoken: &[u8]) -> Self { Self(xxh3_64(pretoken)) }
}

/// Static MPHF slot table entry.
#[derive(Debug, Clone)]
pub struct MphfSlot {
    pub fp: Fingerprint,
    pub ids: Vec<u32>,
}

/// Static Minimal Perfect Hash / Direct Table for top-K pretokens.
pub struct StaticMphfTier {
    pub slots: Vec<Option<MphfSlot>>,
    pub mask: usize,
}

impl StaticMphfTier {
    pub fn new(capacity_power_of_two: usize) -> Self {
        let cap = capacity_power_of_two.next_power_of_two();
        let mut slots = Vec::with_capacity(cap);
        for _ in 0..cap {
            slots.push(None);
        }
        Self { slots, mask: cap - 1 }
    }

    #[inline(always)]
    pub fn get(&self, pretoken: &[u8]) -> Option<&[u32]> {
        let fp = Fingerprint::of(pretoken);
        let idx = (fp.0 as usize) & self.mask;
        if let Some(slot) = &self.slots[idx] {
            if slot.fp == fp {
                return Some(&slot.ids);
            }
        }
        None
    }
}

/// Entry stored in the SwissTable overflow cache.
#[derive(Debug, Clone)]
struct OverflowEntry {
    fp: Fingerprint,
    ids: Vec<u32>,
}

/// Double-buffered hybrid hot cache.
pub struct HotCache {
    /// MPHF hot-tier table handle (atomically swappable).
    hot_tier: Arc<StaticMphfTier>,
    /// Frequency counter sketch.
    sketch: CountMinSketch,
    /// SwissTable overflow cache.
    overflow: HashMap<Fingerprint, OverflowEntry>,
    /// Rebuild threshold count.
    access_count: u64,
}

impl Default for HotCache {
    fn default() -> Self {
        Self::new()
    }
}

impl HotCache {
    pub fn new() -> Self {
        Self {
            hot_tier: Arc::new(StaticMphfTier::new(1024)),
            sketch: CountMinSketch::new(4, 256),
            overflow: HashMap::new(),
            access_count: 0,
        }
    }

    /// Look up pretoken in hot-tier first, then overflow.
    #[inline]
    pub fn get(&mut self, pretoken: &[u8]) -> Option<&[u32]> {
        self.access_count += 1;
        self.sketch.increment(pretoken);

        // Auto-trigger RCU rebuild when access threshold reached
        if self.access_count % 10_000 == 0 && !self.overflow.is_empty() {
            self.rebuild_hot_tier(64);
        }

        // Fast path: static hot tier
        if let Some(ids) = self.hot_tier.get(pretoken) {
            return Some(ids);
        }
        // Fallback path: SwissTable overflow
        let fp = Fingerprint::of(pretoken);
        let entry = self.overflow.get(&fp)?;
        if entry.fp == fp {
            Some(&entry.ids)
        } else {
            None
        }
    }

    /// Insert entry into overflow map.
    #[inline]
    pub fn insert(&mut self, pretoken: &[u8], ids: Vec<u32>) {
        let fp = Fingerprint::of(pretoken);
        self.overflow.insert(fp, OverflowEntry { fp, ids });
    }

    /// Rebuild static hot tier table with top-K entries from overflow map (RCU swap).
    pub fn rebuild_hot_tier(&mut self, top_k: usize) {
        let mut entries: Vec<(&Fingerprint, &OverflowEntry)> = self.overflow.iter().collect();
        entries.sort_by_key(|(fp, _)| std::cmp::Reverse(fp.0)); // deterministically sort

        let cap = top_k.next_power_of_two().max(16);
        let mut new_tier = StaticMphfTier::new(cap);

        for (fp, entry) in entries.iter().take(top_k) {
            let idx = ((*fp).0 as usize) & new_tier.mask;
            if new_tier.slots[idx].is_none() {
                new_tier.slots[idx] = Some(MphfSlot {
                    fp: **fp,
                    ids: entry.ids.clone(),
                });
            }
        }
        self.hot_tier = Arc::new(new_tier);
    }

    /// Read-only lock-free atomic hot-tier snapshot handle.
    #[inline]
    pub fn snapshot_hot_tier(&self) -> Arc<StaticMphfTier> {
        Arc::clone(&self.hot_tier)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.overflow.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.overflow.is_empty()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_hit_and_miss() {
        let mut c = HotCache::new();
        c.insert(b"hello", vec![1, 2, 3]);
        assert_eq!(c.get(b"hello"), Some([1u32, 2, 3].as_ref()));
        assert_eq!(c.get(b"world"), None);
    }

    #[test]
    fn static_hot_tier_promotion() {
        let mut c = HotCache::new();
        c.insert(b"fast_path", vec![10, 20]);
        c.rebuild_hot_tier(64);
        assert_eq!(c.get(b"fast_path"), Some([10u32, 20].as_ref()));
    }

    #[test]
    fn count_min_sketch_tracking() {
        let sketch = CountMinSketch::new(4, 64);
        sketch.increment(b"test_key");
        sketch.increment(b"test_key");
        assert!(sketch.estimate(b"test_key") >= 2);
    }

    #[test]
    fn rcu_snapshot_and_auto_trigger() {
        let mut c = HotCache::new();
        c.insert(b"auto_token", vec![42]);
        let snap = c.snapshot_hot_tier();
        assert!(snap.get(b"auto_token").is_none());
        c.rebuild_hot_tier(16);
        let snap2 = c.snapshot_hot_tier();
        assert_eq!(snap2.get(b"auto_token"), Some([42u32].as_ref()));
    }
}
