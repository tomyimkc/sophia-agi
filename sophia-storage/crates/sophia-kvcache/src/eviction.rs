// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Eviction policy with reference counting.
//!
//! A block is *pinned* while any in-flight request references it (its prefix is
//! live). Only unpinned blocks are eviction candidates. Among candidates we pick
//! the least-recently-used — the classic policy, and a fine default for prefix
//! caches where recency tracks reuse well.
//!
//! Ref-counting is the load-bearing invariant: it is what makes prefix sharing
//! safe. Two requests over the same prompt bump the same block to refcount 2;
//! neither can evict it out from under the other.

use std::collections::HashMap;

use crate::block::BlockId;

#[derive(Default)]
pub struct LruRefCounted {
    /// Logical clock; bumped on every touch.
    tick: u64,
    /// block -> (refcount, last_used_tick).
    meta: HashMap<BlockId, (u32, u64)>,
}

impl LruRefCounted {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a block as resident (refcount starts at 0 = unpinned, cached).
    pub fn track(&mut self, id: BlockId) {
        self.tick += 1;
        self.meta.entry(id).or_insert((0, self.tick));
    }

    /// Pin: a request now depends on this block. Returns the new refcount.
    pub fn pin(&mut self, id: BlockId) -> u32 {
        self.tick += 1;
        let e = self.meta.entry(id).or_insert((0, self.tick));
        e.0 += 1;
        e.1 = self.tick;
        e.0
    }

    /// Unpin: a request released this block. Returns the new refcount.
    pub fn unpin(&mut self, id: BlockId) -> u32 {
        if let Some(e) = self.meta.get_mut(&id) {
            e.0 = e.0.saturating_sub(1);
            e.0
        } else {
            0
        }
    }

    /// Mark a cache hit (read) for LRU recency.
    pub fn touch(&mut self, id: BlockId) {
        self.tick += 1;
        if let Some(e) = self.meta.get_mut(&id) {
            e.1 = self.tick;
        }
    }

    pub fn refcount(&self, id: BlockId) -> u32 {
        self.meta.get(&id).map_or(0, |e| e.0)
    }

    pub fn is_pinned(&self, id: BlockId) -> bool {
        self.refcount(id) > 0
    }

    /// Pick the LRU *unpinned* victim, if any. `among` restricts candidates to
    /// the ids currently resident in the tier being made room in.
    pub fn evict_candidate(&self, among: impl Iterator<Item = BlockId>) -> Option<BlockId> {
        among
            .filter_map(|id| self.meta.get(&id).map(|&(rc, tick)| (id, rc, tick)))
            .filter(|&(_, rc, _)| rc == 0)
            .min_by_key(|&(_, _, tick)| tick)
            .map(|(id, _, _)| id)
    }

    /// Forget a block entirely (after it is fully evicted from all tiers).
    pub fn forget(&mut self, id: BlockId) {
        self.meta.remove(&id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_blocks_are_not_evicted() {
        let mut lru = LruRefCounted::new();
        let a = BlockId(1);
        let b = BlockId(2);
        lru.track(a);
        lru.track(b);
        lru.pin(a); // a is in use
        let victim = lru.evict_candidate([a, b].into_iter());
        assert_eq!(victim, Some(b), "only the unpinned block is evictable");
    }

    #[test]
    fn lru_order_respected() {
        let mut lru = LruRefCounted::new();
        let a = BlockId(1);
        let b = BlockId(2);
        lru.track(a);
        lru.track(b);
        lru.touch(a); // a now more recent than b
        assert_eq!(lru.evict_candidate([a, b].into_iter()), Some(b));
    }

    #[test]
    fn refcount_balances() {
        let mut lru = LruRefCounted::new();
        let a = BlockId(7);
        lru.track(a);
        assert_eq!(lru.pin(a), 1);
        assert_eq!(lru.pin(a), 2);
        assert_eq!(lru.unpin(a), 1);
        assert!(lru.is_pinned(a));
        assert_eq!(lru.unpin(a), 0);
        assert!(!lru.is_pinned(a));
    }
}
