// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Level layout + manifest for leveled compaction.
//!
//! Full-merge-everything (the first cut) re-reads the entire dataset on every
//! compaction — write amplification grows with the data size. Leveled
//! compaction bounds it: a flush lands in **L0** (a few possibly-overlapping
//! tables); when L0 fills it merges into **L1**; each deeper level holds a
//! single sorted run that is `FANOUT`× larger than the one above and is only
//! rewritten when it overflows. A key is rewritten O(levels) times, not O(n).
//!
//! This module is the pure bookkeeping: which table ids sit at which level, how
//! many records each holds, the compaction triggers, and a tiny text manifest so
//! the layout survives a restart. The actual merges (which need the I/O backend)
//! live in [`crate::Engine`].

/// Records grow by this factor per level (L(i+1) budget = L(i) budget × FANOUT).
pub const FANOUT: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableMeta {
    pub id: u64,
    pub records: usize,
}

#[derive(Debug, Default, Clone)]
pub struct Levels {
    /// L0 tables, newest first (overlapping key ranges allowed).
    pub l0: Vec<TableMeta>,
    /// Deeper levels: `deeper[i]` is L(i+1), each a single non-overlapping run.
    pub deeper: Vec<Option<TableMeta>>,
}

impl Levels {
    pub fn new() -> Self {
        Self::default()
    }

    /// A freshly flushed table enters L0 at the front (newest).
    pub fn push_l0(&mut self, meta: TableMeta) {
        self.l0.insert(0, meta);
    }

    /// L1-budget in records; each deeper level is FANOUT× larger.
    pub fn budget(level: usize, l1_base: usize) -> usize {
        l1_base * FANOUT.pow((level - 1) as u32)
    }

    /// The deepest level index (1-based) that currently holds a table.
    pub fn max_level(&self) -> usize {
        for (i, slot) in self.deeper.iter().enumerate().rev() {
            if slot.is_some() {
                return i + 1;
            }
        }
        0
    }

    /// Read-order ids: L0 newest→oldest, then L1, L2, … (each newer level first).
    pub fn read_order(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.l0.iter().map(|t| t.id).collect();
        ids.extend(self.deeper.iter().flatten().map(|t| t.id));
        ids
    }

    pub fn get_deeper(&self, level: usize) -> Option<TableMeta> {
        self.deeper.get(level - 1).copied().flatten()
    }

    pub fn set_deeper(&mut self, level: usize, meta: Option<TableMeta>) {
        if self.deeper.len() < level {
            self.deeper.resize(level, None);
        }
        self.deeper[level - 1] = meta;
    }

    /// Count of live tables across all levels (observability).
    pub fn table_count(&self) -> usize {
        self.l0.len() + self.deeper.iter().filter(|s| s.is_some()).count()
    }

    // --- manifest serialization (one record per line) ---

    pub fn encode(&self) -> String {
        let mut out = String::new();
        for t in &self.l0 {
            out.push_str(&format!("L0 {} {}\n", t.id, t.records));
        }
        for (i, slot) in self.deeper.iter().enumerate() {
            if let Some(t) = slot {
                out.push_str(&format!("L{} {} {}\n", i + 1, t.id, t.records));
            }
        }
        out
    }

    pub fn decode(text: &str) -> Self {
        let mut levels = Levels::new();
        // L0 lines appear newest-first in the file; push_l0 reverses, so feed
        // them in file order via append to preserve it.
        for line in text.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 3 {
                continue;
            }
            let id: u64 = match parts[1].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let records: usize = parts[2].parse().unwrap_or(0);
            let meta = TableMeta { id, records };
            if let Some(level_str) = parts[0].strip_prefix('L') {
                let level: usize = level_str.parse().unwrap_or(usize::MAX);
                if level == 0 {
                    levels.l0.push(meta); // preserve file (newest-first) order
                } else if level != usize::MAX {
                    levels.set_deeper(level, Some(meta));
                }
            }
        }
        levels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_order_is_newest_first() {
        let mut l = Levels::new();
        l.push_l0(TableMeta { id: 1, records: 10 });
        l.push_l0(TableMeta { id: 2, records: 10 }); // newer
        l.set_deeper(1, Some(TableMeta { id: 100, records: 50 }));
        assert_eq!(l.read_order(), vec![2, 1, 100]);
    }

    #[test]
    fn budgets_grow_by_fanout() {
        assert_eq!(Levels::budget(1, 64), 64);
        assert_eq!(Levels::budget(2, 64), 640);
        assert_eq!(Levels::budget(3, 64), 6400);
    }

    #[test]
    fn manifest_round_trips() {
        let mut l = Levels::new();
        l.push_l0(TableMeta { id: 5, records: 3 });
        l.push_l0(TableMeta { id: 6, records: 4 });
        l.set_deeper(1, Some(TableMeta { id: 100, records: 99 }));
        l.set_deeper(3, Some(TableMeta { id: 300, records: 999 }));
        let back = Levels::decode(&l.encode());
        assert_eq!(back.read_order(), l.read_order());
        assert_eq!(back.max_level(), 3);
        assert_eq!(back.get_deeper(1), Some(TableMeta { id: 100, records: 99 }));
    }
}
