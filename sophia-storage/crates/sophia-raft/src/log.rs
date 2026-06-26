// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! The replicated log: an append-only sequence of [`LogEntry`] with the Raft
//! consistency operations (term lookup, conflict truncation, commit tracking).
//!
//! Indexing is 1-based to match the Raft paper: index 0 is the empty sentinel
//! (term 0), the first real entry is index 1.

use crate::types::{Index, LogEntry, Term};

#[derive(Debug, Default, Clone)]
pub struct RaftLog {
    entries: Vec<LogEntry>,
    /// Highest index known committed (a majority has it).
    pub commit_index: Index,
    /// Highest index applied to the state machine.
    pub last_applied: Index,
}

impl RaftLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn last_index(&self) -> Index {
        self.entries.len() as Index
    }

    /// Term of the entry at `index` (0 for the empty sentinel or out of range).
    pub fn term_at(&self, index: Index) -> Term {
        if index == 0 || index > self.last_index() {
            0
        } else {
            self.entries[(index - 1) as usize].term
        }
    }

    pub fn last_term(&self) -> Term {
        self.term_at(self.last_index())
    }

    pub fn entry(&self, index: Index) -> Option<&LogEntry> {
        if index == 0 || index > self.last_index() {
            None
        } else {
            Some(&self.entries[(index - 1) as usize])
        }
    }

    /// Append one entry, returning its index.
    pub fn append(&mut self, entry: LogEntry) -> Index {
        self.entries.push(entry);
        self.last_index()
    }

    /// Entries strictly after `index` (for replication to a follower).
    pub fn entries_after(&self, index: Index) -> Vec<LogEntry> {
        if index >= self.last_index() {
            Vec::new()
        } else {
            self.entries[index as usize..].to_vec()
        }
    }

    /// Does the log contain an entry at `index` with the given `term`? This is
    /// the AppendEntries consistency check on `prev_log_index`/`prev_log_term`.
    pub fn matches(&self, index: Index, term: Term) -> bool {
        index == 0 || (index <= self.last_index() && self.term_at(index) == term)
    }

    /// Splice follower entries in starting at `prev_index + 1`, truncating on the
    /// first term conflict (Raft §5.3). Existing entries that already agree are
    /// kept so we never discard committed suffixes we still match.
    pub fn append_from(&mut self, prev_index: Index, incoming: &[LogEntry]) {
        let mut idx = prev_index + 1;
        for (i, entry) in incoming.iter().enumerate() {
            if idx <= self.last_index() {
                if self.term_at(idx) != entry.term {
                    // Conflict: drop this and everything after, then append rest.
                    self.entries.truncate((idx - 1) as usize);
                    self.entries.extend_from_slice(&incoming[i..]);
                    return;
                }
                // already present and matching — skip
            } else {
                self.entries.extend_from_slice(&incoming[i..]);
                return;
            }
            idx += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(term: Term, c: &str) -> LogEntry {
        LogEntry { term, command: c.as_bytes().to_vec() }
    }

    #[test]
    fn append_and_lookup() {
        let mut log = RaftLog::new();
        assert_eq!(log.last_index(), 0);
        assert_eq!(log.append(e(1, "a")), 1);
        assert_eq!(log.append(e(1, "b")), 2);
        assert_eq!(log.term_at(1), 1);
        assert_eq!(log.last_term(), 1);
        assert!(log.matches(2, 1));
        assert!(!log.matches(2, 2));
    }

    #[test]
    fn append_from_truncates_on_conflict() {
        let mut log = RaftLog::new();
        log.append(e(1, "a"));
        log.append(e(1, "b"));
        log.append(e(2, "stale")); // index 3, term 2 — will conflict
        // Leader says index 3 should be term 3.
        log.append_from(2, &[e(3, "c"), e(3, "d")]);
        assert_eq!(log.last_index(), 4);
        assert_eq!(log.term_at(3), 3);
        assert_eq!(log.entry(3).unwrap().command, b"c");
        assert_eq!(log.entry(4).unwrap().command, b"d");
    }

    #[test]
    fn append_from_is_idempotent_on_match() {
        let mut log = RaftLog::new();
        log.append(e(1, "a"));
        log.append(e(1, "b"));
        // Re-deliver the same suffix: no duplication.
        log.append_from(0, &[e(1, "a"), e(1, "b")]);
        assert_eq!(log.last_index(), 2);
    }
}
