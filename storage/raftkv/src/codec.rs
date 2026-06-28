// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Wire encoding for Raft persistent state (dependency-free, big-endian).
//!
//! ```text
//!   current_term : u64
//!   has_vote     : u8        (1 => voted_for follows)
//!   voted_for    : u64       (present iff has_vote == 1)
//!   log_len      : u32
//!   entries[log_len]:
//!     term       : u64
//!     cmd_len    : u32
//!     cmd        : [u8; cmd_len]
//! ```

use miniraft::{LogEntry, PersistentState};

pub fn encode(state: &PersistentState) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&state.current_term.to_be_bytes());
    match state.voted_for {
        Some(v) => {
            buf.push(1);
            buf.extend_from_slice(&v.to_be_bytes());
        }
        None => buf.push(0),
    }
    buf.extend_from_slice(&(state.log.len() as u32).to_be_bytes());
    for e in &state.log {
        buf.extend_from_slice(&e.term.to_be_bytes());
        buf.extend_from_slice(&(e.command.len() as u32).to_be_bytes());
        buf.extend_from_slice(&e.command);
    }
    buf
}

/// Parse persistent state. Returns `None` on any truncation/format error.
pub fn decode(bytes: &[u8]) -> Option<PersistentState> {
    let mut c = Cursor { b: bytes, pos: 0 };
    let current_term = c.u64()?;
    let voted_for = match c.u8()? {
        0 => None,
        1 => Some(c.u64()?),
        _ => return None,
    };
    let n = c.u32()? as usize;
    let mut log = Vec::with_capacity(n);
    for _ in 0..n {
        let term = c.u64()?;
        let len = c.u32()? as usize;
        let command = c.bytes(len)?.to_vec();
        log.push(LogEntry { term, command });
    }
    Some(PersistentState { current_term, voted_for, log })
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}
impl<'a> Cursor<'a> {
    fn bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        if end > self.b.len() {
            return None;
        }
        let s = &self.b[self.pos..end];
        self.pos = end;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        Some(self.bytes(1)?[0])
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_be_bytes(self.bytes(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_be_bytes(self.bytes(8)?.try_into().ok()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips() {
        let s = PersistentState {
            current_term: 7,
            voted_for: Some(3),
            log: vec![
                LogEntry { term: 1, command: b"set x=1".to_vec() },
                LogEntry { term: 5, command: vec![] },
                LogEntry { term: 7, command: vec![0, 255, 7, 128] },
            ],
        };
        assert_eq!(decode(&encode(&s)), Some(s));

        let empty = PersistentState { current_term: 0, voted_for: None, log: vec![] };
        assert_eq!(decode(&encode(&empty)), Some(empty));
    }

    #[test]
    fn rejects_truncated() {
        let s = PersistentState { current_term: 1, voted_for: Some(2), log: vec![] };
        let enc = encode(&s);
        assert_eq!(decode(&enc[..enc.len() - 1]), None);
    }
}
