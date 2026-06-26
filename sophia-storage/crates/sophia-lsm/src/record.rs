// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! On-disk record framing, shared by the WAL and SSTables.
//!
//! Framing is deliberately boring and self-describing so a half-written tail
//! (torn write after a crash) is *detectable* rather than silently corrupting:
//!
//! ```text
//! [ kind: u8 ][ klen: u32-le ][ vlen: u32-le ][ key ][ value ][ crc32: u32-le ]
//! ```
//!
//! `crc32` covers everything before it. On replay, a record whose length runs
//! past EOF or whose CRC fails truncates the log at that point — at-least-once
//! durability with a clean recovery boundary, matching the semantics the Python
//! `sophia_contract/stores.py` and `queue.py` already promise.

use std::io;

pub const KIND_PUT: u8 = 1;
pub const KIND_DELETE: u8 = 2; // tombstone; value is empty

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    pub kind: u8,
    pub key: Vec<u8>,
    pub value: Vec<u8>,
}

impl Record {
    pub fn put(key: impl Into<Vec<u8>>, value: impl Into<Vec<u8>>) -> Self {
        Record { kind: KIND_PUT, key: key.into(), value: value.into() }
    }

    pub fn delete(key: impl Into<Vec<u8>>) -> Self {
        Record { kind: KIND_DELETE, key: key.into(), value: Vec::new() }
    }

    pub fn is_tombstone(&self) -> bool {
        self.kind == KIND_DELETE
    }

    /// Serialize into the framed wire form (header + payload + trailing CRC).
    pub fn encode(&self) -> Vec<u8> {
        let klen = self.key.len() as u32;
        let vlen = self.value.len() as u32;
        let mut buf = Vec::with_capacity(13 + self.key.len() + self.value.len());
        buf.push(self.kind);
        buf.extend_from_slice(&klen.to_le_bytes());
        buf.extend_from_slice(&vlen.to_le_bytes());
        buf.extend_from_slice(&self.key);
        buf.extend_from_slice(&self.value);
        let crc = crc32(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Decode one record from the front of `buf`, returning the record and the
    /// number of bytes consumed. Returns `Ok(None)` on a clean short/torn tail
    /// (caller should treat this as the durable end of the log).
    pub fn decode(buf: &[u8]) -> io::Result<Option<(Record, usize)>> {
        if buf.len() < 9 {
            return Ok(None);
        }
        let kind = buf[0];
        let klen = u32::from_le_bytes(buf[1..5].try_into().unwrap()) as usize;
        let vlen = u32::from_le_bytes(buf[5..9].try_into().unwrap()) as usize;
        let total = 9 + klen + vlen + 4;
        if buf.len() < total {
            return Ok(None); // torn tail
        }
        let stored_crc = u32::from_le_bytes(buf[total - 4..total].try_into().unwrap());
        if crc32(&buf[..total - 4]) != stored_crc {
            return Ok(None); // corrupt tail — stop here
        }
        if kind != KIND_PUT && kind != KIND_DELETE {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "unknown record kind"));
        }
        let key = buf[9..9 + klen].to_vec();
        let value = buf[9 + klen..9 + klen + vlen].to_vec();
        Ok(Some((Record { kind, key, value }, total)))
    }
}

/// Standard CRC-32 (IEEE polynomial), table-free so we stay dependency-free.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let r = Record::put(b"alpha".to_vec(), b"beta".to_vec());
        let bytes = r.encode();
        let (back, n) = Record::decode(&bytes).unwrap().unwrap();
        assert_eq!(n, bytes.len());
        assert_eq!(back, r);
    }

    #[test]
    fn torn_tail_is_none() {
        let bytes = Record::put(b"k".to_vec(), b"v".to_vec()).encode();
        assert!(Record::decode(&bytes[..bytes.len() - 2]).unwrap().is_none());
    }

    #[test]
    fn bitflip_fails_crc() {
        let mut bytes = Record::put(b"k".to_vec(), b"v".to_vec()).encode();
        bytes[10] ^= 0xFF;
        assert!(Record::decode(&bytes).unwrap().is_none());
    }
}
