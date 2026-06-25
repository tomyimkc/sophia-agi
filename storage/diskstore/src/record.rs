// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! On-disk record framing for the append-only log.
//!
//! Layout (all integers big-endian):
//! ```text
//!   crc32 : u32   # over everything after this field (tstamp..end)
//!   tstamp: u64   # monotonic write sequence, for last-writer-wins on recovery
//!   klen  : u32
//!   vlen  : u32   # == TOMBSTONE marks a delete; no value bytes follow
//!   key   : [u8; klen]
//!   val   : [u8; vlen]   # absent for a tombstone
//! ```

use crate::crc::crc32;

pub const HEADER_LEN: usize = 4 + 8 + 4 + 4; // crc + tstamp + klen + vlen = 20
pub const TOMBSTONE: u32 = u32::MAX;

/// A record parsed off disk during recovery.
pub struct ParsedRecord {
    pub key: Vec<u8>,
    pub tstamp: u64,
    /// `None` for a tombstone; otherwise the value length and its absolute file
    /// offset (so the keydir can point straight at the bytes).
    pub value: Option<(u64, u32)>,
    /// Total bytes this record occupies on disk.
    pub total_len: u64,
}

/// Encode a put (or, with `val == None`, a tombstone) into a single buffer.
pub fn encode(key: &[u8], val: Option<&[u8]>, tstamp: u64) -> Vec<u8> {
    let vlen = match val {
        Some(v) => v.len() as u32,
        None => TOMBSTONE,
    };
    let body_len = 8 + 4 + 4 + key.len() + val.map_or(0, |v| v.len());
    let mut buf = Vec::with_capacity(4 + body_len);
    buf.extend_from_slice(&[0u8; 4]); // crc placeholder
    buf.extend_from_slice(&tstamp.to_be_bytes());
    buf.extend_from_slice(&(key.len() as u32).to_be_bytes());
    buf.extend_from_slice(&vlen.to_be_bytes());
    buf.extend_from_slice(key);
    if let Some(v) = val {
        buf.extend_from_slice(v);
    }
    let crc = crc32(&buf[4..]);
    buf[..4].copy_from_slice(&crc.to_be_bytes());
    buf
}

/// Parse one record starting at `offset`, given the full header bytes plus the
/// key/value bytes. Returns `None` if the CRC fails (torn/garbage tail).
pub fn parse(offset: u64, header: &[u8; HEADER_LEN], body: &[u8]) -> Option<ParsedRecord> {
    let stored_crc = u32::from_be_bytes(header[0..4].try_into().unwrap());
    let tstamp = u64::from_be_bytes(header[4..12].try_into().unwrap());
    let klen = u32::from_be_bytes(header[12..16].try_into().unwrap()) as usize;
    let vlen_raw = u32::from_be_bytes(header[16..20].try_into().unwrap());

    // Recompute CRC over header[4..] ++ body.
    let mut hasher_input = Vec::with_capacity((HEADER_LEN - 4) + body.len());
    hasher_input.extend_from_slice(&header[4..]);
    hasher_input.extend_from_slice(body);
    if crc32(&hasher_input) != stored_crc {
        return None;
    }

    let is_tombstone = vlen_raw == TOMBSTONE;
    let vlen = if is_tombstone { 0 } else { vlen_raw as usize };
    if body.len() != klen + vlen {
        return None; // length mismatch: corrupt
    }
    let key = body[..klen].to_vec();
    let value = if is_tombstone {
        None
    } else {
        let value_offset = offset + HEADER_LEN as u64 + klen as u64;
        Some((value_offset, vlen as u32))
    };
    Some(ParsedRecord {
        key,
        tstamp,
        value,
        total_len: (HEADER_LEN + klen + vlen) as u64,
    })
}

/// Body length (key + value bytes) implied by a header, for the recovery scan.
pub fn body_len(header: &[u8; HEADER_LEN]) -> usize {
    let klen = u32::from_be_bytes(header[12..16].try_into().unwrap()) as usize;
    let vlen_raw = u32::from_be_bytes(header[16..20].try_into().unwrap());
    let vlen = if vlen_raw == TOMBSTONE { 0 } else { vlen_raw as usize };
    klen + vlen
}
