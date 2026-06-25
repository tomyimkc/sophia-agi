// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Length-prefixed binary wire protocol shared by client and server.
//!
//! Every field is big-endian. Keys/values are length-prefixed byte strings, so
//! the protocol is binary-safe (no delimiter escaping). A request is one op
//! byte followed by op-specific fields; a response is one status byte followed
//! by status-specific fields.

use tokio::io::{self, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

// Request ops.
const OP_GET: u8 = 1;
const OP_SET: u8 = 2;
const OP_DEL: u8 = 3;
const OP_PING: u8 = 4;
const OP_STATS: u8 = 5;

// Response status bytes.
const ST_OK: u8 = 0; // del/set acknowledged; for del, a u8 "found" flag follows
const ST_VALUE: u8 = 1; // get hit; a length-prefixed value follows
const ST_NOT_FOUND: u8 = 2; // get/del miss
const ST_PONG: u8 = 3;
const ST_STATS: u8 = 4; // seven u64 counters follow
const ST_ERROR: u8 = 5; // a length-prefixed UTF-8 message follows

/// A guard against malicious/garbled length prefixes (16 MiB).
const MAX_FRAME: u32 = 16 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq)]
pub enum Request {
    Get(Vec<u8>),
    Set { key: Vec<u8>, val: Vec<u8>, ttl_ms: u64 },
    Del(Vec<u8>),
    Ping,
    Stats,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Response {
    Value(Vec<u8>),
    NotFound,
    Deleted(bool),
    Pong,
    Stats(StatsSnapshot),
    Error(String),
}

/// Aggregate counters across all shards. Hits/misses/sets/dels are cache-wide;
/// evictions/expirations are summed from each shard's LRU.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StatsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub sets: u64,
    pub dels: u64,
    pub evictions: u64,
    pub expirations: u64,
    pub entries: u64,
}

async fn read_bytes<R: AsyncRead + Unpin>(r: &mut R) -> io::Result<Vec<u8>> {
    let len = r.read_u32().await?;
    if len > MAX_FRAME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "frame exceeds MAX_FRAME",
        ));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).await?;
    Ok(buf)
}

async fn write_bytes<W: AsyncWrite + Unpin>(w: &mut W, b: &[u8]) -> io::Result<()> {
    w.write_u32(b.len() as u32).await?;
    w.write_all(b).await
}

/// Read one request. Returns `Ok(None)` on a clean EOF at a frame boundary so
/// the server can distinguish "client hung up" from a real I/O error.
pub async fn read_request<R: AsyncRead + Unpin>(r: &mut R) -> io::Result<Option<Request>> {
    let op = match r.read_u8().await {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    };
    let req = match op {
        OP_GET => Request::Get(read_bytes(r).await?),
        OP_SET => {
            let key = read_bytes(r).await?;
            let val = read_bytes(r).await?;
            let ttl_ms = r.read_u64().await?;
            Request::Set { key, val, ttl_ms }
        }
        OP_DEL => Request::Del(read_bytes(r).await?),
        OP_PING => Request::Ping,
        OP_STATS => Request::Stats,
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown op byte {other}"),
            ))
        }
    };
    Ok(Some(req))
}

/// Encode a request into the writer **without flushing**. The caller flushes —
/// this lets a pipelining client pack many requests into one write/syscall
/// batch before a single `flush().await`.
pub async fn write_request<W: AsyncWrite + Unpin>(w: &mut W, req: &Request) -> io::Result<()> {
    match req {
        Request::Get(k) => {
            w.write_u8(OP_GET).await?;
            write_bytes(w, k).await?;
        }
        Request::Set { key, val, ttl_ms } => {
            w.write_u8(OP_SET).await?;
            write_bytes(w, key).await?;
            write_bytes(w, val).await?;
            w.write_u64(*ttl_ms).await?;
        }
        Request::Del(k) => {
            w.write_u8(OP_DEL).await?;
            write_bytes(w, k).await?;
        }
        Request::Ping => w.write_u8(OP_PING).await?,
        Request::Stats => w.write_u8(OP_STATS).await?,
    }
    Ok(())
}

pub async fn write_response<W: AsyncWrite + Unpin>(w: &mut W, resp: &Response) -> io::Result<()> {
    match resp {
        Response::Value(v) => {
            w.write_u8(ST_VALUE).await?;
            write_bytes(w, v).await?;
        }
        Response::NotFound => w.write_u8(ST_NOT_FOUND).await?,
        Response::Deleted(found) => {
            w.write_u8(ST_OK).await?;
            w.write_u8(*found as u8).await?;
        }
        Response::Pong => w.write_u8(ST_PONG).await?,
        Response::Stats(s) => {
            w.write_u8(ST_STATS).await?;
            for c in [s.hits, s.misses, s.sets, s.dels, s.evictions, s.expirations, s.entries] {
                w.write_u64(c).await?;
            }
        }
        Response::Error(msg) => {
            w.write_u8(ST_ERROR).await?;
            write_bytes(w, msg.as_bytes()).await?;
        }
    }
    Ok(())
}

pub async fn read_response<R: AsyncRead + Unpin>(r: &mut R) -> io::Result<Response> {
    let st = r.read_u8().await?;
    let resp = match st {
        ST_VALUE => Response::Value(read_bytes(r).await?),
        ST_NOT_FOUND => Response::NotFound,
        ST_OK => Response::Deleted(r.read_u8().await? != 0),
        ST_PONG => Response::Pong,
        ST_STATS => Response::Stats(StatsSnapshot {
            hits: r.read_u64().await?,
            misses: r.read_u64().await?,
            sets: r.read_u64().await?,
            dels: r.read_u64().await?,
            evictions: r.read_u64().await?,
            expirations: r.read_u64().await?,
            entries: r.read_u64().await?,
        }),
        ST_ERROR => {
            let b = read_bytes(r).await?;
            Response::Error(String::from_utf8_lossy(&b).into_owned())
        }
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown status byte {other}"),
            ))
        }
    };
    Ok(resp)
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn roundtrip_req(req: Request) {
        let mut buf = Vec::new();
        write_request(&mut buf, &req).await.unwrap();
        let mut cur = std::io::Cursor::new(buf);
        let got = read_request(&mut cur).await.unwrap().unwrap();
        assert_eq!(got, req);
    }

    async fn roundtrip_resp(resp: Response) {
        let mut buf = Vec::new();
        write_response(&mut buf, &resp).await.unwrap();
        let mut cur = std::io::Cursor::new(buf);
        let got = read_response(&mut cur).await.unwrap();
        assert_eq!(got, resp);
    }

    #[tokio::test]
    async fn request_roundtrips() {
        roundtrip_req(Request::Get(b"k".to_vec())).await;
        roundtrip_req(Request::Set { key: b"k".to_vec(), val: b"v".to_vec(), ttl_ms: 42 }).await;
        roundtrip_req(Request::Del(b"k".to_vec())).await;
        roundtrip_req(Request::Ping).await;
        roundtrip_req(Request::Stats).await;
    }

    #[tokio::test]
    async fn response_roundtrips() {
        roundtrip_resp(Response::Value(b"v".to_vec())).await;
        roundtrip_resp(Response::NotFound).await;
        roundtrip_resp(Response::Deleted(true)).await;
        roundtrip_resp(Response::Pong).await;
        roundtrip_resp(Response::Stats(StatsSnapshot { hits: 1, entries: 9, ..Default::default() })).await;
        roundtrip_resp(Response::Error("boom".into())).await;
    }

    #[tokio::test]
    async fn clean_eof_is_none() {
        let mut cur = std::io::Cursor::new(Vec::<u8>::new());
        assert_eq!(read_request(&mut cur).await.unwrap(), None);
    }
}
