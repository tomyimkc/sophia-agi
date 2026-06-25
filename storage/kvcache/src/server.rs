// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Tokio TCP server: one async task per connection, all sharing one
//! `Arc<ShardedCache>`. Requests on a single connection are handled in order
//! (request/response, no pipelining yet); concurrency comes from many
//! connections, not many in-flight requests per connection.

use std::sync::Arc;

use tokio::io::{AsyncWriteExt, BufReader, BufWriter};
use tokio::net::{TcpListener, TcpStream};

use crate::cache::ShardedCache;
use crate::protocol::{read_request, write_response, Request, Response};

/// Serve until the listener errors. Takes an already-bound listener so callers
/// (the binary, and integration tests) control the address — binding to port 0
/// lets a test grab an ephemeral port without races.
pub async fn serve(listener: TcpListener, cache: Arc<ShardedCache>) -> std::io::Result<()> {
    loop {
        let (stream, _peer) = listener.accept().await?;
        let cache = Arc::clone(&cache);
        tokio::spawn(async move {
            if let Err(e) = handle_conn(stream, cache).await {
                // A client hanging up mid-frame is normal; log only at debug.
                if e.kind() != std::io::ErrorKind::UnexpectedEof {
                    eprintln!("kvcache: connection error: {e}");
                }
            }
        });
    }
}

async fn handle_conn(stream: TcpStream, cache: Arc<ShardedCache>) -> std::io::Result<()> {
    let _ = stream.set_nodelay(true); // latency over throughput for small frames
    let (rd, wr) = stream.into_split();
    let mut reader = BufReader::new(rd);
    let mut writer = BufWriter::new(wr);

    while let Some(req) = read_request(&mut reader).await? {
        let resp = dispatch(&cache, req);
        write_response(&mut writer, &resp).await?;
        writer.flush().await?;
    }
    Ok(())
}

fn dispatch(cache: &ShardedCache, req: Request) -> Response {
    match req {
        Request::Get(k) => match cache.get(&k) {
            Some(v) => Response::Value(v),
            None => Response::NotFound,
        },
        Request::Set { key, val, ttl_ms } => {
            cache.set(&key, val, ttl_ms);
            Response::Deleted(true) // ST_OK ack; reused for set acknowledgement
        }
        Request::Del(k) => Response::Deleted(cache.del(&k)),
        Request::Ping => Response::Pong,
        Request::Stats => Response::Stats(cache.stats()),
    }
}
