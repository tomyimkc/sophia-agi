// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 tomyimkc
//! Async client. One TCP connection, request/response. Cheap to create per
//! task; the benchmark and integration tests open many in parallel.

use tokio::io::{AsyncWriteExt, BufReader, BufWriter};
use tokio::net::{TcpStream, ToSocketAddrs};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};

use crate::protocol::{read_response, write_request, Request, Response, StatsSnapshot};

pub struct Client {
    reader: BufReader<OwnedReadHalf>,
    writer: BufWriter<OwnedWriteHalf>,
}

impl Client {
    pub async fn connect<A: ToSocketAddrs>(addr: A) -> std::io::Result<Self> {
        let stream = TcpStream::connect(addr).await?;
        let _ = stream.set_nodelay(true);
        let (rd, wr) = stream.into_split();
        Ok(Client {
            reader: BufReader::new(rd),
            writer: BufWriter::new(wr),
        })
    }

    async fn call(&mut self, req: Request) -> std::io::Result<Response> {
        write_request(&mut self.writer, &req).await?;
        self.writer.flush().await?;
        read_response(&mut self.reader).await
    }

    /// Pipeline a batch: write every request in one flush, then read every
    /// response in order. Amortizes per-request round-trip latency across the
    /// batch — the lever for raising throughput past the one-in-flight ceiling.
    /// Responses are positional (`out[i]` answers `reqs[i]`).
    pub async fn pipeline(&mut self, reqs: &[Request]) -> std::io::Result<Vec<Response>> {
        for req in reqs {
            write_request(&mut self.writer, req).await?;
        }
        self.writer.flush().await?;
        let mut out = Vec::with_capacity(reqs.len());
        for _ in reqs {
            out.push(read_response(&mut self.reader).await?);
        }
        Ok(out)
    }

    pub async fn get(&mut self, key: &[u8]) -> std::io::Result<Option<Vec<u8>>> {
        match self.call(Request::Get(key.to_vec())).await? {
            Response::Value(v) => Ok(Some(v)),
            Response::NotFound => Ok(None),
            other => Err(unexpected(other)),
        }
    }

    pub async fn set(&mut self, key: &[u8], val: &[u8], ttl_ms: u64) -> std::io::Result<()> {
        match self.call(Request::Set { key: key.to_vec(), val: val.to_vec(), ttl_ms }).await? {
            Response::Deleted(_) => Ok(()),
            other => Err(unexpected(other)),
        }
    }

    pub async fn del(&mut self, key: &[u8]) -> std::io::Result<bool> {
        match self.call(Request::Del(key.to_vec())).await? {
            Response::Deleted(found) => Ok(found),
            other => Err(unexpected(other)),
        }
    }

    pub async fn ping(&mut self) -> std::io::Result<()> {
        match self.call(Request::Ping).await? {
            Response::Pong => Ok(()),
            other => Err(unexpected(other)),
        }
    }

    pub async fn stats(&mut self) -> std::io::Result<StatsSnapshot> {
        match self.call(Request::Stats).await? {
            Response::Stats(s) => Ok(s),
            other => Err(unexpected(other)),
        }
    }
}

fn unexpected(resp: Response) -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("unexpected response: {resp:?}"),
    )
}
