# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Pure-Python client for the Rust ``kvcache`` server (storage/kvcache).

Speaks the same length-prefixed binary protocol as ``storage/kvcache/src/
protocol.rs`` over a plain TCP socket — no native bindings, no extra
dependencies. This is the seam by which Sophia's Python side (RAG retrieval,
embeddings, decision memory) can offload hot reads to the sharded cache tier.

Design stance, consistent with Sophia's fail-closed ethos: the cache is an
*optimization*, never a source of truth. Every call degrades gracefully — on a
connection error, timeout, or protocol surprise the client returns ``None`` (a
miss) rather than raising, so a dead cache can only cost latency, never
correctness. Enable by setting ``SOPHIA_KVCACHE_ADDR=host:port``.
"""

from __future__ import annotations

import logging
import os
import socket
import struct
from dataclasses import dataclass

_LOG = logging.getLogger("sophia.kvcache")

# Wire constants — keep in lockstep with protocol.rs.
_OP_GET, _OP_SET, _OP_DEL, _OP_PING, _OP_STATS = 1, 2, 3, 4, 5
_ST_OK, _ST_VALUE, _ST_NOT_FOUND, _ST_PONG, _ST_STATS, _ST_ERROR = 0, 1, 2, 3, 4, 5

_DEFAULT_TIMEOUT_S = float(os.environ.get("SOPHIA_KVCACHE_TIMEOUT", "0.5"))


@dataclass
class CacheStats:
    hits: int
    misses: int
    sets: int
    dels: int
    evictions: int
    expirations: int
    entries: int


class KVCacheError(RuntimeError):
    """Raised only by the strict (``*_strict``) methods; the lenient API swallows it."""


class KVCacheClient:
    """One short-lived TCP connection to a kvcache-server.

    Not thread-safe; create one per thread, or one per call. Connection is lazy
    so constructing a client never blocks or raises.
    """

    def __init__(self, addr: str, *, timeout: float = _DEFAULT_TIMEOUT_S) -> None:
        host, _, port = addr.rpartition(":")
        if not host or not port:
            raise ValueError(f"address must be host:port, got {addr!r}")
        self._host = host
        self._port = int(port)
        self._timeout = timeout
        self._sock: socket.socket | None = None

    # --- connection management ---

    def _conn(self) -> socket.socket:
        if self._sock is None:
            s = socket.create_connection((self._host, self._port), timeout=self._timeout)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = s
        return self._sock

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __enter__(self) -> "KVCacheClient":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def _reset(self) -> None:
        # Drop a poisoned socket so the next call reconnects cleanly.
        self.close()

    def _recv_exact(self, n: int) -> bytes:
        sock = self._conn()
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise KVCacheError("server closed connection mid-frame")
            buf.extend(chunk)
        return bytes(buf)

    def _send(self, payload: bytes) -> None:
        self._conn().sendall(payload)

    # --- strict primitives (raise on failure) ---

    def get_strict(self, key: bytes) -> bytes | None:
        self._send(bytes([_OP_GET]) + struct.pack(">I", len(key)) + key)
        status = self._recv_exact(1)[0]
        if status == _ST_VALUE:
            (length,) = struct.unpack(">I", self._recv_exact(4))
            return self._recv_exact(length)
        if status == _ST_NOT_FOUND:
            return None
        raise KVCacheError(f"unexpected status {status} for GET")

    def set_strict(self, key: bytes, val: bytes, ttl_ms: int = 0) -> None:
        payload = (
            bytes([_OP_SET])
            + struct.pack(">I", len(key)) + key
            + struct.pack(">I", len(val)) + val
            + struct.pack(">Q", ttl_ms)
        )
        self._send(payload)
        status = self._recv_exact(1)[0]
        if status != _ST_OK:
            raise KVCacheError(f"unexpected status {status} for SET")
        self._recv_exact(1)  # ack flag byte

    def del_strict(self, key: bytes) -> bool:
        self._send(bytes([_OP_DEL]) + struct.pack(">I", len(key)) + key)
        status = self._recv_exact(1)[0]
        if status == _ST_OK:
            return self._recv_exact(1)[0] != 0
        if status == _ST_NOT_FOUND:
            return False
        raise KVCacheError(f"unexpected status {status} for DEL")

    def ping_strict(self) -> None:
        self._send(bytes([_OP_PING]))
        if self._recv_exact(1)[0] != _ST_PONG:
            raise KVCacheError("bad PING response")

    def stats_strict(self) -> CacheStats:
        self._send(bytes([_OP_STATS]))
        if self._recv_exact(1)[0] != _ST_STATS:
            raise KVCacheError("bad STATS response")
        vals = struct.unpack(">7Q", self._recv_exact(56))
        return CacheStats(*vals)

    # --- lenient API (never raises; a dead cache is just a miss) ---

    def get(self, key: bytes) -> bytes | None:
        try:
            return self.get_strict(key)
        except (OSError, KVCacheError) as e:
            _LOG.debug("kvcache GET failed, treating as miss: %s", e)
            self._reset()
            return None

    def set(self, key: bytes, val: bytes, ttl_ms: int = 0) -> bool:
        try:
            self.set_strict(key, val, ttl_ms)
            return True
        except (OSError, KVCacheError) as e:
            _LOG.debug("kvcache SET failed, ignoring: %s", e)
            self._reset()
            return False


def from_env() -> KVCacheClient | None:
    """Construct a client from ``SOPHIA_KVCACHE_ADDR``, or ``None`` if unset.

    Returns ``None`` (cache disabled) rather than raising when the env var is
    absent or malformed, so callers can write ``client = from_env()`` and guard
    with a simple truthiness check.
    """
    addr = os.environ.get("SOPHIA_KVCACHE_ADDR", "").strip()
    if not addr:
        return None
    try:
        return KVCacheClient(addr)
    except ValueError as e:
        _LOG.warning("ignoring malformed SOPHIA_KVCACHE_ADDR=%r: %s", addr, e)
        return None
