# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Python bridge to the Rust ANN server (`services/ann_serving`, `serve` binary).

Demonstrates the architecture-track wiring: the dense recall view served by the compiled Rust
HNSW core while Python keeps query understanding, fusion, rerank, and provenance. The bridge is
**optional and fail-soft** — if the release binary or the exported vectors file is missing it
reports ``available() is False`` and callers fall back to the pure-Python vector path
(`agent.vector_store.search`). Nothing here is on the default retrieval path; it's an opt-in,
measured alternative for when the Rust core is built and the index exported.

Protocol (see `services/ann_serving/src/serve.rs`): spawn `serve <vectors_file>`, read the
`READY <n> <dim>` handshake, then write `"<k> <ef> f0 f1 …\\n"` per query and read back
`"id:score id:score …"`. Ids are row indices into `agent.vector_store.load_index`, so they map
straight back to chunks.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from agent.config import ROOT

DEFAULT_BINARY = ROOT / "services" / "ann_serving" / "target" / "release" / "serve"
DEFAULT_VECTORS = ROOT / "rag" / "index" / "vectors.txt"


class AnnClient:
    """Manages a `serve` subprocess and issues ANN queries over its stdio protocol."""

    def __init__(
        self,
        vectors_path: Path | None = None,
        binary: Path | None = None,
        *,
        m: int = 16,
        ef_construction: int = 200,
        shards: int = 1,
    ) -> None:
        # vectors_path may be a text vectors file (built on start) or a packed `.idx`
        # (loaded instantly; `shards` is then ignored, the shard count is baked into the file).
        self.vectors_path = Path(vectors_path) if vectors_path else DEFAULT_VECTORS
        self.binary = Path(binary) if binary else DEFAULT_BINARY
        self.m = m
        self.ef_construction = ef_construction
        self.shards = max(1, shards)
        self._proc: subprocess.Popen | None = None
        self.size = 0
        self.dim = 0

    def available(self) -> bool:
        """True iff both the compiled server and an exported vectors file are present."""
        return self.binary.exists() and self.vectors_path.exists()

    def start(self) -> "AnnClient":
        if not self.available():
            raise RuntimeError(
                f"ANN bridge unavailable: binary={self.binary.exists()} "
                f"vectors={self.vectors_path.exists()} (build with `cargo build --release` and "
                "`python tools/export_rag_index.py`)"
            )
        cmd = [str(self.binary), str(self.vectors_path), str(self.m), str(self.ef_construction)]
        # Sharding applies only when building from a text file; a `.idx` carries its own shards.
        if self.shards > 1 and self.vectors_path.suffix != ".idx":
            cmd += ["--shards", str(self.shards)]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        ready = self._proc.stdout.readline().strip()  # type: ignore[union-attr]
        if not ready.startswith("READY"):
            self.close()
            raise RuntimeError(f"ANN server failed to start (got {ready!r})")
        _, n, dim = ready.split()
        self.size, self.dim = int(n), int(dim)
        return self

    def search(self, embedding, k: int = 8, ef: int = 64) -> "list[tuple[int, float]]":
        """Return ``[(row_id, score)]`` best-first for one query embedding."""
        if self._proc is None:
            raise RuntimeError("call start() first")
        vec = " ".join(repr(float(x)) for x in embedding)
        self._proc.stdin.write(f"{k} {ef} {vec}\n")  # type: ignore[union-attr]
        self._proc.stdin.flush()  # type: ignore[union-attr]
        line = self._proc.stdout.readline().strip()  # type: ignore[union-attr]
        if not line or line.startswith("ERR"):
            return []
        out: list[tuple[int, float]] = []
        for tok in line.split():
            sid, _, sscore = tok.partition(":")
            try:
                out.append((int(sid), float(sscore)))
            except ValueError:
                continue
        return out

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.write("QUIT\n")
                    self._proc.stdin.flush()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            finally:
                self._proc = None

    def __enter__(self) -> "AnnClient":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.close()


__all__ = ["AnnClient", "DEFAULT_BINARY", "DEFAULT_VECTORS"]
