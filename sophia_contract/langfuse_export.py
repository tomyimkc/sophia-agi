# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Ship the contract's Langfuse-compatible trace spans to a live Langfuse instance.

``sophia_contract.trace.Tracer`` already records spans in Langfuse's shape
(``id, name, startTime, endTime, input, output, level, metadata``) to
``traces.jsonl``. This adapter batches them to Langfuse's public ingestion API
(``POST {host}/api/public/ingestion``, basic-auth public:secret), mapping each
span to a ``trace-create`` event.

Dependency-free (urllib, like agent/model.py) and offline-testable: with no creds
or ``dry_run=True`` it builds the batch and returns it WITHOUT any network call, so
CI can assert the payload shape. Credentials come from the environment:
``LANGFUSE_HOST`` (default https://cloud.langfuse.com), ``LANGFUSE_PUBLIC_KEY``,
``LANGFUSE_SECRET_KEY``.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request


def _event(span: dict) -> dict:
    """One Langfuse ingestion event (trace-create) for a contract span."""
    return {
        "id": span["id"],                      # event id (idempotent ingestion)
        "type": "trace-create",
        "timestamp": span.get("startTime") or "",
        "body": {
            "id": span["id"],
            "name": span.get("name", "sophia-contract"),
            "timestamp": span.get("startTime") or "",
            "input": span.get("input"),
            "output": span.get("output"),
            "metadata": {**(span.get("metadata") or {}), "level": span.get("level", "DEFAULT"),
                         "source": "sophia-contract"},
        },
    }


def build_batch(spans: "list[dict]") -> dict:
    """The exact JSON body Langfuse's /api/public/ingestion expects."""
    return {"batch": [_event(s) for s in spans]}


def export_spans(
    spans: "list[dict]",
    *,
    host: "str | None" = None,
    public_key: "str | None" = None,
    secret_key: "str | None" = None,
    dry_run: bool = False,
    timeout_sec: int = 30,
) -> dict:
    """POST spans to Langfuse. Returns {sent, count, batch?, status?/error?}.

    No-ops safely (sent=False) when creds are missing or dry_run — returning the
    batch it *would* have sent, so callers/tests can inspect it offline."""
    host = (host or os.environ.get("LANGFUSE_HOST") or "https://cloud.langfuse.com").rstrip("/")
    public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
    batch = build_batch(spans)

    if dry_run or not (public_key and secret_key):
        return {"sent": False, "count": len(batch["batch"]),
                "reason": "dry_run" if dry_run else "missing LANGFUSE_PUBLIC_KEY/SECRET_KEY",
                "batch": batch, "host": host}

    token = base64.b64encode(f"{public_key}:{secret_key}".encode("utf-8")).decode("ascii")
    request = urllib.request.Request(
        f"{host}/api/public/ingestion",
        data=json.dumps(batch).encode("utf-8"),
        headers={"Authorization": f"Basic {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as resp:
            return {"sent": True, "count": len(batch["batch"]), "status": resp.status}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300] if hasattr(exc, "read") else ""
        return {"sent": False, "count": len(batch["batch"]), "error": f"HTTP {exc.code}: {body}"}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"sent": False, "count": len(batch["batch"]), "error": repr(exc)}


def export_traces_file(path: "str", **kw) -> dict:
    """Read a traces.jsonl (one span per line) and export it."""
    from pathlib import Path

    spans = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                spans.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return export_spans(spans, **kw)
