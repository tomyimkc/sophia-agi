# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Scrub PII / secrets / canaries from any corpus before it is published.

Run over every record bound for the public Hugging Face dataset. The scrubber is
conservative-by-redaction: it replaces matches with typed ``[REDACTED:…]`` tokens
rather than dropping records, so the corpus stays usable and the redaction is
auditable. A record that still contains a *canary* after scrubbing is a hard fail
(a canary in the public corpus means the private mixture leaked) and is dropped.

Pure, deterministic, dependency-free.
"""

from __future__ import annotations

from typing import Any, Iterable

from agent.canary import scan_for_canaries
from agent.secret_patterns import find_pii, find_secrets, redact

# Keys whose string values are scrubbed in a record. Extend for your schema.
_TEXT_KEYS = ("text", "input", "output", "prompt", "completion", "content",
              "question", "answer", "instruction", "response")


def scrub_text(text: str) -> str:
    """Redact secrets, internal identifiers, and PII from a single string."""
    return redact(text, secrets=True, internal=True, pii=True)


def scrub_record(record: dict[str, Any]) -> "tuple[dict[str, Any], dict]":
    """Return ``(scrubbed_record, report)``.

    ``report`` has ``secrets`` / ``pii`` counts and ``canaries`` (any canary found
    BEFORE scrubbing — its presence is a leak the caller must treat as fatal).
    """
    secrets = pii = 0
    canaries: list[str] = []
    out = dict(record)
    for key, value in record.items():
        if key in _TEXT_KEYS and isinstance(value, str):
            secrets += len(find_secrets(value))
            pii += len(find_pii(value))
            canaries += scan_for_canaries(value)
            out[key] = scrub_text(value)
    return out, {"secrets": secrets, "pii": pii, "canaries": sorted(set(canaries))}


def scrub_corpus(records: Iterable[dict[str, Any]]) -> "tuple[list[dict], dict]":
    """Scrub an iterable of records. Records whose source contained a canary are
    DROPPED (not just redacted) and counted. Returns ``(clean_records, summary)``.
    """
    clean: list[dict] = []
    total_secrets = total_pii = dropped = 0
    leaked_canaries: list[str] = []
    for rec in records:
        scrubbed, rep = scrub_record(rec)
        total_secrets += rep["secrets"]
        total_pii += rep["pii"]
        if rep["canaries"]:
            dropped += 1
            leaked_canaries += rep["canaries"]
            continue  # never publish a record that carried a canary
        clean.append(scrubbed)
    summary = {
        "in": clean.__len__() + dropped,
        "out": len(clean),
        "redacted_secrets": total_secrets,
        "redacted_pii": total_pii,
        "dropped_canary_records": dropped,
        "leaked_canaries": sorted(set(leaked_canaries)),
        "ok": not leaked_canaries,
    }
    return clean, summary


__all__ = ["scrub_text", "scrub_record", "scrub_corpus"]
