# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Canonical secret / PII / internal-identifier patterns, shared by the prompt
hygiene linter (``tools/check_prompt_hygiene.py``), the corpus scrubber
(``agent/corpus_scrub.py``), and the output-leakage filter.

Deterministic and dependency-free. The gateway keeps its OWN copy (see
``gateway/output_guard.py``) so it stays standalone; this module is the source of
truth for the offline tools and is the place to extend a pattern once.
"""

from __future__ import annotations

import re

# Provider / cloud credential shapes. Conservative — aimed at high precision so
# the CI gate does not cry wolf on ordinary prose.
SECRET_PATTERNS: dict[str, str] = {
    "openai_key": r"sk-[A-Za-z0-9]{20,}",
    "anthropic_key": r"sk-ant-[A-Za-z0-9_\-]{20,}",
    "hf_token": r"hf_[A-Za-z0-9]{20,}",
    "xai_key": r"xai-[A-Za-z0-9]{20,}",
    "google_key": r"AIza[0-9A-Za-z_\-]{30,}",
    "aws_access_key": r"AKIA[0-9A-Z]{16}",
    "github_pat": r"gh[pousr]_[A-Za-z0-9]{30,}",
    "slack_token": r"xox[baprs]-[A-Za-z0-9-]{10,}",
    "private_key_block": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "generic_bearer": r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}",
    "aws_secret_assignment": r"(?i)aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+]{30,}",
}

# Internal identifiers that should never appear in a public prompt or corpus.
INTERNAL_PATTERNS: dict[str, str] = {
    "private_ip": r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b",
    "localhost_url_with_cred": r"https?://[^\s/@]+:[^\s/@]+@",
    "home_path": r"/(?:home|Users)/[A-Za-z0-9._-]+/",
    "internal_tld": r"\b[a-z0-9.-]+\.(?:internal|local|corp|intranet)\b",
}

# PII shapes for the corpus scrubber. Email is intentionally broad; the others are
# precise. We do NOT try to catch names — that needs NER and is out of scope here.
PII_PATTERNS: dict[str, str] = {
    "email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "us_ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]?){13,16}\b",
    "us_phone": r"\b(?:\+?1[ \-.]?)?\(?\d{3}\)?[ \-.]\d{3}[ \-.]\d{4}\b",
}

_SECRET_RX = {k: re.compile(v) for k, v in SECRET_PATTERNS.items()}
_INTERNAL_RX = {k: re.compile(v) for k, v in INTERNAL_PATTERNS.items()}
_PII_RX = {k: re.compile(v) for k, v in PII_PATTERNS.items()}


def _matches(text: str, table: "dict[str, re.Pattern[str]]") -> "list[dict]":
    out: list[dict] = []
    for name, rx in table.items():
        for m in rx.finditer(text or ""):
            out.append({"kind": name, "match": m.group(0), "start": m.start(), "end": m.end()})
    return out


def find_secrets(text: str) -> "list[dict]":
    """Credential-shaped substrings (the highest-severity finding)."""
    return _matches(text, _SECRET_RX)


def find_internal(text: str) -> "list[dict]":
    """Internal hostnames / IPs / home paths that leak deployment detail."""
    return _matches(text, _INTERNAL_RX)


def find_pii(text: str) -> "list[dict]":
    """Email / SSN / card / phone shapes for corpus scrubbing."""
    return _matches(text, _PII_RX)


def find_all(text: str) -> "list[dict]":
    return find_secrets(text) + find_internal(text) + find_pii(text)


def redact(text: str, *, secrets: bool = True, internal: bool = True, pii: bool = True) -> str:
    """Replace every matched span with a typed ``[REDACTED:<kind>]`` token.

    Applied longest-first per pass so overlapping matches don't corrupt offsets.
    """
    tables: list[dict] = []
    if secrets:
        tables.append(_SECRET_RX)
    if internal:
        tables.append(_INTERNAL_RX)
    if pii:
        tables.append(_PII_RX)
    out = text or ""
    for table in tables:
        for name, rx in table.items():
            out = rx.sub(f"[REDACTED:{name}]", out)
    return out


__all__ = [
    "SECRET_PATTERNS", "INTERNAL_PATTERNS", "PII_PATTERNS",
    "find_secrets", "find_internal", "find_pii", "find_all", "redact",
]
