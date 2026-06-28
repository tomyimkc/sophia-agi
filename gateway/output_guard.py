# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Output-leakage guard (P3) — the egress mirror of the firewall.

The firewall screens what comes IN (tool descriptions, call args). This screens
what goes OUT to the user, for three leak classes (OWASP LLM02 / LLM07):

  1. secret / credential / internal-identifier shapes (a leaked key or hostname),
  2. canary tokens (a confirmed system-prompt / training-data leak),
  3. verbatim system-prompt echo ("repeat your instructions" attacks).

Three actions, escalating: ``allow`` (clean) → ``redact`` (replace secret/PII
spans in place, still safe to surface) → ``block`` (a canary leaked or the system
prompt was echoed — withhold entirely, fail-closed).

Deterministic and dependency-free, so the gateway stays standalone (it keeps its
own pattern copy rather than importing ``agent.secret_patterns`` — same policy as
``gateway/firewall.py``).
"""

from __future__ import annotations

import re

# Mirror of agent.secret_patterns (kept local on purpose — see module docstring).
# Keep in sync with agent/secret_patterns.py so egress redaction is not weaker
# than the prompt-hygiene / corpus-scrub layers.
_LEAK_PATTERNS = {
    "openai_key": r"sk-[A-Za-z0-9]{20,}",
    "anthropic_key": r"sk-ant-[A-Za-z0-9_\-]{20,}",
    "hf_token": r"hf_[A-Za-z0-9]{20,}",
    "xai_key": r"xai-[A-Za-z0-9]{20,}",
    "google_key": r"AIza[0-9A-Za-z_\-]{30,}",
    "aws_access_key": r"AKIA[0-9A-Z]{16}",
    "aws_secret_assignment": r"(?i)aws_secret_access_key\s*[=:]\s*['\"]?[A-Za-z0-9/+]{30,}",
    "github_pat": r"gh[pousr]_[A-Za-z0-9]{30,}",
    "slack_token": r"xox[baprs]-[A-Za-z0-9-]{10,}",
    "generic_bearer": r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}",
    "private_key_block": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "private_ip": r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b",
    "home_path": r"/(?:home|Users)/[A-Za-z0-9._-]+/",
    "internal_tld": r"\b[a-z0-9.-]+\.(?:internal|local|corp|intranet)\b",
    # ReDoS-safe email (non-overlapping domain labels); mirrors agent.secret_patterns.
    "email": r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+",
    "us_ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d(?:[ -]?\d){12,15}\b",
    "us_phone": r"\b(?:\+?1[ \-.]?)?\(?\d{3}\)?[ \-.]\d{3}[ \-.]\d{4}\b",
}
_RX = {k: re.compile(v) for k, v in _LEAK_PATTERNS.items()}
_CANARY_RX = re.compile(r"SOPHIA-CANARY-[0-9a-f]{16}", re.IGNORECASE)

# Length of a verbatim system-prompt substring that counts as an "echo" leak.
_ECHO_MIN_CHARS = 40


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _echoes_system_prompt(text: str, system_prompt: str) -> bool:
    """True if a long contiguous window of the system prompt appears in ``text``.

    Cheap O(n) sliding-window over normalized text — catches "repeat everything
    above verbatim" without needing the exact whitespace to match.
    """
    sp = _normalize(system_prompt)
    out = _normalize(text)
    if len(sp) < _ECHO_MIN_CHARS or not out:
        return False
    step = max(1, _ECHO_MIN_CHARS // 2)
    for i in range(0, len(sp) - _ECHO_MIN_CHARS + 1, step):
        if sp[i:i + _ECHO_MIN_CHARS] in out:
            return True
    return False


def guard_output(text: str, *, system_prompt: "str | None" = None,
                 canaries: "list[str] | None" = None) -> dict:
    """Screen an outgoing user-facing string.

    Returns ``{action, clean, redacted, findings}`` where ``action`` is
    ``allow``/``redact``/``block`` and ``redacted`` is the safe-to-surface text
    (only meaningful for allow/redact; for block the caller must withhold).

    ``findings`` carry ONLY non-sensitive metadata (``kind``/``severity``/
    ``count``) — never the matched value — so returning/logging them cannot
    re-leak the secret the guard is suppressing.

    Canary handling: if ``canaries`` (the minted set) is provided, only those
    EXACT tokens are treated as a confirmed leak — an attacker cannot force a
    false-positive block (DoS) by emitting a random ``SOPHIA-CANARY-*`` string.
    If ``canaries`` is None, any canary-shaped token is treated as a leak (the
    conservative default when the caller has not supplied its canary set).
    """
    text = text or ""
    findings: list[dict] = []

    shaped = {m.group(0) for m in _CANARY_RX.finditer(text)}
    if canaries is not None:
        canary_hits = sorted(shaped & set(canaries))   # only KNOWN canaries
    else:
        canary_hits = sorted(shaped)                    # no allowlist → shape-based
    if canary_hits:
        findings.append({"kind": "canary", "severity": "critical", "count": len(canary_hits)})

    echoed = bool(system_prompt) and _echoes_system_prompt(text, system_prompt)
    if echoed:
        findings.append({"kind": "system_prompt_echo", "severity": "critical"})

    redacted = text
    for name, rx in _RX.items():
        n = len(rx.findall(text))
        if n:
            findings.append({"kind": name, "severity": "high", "count": n})
        redacted = rx.sub(f"[REDACTED:{name}]", redacted)

    if canary_hits or echoed:
        action = "block"            # confirmed leak — withhold entirely
    elif any(f["severity"] == "high" for f in findings):
        action = "redact"           # secret/PII present — surface the redacted form
    else:
        action = "allow"
    return {"action": action, "clean": action == "allow", "redacted": redacted,
            "findings": findings}


__all__ = ["guard_output"]
