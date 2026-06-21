"""LoRA leakage guard — keep confidential/secret/PII data OUT of the weights.

Anything in a fine-tuned model's weights is extractable (membership inference,
canary regurgitation), so the only safe rule is: *never train on confidential
data*. This is a deterministic pre-export filter that drops any example which
carries a secret/canary, matches a PII pattern, or is metadata-flagged
(`classification` ∈ confidential/secret/restricted, or `doNotTrain: true`) — so a
sensitive example can never reach `training/corpus.jsonl`.

It also ships a **canary harness**: inject unique canary strings, then after a
training run measure how often the model regurgitates them — a direct, falsifiable
leakage test the maintainer runs post-train. (The filter guarantees a
*confidential* canary never enters training in the first place.)

Deterministic, no model; designed for ~0 false positives on a public historical/
philosophical corpus (it flags secrets/PII, never ordinary names or years).
"""

from __future__ import annotations

import hashlib
import re

# PII / secret patterns chosen to fire on genuine secrets, NOT on historical prose.
_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b"),
    "phone": re.compile(r"\b\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "secret_kv": re.compile(r"(?i)\b(?:api[_-]?key|secret|password|passwd|access[_-]?token|bearer)\b\s*[:=]\s*\S+"),
}
_UNSAFE_CLASSES = {"confidential", "secret", "restricted", "private"}


def _example_text(example: dict) -> str:
    """All free text in an example (message contents)."""
    parts = []
    for msg in example.get("messages", []) or []:
        parts.append(str(msg.get("content", "")))
    if "text" in example:
        parts.append(str(example.get("text", "")))
    return "\n".join(parts)


def unsafe_reasons(example: dict, *, secrets: "list | None" = None) -> list:
    """Why this example must NOT be trained on (empty list = safe)."""
    reasons: list = []
    meta = example.get("metadata") or {}
    cls = str(meta.get("classification", "")).strip().lower()
    if cls in _UNSAFE_CLASSES:
        reasons.append(f"classification:{cls}")
    if meta.get("doNotTrain") is True:
        reasons.append("doNotTrain")

    text = _example_text(example)
    for name, pat in _PATTERNS.items():
        if pat.search(text):
            reasons.append(f"pii:{name}")
    for s in (secrets or []):
        if s and s in text:
            reasons.append("secret-value")
            break
    return reasons


def is_safe_to_train(example: dict, *, secrets: "list | None" = None) -> bool:
    return not unsafe_reasons(example, secrets=secrets)


def filter_examples(examples: list, *, secrets: "list | None" = None) -> dict:
    """Split a corpus into safe-to-train vs dropped (with reasons)."""
    safe, dropped = [], []
    for ex in examples:
        rs = unsafe_reasons(ex, secrets=secrets)
        (dropped if rs else safe).append({"example": ex, "reasons": rs} if rs else ex)
    return {
        "safe": safe,
        "dropped": dropped,
        "nIn": len(examples),
        "nSafe": len(safe),
        "nDropped": len(dropped),
        "reasonsHistogram": _histogram([r for d in dropped for r in d["reasons"]]),
    }


def _histogram(items: list) -> dict:
    h: dict = {}
    for it in items:
        key = it.split(":")[0]
        h[key] = h.get(key, 0) + 1
    return h


# --------------------------------------------------------------------------- #
# Canary harness — a direct, falsifiable post-train leakage test.
# --------------------------------------------------------------------------- #


def make_canary(seed: str) -> str:
    """A unique, unguessable marker string."""
    return "CANARY-" + hashlib.sha256(f"sophia-canary::{seed}".encode()).hexdigest()[:16]


def canary_extraction_rate(model_outputs: list, canaries: list) -> float:
    """Fraction of canaries that appear verbatim in any model output. A model NOT
    trained on a canary should score 0; a memorising model regurgitates them."""
    if not canaries:
        return 0.0
    blob = "\n".join(str(o) for o in model_outputs)
    leaked = sum(1 for c in canaries if c and c in blob)
    return round(leaked / len(canaries), 4)
