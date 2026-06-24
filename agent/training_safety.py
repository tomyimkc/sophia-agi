# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
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

It scans EVERY string in an example (messages, metadata free-text, and alternate
schemas like prompt/completion or DPO chosen/rejected), and reads sensitivity from
synonymous metadata keys (classification/sensitivity/dataClass/visibility/label/pii)
and a truthy ``doNotTrain``.

Deterministic, no model; tuned for ~0 false positives on a public historical/
philosophical corpus (verified 0/518). Honest scope: the PII patterns are
**precision-over-recall** — they catch canonically-formatted secrets/PII, not every
obfuscation; pair with the metadata flags and the post-train canary test for depth.
"""

from __future__ import annotations

import hashlib
import re

# PII / secret patterns chosen to fire on genuine secrets, NOT on historical prose.
# Secret values require a >=6-char token after the key so prose like "the secret: be
# kind" does not match. Phone requires separators (so it catches NNN-NNN-NNNN without
# false-positiving on bare long IDs/ISBNs).
_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "ssn": re.compile(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+\d{1,3}[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"),
    "secret_kv": re.compile(r"(?i)\b(?:api[_-]?key|secret[_-]?key|secret|password|passwd|access[_-]?token|bearer)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-./+]{6,}"),
}
_UNSAFE_CLASSES = {"confidential", "secret", "restricted", "private"}
# Synonymous metadata keys a producer might use to mark sensitivity.
_CLASS_KEYS = {"classification", "sensitivity", "dataclass", "visibility", "label", "pii"}
_TRUTHY = {True, "true", "yes", "1", 1}


def _all_strings(obj) -> list:
    """Every string leaf anywhere in the example (messages, metadata, prompt/
    completion, chosen/rejected, instruction/input/output, nested, future fields)."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _all_strings(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in _all_strings(v)]
    return []


def unsafe_reasons(example: dict, *, secrets: "list | None" = None) -> list:
    """Why this example must NOT be trained on (empty list = safe)."""
    reasons: list = []
    meta = example.get("metadata") or {}
    for k, v in meta.items():
        kl = str(k).lower()
        if kl == "donottrain" and v in _TRUTHY:
            reasons.append("doNotTrain")
        elif kl in _CLASS_KEYS:
            if v in _TRUTHY or str(v).strip().lower() in _UNSAFE_CLASSES:
                reasons.append(f"{kl}:{str(v).strip().lower()}")

    text = "\n".join(_all_strings(example))   # scan the WHOLE example, not just messages
    for name, pat in _PATTERNS.items():
        if pat.search(text):
            reasons.append(f"pii:{name}")
    for s in (secrets or []):
        if s and s in text:
            reasons.append("secret-value")
            break
    return sorted(set(reasons))


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
