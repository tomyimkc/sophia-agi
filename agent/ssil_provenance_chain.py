# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Cryptographic provenance chain (Merkle lineage) gate — SSIL stage G9C.

Recursive self-improvement promotes a model only if it descends *exclusively* from
gated ancestors. But a flat registry (``agent/ssil_registry.py``) is mutable: an
attacker (or a buggy round) can rewrite a parent's gate verdict, splice in an
ungated ancestor, or reorder history, and nothing in the registry detects it. G9C
binds the lineage with a hash chain so any such edit is *tamper-evident*, and then
checks that the leaf's entire ancestry was promote-gated.

Mechanism (real, tractable, pure-stdlib):
  - Each entry = ``{id, parentHash, spec, metric, gateVerdict}``.
  - ``entryHash = sha256(parentHash + canonical_json(id, spec, metric, gateVerdict))``,
    where ``canonical_json`` is ``json.dumps(..., sort_keys=True, separators=(",",":"))``
    so the digest is stable regardless of key order or whitespace.
  - ``append_entry`` computes ``entryHash`` from the current chain tip and returns the
    entry with its hash filled in (the chain is a singly-linked Merkle-style list).
  - ``verify_chain`` recomputes every hash and checks each ``parentHash`` points at the
    prior entry's recomputed hash; the FIRST index that fails is reported.
  - ``lineage`` walks ``parentHash`` links from a leaf up to the genesis root.
  - ``assert_gated_lineage`` requires every ancestor (and the leaf) to carry
    ``gateVerdict == "promote"``.

Optional integrity signing: if ``bundle["hmacKey"]`` is set, each entry's hash is an
HMAC-SHA256 keyed by that secret instead of a bare SHA-256, so an attacker who cannot
forge the key cannot recompute a valid chain after editing it.

Verdict semantics (fail-closed; breaches win):
  - ``reject``     — chain fails verification (tampering/reorder) OR the leaf's
                     lineage contains an ungated (non-promote) ancestor.
  - ``quarantine`` — required input missing (no chain / leaf not in chain / broken
                     parent link / unsigned chain when a key is required) — cannot
                     verify lineage, so never promote.
  - ``promote``    — chain verifies and the full leaf lineage is promote-gated.

HONEST BOUNDARY: an HMAC with a shared secret is *integrity*, not a full PKI/HSM —
anyone holding the key can forge the chain, and we do not manage key custody here. A
tamper-evident chain proves LINEAGE (which gated ancestors produced this leaf), NOT
that the gating *decisions themselves* were correct. canClaimAGI is always False;
candidateOnly is always True.

This is SSIL gate G9C. See docs/11-Platform/Safe-Self-Improvement-Loop.md (gate G9C).
"""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GATE_ID = "G9C"
GATE_NAME = "Cryptographic provenance chain (Merkle lineage)"
SCHEMA = "sophia.provenance_chain_decision.v1"

_GENESIS = "0" * 64  # parentHash of the genesis (root) entry
_BOUNDARY = (
    "tamper-evident hash chain (sha256, optional HMAC-sha256). An HMAC with a shared "
    "secret is integrity, not full PKI/HSM key custody — a key holder can forge the "
    "chain. The chain proves LINEAGE (which gated ancestors produced this leaf), NOT "
    "that the gating decisions themselves were correct. Candidate-only signal."
)


def _canonical_json(entry: dict[str, Any]) -> str:
    """Canonical serialization of an entry's content fields (order-independent).

    Only the content-bearing fields participate; ``parentHash`` and ``entryHash`` are
    chain bookkeeping and are mixed in separately (parentHash) or are the output.
    """
    payload = {
        "id": entry.get("id"),
        "spec": entry.get("spec"),
        "metric": entry.get("metric"),
        "gateVerdict": entry.get("gateVerdict"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_hash(entry: dict[str, Any], parent_hash: str, *, hmac_key: str | None = None) -> str:
    """entryHash = digest(parentHash + canonical_json(content)).

    Uses HMAC-SHA256 keyed by ``hmac_key`` when given, else bare SHA-256. Deterministic
    and offline; no wall-clock or randomness enters the digest.
    """
    message = (parent_hash + _canonical_json(entry)).encode("utf-8")
    if hmac_key is not None:
        return hmac.new(hmac_key.encode("utf-8"), message, hashlib.sha256).hexdigest()
    return hashlib.sha256(message).hexdigest()


def append_entry(chain: list[dict[str, Any]], entry: dict[str, Any], *, hmac_key: str | None = None) -> dict[str, Any]:
    """Return ``entry`` with ``parentHash`` (chain tip) and ``entryHash`` filled in.

    Does not mutate the input ``entry``; the caller appends the returned dict. The
    parent of the first entry is the genesis hash (all zeros).
    """
    parent_hash = chain[-1]["entryHash"] if chain else _GENESIS
    linked = dict(entry)
    linked["parentHash"] = parent_hash
    linked["entryHash"] = compute_hash(linked, parent_hash, hmac_key=hmac_key)
    return linked


def verify_chain(chain: list[dict[str, Any]], *, hmac_key: str | None = None) -> tuple[bool, int]:
    """Recompute every hash; return (ok, firstBadIndex).

    ``firstBadIndex`` is -1 when the chain is intact, else the index of the first entry
    whose stored ``entryHash`` does not match the recomputed value or whose
    ``parentHash`` does not point at the prior recomputed hash. An empty chain is
    vacuously ok.
    """
    prev_hash = _GENESIS
    for i, entry in enumerate(chain):
        if entry.get("parentHash") != prev_hash:
            return (False, i)
        expected = compute_hash(entry, prev_hash, hmac_key=hmac_key)
        if entry.get("entryHash") != expected:
            return (False, i)
        prev_hash = expected
    return (True, -1)


def _index_by_hash(chain: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {e.get("entryHash"): e for e in chain if e.get("entryHash") is not None}


def lineage(chain: list[dict[str, Any]], leaf_id: str) -> list[dict[str, Any]]:
    """Ancestor chain for ``leaf_id``, leaf first up to the genesis root.

    Walks ``parentHash`` links. Raises ``KeyError`` if the leaf id is absent and
    ``ValueError`` if a ``parentHash`` points at a missing entry (broken link) or a
    cycle is detected. Returns ``[leaf, parent, ..., root]``.
    """
    by_id = {e.get("id"): e for e in chain}
    if leaf_id not in by_id:
        raise KeyError(leaf_id)
    by_hash = _index_by_hash(chain)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    node: dict[str, Any] | None = by_id[leaf_id]
    while node is not None:
        h = node.get("entryHash")
        if h in seen:
            raise ValueError(f"cycle detected at entry {node.get('id')!r}")
        seen.add(h)
        out.append(node)
        parent = node.get("parentHash")
        if parent == _GENESIS:
            break
        if parent not in by_hash:
            raise ValueError(f"broken parent link at entry {node.get('id')!r}: parentHash not in chain")
        node = by_hash[parent]
    return out


def assert_gated_lineage(chain: list[dict[str, Any]], leaf_id: str) -> tuple[bool, list[str]]:
    """Every ancestor (and the leaf) must carry ``gateVerdict == "promote"``.

    Returns (ok, ungated_ids). Does not raise on lineage errors here — the gate's
    ``evaluate`` handles those as fail-closed quarantine — but propagates KeyError /
    ValueError from ``lineage`` so callers can distinguish "ungated" from "unwalkable".
    """
    chain_path = lineage(chain, leaf_id)
    ungated = [e.get("id") for e in chain_path if e.get("gateVerdict") != "promote"]
    return (not ungated, [str(i) for i in ungated])


def _decision(*, verdict: str, reasons: list[str], metrics: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    """Build the standard SSIL decision dict (exact key set/order for G9C)."""
    return {
        "schema": SCHEMA,
        "gate": GATE_ID,
        "gateName": GATE_NAME,
        "candidateOnly": True,
        "level3Evidence": False,
        "canClaimAGI": False,
        "candidateId": candidate_id,
        "verdict": verdict,
        "reasons": list(reasons),
        "metrics": metrics,
        "boundary": _BOUNDARY,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def evaluate(bundle: dict[str, Any], *, candidate_id: str = "sophia-rlvr-v1") -> dict[str, Any]:
    """Verify the provenance chain AND that the leaf lineage is fully promote-gated.

    Fail-closed: missing ``chain`` / missing ``leafId`` / leaf absent / broken link ->
    ``quarantine`` naming the missing input. Tampering (chain fails verification) or an
    ungated ancestor -> ``reject``. Chain verifies and lineage fully promote-gated ->
    ``promote``.
    """
    if bundle is None:
        return _decision(
            verdict="quarantine",
            reasons=["missing required input: bundle is None"],
            metrics={},
            candidate_id=candidate_id,
        )

    chain = bundle.get("chain")
    if chain is None:
        return _decision(
            verdict="quarantine",
            reasons=["fail-closed: missing required input 'chain'"],
            metrics={},
            candidate_id=candidate_id,
        )
    if not isinstance(chain, list):
        return _decision(
            verdict="quarantine",
            reasons=["fail-closed: required input 'chain' is not a list"],
            metrics={"chainType": type(chain).__name__},
            candidate_id=candidate_id,
        )

    leaf_id = bundle.get("leafId")
    if leaf_id is None:
        return _decision(
            verdict="quarantine",
            reasons=["fail-closed: missing required input 'leafId'"],
            metrics={"chainLength": len(chain)},
            candidate_id=candidate_id,
        )

    hmac_key = bundle.get("hmacKey")  # optional; None -> bare SHA-256

    # (1) INTEGRITY — recompute every hash; a mismatch means tampering / reorder.
    ok, bad_index = verify_chain(chain, hmac_key=hmac_key)
    if not ok:
        return _decision(
            verdict="reject",
            reasons=[
                f"provenance chain failed verification (tampering or reorder) at index {bad_index}",
            ],
            metrics={
                "chainLength": len(chain),
                "chainVerified": False,
                "firstBadIndex": bad_index,
                "signed": hmac_key is not None,
            },
            candidate_id=candidate_id,
        )

    # (2) LINEAGE walk — leaf must exist and links must be intact.
    try:
        chain_path = lineage(chain, leaf_id)
    except KeyError:
        return _decision(
            verdict="quarantine",
            reasons=[f"fail-closed: leafId {leaf_id!r} not present in chain (cannot verify lineage)"],
            metrics={"chainLength": len(chain), "chainVerified": True, "leafId": leaf_id},
            candidate_id=candidate_id,
        )
    except ValueError as exc:
        return _decision(
            verdict="quarantine",
            reasons=[f"fail-closed: cannot walk lineage of {leaf_id!r}: {exc}"],
            metrics={"chainLength": len(chain), "chainVerified": True, "leafId": leaf_id},
            candidate_id=candidate_id,
        )

    lineage_ids = [e.get("id") for e in chain_path]
    ungated = [e.get("id") for e in chain_path if e.get("gateVerdict") != "promote"]

    # (3) GATED-LINEAGE — every ancestor (and the leaf) must be promote-gated.
    if ungated:
        return _decision(
            verdict="reject",
            reasons=[
                f"ungated ancestor(s) in leaf lineage (not 'promote'): {ungated}; "
                f"leaf does not descend exclusively from gated ancestors",
            ],
            metrics={
                "chainLength": len(chain),
                "chainVerified": True,
                "leafId": leaf_id,
                "lineageIds": lineage_ids,
                "lineageDepth": len(chain_path),
                "ungatedAncestors": ungated,
                "signed": hmac_key is not None,
            },
            candidate_id=candidate_id,
        )

    return _decision(
        verdict="promote",
        reasons=[
            f"chain verified ({len(chain)} entries); leaf {leaf_id!r} lineage "
            f"({len(chain_path)} ancestors) is fully promote-gated",
        ],
        metrics={
            "chainLength": len(chain),
            "chainVerified": True,
            "leafId": leaf_id,
            "lineageIds": lineage_ids,
            "lineageDepth": len(chain_path),
            "ungatedAncestors": [],
            "signed": hmac_key is not None,
        },
        candidate_id=candidate_id,
    )


def append_decision_ledger(decision: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decision, ensure_ascii=False) + "\n")


def build_chain(entries: list[dict[str, Any]], *, hmac_key: str | None = None) -> list[dict[str, Any]]:
    """Convenience: hash-link a list of content entries into a verified chain."""
    chain: list[dict[str, Any]] = []
    for e in entries:
        chain.append(append_entry(chain, e, hmac_key=hmac_key))
    return chain


def demo_bundle() -> dict[str, Any]:
    """A genesis -> two-child lineage where every ancestor is promote-gated and the
    chain hashes verify. This bundle makes G9C return ``promote``."""
    content = [
        {"id": "genesis", "spec": {"adapter": "base", "kind": "lora_adapter"}, "metric": 0.0, "gateVerdict": "promote"},
        {"id": "round-1", "spec": {"adapter": "sophia-rlvr-v1", "kind": "lora_adapter"}, "metric": 0.7149, "gateVerdict": "promote"},
        {"id": "round-2", "spec": {"adapter": "sophia-rlvr-v2", "kind": "lora_adapter"}, "metric": 0.7310, "gateVerdict": "promote"},
    ]
    chain = build_chain(content)
    return {"chain": chain, "leafId": "round-2", "hmacKey": None}


if __name__ == "__main__":
    print(json.dumps(evaluate(demo_bundle()), ensure_ascii=False, indent=2))
