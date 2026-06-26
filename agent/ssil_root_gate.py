# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Root-of-trust / meta-gate — SSIL stage G0.

Recursive self-improvement eventually points the optimizer at its OWN gates and
rewards: the "raise capability" gradient becomes indistinguishable from "weaken the
scorer". The most dangerous self-edit is therefore not a change to the model weights
but a change to the *gate code* that decides whether weights may be promoted. G0 is
the root of trust the other gates hang from: nothing in the pipeline may silently
edit a gate.

Mechanism (deterministic, offline, stdlib only). A signed GATE MANIFEST pins, per
gate, ``{sha256, version, declaredInvariants:[names]}``. ``evaluate`` re-hashes the
current gate sources (caller supplies ``gateHashes`` name->sha256) and compares:

  - Any gate whose hash differs from the manifest with NO two-key approval token is an
    unauthorised edit -> ``reject``.
  - A change to the ROOT gate itself (``rootGateName``) is the highest-privilege edit:
    it requires an offline two-key ``rootApprovalToken`` AND that the proposed change
    PRESERVE every declared invariant (the new invariant-name set must be a superset of
    the manifest's). A root edit without the token, or one that drops a declared
    invariant, is a ``reject``.
  - A brand-new gate name not in the manifest is unreviewed -> ``quarantine``.
  - All hashes match the manifest -> ``promote``.

Verdict semantics (fail-closed): missing ``gateHashes`` or ``signedManifest`` (the
inputs that define authorised state) is a hard safety failure -> ``reject``; a missing
``rootGateName`` (we cannot tell which edit is privileged) -> ``quarantine``.

BOUNDARY (honest): G0 checks declared-invariant NAMES plus content hashes. It verifies
that a gate edit was human-AUTHORISED (two-key) and STRUCTURALLY invariant-preserving
(no declared invariant name was dropped). It does NOT and cannot prove the edit is
semantically safe — that the new gate code actually still enforces those invariants.
It abstains from any such semantic guarantee.

See docs/11-Platform/Safe-Self-Improvement-Loop.md (gate G0).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

GATE_ID = "G0"
GATE_NAME = "Root-of-trust / meta-gate"
SCHEMA = "sophia.root_gate_decision.v1"

_BOUNDARY = (
    "Checks declared-invariant NAMES plus content hashes only: confirms a gate edit was "
    "human-authorised (offline two-key token) and structurally invariant-preserving (no "
    "declared invariant name dropped). It cannot prove the edited gate code still SEMANTICALLY "
    "enforces those invariants; semantic safety of an edit is abstained, not asserted."
)


def sha256_of_text(s: str) -> str:
    """sha256 hex digest of a UTF-8 string (the hash a manifest pins per gate source)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _decision(
    *,
    verdict: str,
    reasons: tuple[str, ...],
    metrics: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
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
    """Gate the integrity of the gate set itself against a signed manifest.

    Breach (unauthorised or invariant-dropping gate edit) -> reject. Unreviewed new
    gate, or missing non-safety routing input -> quarantine. All hashes match -> promote.
    """
    # --- Fail-closed on a missing bundle (the highest-privilege gate must never
    # crash on None; like its siblings it rejects rather than raising). ---
    if bundle is None:
        return _decision(
            verdict="reject",
            reasons=("missing required input: bundle is None",),
            metrics={"missingInput": "bundle"},
            candidate_id=candidate_id,
        )

    gate_hashes = bundle.get("gateHashes")
    signed_manifest = bundle.get("signedManifest")

    # --- Fail-closed on hard safety inputs (these define the authorised state). ---
    if gate_hashes is None:
        return _decision(
            verdict="reject",
            reasons=("missing safety input: gateHashes (cannot establish current gate state)",),
            metrics={"missingInput": "gateHashes"},
            candidate_id=candidate_id,
        )
    if signed_manifest is None:
        return _decision(
            verdict="reject",
            reasons=("missing safety input: signedManifest (no signed root of trust to compare against)",),
            metrics={"missingInput": "signedManifest"},
            candidate_id=candidate_id,
        )

    root_gate_name = bundle.get("rootGateName")
    if root_gate_name is None:
        # Without knowing which gate is the root we cannot tell which edit is privileged.
        return _decision(
            verdict="quarantine",
            reasons=("missing input: rootGateName (cannot identify the privileged root edit)",),
            metrics={"missingInput": "rootGateName"},
            candidate_id=candidate_id,
        )

    root_token = bundle.get("rootApprovalToken")  # None unless an offline two-key token was supplied
    proposed_invariants = bundle.get("proposedInvariants") or {}

    breach: list[str] = []       # unauthorised / invariant-dropping edits -> reject
    unreviewed: list[str] = []   # brand-new unknown gates -> quarantine
    changed: list[str] = []
    matched: list[str] = []

    manifest_names = set(signed_manifest)
    current_names = set(gate_hashes)

    # A gate pinned by the manifest that has DISAPPEARED from the current set is a
    # structural edit to the gate graph: treat as an unauthorised change.
    for name in sorted(manifest_names - current_names):
        breach.append(f"gate removed from set without authorisation: {name!r}")

    for name in sorted(current_names):
        current_sha = gate_hashes[name]
        if name not in signed_manifest:
            unreviewed.append(name)
            continue
        entry = signed_manifest[name]
        manifest_sha = entry.get("sha256")
        if current_sha == manifest_sha:
            matched.append(name)
            continue

        # Hash mismatch: this gate's source was edited.
        changed.append(name)
        is_root = name == root_gate_name
        if is_root:
            # Highest-privilege edit: needs the offline two-key token AND must preserve
            # every declared invariant (new invariant set must be a superset).
            if not root_token:
                breach.append(
                    f"root gate {name!r} edited without offline two-key rootApprovalToken"
                )
            declared = set(entry.get("declaredInvariants", ()))
            proposed = set(proposed_invariants.get(name, ()))
            dropped = sorted(declared - proposed)
            if dropped:
                breach.append(
                    f"root gate {name!r} edit drops declared invariant(s): {dropped}"
                )
        else:
            # Any other gate edited with no two-key authorisation is unauthorised.
            if not root_token:
                breach.append(
                    f"unauthorised gate edit (hash mismatch, no two-key token): {name!r}"
                )
            else:
                # An authorised non-root edit must still preserve its declared invariants.
                declared = set(entry.get("declaredInvariants", ()))
                proposed = set(proposed_invariants.get(name, ()))
                dropped = sorted(declared - proposed)
                if dropped:
                    breach.append(
                        f"authorised gate edit to {name!r} drops declared invariant(s): {dropped}"
                    )

    if breach:
        verdict = "reject"
        reasons = tuple(breach)
    elif unreviewed:
        verdict = "quarantine"
        reasons = (
            f"unreviewed gate(s) absent from signed manifest (cannot vouch): {sorted(unreviewed)}",
        )
    else:
        verdict = "promote"
        reasons = ("all gate hashes match the signed manifest; root of trust intact",)

    metrics = {
        "rootGateName": root_gate_name,
        "rootApprovalTokenPresent": bool(root_token),
        "manifestGates": sorted(manifest_names),
        "currentGates": sorted(current_names),
        "matched": sorted(matched),
        "changed": sorted(changed),
        "unreviewed": sorted(unreviewed),
        "rootChanged": root_gate_name in changed,
    }
    return _decision(verdict=verdict, reasons=reasons, metrics=metrics, candidate_id=candidate_id)


def _manifest_from_sources(sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build a signed-manifest dict by hashing each gate's source text."""
    return {
        name: {
            "sha256": sha256_of_text(spec["source"]),
            "version": spec.get("version", "v1"),
            "declaredInvariants": list(spec.get("declaredInvariants", ())),
        }
        for name, spec in sources.items()
    }


def demo_bundle() -> dict[str, Any]:
    """A bundle in which every current gate hash matches its signed manifest entry,
    so this gate returns verdict ``promote`` (root of trust intact)."""
    sources = {
        "G0_root": {
            "source": "def evaluate(bundle): ...  # root gate source v1",
            "version": "v1",
            "declaredInvariants": ["two_key_for_root", "invariant_superset"],
        },
        "G2_reward_isolation": {
            "source": "def evaluate_reward_isolation(...): ...  # v1",
            "version": "v1",
            "declaredInvariants": ["no_reward_write", "monitored_access"],
        },
        "G5_honeypots": {
            "source": "def evaluate_honeypots(...): ...  # v1",
            "version": "v1",
            "declaredInvariants": ["no_bait_taken", "all_modes_covered"],
        },
        "G6_corrigibility": {
            "source": "def evaluate_corrigibility(...): ...  # v1",
            "version": "v1",
            "declaredInvariants": ["kill_switch_present", "operator_can_edit_gate"],
        },
    }
    manifest = _manifest_from_sources(sources)
    gate_hashes = {name: manifest[name]["sha256"] for name in manifest}
    return {
        "gateHashes": gate_hashes,
        "signedManifest": manifest,
        "rootGateName": "G0_root",
        "rootApprovalToken": None,
        "proposedInvariants": None,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(demo_bundle()), ensure_ascii=False, indent=2))
