"""Vault ↔ contract bridge — the Stage-0 ``provenance.py`` of the playbook.

Turns an Obsidian/Markdown note into a provenance-gated artifact: read the note's
frontmatter, ``record_claim`` it, ``verify_claim`` it, and stamp the verdict back
into the frontmatter (``provenance_id``, ``gate_status``, …). The core invariant
holds end to end: **a note is publishable only when ``gate_status == accepted``.**

Dependency-free: uses ``okf.frontmatter`` (the repo's own YAML codec), not
``python-frontmatter`` — so the bridge ships with sophia-agi and adds no install.

Idempotency is keyed by *content version*: ``vault:<relpath>:<body-fingerprint>``,
so re-gating an unchanged note returns the same claim (idempotent), while an edit
is a new claim version — never a fail-closed key/content conflict.

    from sophia_contract import SophiaContract
    from sophia_contract.vault import VaultGate
    gate = VaultGate(SophiaContract(store_dir="var/contract"))
    verdict = gate.gate_note("vault/06_Review/draft.md", role="role_05_copywriting")
    if gate.is_publishable("vault/06_Review/draft.md"): ...
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from okf import frontmatter
from sophia_contract.models import content_fingerprint
from sophia_contract.service import SophiaContract

# Frontmatter keys the bridge reads (inputs) and writes (gate outputs).
BLP_KEY = "blp_level"
ROLE_KEY = "role"
STATUS_KEY = "gate_status"
PROV_KEY = "provenance_id"


def _as_list(value) -> "list":
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


class VaultGate:
    def __init__(self, contract: "SophiaContract | None" = None, *, vault_root: "str | Path | None" = None):
        self.contract = contract or SophiaContract()
        self.vault_root = Path(vault_root) if vault_root else None

    def _relpath(self, path: Path) -> str:
        if self.vault_root:
            try:
                return str(path.resolve().relative_to(self.vault_root.resolve()))
            except ValueError:
                pass
        return path.name

    def idempotency_key(self, path: Path, body: str) -> str:
        """Content-versioned key: same note + same body -> same claim (idempotent);
        an edited body -> a new claim version."""
        return f"vault:{self._relpath(path)}:{content_fingerprint(body)}"

    def gate_note(self, path: "str | Path", *, role: "str | None" = None,
                  clearance: "str | None" = None, default_blp: str = "UNCLASSIFIED",
                  write_back: bool = True) -> dict:
        """Record + verify a note, stamp the verdict into its frontmatter, and return
        the Verdict (or an error dict). Fail-closed: any contract error leaves the
        note un-stamped/unpublishable."""
        path = Path(path)
        meta, body = frontmatter.parse(path.read_text(encoding="utf-8"))
        blp_level = str(meta.get(BLP_KEY, default_blp))
        note_role = role or meta.get(ROLE_KEY)

        request = {
            "idempotency_key": self.idempotency_key(path, body),
            "content": body,
            "sources": _as_list(meta.get("sources") or meta.get("source")),
            "parents": _as_list(meta.get("parents")),
            "blp_level": blp_level,
        }
        if note_role:
            request["role"] = note_role
        claim = self.contract.record_claim(request)
        if "error" in claim:
            return claim  # fail closed: not recorded -> not gated -> not publishable

        verify_req = {"claim_id": claim["claim_id"]}
        if note_role:
            verify_req["role"] = note_role
        verdict = self.contract.verify_claim(verify_req, clearance=clearance or blp_level)
        if "error" in verdict:
            return verdict

        if write_back:
            meta[PROV_KEY] = claim["claim_id"]
            meta[STATUS_KEY] = verdict["verdict"]
            meta["gate_confidence"] = verdict["confidence"]
            if verdict.get("held_reason"):
                meta["gate_held_reason"] = verdict["held_reason"]
            elif "gate_held_reason" in meta:
                del meta["gate_held_reason"]
            meta["gate_reasons"] = verdict["reasons"]
            path.write_text(frontmatter.serialize(meta, body), encoding="utf-8")
        return verdict

    def is_publishable(self, path: "str | Path") -> bool:
        """True only if the note's stamped gate_status is 'accepted' (fail-closed)."""
        meta, _ = frontmatter.parse(Path(path).read_text(encoding="utf-8"))
        return meta.get(STATUS_KEY) == "accepted"

    def publish_if_accepted(self, path: "str | Path", publish):
        """Run ``publish(path)`` only when the note is accepted; else return the
        blocking verdict status. The single choke point for 'never publish without
        accepted'."""
        if self.is_publishable(path):
            return {"published": True, "result": publish(Path(path))}
        meta, _ = frontmatter.parse(Path(path).read_text(encoding="utf-8"))
        return {"published": False, "gate_status": meta.get(STATUS_KEY),
                "held_reason": meta.get("gate_held_reason")}
