#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Provenance-of-provenance — stamp a claim receipt with the fingerprint of the
instrument that certified it, so a later change to the instrument re-opens the claim.

A GO/NO-GO receipt (``tools/claim_gate.py``) is only trustworthy if the *code and
spec that produced it* are exactly the ones on disk now. If someone edits
``claim_gate.py``, ``eval_stats.py``, or the ``measurement_spec.json`` after the
receipt was written, the receipt is stale: it certifies a claim under an instrument
that no longer exists. This module makes that detectable.

``gate_fingerprint(paths)`` is a SHA-256 over the *exact bytes* of the certifying
tool sources + spec (canonicalised: each file hashed with its relative name, then
the per-file digests folded in sorted order — so the fingerprint is stable across
machines and independent of argument order). ``stamp_receipt`` writes a
``gateProvenance`` block into the receipt; ``verify_receipt`` recomputes the
fingerprint and returns FRESH if it matches, STALE if any certifying file changed.

This mirrors ``okf.forgetting_audit``'s hash-chain discipline (a single bit-flip is
detectable by re-hashing) applied to the *provenance of the gate itself*.

    python3 tools/gate_provenance.py --stamp   RECEIPT.json --files a.py b.py spec.json
    python3 tools/gate_provenance.py --verify  RECEIPT.json

NO-WALLCLOCK RULE: pure functions never read the system clock. ``stamp_receipt``
takes ``stamped_at`` as an argument (default ``None``); the CLI is the only place a
timestamp may be injected, and even there it defaults to ``None`` unless
``--stamped-at`` is passed — so tests are deterministic.

Exit codes: 0 = FRESH (fingerprint matches / stamp succeeded),
1 = STALE (certifying files changed since stamping), 2 = unreadable/missing input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The default certifying set: the gate code, the statistics library it leans on, and
# the pre-registered measurement spec. Callers may override via --files.
DEFAULT_CERTIFIERS = [
    ROOT / "tools" / "claim_gate.py",
    ROOT / "tools" / "eval_stats.py",
    ROOT / "agi-proof" / "benchmark-results" / "wisdom-market" / "measurement_spec.json",
]

FINGERPRINT_VERSION = "gate-provenance/1"


def _rel(path: Path) -> str:
    """Path label used inside the fingerprint: repo-relative if possible, else basename.

    Keeping the label stable (not an absolute path) makes the fingerprint identical
    across checkouts/machines — only file *content* and *identity* move it."""
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return p.name


def _file_digest(path: Path) -> str:
    """SHA-256 of one file's exact bytes. Raises if the file is unreadable — the
    fingerprint must never silently skip a certifier (fail-closed)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fold(lines: "list[str]") -> str:
    """Fold per-file ``"<rel-name>\\n<sha256>"`` lines into one outer SHA-256.

    Lines are sorted (argument order is irrelevant) and separated by a NUL, with a
    leading version tag. This is the single source of truth for the folding algorithm,
    shared by ``gate_fingerprint`` (stamp side) and ``verify_receipt`` (verify side)."""
    h = hashlib.sha256()
    h.update(FINGERPRINT_VERSION.encode("utf-8"))
    for line in sorted(lines):
        h.update(b"\x00")
        h.update(line.encode("utf-8"))
    return h.hexdigest()


def _locate(entry: "dict") -> "Path | None":
    """Find the on-disk bytes for a stamped manifest entry.

    Prefers the recorded absolute path (handles certifiers outside the repo), then
    falls back to ``ROOT/<rel>`` for in-tree files that moved with the checkout.
    Returns None if neither exists (fail-closed: caller treats as missing certifier)."""
    ab = entry.get("abspath")
    if ab and Path(ab).is_file():
        return Path(ab)
    rel = entry.get("file")
    cand = (ROOT / rel) if rel else None
    if cand is not None and cand.is_file():
        return cand
    return None


def gate_fingerprint(paths: "list[str | Path]") -> str:
    """A single SHA-256 over the exact certifying tool sources + spec.

    Each file contributes ``"<rel-name>\\n<sha256-of-bytes>"``; the per-file lines are
    sorted (so argument order is irrelevant) and folded into one outer SHA-256 with a
    version tag. A change to any byte of any file — or adding/removing a certifier —
    changes the fingerprint. Missing/unreadable files raise (fail-closed)."""
    if not paths:
        raise ValueError("gate_fingerprint requires at least one certifying file")
    return _fold([f"{_rel(p)}\n{_file_digest(p)}" for p in paths])


def fingerprint_manifest(paths: "list[str | Path]") -> "dict":
    """The list of {file, sha256, abspath} that a fingerprint is built from.

    Recorded in the stamp so a STALE result can name *which* certifier drifted. The
    ``file`` label (repo-relative name) is what the portable ``gateHash`` is built
    from; ``abspath`` is a locate-hint for verification of files that live *outside*
    the repo (temp files, an out-of-tree spec) — verify prefers ``abspath`` and falls
    back to ``ROOT / file`` so an in-tree certifier still resolves after a checkout
    move."""
    return {"version": FINGERPRINT_VERSION,
            "files": sorted(({"file": _rel(p), "sha256": _file_digest(p),
                              "abspath": str(Path(p).resolve())} for p in paths),
                            key=lambda r: r["file"])}


def stamp_receipt(receipt_path: "str | Path",
                  paths: "list[str | Path] | None" = None,
                  *, stamped_at: "str | None" = None,
                  write: bool = True) -> "dict":
    """Add a ``gateProvenance`` block to a receipt and (optionally) write it back.

    The block records the fingerprint, the per-file manifest, and ``stampedAt`` —
    which is whatever the caller injected (``None`` by default: no wallclock is read
    here). Returns the updated receipt dict. Raises on unreadable receipt or files."""
    receipt_path = Path(receipt_path)
    certifiers = list(paths) if paths else list(DEFAULT_CERTIFIERS)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("receipt is not a JSON object")
    manifest = fingerprint_manifest(certifiers)
    receipt["gateProvenance"] = {
        "gateHash": gate_fingerprint(certifiers),
        "stampedFiles": manifest["files"],
        "fingerprintVersion": FINGERPRINT_VERSION,
        "stampedAt": stamped_at,   # injected; None in tests — never a wallclock read
    }
    if write:
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    return receipt


def verify_receipt(receipt_path: "str | Path") -> "dict":
    """Recompute the gate fingerprint and compare it to the stamped one.

    Returns a verdict dict with ``status`` in {FRESH, STALE, UNSTAMPED} and, on STALE,
    the exact files whose hash drifted. FRESH means every certifier is byte-identical
    to when the receipt was stamped; STALE means CI must re-open the claim. A receipt
    with no ``gateProvenance`` block is UNSTAMPED (treated as not-fresh -> exit 1)."""
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    prov = (receipt or {}).get("gateProvenance") if isinstance(receipt, dict) else None
    if not prov:
        return {"status": "UNSTAMPED", "fresh": False,
                "detail": "receipt carries no gateProvenance block — nothing to verify"}

    stamped_files = prov.get("stampedFiles") or []
    stamped_hash = prov.get("gateHash")
    # Recompute per-file digests for the SAME set of files the stamp recorded. Locate
    # each by its recorded abspath (handles out-of-tree certifiers), falling back to
    # ROOT/<rel> for in-tree files that moved with the checkout — so we detect content
    # drift regardless of where the certifier lives.
    drifted: list[dict] = []
    missing: list[str] = []
    located_lines: list[str] = []   # rel-name + current-digest, to recompute the outer hash
    for entry in stamped_files:
        rel = entry.get("file")
        was = entry.get("sha256")
        located = _locate(entry)
        if located is None:
            missing.append(rel)
            continue
        now = _file_digest(located)
        located_lines.append(f"{rel}\n{now}")
        if now != was:
            drifted.append({"file": rel, "stamped": was, "now": now})

    if missing:
        # A vanished certifier is fail-closed STALE (we cannot prove the instrument
        # is unchanged if part of it is gone).
        return {"status": "STALE", "fresh": False, "reason": "missing_certifier",
                "missing": missing, "drifted": [d["file"] for d in drifted],
                "stampedHash": stamped_hash,
                "detail": f"{len(missing)} certifier(s) missing; cannot verify instrument"}

    recomputed_hash = _fold(located_lines) if located_lines else None
    fresh = bool(stamped_hash) and recomputed_hash == stamped_hash and not drifted
    return {
        "status": "FRESH" if fresh else "STALE",
        "fresh": fresh,
        "stampedHash": stamped_hash,
        "recomputedHash": recomputed_hash,
        "drifted": [d["file"] for d in drifted],
        "drift": drifted,
        "detail": ("instrument unchanged since stamping" if fresh
                   else f"{len(drifted)} certifier(s) changed since stamping — re-open claim"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stamp", metavar="RECEIPT", type=Path,
                      help="stamp a receipt with the current gate fingerprint")
    mode.add_argument("--verify", metavar="RECEIPT", type=Path,
                      help="verify a stamped receipt against the current certifying files")
    ap.add_argument("--files", nargs="*", type=Path, default=None,
                    help="certifying files (default: claim_gate.py + eval_stats.py + measurement_spec.json)")
    ap.add_argument("--stamped-at", default=None,
                    help="OPTIONAL injected ISO timestamp for the stamp (no wallclock is read otherwise)")
    args = ap.parse_args()

    if args.stamp is not None:
        try:
            receipt = stamp_receipt(args.stamp, args.files, stamped_at=args.stamped_at)
        except FileNotFoundError as e:
            print(json.dumps({"status": "ERROR", "reason": "unreadable", "detail": str(e), "code": 2}))
            print(f"gate-provenance STAMP: unreadable input: {e}", file=sys.stderr)
            return 2
        except Exception as e:
            print(json.dumps({"status": "ERROR", "reason": "bad_input", "detail": str(e), "code": 2}))
            print(f"gate-provenance STAMP: bad input: {e}", file=sys.stderr)
            return 2
        prov = receipt["gateProvenance"]
        print(f"gate-provenance STAMP {args.stamp}: gateHash={prov['gateHash'][:12]}… "
              f"over {len(prov['stampedFiles'])} certifier(s)", file=sys.stderr)
        print(json.dumps({"status": "STAMPED", "gateHash": prov["gateHash"],
                          "stampedFiles": [f["file"] for f in prov["stampedFiles"]],
                          "stampedAt": prov["stampedAt"], "code": 0}))
        return 0

    # verify
    try:
        result = verify_receipt(args.verify)
    except FileNotFoundError as e:
        print(json.dumps({"status": "ERROR", "reason": "unreadable", "detail": str(e), "code": 2}))
        print(f"gate-provenance VERIFY: unreadable receipt: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(json.dumps({"status": "ERROR", "reason": "bad_input", "detail": str(e), "code": 2}))
        print(f"gate-provenance VERIFY: bad input: {e}", file=sys.stderr)
        return 2
    result["code"] = 0 if result["fresh"] else 1
    print(f"gate-provenance VERIFY {args.verify}: {result['status']} — {result['detail']}",
          file=sys.stderr)
    print(json.dumps({"status": result["status"], "fresh": result["fresh"],
                      "drifted": result.get("drifted", []), "code": result["code"]}))
    return int(result["code"])


if __name__ == "__main__":
    raise SystemExit(main())
