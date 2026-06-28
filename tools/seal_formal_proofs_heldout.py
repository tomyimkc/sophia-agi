# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Seal the held-out formal-proofs evidence split (miniF2F-v2 test) → public hashes only.

Thin formal-proofs analogue of ``tools/seal_math_code_heldout.py`` (design:
``docs/06-Roadmap/Formal-Proofs-Eval-Design.md`` §2; registered in
``agi-proof/formal-proofs-curriculum/preregistration.json``).

The leakage firewall for Phase 1. The miniF2F-v2c ``test`` (244) theorem STATEMENTS
(no proofs) are the evidence split; they must be fixed to a hash manifest BEFORE any
proposer runs, and the proposer must be unable to read them. Because the statements
come from an external public repo (the miniF2F-Lean Revisited authors' data release,
``roozbeh-mohit/miniF2F_v2``) we keep the payload **out of git**: the committed
manifest carries SHA-256 hashes + ids + the pinned source commit only; the full
statements live under gitignored
``private/formal-proofs-heldout/``.

Source layout (gitignored):
    private/formal-proofs-heldout/
        source.json            # {"repo","commit","split","dataset","paper","license"}
        minif2f-v2-test.jsonl  # one obj/line; statement only, e.g.
                               # {"claim_id": "...", "proposition": "...",
                               #  "lean_statement": "theorem ... : ... := by"}

Usage:
    # Seal (writes the committed hash manifest + refreshes the private manifest):
    python tools/seal_formal_proofs_heldout.py
    # Verify the committed manifest still matches the private source:
    python tools/seal_formal_proofs_heldout.py --check
    # Seal from an explicit source dir (used by tests with a fixture):
    python tools/seal_formal_proofs_heldout.py --source <dir> [--out <manifest>]

Exit codes: 0 = OK; 1 = stale/missing/not-yet-sealed (e.g. source absent → the
preregistration openChecklist seal step has not been done).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MANIFEST_OUT = ROOT / "agi-proof" / "formal-proofs-curriculum" / "heldout-seal.manifest.json"
PRIVATE_DIR = ROOT / "private" / "formal-proofs-heldout"
SOURCE_FILE = "source.json"            # provenance sidecar inside the private dir
SPLIT_FILE = "minif2f-v2-test.jsonl"   # the sealed evidence statements (no proofs)

SCHEMA = "sophia.formal_proofs_heldout_seal.v1"

# The formal-proofs analogue of the pretraining-contamination caveat. miniF2F is public
# (the miniF2F-Lean Revisited paper, arXiv 2511.03108, exists because of it), so a pass
# rate is suggestive, NOT contamination-free. The clean external path is the (empty)
# third-party-authored pack.
PRETRAINING_CAVEAT = (
    "miniF2F is a public benchmark; the proposer model may have seen its proofs during "
    "pretraining (the miniF2F-Lean Revisited paper documents this). A pass rate on this "
    "split is suggestive, NOT contamination-free proof. Controls below reduce and make "
    "the residual auditable; a clean external claim requires the third-party-authored "
    "pack (agi-proof/third-party-heldout/)."
)

# The three machine-checkable leakage controls (design §2 / preregistration.leakageControls).
LEAKAGE_CONTROLS = {
    "1_knowledgeCutoff": "Proposer model cutoff recorded in the preregistration, dated "
                         "relative to the miniF2F-v2 revision pinned by source.commit.",
    "2_sealedSplit": "These statements are hashed here before any proposer runs; the "
                     "proposer's data paths are guarded by tools/heldout_seal_guard.py.",
    "3_noExactLibraryLemma": "A proof closing by citing the target theorem's own "
                             "library declaration is plagiarism, not evidence; the reward "
                             "function rejects it. (Enforced at eval time, not here.)",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _item_digest(row: dict) -> str:
    return _sha256_text(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def _item_id(row: dict, idx: int) -> str:
    for key in ("claim_id", "id", "name", "full_name"):
        if row.get(key):
            return str(row[key])
    return str(idx)


def build_manifest(source_dir: Path) -> dict:
    """Build the (hashes-only) manifest from a private source dir. Raises if absent."""
    split_path = source_dir / SPLIT_FILE
    if not split_path.exists():
        raise FileNotFoundError(
            f"held-out split not found: {split_path}. The seal step of the "
            f"preregistration openChecklist has not been done — extract the miniF2F-v2 "
            f"test statements to {source_dir}/ first (see this file's docstring)."
        )
    source_meta_path = source_dir / SOURCE_FILE
    source_meta = (
        json.loads(source_meta_path.read_text(encoding="utf-8"))
        if source_meta_path.exists()
        else {"repo": "PIN-BEFORE-OPEN", "commit": "PIN-BEFORE-OPEN", "split": "test"}
    )

    raw = split_path.read_bytes()
    items = _load_jsonl(split_path)
    sealed_at = datetime.now(timezone.utc).isoformat()
    return {
        "schema": SCHEMA,
        "packId": "formal-proofs-curriculum-heldout",
        "sealedAt": sealed_at,
        "visibility": "public-hash-only",
        "privateCopy": "private/formal-proofs-heldout/ (gitignored)",
        "source": {
            "benchmark": "miniF2F-v2",
            "repo": source_meta.get("repo", "PIN-BEFORE-OPEN"),
            "commit": source_meta.get("commit", "PIN-BEFORE-OPEN"),
            "split": source_meta.get("split", "test"),
            "dataset": source_meta.get("dataset", "datasets/miniF2F_v2c.jsonl"),
            "paper": source_meta.get("paper", "miniF2F-Lean Revisited (arXiv 2511.03108)"),
            "license": source_meta.get("license", "Lean statements Apache-2.0; Metamath MIT"),
            **(
                {"legacyRepoAlias": source_meta["legacyRepoAlias"]}
                if source_meta.get("legacyRepoAlias")
                else {}
            ),
        },
        "generatorPolicy": "Proposers/generators MUST NOT read sealed paths "
                           "(tools/heldout_seal_guard.py enforces).",
        "leakageControls": LEAKAGE_CONTROLS,
        "pretrainingContaminationPolicy": PRETRAINING_CAVEAT,
        "cleanExternalClaimPath": "agi-proof/third-party-heldout/ (third-party-authored; "
                                  "currently EMPTY)",
        "files": [
            {
                "path": f"private/formal-proofs-heldout/{SPLIT_FILE}",
                "sha256": _sha256_bytes(raw),
                "itemCount": len(items),
                "items": [
                    {"id": _item_id(it, i), "sha256": _item_digest(it)}
                    for i, it in enumerate(items)
                ],
            }
        ],
    }


def _copy_private(manifest: dict, source_dir: Path) -> None:
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    src = source_dir / SPLIT_FILE
    if src.resolve() != (PRIVATE_DIR / SPLIT_FILE).resolve():
        shutil.copy2(src, PRIVATE_DIR / SPLIT_FILE)
    (PRIVATE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def check_manifest(source_dir: Path = PRIVATE_DIR, manifest_out: Path = MANIFEST_OUT) -> int:
    if not manifest_out.exists():
        print(f"NOT SEALED: manifest missing ({manifest_out}). Run "
              f"tools/seal_formal_proofs_heldout.py once the split is staged.", file=sys.stderr)
        return 1
    try:
        fresh = build_manifest(source_dir)
    except FileNotFoundError as exc:
        print(f"NOT SEALED: {exc}", file=sys.stderr)
        return 1
    on_disk = json.loads(manifest_out.read_text(encoding="utf-8"))
    # Compare the content that matters for integrity (hashes + ids + source), not sealedAt.
    if (on_disk.get("files") != fresh["files"]) or (on_disk.get("source") != fresh["source"]):
        print("held-out seal manifest is STALE — re-run tools/seal_formal_proofs_heldout.py",
              file=sys.stderr)
        return 1
    n = fresh["files"][0]["itemCount"]
    print(f"formal-proofs held-out seal OK ({n} sealed items, commit "
          f"{fresh['source']['commit']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify the committed manifest matches the private source")
    ap.add_argument("--source", type=Path, default=PRIVATE_DIR,
                    help="source dir holding the split + source.json (default: private dir)")
    ap.add_argument("--out", type=Path, default=MANIFEST_OUT,
                    help="manifest output path (default: the committed manifest)")
    args = ap.parse_args(argv)

    if args.check:
        return check_manifest(args.source, args.out)

    try:
        manifest = build_manifest(args.source)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    refresh_private = args.out.resolve() == MANIFEST_OUT.resolve()
    if refresh_private:
        _copy_private(manifest, args.source)
    n = manifest["files"][0]["itemCount"]
    print(f"wrote {args.out} ({n} sealed items, commit {manifest['source']['commit']})")
    if refresh_private:
        print(f"private payload under {PRIVATE_DIR} (gitignored)")
    else:
        print("private payload not refreshed (--out is not the committed manifest path)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
