# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Load, seal, and verify the HK bilingual advisor held-out benchmark (Phase 0)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "data" / "hk_advisor_benchmark"
HELDOUT = BENCH_DIR / "heldout_v1.jsonl"
MANIFEST = BENCH_DIR / "manifest.json"


def load_cases(path: Path | None = None) -> list[dict]:
    p = path or HELDOUT
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def content_hash(cases: list[dict]) -> str:
    payload = [{"id": c["id"], "prompt": c["prompt"]} for c in cases]
    payload.sort(key=lambda x: x["id"])
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def balance_counts(cases: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {
        "answerable": 0,
        "abstain": 0,
        "traps": 0,
        "total": len(cases),
        "trap_fabrication_bait": 0,
        "trap_fake_citation": 0,
        "trap_unanswerable": 0,
    }
    for c in cases:
        trap = c.get("trap", "none")
        decision = c.get("label", {}).get("decision", "")
        if trap in ("fabrication_bait", "fake_citation", "unanswerable"):
            out["traps"] += 1
            out[f"trap_{trap}"] += 1
        elif decision == "answerable":
            out["answerable"] += 1
        elif decision == "abstain":
            out["abstain"] += 1
        else:
            out["abstain"] += 1
    return out


def bilingual_split(cases: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {"yue": 0, "en": 0}
    for c in cases:
        lang = c.get("language", "en")
        if lang in out:
            out[lang] += 1
    return out


def verify_manifest(*, root: Path = ROOT) -> dict:
    manifest_path = root / "data" / "hk_advisor_benchmark" / "manifest.json"
    heldout_path = root / "data" / "hk_advisor_benchmark" / "heldout_v1.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = load_cases(heldout_path)
    computed = content_hash(cases)
    balance = balance_counts(cases)
    bilingual = bilingual_split(cases)
    ok = (
        manifest.get("contentHash") == computed
        and balance["total"] == manifest.get("nCases", 0)
        and balance["answerable"] == 30
        and balance["abstain"] == 30
        and balance["traps"] == 30
        and bilingual.get("yue") == 45
        and bilingual.get("en") == 45
        and manifest.get("candidateOnly") is True
        and manifest.get("canClaimAGI") is False
        and manifest.get("sealed") is True
    )
    return {
        "ok": ok,
        "contentHash": computed,
        "manifestHash": manifest.get("contentHash"),
        "balance": balance,
        "bilingualSplit": bilingual,
        "nCases": len(cases),
    }
