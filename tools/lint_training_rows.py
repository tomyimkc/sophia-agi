#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Habit-not-fact linter for the source-discipline training packs.

Sophia's design contract: the weights learn a HABIT (route, qualify, refuse, keep traditions
distinct) while the external gate/tools enforce TRUTH. A training target that bakes in a bare
ground-truth fact ("Laozi wrote the Dao De Jing") instead of the habit ("that attribution is
legendary; route to the source") quietly violates that split and teaches the model to MEMORIZE
answers. This deterministic linter scans the committed discipline-family targets and fails any that
carry NEITHER a routing/epistemic structure NOR qualification vocabulary — i.e. a bare assertion.

    python3 tools/lint_training_rows.py
    python3 tools/lint_training_rows.py --max-flag 30
Exit 0 = every discipline-family target teaches a habit; 1 = bare-fact target(s) found.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRAIN = ["training/local_sophia_v3/mlx/train.jsonl", "training/local_sophia_v3/mlx/valid.jsonl"]
# Verifier-gated distillation packs. Every row here is teacher-generated, so it MUST carry a
# verification_provenance tag (HOW it was verified) — the contract that lets the builder
# oversample hard-won rows and keeps "small but high-signal" auditable. Absent files are skipped.
DISTILL_PACKS = ["training/distill_sft.jsonl"]
# Must mirror tools/distill_export.PROV_VALUES (kept local so this linter stays pure-stdlib).
PROV_VALUES = {"passed_first_try", "patched_after_failure", "self_consistent"}
# Families whose targets MUST teach the source-discipline / moral-routing HABIT (not a bare fact).
# general_retention, tool_mcp, hk_bilingual, and the multi-voice `council` (direct reasoned analysis)
# legitimately give direct answers, so they are exempt.
DISCIPLINE_FAMILIES = {"source_discipline", "moral_gate"}

# Structured-habit keys (the target opens with a route/epistemic header) — strongest signal.
STRUCT_KEYS = ("route", "epistemic_status", "risk_flags", "needed_sources", "confidence",
               "verdict", "reason")
# Qualification / routing / refusal / distinction / debunking vocabulary — the prose forms the
# source-discipline habit takes. A target with NONE of these AND no struct header is a bare fact.
HABIT_MARKERS = re.compile(
    r"(\battribut|\btradition|\bcontest|\bdisput|\blegendary|\bpseudonym|\banonymous|\bcomposite|"
    r"\bcompiled|\buncertain|cannot (confirm|verify|attribute)|do not (over|attribute|merge|conflate|"
    r"equate)|\broute\b|needed_sources|source discipline|provenance|scholars|consensus|qualif|"
    r"epistemic|may reflect|\bmyth\b|misconception|mix-?up|oversimplif|propaganda|"
    r"\bno single\b|not (settled|definitive|reliabl|accurat|establish|typical|true that)|"
    r"did not (write|author|compose|pen)|was (written|composed|authored) by|distinct from|"
    r"differ(s|ent)? (from|between)|separate(s|d)? (the|from|when)|split (when|the|appropriate)|"
    r"both are|^no[,. ]|據傳|傳統|存疑|無法確認|來源|並非|誤解)", re.I)


def _target(row: dict) -> str:
    for m in row.get("messages", []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m.get("content") or ""
    return ""


def _has_struct(text: str) -> bool:
    head = text.lstrip()[:1]
    if head != "{":
        return False
    try:
        obj = json.loads(text[: text.index("}") + 1]) if "}" in text else {}
    except Exception:
        return False
    return any(k in obj for k in STRUCT_KEYS)


def _check_distill_provenance(max_flag: int) -> list[tuple[str, str]]:
    """Flag any verifier-gated distillation row missing a valid verification_provenance tag.

    Only fires on rows whose metadata.source marks them as distilled, so existing
    hand-curated packs are unaffected; absent distill packs are simply skipped (no-op in CI
    until a distillation run writes one).
    """
    bad: list[tuple[str, str]] = []
    for rel in DISTILL_PACKS:
        p = ROOT / rel
        if not p.exists():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            meta = row.get("metadata") or {}
            if "distill" not in str(meta.get("source", "")).lower():
                continue
            prov = meta.get("verification_provenance")
            if prov not in PROV_VALUES:
                user = next((m.get("content", "") for m in row.get("messages", [])
                             if m.get("role") == "user"), "")
                bad.append((rel, f"prov={prov!r} user«{user[:60]}»"))
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-flag", type=int, default=25)
    args = ap.parse_args()

    flagged, n_checked = [], 0
    for rel in TRAIN:
        p = ROOT / rel
        if not p.exists():
            continue
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            row = json.loads(ln)
            fam = (row.get("metadata") or {}).get("task_family")
            if fam not in DISCIPLINE_FAMILIES:
                continue
            n_checked += 1
            tgt = _target(row)
            if _has_struct(tgt) or HABIT_MARKERS.search(tgt):
                continue
            user = next((m.get("content", "") for m in row.get("messages", [])
                         if m.get("role") == "user"), "")
            flagged.append((fam, user[:70], tgt[:90]))

    prov_bad = _check_distill_provenance(args.max_flag)

    print(f"TRAINING-ROW HABIT LINT: checked {n_checked} discipline-family targets, "
          f"{len(flagged)} bare-fact (no route/qualification); "
          f"distill-provenance violations={len(prov_bad)}.")
    for fam, u, t in flagged[: args.max_flag]:
        print(f"  BARE [{fam}] user«{u}» -> target«{t}»")
    for rel, why in prov_bad[: args.max_flag]:
        print(f"  NO-PROVENANCE [{rel}] {why}")
    if flagged:
        print("FAIL — a discipline target must teach a HABIT (route/qualify/refuse), not a bare fact.")
        return 1
    if prov_bad:
        print("FAIL — a distilled row must carry a valid verification_provenance tag "
              f"(one of {sorted(PROV_VALUES)}).")
        return 1
    print("OK — every discipline-family target encodes the source-discipline habit; "
          "distilled rows are provenance-tagged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
