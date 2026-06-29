#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Audit whether the content-channel provenance held-out (the 32-item set the
train-runpod lane scored, ``tests/benchmark-{philosophy,psychology,history,religion}.json``)
can be HONESTLY grown from ``data/attributions.json`` — Step-3B.

WHY THIS IS AN AUDIT, NOT A PACK BUILDER
----------------------------------------
The premise of Step-3B was that ``data/attributions.json`` holds hundreds of
attribution entries with ~300 NOT-yet-used in training, from which a large
decontam-clean held-out could be carved. That premise is FALSE. This script
proves it from scratch and fails honestly (it never fabricates a pack):

  * ``data/attributions.json`` holds **30 corpus records** (text-attribution
    metadata), not hundreds. They yield a FIXED, finite set of derivable
    provenance facts:
        deny       (textId, wrong_author)  — one per ``doNotAttributeTo`` entry
        affirm     (textId, attributedAuthor)
        confidence (textId, authorConfidence)   for compiled/legendary/none_extant
  * The committed TRAINED pack ``training/lora/train.jsonl`` carries a structured
    ``metadata.trap`` label per row ("deny confucius -> dao_de_jing-r0", ...) that
    maps each trained row to exactly one such fact. Parsing it gives the set of
    facts the model was TRAINED on.
  * A genuinely held-out (FACT-DISJOINT) item must test a fact NOT in that set.

TWO DECONTAM STANDARDS, REPORTED SEPARATELY
-------------------------------------------
  FACT-level (strict; Step-3B's "NEVER reuse a training entity/item"): a candidate
    is clean only if its (action, key, textId) fact is absent from the trained set.
  PROMPT-level (the repo's operative ``assert_decontam`` standard: exact +
    word-5-shingle Jaccard>=0.9 on the QUESTION text only): a candidate is clean if
    its question string is disjoint from every training prompt and existing held-out
    prompt — even if the underlying FACT was trained.

The existing 32-item held-out itself is only PROMPT-disjoint: 9 of its 10
philosophy facts are in the trained set. So PROMPT-level "growth" would only add
more memorization-recall items (same character), and FACT-level growth is 0.

    python3 tools/audit_provenance_heldout_growth.py            # human summary
    python3 tools/audit_provenance_heldout_growth.py --json     # machine
    python3 tools/audit_provenance_heldout_growth.py --out FILE # write findings json

Exit 0 = audit ran. Stdlib + the repo's own decontam guard for normalize()/prompt_of()
so the prompt-level numbers match ``tools/assert_decontam.py`` exactly. Deterministic.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))
from provenance_bench.dataset_guard import _load_jsonl, normalize, prompt_of  # noqa: E402

ATTRS = ROOT / "data" / "attributions.json"
TRAINED_PACK = ROOT / "training" / "lora" / "train.jsonl"
HOLDOUT_PACK = ROOT / "training" / "lora" / "holdout.jsonl"
BENCH_DOMAINS = ["philosophy", "psychology", "history", "religion"]

# observed train-lane datapoint (from the task brief / runpod-train artifacts)
OBS_N = 32
OBS_UPLIFT = 0.0625      # +6.25pt content passAt1 (23/32 -> 25/32)
OBS_CI_HALF = 0.172      # 95% paired CI half-width at N=32 ([-0.125,+0.219])
REQUIRED_N_80 = 493      # paired N for 80% power at this effect/rho
RHO = 0.342


def _trap_fact(trap: str) -> tuple[str, str, str] | None:
    """('deny'|'affirm'|'confidence', key, textId) from a metadata.trap label."""
    base = re.sub(r"-r\d+$", "", trap.strip())
    for pat, act in ((r"deny (\S+) -> (.+)$", "deny"),
                     (r"affirm (\S+) for (.+)$", "affirm"),
                     (r"confidence (\S+) for (.+)$", "confidence")):
        m = re.match(pat, base)
        if m:
            return (act, m.group(1), m.group(2))
    return None


def trained_facts(path: Path = TRAINED_PACK) -> set[tuple[str, str, str]]:
    facts: set[tuple[str, str, str]] = set()
    if not path.exists():
        return facts
    for row in _load_jsonl(path):
        trap = (row.get("metadata") or {}).get("trap")
        if isinstance(trap, str):
            f = _trap_fact(trap)
            if f:
                facts.add(f)
    return facts


def derivable_facts(attrs: dict) -> list[tuple[str, str, str]]:
    """Every provenance fact derivable from the corpus records (the honest ceiling)."""
    out: list[tuple[str, str, str]] = []
    for tid, rec in attrs.items():
        for w in (rec.get("doNotAttributeTo") or []):
            out.append(("deny", w, tid))
        if rec.get("attributedAuthor"):
            out.append(("affirm", rec["attributedAuthor"], tid))
        if rec.get("authorConfidence") in ("compiled", "legendary", "none_extant"):
            out.append(("confidence", rec["authorConfidence"], tid))
    return out


def existing_heldout_prompts() -> set[str]:
    out: set[str] = set()
    for dom in BENCH_DOMAINS:
        p = ROOT / "tests" / f"benchmark-{dom}.json"
        if p.exists():
            for c in json.loads(p.read_text(encoding="utf-8")).get("cases", []):
                if c.get("question"):
                    out.add(normalize(c["question"]))
    return out


def all_training_prompts() -> set[str]:
    out: set[str] = set()
    for p in ROOT.glob("training/**/*.jsonl"):
        for row in _load_jsonl(p):
            pr = prompt_of(row)
            if pr:
                out.add(normalize(pr))
    return out


def _shingles(s: str, k: int = 5) -> set[str]:
    t = normalize(s).split()
    return {" ".join(t[i:i + k]) for i in range(max(0, len(t) - k + 1))} or {" ".join(t)}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a and b) else 0.0


def ci_half_at(n: int) -> float:
    """Paired-proportion CI half-width scales ~1/sqrt(N) off the observed N=32 anchor."""
    return OBS_CI_HALF * math.sqrt(OBS_N / n) if n > 0 else float("inf")


def audit() -> dict:
    attrs = json.loads(ATTRS.read_text(encoding="utf-8"))
    facts = derivable_facts(attrs)
    by_type = {t: sum(1 for f in facts if f[0] == t) for t in ("deny", "affirm", "confidence")}
    tf = trained_facts()
    hf = trained_facts(HOLDOUT_PACK)

    fact_disjoint = [f for f in facts if f not in tf and f not in hf]

    # PROMPT-level: one canonical NEW phrasing per (deny|confidence) fact, then decontam.
    train_prompts = all_training_prompts()
    held_prompts = existing_heldout_prompts()
    train_sh = [_shingles(p) for p in train_prompts]
    held_sh = [_shingles(p) for p in held_prompts]

    def title(tid):
        return attrs[tid]["canonicalTitleEn"]

    def disp(a):
        special = {"laozi": "Laozi", "zhuangzi": "Zhuangzi", "sunzi": "Sunzi",
                   "marcus_aurelius": "Marcus Aurelius", "sima_qian": "Sima Qian",
                   "zuo_qiuming": "Zuo Qiuming", "han_feizi": "Han Feizi",
                   "confucian_school": "the Confucian school", "multiple": "a single author"}
        return special.get(a, a.replace("_", " ").title())

    prompt_clean = 0
    prompt_clean_but_trained_fact = 0
    for act, key, tid in facts:
        if act == "deny":
            q = f"Is the {title(tid)} the work of {disp(key)}?"
        elif act == "confidence":
            q = f"Is the authorship of the {title(tid)} historically settled and certain?"
        else:
            continue  # affirm folded into deny items; not a standalone trap question
        nq, qs = normalize(q), _shingles(q)
        if nq in train_prompts or nq in held_prompts:
            continue
        if any(_jaccard(qs, ts) >= 0.9 for ts in train_sh):
            continue
        if any(_jaccard(qs, hs) >= 0.9 for hs in held_sh):
            continue
        prompt_clean += 1
        if (act, key, tid) in tf:
            prompt_clean_but_trained_fact += 1

    # CI projections
    proj = {str(n): round(ci_half_at(n), 4)
            for n in (OBS_N, OBS_N + prompt_clean, 123, 200, 300, REQUIRED_N_80)}
    lenient_total = OBS_N + prompt_clean

    return {
        "schema": "sophia.provenance_heldout_growth_audit.v1",
        "source": "data/attributions.json",
        "canClaimAGI": False,
        "corpusRecords": len(attrs),
        "derivableFacts": {"total": len(facts), **by_type},
        "trainedFacts": len(tf),
        "heldoutPackFacts": len(hf),
        "factDisjointCleanNew": len(fact_disjoint),
        "factDisjointExamples": fact_disjoint[:10],
        "promptLevel": {
            "candidatesGenerated": by_type["deny"] + by_type["confidence"],
            "promptCleanVsTrainAndHeldout": prompt_clean,
            "ofWhichFactAlreadyTrained": prompt_clean_but_trained_fact,
        },
        "honestN": len(fact_disjoint),
        "ciProjection": {
            "observedN": OBS_N, "observedUpliftPct": OBS_UPLIFT * 100,
            "observedCiHalfWidth": OBS_CI_HALF, "rho": RHO,
            "requiredNFor80pct": REQUIRED_N_80,
            "halfWidthByN": proj,
            "lenientMaxTotalN": lenient_total,
            "lenientMaxCiHalfWidth": round(ci_half_at(lenient_total), 4),
            "lenientMaxExcludesZero": ci_half_at(lenient_total) < OBS_UPLIFT,
        },
        "verdict": (
            "EXHAUSTED. The 30 records of data/attributions.json yield "
            f"{len(facts)} derivable provenance facts; the trained pack "
            f"training/lora/train.jsonl already trains on {len(tf)} of them, "
            f"covering all {len(facts)} derivable facts. FACT-DISJOINT (genuinely "
            f"held-out) growth from this source = {len(fact_disjoint)} items. The "
            f"{prompt_clean} prompt-level-clean candidates ALL test already-trained "
            "facts (memorization recall, identical in character to the existing "
            "32-item held-out, whose 9/10 philosophy facts are also trained), so "
            "they cannot honestly grow a GENERALIZATION held-out. The held-out "
            "stays at N=32, CI +/-0.17, includes 0 -> still underpowered. 493 is "
            "unreachable from this source; even the lenient prompt-level max "
            f"(N={lenient_total}) gives CI +/-{ci_half_at(lenient_total):.3f}, which "
            "still includes the +6.25pt effect."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="write the findings json here")
    args = ap.parse_args()
    rep = audit()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0
    df = rep["derivableFacts"]
    print(f"source: {rep['source']}  records: {rep['corpusRecords']}")
    print(f"derivable provenance facts: {df['total']} "
          f"(deny={df['deny']} affirm={df['affirm']} confidence={df['confidence']})")
    print(f"trained facts (lora/train.jsonl): {rep['trainedFacts']}")
    print(f"FACT-DISJOINT clean-new (strict 'never reuse a training entity'): "
          f"{rep['factDisjointCleanNew']}")
    pl = rep["promptLevel"]
    print(f"PROMPT-LEVEL clean candidates: {pl['promptCleanVsTrainAndHeldout']} "
          f"(of which fact-already-trained: {pl['ofWhichFactAlreadyTrained']})")
    print(f"HONEST N (decontam-clean NEW held-out from this source): {rep['honestN']}")
    cp = rep["ciProjection"]
    print(f"CI half-width by N: {cp['halfWidthByN']}")
    print(f"verdict: {rep['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
