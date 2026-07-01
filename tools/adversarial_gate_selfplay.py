#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""W4 — adversarial gate self-play curriculum (drop-in, fail-closed).

Thesis: hard negatives (training/hard_negatives_dpo.jsonl) are a FROZEN file. Make them a
self-generating flywheel: a proposer mines realistic inputs that make the base model
fabricate-and-still-pass the gate; those become fresh DPO negatives; model + gate co-train;
repeat. Reuses the repo's existing "an LLM proposes, held-out validation gates trust"
pattern (agent.verifier_proposer.make_model_proposer) and the fabrication-temptation signal
(agent.temptation.prompt_fabrication_temptation).

WHAT THIS DOES (runnable offline):
  * scores candidate adversarial prompts by the REAL prompt_fabrication_temptation;
  * runs one self-play ROUND: proposer emits candidates -> temptation filter -> (model
    generates; gate judges) -> keep only cases that are BOTH high-temptation AND slip past
    the gate (fabricate-and-pass) as new negatives;
  * enforces two anti-collapse guards the thesis names: a novelty floor (dedup by shingle
    overlap so the proposer can't repeat one exploit) and a realism floor (temptation in a
    plausible band, not degenerate);
  * writes new negatives in the hard_negatives DPO shape for the trainer.

WHAT THIS DOES NOT DO (honest seam):
  * with no model/gate backend it runs in DRY mode over provided candidates and marks
    trained:false; it does NOT fine-tune. Wiring a live model+gate and appending to the DPO
    file across rounds is the maintainer step. candidateOnly:true.

Usage:
  python3 tools/adversarial_gate_selfplay.py --candidates cand.jsonl \
      --novelty 0.6 --out new_negatives.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    from agent.temptation import prompt_fabrication_temptation
    _REPO_OK = True
    _IMPORT_ERR = ""
except Exception as e:  # pragma: no cover
    _REPO_OK = False
    _IMPORT_ERR = f"{type(e).__name__}: {e}"


def _env_artifact(reason: str) -> dict[str, Any]:
    return {"schema": "sophia.adversarial_selfplay.v1", "ok": False, "reason": reason,
            "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False}


def _shingles(text: str, n: int = 4) -> set[str]:
    toks = str(text).lower().split()
    return {" ".join(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))} or {text.lower()}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def novelty_filter(candidates: list[str], *, threshold: float) -> list[str]:
    """Keep a candidate only if it is < threshold Jaccard-similar to all kept ones.
    Prevents adversary collapse onto one repeated exploit."""
    kept: list[str] = []
    kept_sh: list[set[str]] = []
    for c in candidates:
        sh = _shingles(c)
        if all(_jaccard(sh, k) < threshold for k in kept_sh):
            kept.append(c)
            kept_sh.append(sh)
    return kept


def selfplay_round(
    candidates: list[str], *,
    model_gate: Callable[[str], dict[str, Any]] | None = None,
    novelty: float = 0.6, realism_band: tuple[float, float] = (0.2, 0.95),
) -> dict[str, Any]:
    """One round. model_gate(prompt) -> {"completion": str, "fabricated": bool,
    "passed_gate": bool}. When None -> DRY mode (temptation scoring + filters only)."""
    if not _REPO_OK:
        return _env_artifact(f"repo instruments unavailable ({_IMPORT_ERR}); run with "
                             "PYTHONPATH=. inside the sophia-agi tree")
    if not candidates:
        return _env_artifact("no candidate prompts provided (fail-closed)")

    # 1) realism: keep prompts whose fabrication-temptation is in a plausible band
    scored = [(c, float(prompt_fabrication_temptation(c))) for c in candidates]
    lo, hi = realism_band
    realistic = [c for c, t in scored if lo <= t <= hi]

    # 2) novelty: dedup so the proposer cannot repeat one exploit
    novel = novelty_filter(realistic, threshold=novelty)

    trained = model_gate is not None
    new_negatives: list[dict[str, Any]] = []
    slipped = 0
    if trained:
        for c in novel:
            res = model_gate(c)
            # a NEW negative = fabricated AND slipped past the gate (fabricate-and-pass)
            if bool(res.get("fabricated")) and bool(res.get("passed_gate")):
                slipped += 1
                new_negatives.append({
                    "prompt": c,
                    "rejected": res.get("completion", ""),   # the fabrication (DPO negative)
                    "chosen": res.get("reference_abstain", "I can't verify that."),
                    "source": "adversarial-selfplay",
                })

    return {
        "schema": "sophia.adversarial_selfplay.v1", "ok": True, "trained": trained,
        "nCandidates": len(candidates), "nRealistic": len(realistic), "nNovel": len(novel),
        "temptationScores": [round(t, 3) for _, t in scored],
        "nNewNegatives": len(new_negatives),
        "slipPastGateRate": (round(slipped / len(novel), 4) if (trained and novel) else None),
        "newNegatives": new_negatives,
        "note": ("DRY mode: no model/gate backend, so temptation+novelty+realism filters ran "
                 "but no fabricate-and-pass mining occurred (trained:false). Wire a live "
                 "model+gate and append newNegatives to the DPO file across rounds — the "
                 "maintainer seam. Novelty+realism floors are the anti-collapse guards."),
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
    }


def load_jsonl(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        out.append(obj["prompt"] if isinstance(obj, dict) else str(obj))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="W4 adversarial gate self-play")
    ap.add_argument("--candidates", required=True, help="JSONL of {prompt} or bare strings")
    ap.add_argument("--novelty", type=float, default=0.6, help="max Jaccard to keep as novel")
    ap.add_argument("--out", default=None, help="write new negatives JSONL here")
    args = ap.parse_args(argv)

    cands = load_jsonl(Path(args.candidates))
    report = selfplay_round(cands, model_gate=None, novelty=args.novelty)  # DRY (no backend)
    if args.out and report.get("ok"):
        with open(args.out, "w", encoding="utf-8") as fh:
            for neg in report["newNegatives"]:
                fh.write(json.dumps(neg) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())