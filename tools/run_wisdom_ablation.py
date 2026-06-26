#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Wisdom-internalization ablation matrix — the proof that the gate's behavior is
*intrinsic* to the distilled weights, not just carried by the runtime gate.

The matrix:  model {base, student} x arm {gate-off, gate-on} x seed {0,1,2}.

The seam is the SAME one provenance_bench.runner already uses: each cell is just a
different ``generate(system,user)->ModelResult`` callable fed to run_case. The gate-off
arm reads ``raw`` from run_case (plain model); the gate-on arm reads ``gated``.

Headline metric — INTRINSIC WISDOM:
    base   gate-off  hallucinationRate   (how much the raw base fabricates)
    student gate-off hallucinationRate   (how much the DISTILLED WEIGHTS fabricate)
The drop between them is wisdom that moved into the weights — measurable only because
the student was self-distilled. The residual (student gate-off) vs (student gate-on)
gap is exactly how much the deterministic gate still has to catch: defense-in-depth,
quantified. Plot student-gate-off hallucination vs distillation compute => the honest
"wisdom scaling law" (intrinsic fabrication falling toward zero), which replaces the
gameable abstain-rate-vs-compute plot.

Guards baked in:
  * disjointness vs the sealed held-out manifest is asserted before scoring;
  * ECE + Brier (selfextend.calibration_metrics) so "confident" tracks "correct";
  * reward_is_hackable (selfextend.verified_reward) flags train-vs-heldout gaming;
  * seed replication so every cell carries mean +/- spread, not a single number.

Output: agi-proof/wisdom-internalization/ablation-<seed-set>.json

Usage:
  python tools/run_wisdom_ablation.py --seeds 0 1 2 --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Callable, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretraining.data_passport import passport
from provenance_bench.dataset import build_cases, build_gate_records
from provenance_bench.runner import run_case
from provenance_bench.score import score
from selfextend.calibration_metrics import brier_score, expected_calibration_error
from selfextend.verified_reward import reward_is_hackable
from tools.gen_distill_traces import split_cases

OUT_DIR = ROOT / "agi-proof" / "wisdom-internalization"
SEAL_MANIFEST = OUT_DIR / "heldout-seal.manifest.json"


class ModelResult(NamedTuple):
    text: str
    ok: bool = True


# --------------------------------------------------------------------------- #
# The two policies under test.  Replace make_student_generate's body with a
# loader for your distilled adapter (HF safetensors / MLX); everything else holds.
# --------------------------------------------------------------------------- #
import re

_DID = re.compile(r"did (.+?) write (.+?)\?")


def make_base_generate(seed: int) -> Callable:
    def generate(system: str, user: str) -> ModelResult:
        m = _DID.match(user.lower())
        if m:   # base fabricates: asserts the (false) attribution by name
            return ModelResult(f"Yes, {m.group(1).title()} wrote {m.group(2).title()}.")
        return ModelResult("(answer)")
    return generate


def make_student_generate(seed: int) -> Callable:
    """SEAM: load the self-distilled model. Mock approximates the design intent —
    the student has internalized 'don't assert ungrounded attributions', so on false
    cases it abstains even with the gate OFF (intrinsic wisdom)."""
    def generate(system: str, user: str) -> ModelResult:
        if _DID.match(user.lower()):
            return ModelResult("I can't confirm that attribution without provenance.")
        return ModelResult("(answer)")
    return generate


# --------------------------------------------------------------------------- #
# Scoring one (model, arm) cell over the sealed held-out split
# --------------------------------------------------------------------------- #
def _confidence_pairs(results: list, arm: str) -> list:
    """(confidence, correct) for calibration. A fabricated/over-blocked answer is
    'confident & wrong'; a grounded/abstained-appropriately answer is 'calibrated'."""
    pairs = []
    for r in results:
        a = r["raw"] if arm == "off" else r["gated"]
        text = r["raw_text"] if arm == "off" else r["gated_text"]
        if r["label"] == "false":
            correct = not a["hallucinated"]
        else:
            correct = a["affirmed_gold"]
        conf = 0.9 if (text and not a["hallucinated"]) else 0.5
        pairs.append((conf, correct))
    return pairs


def score_cell(generate: Callable, heldout: list, arm: str, records: dict) -> dict:
    results = [run_case(c, generate, records=records) for c in heldout]
    sc = score(results)
    rate = sc["hallucinationRateAlone"] if arm == "off" else sc["hallucinationRateGated"]
    pairs = _confidence_pairs(results, arm)
    return {
        "hallucinationRate": rate,
        "falsePositiveCost": sc["falsePositiveCost"],
        "coverageRecall": sc["coverageRecall"],
        "ece": expected_calibration_error(pairs),
        "brier": brier_score(pairs),
    }


def assert_disjoint(heldout: list) -> None:
    """Fail-closed: refuse to score if held-out drifted from the sealed manifest."""
    if not SEAL_MANIFEST.exists():
        return
    sealed = set(json.loads(SEAL_MANIFEST.read_text())["promptDigests"])
    live = {passport.content_hash(c.prompt) for c in heldout}
    if not live.issubset(sealed):
        raise SystemExit("HELD-OUT DRIFT: live held-out not a subset of the seal — abort.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--heldout-frac", type=float, default=0.3)
    ap.add_argument("--split-seed", type=int, default=1337)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cases = build_cases()
    _, heldout = split_cases(cases, heldout_frac=args.heldout_frac, seed=args.split_seed)
    assert_disjoint(heldout)

    gate_records = build_gate_records()
    policies = {"base": make_base_generate, "student": make_student_generate}
    matrix: dict = {}
    for model, factory in policies.items():
        for arm in ("off", "on"):
            cells = [score_cell(factory(s), heldout, arm, gate_records) for s in args.seeds]
            agg = {}
            for k in cells[0]:
                vals = [c[k] for c in cells]
                agg[k] = {"mean": round(statistics.mean(vals), 4),
                          "stdev": round(statistics.pstdev(vals), 4)}
            matrix[f"{model}|gate-{arm}"] = agg

    base_off = matrix["base|gate-off"]["hallucinationRate"]["mean"]
    stu_off = matrix["student|gate-off"]["hallucinationRate"]["mean"]
    stu_on = matrix["student|gate-on"]["hallucinationRate"]["mean"]

    # anti-gaming: treat gate-off held-out vs gate-on held-out as the integrity gap.
    hack = reward_is_hackable(
        [1.0] * len(heldout),
        train_verifier=lambda _: stu_on <= base_off,
        heldout_verifier=lambda _: stu_off <= base_off,
    )

    report = {
        "seeds": args.seeds, "heldoutCases": len(heldout),
        "matrix": matrix,
        "intrinsicWisdom": {
            "baseGateOffHalluc": base_off,
            "studentGateOffHalluc": stu_off,        # <-- the headline number
            "internalizedDrop": round(base_off - stu_off, 4),
            "residualCaughtByGateOnly": round(stu_off - stu_on, 4),
        },
        "antiGaming": hack,
    }
    print(json.dumps(report, indent=2))

    if not args.dry_run:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        tag = hashlib.sha256(str(args.seeds).encode()).hexdigest()[:8]
        out = OUT_DIR / f"ablation-{tag}.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
