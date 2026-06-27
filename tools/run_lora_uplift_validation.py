#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""VALIDATED-bar aggregator for a QLoRA adapter's CONTENT uplift.

The run #5/#7 eval ladders are single-judge / single-seed -> candidate-only. This harness
takes the per-seed, multi-judge-family CONTENT judgments (base vs adapter) and applies the
no-overclaim VALIDATED gate (the same one provenance_bench uses), reusing its primitives:

  - provenance_bench.consensus.cohen_kappa  -> inter-judge agreement
  - provenance_bench.aggregate._ci          -> 95% bootstrap CI on the uplift
  - provenance_bench.aggregate._distinct_families / KAPPA_FLOOR  -> the gate thresholds

A result is VALIDATED only if ALL hold (provenance_bench definition):
  notMock, >=2 judge families, NO judge family == the subject's family (judge != subject),
  mean pairwise Cohen's kappa >= 0.40, >=3 seeds, and a 95% CI on the content-uplift delta
  that excludes zero. Otherwise: candidate-only / illustrative.

This is the AGGREGATION + GATE half of the protocol (offline, deterministic, tested via
--mock). Producing the judgments it consumes (2 independent LLM judges, judge != Qwen2.5-3B,
labeling the CONTENT channel over each seed's eval transcripts) is the upstream step the
preregistration (docs/06-Roadmap/P6-LoRA-Uplift-Validation-Preregistration.md) specifies.

Judgments schema (``--judgments file.json``):
  {
    "subjectModel": "Qwen/Qwen2.5-3B-Instruct",   # NOT a judge (subject family = 'qwen')
    "judges": ["openrouter:deepseek/deepseek-chat",          # families keyed by VENDOR:
               "openrouter:meta-llama/llama-3.1-70b-instruct"],  # 'deepseek', 'meta-llama'
    "seeds": [
      {"seed": 0, "items": [
        {"id": "philosophy/...", "baseContent": {"deepseek": true, "meta-llama": false},
                                  "adapterContent": {"deepseek": true, "meta-llama": true}}, ...]},
      ... (>=3 seeds for VALIDATED) ...
    ]
  }

Usage:
  python tools/run_lora_uplift_validation.py --judgments judgments.json --out report.json
  python tools/run_lora_uplift_validation.py --mock          # offline self-test of the gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.aggregate import (  # noqa: E402
    _AGGREGATOR_PROVIDERS, KAPPA_FLOOR, _ci, _distinct_families, _llmhub_family,
)
from provenance_bench.consensus import cohen_kappa  # noqa: E402


def _family_key(judge_spec: str) -> str:
    """Provider family for a judge spec, IDENTICAL to aggregate._distinct_families'
    per-judge derivation, so the per-family label vectors key the SAME way the gate
    counts families (vendor, gateway-aware). Examples:
      'openrouter:meta-llama/llama-3.1-70b-instruct' -> 'meta-llama'  (NOT 'llama')
      'openrouter:deepseek/deepseek-chat'            -> 'deepseek'
      'llmhub:gpt-4o'                                -> 'openai'
    The old 'last path component, first token' rule diverged from the gate's vendor
    family for vendor-prefixed names (meta-llama -> 'llama'), which skipped items and
    corrupted the kappa. Keeping the two in lockstep is the whole point."""
    if not judge_spec or not judge_spec.strip():
        return ""
    prov, _, model = judge_spec.partition(":")
    prov = prov.strip().lower()
    model = model.strip().split("@", 1)[0]  # drop any 'model@http://host/v1' base_url suffix
    if prov == "llmhub":
        return _llmhub_family(model)
    if prov in _AGGREGATOR_PROVIDERS and "/" in model:
        return model.split("/", 1)[0].lower()
    return prov


def _subject_family(subject_spec: str) -> str:
    """Vendor family of the subject model. The subject is usually a bare HF id
    ('Qwen/Qwen2.5-3B-Instruct' -> 'qwen'); also accept a 'provider:vendor/model' spec."""
    s = (subject_spec or "").strip()
    if not s:
        return ""
    if ":" in s and "/" in s.split(":", 1)[1]:
        return _family_key(s)
    if "/" in s:
        return s.split("/", 1)[0].lower()
    if ":" in s:
        return s.split(":", 1)[0].lower()
    return s.lower()


def _majority(labels: list[bool]) -> bool:
    """Strict majority, matching provenance_bench.consensus.make_consensus_judge
    (``sum(True) * 2 > n``). A tie (e.g. 1-1 across two judges) does NOT pass, so a
    2-judge 'consensus' is unanimity, not OR — otherwise pass rates / uplift inflate."""
    return sum(1 for x in labels if x) * 2 > len(labels)


def mean_pairwise_kappa(judgments: dict) -> "tuple[float | None, dict]":
    """Mean pairwise Cohen's kappa across judge families on the ADAPTER content labels,
    pooled over all (seed, item) rows. Mirrors consensus.percent_agreement's pairwise kappa."""
    judges = judgments.get("judges", [])
    fams = [_family_key(j) for j in judges]
    # Per-family vote vector over all rows where every family judged the item.
    vectors: dict[str, list[int]] = {f: [] for f in fams}
    for srun in judgments.get("seeds", []):
        for item in srun.get("items", []):
            ac = item.get("adapterContent", {})
            if not all(f in ac for f in fams):
                continue
            for f in fams:
                vectors[f].append(1 if ac[f] else 0)
    pair_kappas: list[float] = []
    detail: dict = {}
    for i in range(len(fams)):
        for j in range(i + 1, len(fams)):
            k = cohen_kappa(vectors[fams[i]], vectors[fams[j]])
            if k is not None:
                pair_kappas.append(k)
                detail[f"{fams[i]}_vs_{fams[j]}"] = round(k, 4)
    mean_k = round(sum(pair_kappas) / len(pair_kappas), 4) if pair_kappas else None
    return mean_k, detail


def _consensus_rates(srun: dict, fams: list[str]) -> "tuple[float, float, int]":
    """Per-seed consensus (majority-of-families) CONTENT pass rates: (base, adapter, n).

    A row counts only if EVERY judge family labelled it on BOTH base and adapter — else
    the majority-of-families vote is ill-defined and a single-judge row would silently
    enter the uplift/CI while mean_pairwise_kappa() drops it. The vote is taken over the
    fixed family set (in order), never over raw dict.values()."""
    base_pass, adapter_pass, n = 0, 0, 0
    for item in srun.get("items", []):
        bc = item.get("baseContent", {})
        ac = item.get("adapterContent", {})
        if not all(f in bc for f in fams) or not all(f in ac for f in fams):
            continue
        n += 1
        base_pass += 1 if _majority([bool(bc[f]) for f in fams]) else 0
        adapter_pass += 1 if _majority([bool(ac[f]) for f in fams]) else 0
    if n == 0:
        return 0.0, 0.0, 0
    return base_pass / n, adapter_pass / n, n


def _bootstrap_delta_ci(per_seed_deltas: list[float], n_boot: int = 2000,
                        seed: int = 0) -> "list[float]":
    """95% CI on the mean content-uplift delta via bootstrap over per-seed deltas."""
    import numpy as np
    if not per_seed_deltas:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    arr = np.asarray(per_seed_deltas, dtype=float)
    boots = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_boot)]
    return _ci(boots)


def aggregate(judgments: dict, *, n_boot: int = 2000) -> dict:
    subject = judgments.get("subjectModel", "")
    judges = judgments.get("judges", [])
    seeds = judgments.get("seeds", [])
    fams = [_family_key(j) for j in judges]
    subject_fam = _subject_family(subject)

    per_seed = []
    for srun in seeds:
        b, a, n = _consensus_rates(srun, fams)
        per_seed.append({"seed": srun.get("seed"), "baseContent": round(b, 4),
                         "adapterContent": round(a, 4), "delta": round(a - b, 4), "n": n})
    deltas = [s["delta"] for s in per_seed if s["n"] > 0]
    mean_delta = round(sum(deltas) / len(deltas), 4) if deltas else 0.0
    ci = [round(x, 4) for x in _bootstrap_delta_ci(deltas, n_boot=n_boot)]
    mean_k, kappa_detail = mean_pairwise_kappa(judgments)
    n_families = _distinct_families(judges)

    checks = {
        "notMock": bool(subject) and "mock" not in subject.lower(),
        "multiFamilyJudges": n_families >= 2,
        "kappaAboveFloor": mean_k is not None and mean_k >= KAPPA_FLOOR,
        "atLeast3Seeds": len([s for s in per_seed if s["n"] > 0]) >= 3,
        "ciExcludesZero": bool(ci) and (ci[0] > 0 or ci[1] < 0),
        # Enforce the honest_scope promise: no judge family may be the subject's own
        # family (judge != subject). A qwen judge over a qwen subject is self-grading.
        "judgeNotSubject": bool(judges) and bool(subject_fam)
        and subject_fam not in set(fams),
    }
    validated = all(checks.values())

    return {
        "benchmark": "lora-content-uplift-validation",
        "subjectModel": subject,
        "subjectFamily": subject_fam,
        "judges": judges,
        "judgeFamilies": n_families,
        "judgeFamilyKeys": fams,
        "metric": "consensus CONTENT pass-rate uplift (adapter - base), per seed",
        "perSeed": per_seed,
        "meanDelta": mean_delta,
        "ciDelta": ci,
        "meanPairwiseKappa": mean_k,
        "kappaPairs": kappa_detail,
        "kappaFloor": KAPPA_FLOOR,
        "validated": validated,
        "validatedChecks": checks,
        "candidateOnly": not validated,
        "canClaimAGI": False,
        "honest_scope": (
            "VALIDATED requires ALL of: non-mock subject, >=2 independent judge families "
            "(judge != subject), mean pairwise Cohen's kappa >= 0.40, >=3 seeds, and a 95% "
            "bootstrap CI on the content-uplift delta excluding zero. If any fails, the "
            "result is candidate-only / illustrative, NOT a capability claim. This harness "
            "is the aggregation+gate half; the upstream 2-judge labelling step is specified "
            "in the P6 preregistration."
        ),
    }


# --------------------------------------------------------------------------- #
# Mock self-test: a clear-signal synthetic scenario. notMock is False by design
# (mock can never VALIDATE), so this verifies the OTHER four checks compute correctly.
# --------------------------------------------------------------------------- #
def mock_judgments(*, seeds: int = 3, n: int = 32) -> dict:
    import numpy as np
    rng = np.random.default_rng(0)
    seed_runs = []
    for s in range(seeds):
        items = []
        for i in range(n):
            # Base content ~0.72; adapter ~0.84; two judges agree ~90% of the time.
            base_true = bool(rng.random() < 0.72)
            adapter_true = bool(rng.random() < 0.84)
            def pair(truth):
                a = truth if rng.random() < 0.9 else (not truth)
                b = truth if rng.random() < 0.9 else (not truth)
                return {"deepseek": bool(a), "meta-llama": bool(b)}
            items.append({"id": f"item_{i}", "baseContent": pair(base_true),
                          "adapterContent": pair(adapter_true)})
        seed_runs.append({"seed": s, "items": items})
    return {
        "subjectModel": "mock:Qwen2.5-3B",   # mock -> notMock check is False on purpose
        # Two NON-qwen families (judge != subject), per the P6 preregistration.
        "judges": ["openrouter:deepseek/deepseek-chat",
                   "openrouter:meta-llama/llama-3.1-70b-instruct"],
        "seeds": seed_runs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judgments", type=Path, help="judgments JSON (see module docstring)")
    ap.add_argument("--mock", action="store_true", help="offline self-test on synthetic data")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.mock:
        judgments = mock_judgments()
    elif args.judgments:
        judgments = json.loads(args.judgments.read_text(encoding="utf-8"))
    else:
        ap.error("provide --judgments FILE or --mock")
        return 2

    report = aggregate(judgments)
    out = args.out
    if out:
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "validated", "validatedChecks", "meanDelta", "ciDelta", "meanPairwiseKappa",
        "judgeFamilies")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
