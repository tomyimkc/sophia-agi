#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""No-answer-leakage feature audit — the T3 calibration-verifier guardrail, made runnable.

The T3 measurement_spec names a guardrail (``noAnswerLeakage``): the verifier's features must
be derived ONLY from the reasoning trace and retrieved sources, NEVER from the gold answer or
the correctness label. A feature that secretly reads the label inflates AUROC and is not a
result. This tool makes that audit a real, deterministic, offline check instead of a promise.

Mechanism (metamorphic test). For each trace we re-extract features after perturbing ONLY the
fields that must not influence them — the correctness label and any gold-answer field — leaving
the trace/source fields untouched. If a feature value CHANGES under that perturbation, the
extractor is reading the answer: leakage. If every feature is invariant across all perturbations,
the extractor passes (it cannot be reading a field it never saw move).

BOUNDARY (honest): this proves the extractor is INVARIANT to the named answer/label fields on
the audited traces — it does NOT prove the features are causally innocent of correctness through
some OTHER channel (a feature can legitimately correlate with correctness; that is the point).
It catches the specific failure of an extractor that mechanically copies the label/answer.
Deterministic, offline, pure stdlib.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Fields a correctness-prediction feature extractor must NEVER depend on.
_FORBIDDEN_FIELDS_DEFAULT = ("correct", "label", "gold", "goldAnswer", "answer", "isCorrect")

# Perturbations applied to a forbidden field to test invariance (covers bit-flip + value swap).
def _perturbations(value: Any) -> list[Any]:
    out: list[Any] = []
    if isinstance(value, bool):
        out.append(not value)
    elif isinstance(value, (int, float)):
        out.extend([1 - value if value in (0, 1) else -value, 0, 1, 999])
    elif isinstance(value, str):
        out.extend(["", value + "_PERTURBED", "ZZZ"])
    else:
        out.append(None)
    return out


def audit_trace(
    trace: dict,
    feature_fn: Callable[[dict], dict],
    *,
    forbidden_fields: Sequence[str] = _FORBIDDEN_FIELDS_DEFAULT,
) -> dict:
    """Audit one trace: features must be invariant to every perturbation of every forbidden
    field present. Returns {leak: bool, offendingFields: [...], baseline: {...}}."""
    baseline = feature_fn(trace)
    offending: list[str] = []
    for field in forbidden_fields:
        if field not in trace:
            continue
        for pert in _perturbations(trace[field]):
            mutated = copy.deepcopy(trace)
            mutated[field] = pert
            try:
                got = feature_fn(mutated)
            except Exception:
                # An extractor that CRASHES when the label changes is also coupling to it.
                offending.append(field)
                break
            if got != baseline:
                offending.append(field)
                break
    return {"leak": bool(offending), "offendingFields": sorted(set(offending)), "baseline": baseline}


def audit(
    traces: list[dict],
    feature_fn: Callable[[dict], dict],
    *,
    forbidden_fields: Sequence[str] = _FORBIDDEN_FIELDS_DEFAULT,
) -> dict:
    """Audit a trace set. PASS only if no trace leaks any forbidden field."""
    per = [audit_trace(t, feature_fn, forbidden_fields=forbidden_fields) for t in traces]
    leaks = [{"id": t.get("id"), "offendingFields": r["offendingFields"]}
             for t, r in zip(traces, per) if r["leak"]]
    passed = not leaks
    return {
        "schema": "sophia.feature_leakage_audit.v1",
        "n": len(traces),
        "forbiddenFields": list(forbidden_fields),
        "passed": passed,
        "verdict": "no-leakage" if passed else "LEAKAGE",
        "leakingTraces": leaks,
        "candidateOnly": True,
        "canClaimAGI": False,
        "boundary": (
            "Proves the extractor is INVARIANT to the named answer/label fields on these traces; "
            "it does NOT prove features are causally innocent of correctness through other channels."
        ),
    }


def audit_t3_extractor(n: int = 200, seed: int = 0) -> dict:
    """Run the audit against the real T3 feature extractor over synthetic traces.

    The T3 extractor must pass by construction (it reads only samples/evidence/authorConfidence),
    so a FAILURE here is a regression that coupled features to the label."""
    from tools.run_calibration_verifier_eval import _synthetic_traces, extract_features

    traces = _synthetic_traces(n, seed=seed)
    # Evidence objects aren't JSON-friendly for deepcopy of forbidden fields, but the forbidden
    # fields ('correct' etc.) are plain scalars; deepcopy handles the dataclass list fine.
    return audit(traces, extract_features)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--t3", action="store_true", help="audit the real T3 feature extractor (synthetic traces)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.t3:
        report = audit_t3_extractor(n=args.n, seed=args.seed)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["passed"] else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
