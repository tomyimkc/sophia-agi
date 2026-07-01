#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""W1 — verifier-distilled Process Reward Model (drop-in, fail-closed).

Thesis: the fail-closed symbolic verifier stack (agent.step_verifier.verify_derivation,
which returns a per-step StepVerdict with .ok) is a FREE, high-precision, per-step
labeler. Distill a soft neural PRM from its verdicts so reward can be scored densely at
training scale and can GENERALIZE to steps the symbolic checker can't reach.

WHAT THIS DOES (runnable offline, no GPU):
  * takes derivations (ordered expression lists), runs the REAL verify_derivation to get
    a ground-truth pass/fail for every transition;
  * builds (step_text, label) rows from those verdicts — the distillation dataset;
  * trains a stand-in PRM via the repo's OWN agent.activation_probes.train_centroid_probe
    (a linear probe over transparent features — the same fail-closed contract the repo
    already ships), and measures held-out agreement with the symbolic oracle;
  * CRITICAL honesty control: reports agreement on a HELD-OUT split AND on a held-out
    DOMAIN, so PRM generalization is measured, not memorization.

WHAT THIS DOES NOT DO (honest seam):
  * the stand-in PRM uses transparent features, not LM hidden states, and is not wired
    as an RLVR reward. Replacing featurize_text with real residual-stream vectors and
    plugging the PRM into tools/run_rlvr.py is the maintainer step. This tool proves the
    label pipeline + generalization measurement first. candidateOnly:true.

Derivation schema: {"id": str, "domain": "math"|"physics"|..., "steps": [expr, expr, ...]}
Usage:
  python3 tools/distill_process_reward_model.py --derivations d.jsonl \
      --holdout-domain physics --out prm_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

try:
    from agent.step_verifier import verify_derivation
    from agent.activation_probes import train_centroid_probe, evaluate_probe
    _REPO_OK = True
    _IMPORT_ERR = ""
except Exception as e:  # pragma: no cover
    _REPO_OK = False
    _IMPORT_ERR = f"{type(e).__name__}: {e}"


def _env_artifact(reason: str) -> dict[str, Any]:
    return {
        "schema": "sophia.prm_distillation.v1", "ok": False, "reason": reason,
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
    }


def label_steps(derivations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the fail-closed symbolic verifier and emit (text, label) rows.

    The real agent.step_verifier contract (verified against the tree):
      * each step must be a dict {"expr": ...} (or a Step); bare strings are coerced here;
      * StepVerdict.verdict is three-way: "accepted" | "rejected" | "abstain".

    We map to PRM labels HONESTLY:
      verdict == "accepted"  -> label True   (symbolic oracle confirmed the transition)
      verdict == "rejected"  -> label False  (symbolic oracle refuted it)
      verdict == "abstain"   -> DROPPED      (unchecked: NOT a training label — an
                                              abstain is 'no oracle', not 'bad step')
    Dropping abstains is the fail-closed choice: distilling on unchecked steps would teach
    the PRM the verifier's silence, not its judgement.
    """
    rows: list[dict[str, Any]] = []
    dropped_abstain = 0
    for d in derivations:
        raw = d.get("steps", [])
        domain = d.get("domain", "math")
        if len(raw) < 2:
            continue
        # coerce bare strings -> {"expr": ...} dicts (the verifier requires dicts/Step)
        steps = [s if isinstance(s, dict) else {"expr": str(s)} for s in raw]
        res = verify_derivation(steps, default_domain=domain)
        for v in res.steps:
            if v.verdict == "abstain":
                dropped_abstain += 1
                continue
            rows.append({
                "id": f"{d.get('id','?')}#{v.index}",
                "domain": domain,
                "text": f"{v.from_expr} -> {v.to_expr}",
                "label": v.verdict == "accepted",   # the symbolic oracle's verdict = PRM target
                "checker": v.checker,
            })
    # stash the abstain count on the function's return via a module-level marker
    label_steps.last_dropped_abstain = dropped_abstain  # type: ignore[attr-defined]
    return rows


def run(derivations: list[dict[str, Any]], *, holdout_domain: str | None = None,
        holdout_frac: float = 0.3, seed: int = 0) -> dict[str, Any]:
    if not _REPO_OK:
        return _env_artifact(f"repo instruments unavailable ({_IMPORT_ERR}); run with "
                             "PYTHONPATH=. inside the sophia-agi tree")
    if not derivations:
        return _env_artifact("no derivations provided (fail-closed)")

    rows = label_steps(derivations)
    if len(rows) < 4:
        return _env_artifact(f"only {len(rows)} labeled steps; need >=4 to split (fail-closed)")

    pos = sum(1 for r in rows if r["label"])
    neg = len(rows) - pos
    if pos == 0 or neg == 0:
        return _env_artifact(
            f"degenerate labels (pos={pos}, neg={neg}); the symbolic verifier gave a "
            "single class, so a PRM cannot be trained/measured. Provide derivations with "
            "both accepted and rejected steps.")

    # ---- split: random held-out, and (if requested) a held-out DOMAIN ----
    import random
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)

    dom_test = [r for r in rows if holdout_domain and r["domain"] == holdout_domain]
    in_domain = [r for r in rows if not (holdout_domain and r["domain"] == holdout_domain)]

    k = max(1, int(len(in_domain) * holdout_frac))
    rng.shuffle(in_domain)
    test_rand = in_domain[:k]
    train = in_domain[k:]
    if not train or not test_rand:
        return _env_artifact("split left an empty train or test set (too few steps); add data")

    probe = train_centroid_probe(train, name="process_reward_prm")
    eval_rand = evaluate_probe(probe, test_rand)

    report = {
        "schema": "sophia.prm_distillation.v1", "ok": True,
        "nDerivations": len(derivations), "nLabeledSteps": len(rows),
        "nDroppedAbstain": getattr(label_steps, "last_dropped_abstain", 0),
        "labelBalance": {"accepted": pos, "rejected": neg},
        "checkers": sorted({r["checker"] for r in rows}),
        "heldOutRandom": {"n": eval_rand["n"], "metrics": eval_rand["metrics"]},
        "note": "PRM target = fail-closed verify_derivation verdicts. Stand-in probe uses "
                "transparent features (agent.activation_probes), NOT LM hidden states, and "
                "is NOT yet wired as an RLVR reward — those are the maintainer seams. "
                "Held-out-domain agreement is the honest generalization test.",
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
    }
    if dom_test:
        eval_dom = evaluate_probe(probe, dom_test)
        report["heldOutDomain"] = {
            "domain": holdout_domain, "n": eval_dom["n"], "metrics": eval_dom["metrics"],
            "warning": "low held-out-domain accuracy => the PRM inherits the symbolic "
                       "verifier's coverage gaps; keep the symbolic checker as a periodic "
                       "ground-truth audit before trusting the PRM out-of-domain.",
        }
    return report


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="W1 verifier-distilled PRM")
    ap.add_argument("--derivations", required=True, help="JSONL {id,domain,steps[]}")
    ap.add_argument("--holdout-domain", default=None)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    derivs = load_jsonl(Path(args.derivations))
    report = run(derivs, holdout_domain=args.holdout_domain,
                 holdout_frac=args.holdout_frac, seed=args.seed)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())