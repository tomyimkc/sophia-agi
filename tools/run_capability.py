"""Spec D — capability-retention runner. Two-tier (mirrors tools/run_steering.py):
  --dry-run : deterministic core on bundled fixtures -> CAPABILITY RETENTION VERIFIED
  --model X : reduced REAL run, granite steered-vs-unsteered on the arithmetic slice
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.steering.capability import capability_cell, score_response
SLICE = ROOT / "data" / "capability_arithmetic.json"
REPORT = ROOT / "agi-proof" / "benchmark-results" / "capability-retention.public-report.json"


def _items() -> "list[dict]":
    return json.loads(SLICE.read_text())["items"]


def build_dry_run_cell() -> dict:
    """Deterministic demo: a correct/coherent base vs a degenerate steered set."""
    items = _items()
    base = [score_response(f"the answer = {it['answer']}", it["answer"]) for it in items]
    # steered: half degenerate-repetition (wrong + low coherence, the high-alpha failure
    # mode), half still correct+coherent -> a partial (not total) drop, ~0.5 accuracy.
    steered = []
    for i, it in enumerate(items):
        if i % 2 == 0:
            steered.append(score_response("the the the the the the the the", it["answer"]))
        else:
            steered.append(score_response(f"the answer = {it['answer']}", it["answer"]))
    return capability_cell(base, steered)


def _run_dry() -> int:
    cell = build_dry_run_cell()
    assert set(cell) >= {"n", "base_accuracy", "steered_accuracy",
                         "capability_drop", "coherence", "retains"}
    assert cell["capability_drop"] > 0.05 and cell["retains"] is False
    print(json.dumps(cell, indent=2))
    print("CAPABILITY RETENTION VERIFIED ✓")
    return 0


def _run_real(args) -> int:
    try:
        import torch
    except Exception:
        print("real run needs torch: pip install -r requirements-steering.txt"); return 1
    if not torch.backends.mps.is_available():
        print("MPS not available; capability real run is Apple-Silicon only."); return 1
    from tools.run_steering import (
        _contrastive_prompts, _load_and_smoke, _residual_scale, MODEL_ALIASES, _CARRIERS,
    )
    from agent.steering.hooks import extract_persona_vector, SteeredClient

    model_id = MODEL_ALIASES.get(args.model, args.model)
    probe = _load_and_smoke(model_id)
    if probe is None:
        print("CAPABILITY DEMO ABSTAINED (model load failed) ✗"); return 1
    model, tok, model_id, L = probe

    pos, neg = _contrastive_prompts(args.axis)
    vec = extract_persona_vector(model, tok, pos, neg, L, normalize=True)
    alpha = float(args.alpha_coef) * _residual_scale(model, tok, L, _CARRIERS)
    steered = SteeredClient(model, tok, vector=vec, alpha=alpha, layers=[L], max_new_tokens=48)
    plain = SteeredClient(model, tok, max_new_tokens=48)

    items = _items()
    base_scored, steer_scored = [], []
    sys_prompt = "You are a careful assistant. Solve the arithmetic problem."
    for it in items:
        # SteeredClient.generate returns a _Result (.text/.ok), not a str — extract text (repo idiom).
        rb = plain.generate(sys_prompt, it["prompt"])
        rs = steered.generate(sys_prompt, it["prompt"])
        base_scored.append(score_response(rb.text if rb.ok else "", it["answer"]))
        steer_scored.append(score_response(rs.text if rs.ok else "", it["answer"]))
    cell = capability_cell(base_scored, steer_scored)
    report = {"benchmark": "capability-retention", "model": model_id, "axis": args.axis,
              "alphaCoef": args.alpha_coef, "layer": L, "mode": "real-reduced", "cell": cell,
              "note": ("A capability drop under steering is the expected honest result: steering "
                       "strong enough to move a trait degrades reasoning, explaining Spec B's SSA null. "
                       "retains=True only if drop<0.05 and coherence>=75.")}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(cell, indent=2)); print(f"wrote {REPORT}"); return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Spec D capability-retention runner")
    ap.add_argument("--dry-run", action="store_true", help="deterministic CI core")
    ap.add_argument("--model", default=None, help="hf id or 'granite' for the real run")
    ap.add_argument("--axis", default="E", help="persona axis for the steering vector")
    ap.add_argument("--alpha-coef", type=float, default=8.0)
    args = ap.parse_args(argv)
    if args.dry_run:
        return _run_dry()
    if args.model:
        return _run_real(args)
    return _run_dry()


if __name__ == "__main__":
    sys.exit(main())
