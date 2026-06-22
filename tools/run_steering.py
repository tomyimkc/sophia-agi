"""Spec B — Level-3 activation-steering runner.

OFFLINE (default, no torch/GPU/network): --model mock / --dry-run runs the
steering-machinery invariants through the shipping functions.
REAL (gated, MPS): --model phi3.5 downloads + steers microsoft/Phi-3.5-mini-instruct
and runs the Ollama-judged battery. LIVE SSA is OPEN in agi-proof/failure-ledger.md
until a gated run (entry id: steering-live-run-not-yet-gated-2026-06-23).
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_JSON = ROOT / "agi-proof" / "benchmark-results" / "steering.public-report.json"
DEFAULT_MODEL = "microsoft/Phi-3.5-mini-instruct"
FALLBACK_CHAIN = [
    "microsoft/Phi-3.5-mini-instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "ibm-granite/granite-3.1-2b-instruct", "stabilityai/stablelm-2-1_6b-chat",
]


def _offline_invariants() -> "tuple[bool, dict]":
    """Steering-machinery invariants (no torch, no GPU, no network)."""
    from agent.steering import vectors as vec
    from agent.steering import compose, stats
    from provenance_bench import steering_dataset as sds

    # mock extractor is deterministic + unit
    m1 = vec.mock_vector(3072, seed=1)
    m2 = vec.mock_vector(3072, seed=1)
    mock_det = (m1 == m2) and abs(vec.norm(m1) - 1.0) < 1e-9

    # composition with soft-projection reduces pairwise overlap vs raw
    vs = {"E": vec.normalize([1.0, 0.0]), "O": vec.normalize([1.0, 1.0])}
    raw_cos = abs(vec.cosine(vs["E"], vs["O"]))
    sp = compose.soft_project(vs)
    compose_reduces = abs(vec.cosine(sp["E"], sp["O"])) < raw_cos

    # SSA verdict enacts on a strong synthetic cell, abstains on a weak one
    strong = {"delta_ci": [0.4, 0.9], "delta_point": 0.6, "steered_d": 0.8,
              "off_target_d": {"O": 0.1}, "kappa": 0.55, "capability_drop": 0.02,
              "coherence": 90.0, "is_mock": False}
    weak = {**strong, "delta_ci": [-0.1, 0.5], "delta_point": 0.1}
    enacts = stats.ssa_verdict(strong)["status"] == "enacted"
    abstains = stats.ssa_verdict(weak)["status"] == "abstained"

    split = sds.build_steering_split(eval_frac=0.3, seed=0)

    checks = {
        "mockExtractDeterministic": mock_det,
        "composeOrthogonalReduces": compose_reduces,
        "verdictEnactsWhenStrong": enacts,
        "verdictAbstainsWhenWeak": abstains,
        "contaminationFree": split["item_intersection"] == [],
    }
    detail = {
        "checks": checks,
        "extractItems": len(split["extract_items"]),
        "measureItems": len(split["measure_items"]),
        "extractSealed": split["extract_sealed"],
        "measureSealed": split["measure_sealed"],
        "ssaThresholds": stats.SSA_THRESHOLDS,
        "fallbackChain": FALLBACK_CHAIN,
    }
    return all(checks.values()), detail


def _write_report(detail: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(detail, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def _run_real(args) -> int:
    """Gated real Phi-3.5 MPS run (Task 9 wires the full battery). Bails cleanly
    if torch/MPS is unavailable, mirroring run_rlvr's cuda guard."""
    try:
        import torch
    except Exception:
        print("real run needs torch: pip install -r requirements-steering.txt", file=sys.stdout)
        return 1
    if not torch.backends.mps.is_available():
        print("MPS not available; steering real run is Apple-Silicon only.", file=sys.stdout)
        return 1
    # Full real pipeline is filled in by Task 9 (load-and-smoke probe → extract →
    # steer → measure → judge → SSA). Until then, record the OPEN live claim.
    report = {
        "benchmark": "steering", "model": args.model, "visibility": "public-aggregate",
        "claimStatus": "Open — capability claim requires a gated run; "
                       "this artifact records config only",
        "ssaThresholds": __import__("agent.steering.stats", fromlist=["SSA_THRESHOLDS"]).SSA_THRESHOLDS,
        "fallbackChain": FALLBACK_CHAIN,
    }
    _write_report(report, args.out)
    print("Real steering run scaffolded. Full battery + SSA is the gated step.")
    return 0


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mock",
                    help=f'subject (default "mock"; real: "phi3.5" → {DEFAULT_MODEL})')
    ap.add_argument("--dry-run", action="store_true", help="offline invariants only (no torch)")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.model == "mock" or args.dry_run:
        ok, detail = _offline_invariants()
        detail["benchmark"] = "steering"
        detail["mode"] = "mock-offline"
        detail["claim"] = "steering-machinery invariants (NOT a capability claim)"
        detail["liveClaimStatus"] = (
            "Open — see agi-proof/failure-ledger.md steering-live-run-not-yet-gated-2026-06-23"
        )
        _write_report(detail, args.out)
        print("STEERING WIRING VERIFIED ✓" if ok else "STEERING INVARIANTS NOT MET ✗")
        return 0 if ok else 1

    return _run_real(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc(file=sys.stdout)
        raise SystemExit(1)
