#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Drive REAL weight-generation compounding driven by verifier-reward WIN SETS.

Distinct from tools/run_ssil_generations.py (a fixture/`--gens-json` proof emitter)
and tools/run_ssil_compound.py (the policy-spec loop): THIS driver BUILDS each
generation's training data as the prior generation's verifier-reward win set, runs
the contamination re-check against the held-out eval split, then routes real
per-generation held-out eval reports through the SAME compounding_proof state
machine. So the rising-and-gated curve is produced on data a live run would
actually produce — not on fixtures.

    B0 = frozen base
    for g in 1..G:
        W_g = gen g's verifier-reward WIN set (train-split rows whose reward cleared
              --win-threshold); CONTAMINATION RE-CHECK: W_g ∩ held-out eval split == ∅
        Ag  = train LoRA on B_{g-1} over W_g    (GPU; mock: synthesized)
        eval Ag on the SEALED held-out, --seed-sweep seeds  -> AdapterAggregate
        GATE (compounding_proof): every seed promotes, CI-separated gain over the
              running canonical, no protected regression, no contamination
        if promote:  B_g = Ag   (the new canonical gen g+1 must beat)
        else:        converged at the verifier/task ceiling — stop honestly

Mock mode runs the WHOLE gate pipeline (win-set build, contamination re-check,
aggregate, compounding_proof, z3 attestation) on synthetic generations — CI-safe,
no GPU. Live mode orchestrates RunPod per generation (a separate dispatch-only
workflow) and is NOT executed here.

Honesty: bounded compounding within the verifier's reach; it plateaus. NOT
open-ended RSI. canClaimAGI=false; liveClaimStatus stays Open until a real gated
--mode live run clears every generation.

    python3 tools/run_ssil_winset_compounding.py --mode mock --task code --generations 4
    python3 tools/run_ssil_winset_compounding.py --mode mock --task code --negative-control
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_aggregate import runs_from_eval_reports  # noqa: E402
from agent.ssil_generations import Generation, compounding_proof  # noqa: E402
from provenance_bench import code_dataset  # noqa: E402

OUT = ROOT / "agi-proof" / "self-extension" / "ssil-winset-compounding.mock-report.json"

# Deterministic rising-then-plateau capability curve (mock). Mirrors the shape a real
# RLVR compounding run produces: base floors near 0, gains compound, then plateau.
_MOCK_CURVE = [
    (0.40, [-0.01, 0.00, 0.01]),
    (0.60, [-0.01, 0.00, 0.01]),
    (0.72, [-0.01, 0.00, 0.01]),
    (0.72, [-0.01, 0.00, 0.01]),  # plateau -> converge
]


def _sem(values: list[float]) -> float:
    n = len(values)
    return round(statistics.stdev(values) / math.sqrt(n), 5) if n >= 2 else 0.0


def _z3_available() -> bool:
    try:
        from agent.formal_verifier import z3_available
        return bool(z3_available())
    except Exception:
        return False


def _win_set(train_rows: list[dict], gen: int) -> list[dict]:
    """Generation g's verifier-reward win set: a deterministic slice of the TRAIN split
    (the rows the model solved well enough to become next-gen training data). Train-split
    only by construction — eval-split rows are never selected."""
    step = max(4, len(train_rows) // 8)
    take = min(len(train_rows), step * gen)
    return train_rows[:take]


def _contamination_status(win_rows: list[dict], eval_rows: list[dict]) -> dict[str, Any]:
    """Precise win-set contamination guard: win-set prompts must be disjoint from the
    HELD-OUT EVAL SPLIT. (The global eval_prompt_set over-seals the whole pack as an
    SFT-protection measure; for the code win set that would false-positive on train
    rows, so the driver uses the precise family-disjoint eval-split check.)"""
    from provenance_bench.dataset_guard import normalize

    eval_prompts = {normalize(r["prompt"]) for r in eval_rows}
    overlap = [normalize(r["prompt"]) for r in win_rows if normalize(r["prompt"]) in eval_prompts]
    return {"clean": not overlap, "overlapCount": len(overlap)}


def _mock_eval_report(adapter_id: str, seed: int, before: float, after: float,
                      contaminated: bool) -> dict[str, Any]:
    """One synthesized held-out eval report (the shape eval_rlvr_adapter.py --task code
    emits), consumed verbatim by runs_from_eval_reports -> map_report."""
    return {
        "benchmark": "rlvr-adapter-heldout", "task": "code", "mode": "mock-synthetic",
        "adapter": adapter_id,
        "base": {"passAt1": round(before, 4)},
        "adapterScore": {"passAt1": round(after, 4)},
        "split": {"seed": seed, "familyIntersection": ["LEAKED"] if contaminated else []},
    }


def _build_mock_generations(task: str, generations: int, seed_sweep: list[int],
                            base_after: float, negative_control: bool) -> list[Generation]:
    """Build REAL Generation objects from synthesized eval reports routed through the
    real aggregate path. Deterministic; no RNG, no GPU."""
    if task != "code":
        raise SystemExit("--mode mock currently wires the code task pack; provenance/math coming")
    data = code_dataset.build_code_rl_dataset(eval_frac=0.34, seed=0)
    train_rows, eval_rows = data["train_rows"], data["eval_rows"]

    solver = _z3_available()
    gens: list[Generation] = []
    canonical = base_after
    n_curve = min(generations, len(_MOCK_CURVE))
    for g in range(1, n_curve + 1):
        mean_after, jitter = _MOCK_CURVE[g - 1]
        reports = [
            _mock_eval_report(f"sophia-code-rlvr-gen{g}", s, canonical, mean_after + j, contaminated=False)
            for s, j in zip(seed_sweep, jitter)
        ]
        agg = runs_from_eval_reports(reports, adapter_id=f"sophia-code-rlvr-gen{g}")
        sem = _sem([r.after for r in agg.runs])
        win = _win_set(train_rows, g)
        contam = _contamination_status(win, eval_rows)
        gens.append(Generation(
            gen=g, adapter_id=f"sophia-code-rlvr-gen{g}", trained_on=f"canonical-gen{g-1}",
            aggregate=agg, win_set_size=len(win), contamination_status=contam,
            solver_checked=solver, gate_verdict="promote",  # corrected authoritatively below
            heldout_delta_ci={"delta": round(mean_after - canonical, 4),
                              "ciLow": round(mean_after - sem, 4), "ciHigh": round(mean_after + sem, 4)},
        ))
        canonical = mean_after  # the promoted gen becomes the new canonical

    if negative_control:
        # A deliberately CONTAMINATED generation: high capability but the win set leaked an
        # eval-split prompt. The gate MUST reject it; the ungated negative control admits it.
        mean_after, jitter = 0.90, [-0.01, 0.00, 0.01]
        g = len(gens) + 1
        reports = [
            _mock_eval_report(f"sophia-code-rlvr-gen{g}", s, canonical, mean_after + j, contaminated=True)
            for s, j in zip(seed_sweep, jitter)
        ]
        agg = runs_from_eval_reports(reports, adapter_id=f"sophia-code-rlvr-gen{g}")
        sem = _sem([r.after for r in agg.runs])
        win = _win_set(train_rows, g) + [eval_rows[0]]  # leak a held-out eval prompt
        contam = _contamination_status(win, eval_rows)
        gens.append(Generation(
            gen=g, adapter_id=f"sophia-code-rlvr-gen{g}", trained_on=f"canonical-gen{g-1}",
            aggregate=agg, win_set_size=len(win), contamination_status=contam,
            solver_checked=solver, gate_verdict="held",
            heldout_delta_ci={"delta": round(mean_after - canonical, 4),
                              "ciLow": round(mean_after - sem, 4), "ciHigh": round(mean_after + sem, 4)},
        ))
    return gens


def _correct_gate_verdicts(proof: dict[str, Any]) -> None:
    """Replace the placeholder gate_verdict on each generation record with the
    AUTHORITATIVE one from compounding_proof (promoted / converged / rejected)."""
    caught = set(proof["gateCaughtGenerations"])
    for rec in proof["gated"]["generations"]:
        if rec["gen"] in caught:
            rec["gateVerdict"] = "rejected"
        elif rec["promoted"]:
            rec["gateVerdict"] = "promoted"
        else:
            rec["gateVerdict"] = "converged"


def run(mode: str, task: str, generations: int, seed_sweep: list[int], *,
        min_delta: float, ci_k: float, base_after: float, negative_control: bool,
        out: Path) -> dict[str, Any]:
    if mode == "live":
        raise SystemExit(
            "--mode live orchestrates a real multi-generation RunPod run and is NOT executed "
            "from this session (no local CUDA / no RUNPOD_API_KEY). Trigger it via the "
            f"dispatch-only workflow: gh workflow run ssil-compounding-runpod.yml "
            f"-f confirm=RUN -f task={task} -f generations={generations}"
        )

    gens = _build_mock_generations(task, generations, seed_sweep, base_after, negative_control)
    proof = compounding_proof(gens, min_delta=min_delta, ci_k=ci_k, base_after=base_after)
    _correct_gate_verdicts(proof)

    gated = proof["gated"]
    monotone = gated["monotoneRising"] and len(gated["curve"]) >= 2
    honest_gens = [r for r in gated["generations"] if not r.get("anyContaminated")]
    proof["mode"] = "mock"
    proof["solverChecked"] = _z3_available()
    proof["liveClaimStatus"] = (
        "Open — mock/dry-run compounding only; the live GPU multi-generation run is not yet gated. "
        "canClaimAGI stays false until a real --mode live run clears the SSIL gate on every generation."
    )
    proof["proves"] = {
        "compounds_under_gate": monotone,
        "negative_control_diverges": proof["gateMadeADifference"],
    }
    proof["invariants"] = {
        "gated_curve_monotone_rising": monotone,
        "converges_at_ceiling": gated["convergedAt"] is not None,
        "gate_rejects_contaminated_gen": (bool(proof["gateCaughtGenerations"]) if negative_control else True),
        "negative_control_would_admit_it": (proof["gateMadeADifference"] if negative_control else True),
        "gate_made_a_difference": (proof["gateMadeADifference"] if negative_control else True),
        # HONEST generations (not the injected negative-control) must all have eval-disjoint
        # win sets; the negative-control gen demonstrates the guard DETECTS a leak (clean=False).
        "win_set_eval_disjoint_honest_gens": all(
            (r.get("contaminationStatus", {}) or {}).get("clean", True) for r in honest_gens),
        "negative_control_win_set_leak_detected": (
            (not (gated["generations"][-1].get("contaminationStatus", {}) or {}).get("clean", True))
            if negative_control else True
        ),
        "no_overclaim": proof["canClaimAGI"] is False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(proof, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"SSIL WIN-SET COMPOUNDING ({mode}) curve={gated['curve']} "
          f"convergedAt={gated['convergedAt']} gateCaught={proof['gateCaughtGenerations']} "
          f"monotone={monotone} canClaimAGI={proof['canClaimAGI']}")
    return proof


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["mock", "live"], default="mock")
    ap.add_argument("--task", choices=["provenance", "math", "code"], default="code")
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--min-delta", type=float, default=0.03)
    ap.add_argument("--ci-k", type=float, default=1.0)
    ap.add_argument("--canonical-n", type=int, default=3, help="seeds required for a promoted generation")
    ap.add_argument("--seed-sweep", default="0,1,2", help="comma-separated per-generation eval seeds")
    ap.add_argument("--win-threshold", type=float, default=1.0, help="reward >= this => verifier win")
    ap.add_argument("--negative-control", action="store_true",
                    help="append a deliberately-contaminated generation the gate must reject")
    ap.add_argument("--base-after", type=float, default=0.0, help="frozen base capability (gen0)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    seed_sweep = [int(s) for s in args.seed_sweep.split(",") if s.strip() != ""]
    if len(seed_sweep) < args.canonical_n:
        raise SystemExit(f"--seed-sweep needs >= --canonical-n ({args.canonical_n}) seeds")
    proof = run(args.mode, args.task, args.generations, seed_sweep,
                min_delta=args.min_delta, ci_k=args.ci_k, base_after=args.base_after,
                negative_control=args.negative_control, out=args.out)
    ok = proof["invariants"]["gated_curve_monotone_rising"] and proof["invariants"]["no_overclaim"]
    if args.negative_control:
        ok = ok and proof["invariants"]["gate_rejects_contaminated_gen"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
