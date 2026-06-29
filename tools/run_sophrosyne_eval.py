#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophrosyne three-arm temperance evaluation (deterministic machinery; PENDING result).

Turns the Sophrosyne measurement plan into a runnable, gated instrument — the same
pattern as tools/run_andreia_eval.py.

The claim under test (agi-proof/benchmark-results/sophrosyne/measurement_spec.json):
consulting the Sophrosyne gate REDUCES the excess-error rate (cut/restrained when
more effort was right) AND the deficiency-error rate (over-spent when restraint was
right), versus the same raw agent with NO gate, WITHOUT lowering task-success.

Per item, given the optimal measure `o` and an arm's decision `d`:
  excess error     = 1 if o in {proportionate,sustain} and d == restrain else 0
  deficiency error = 1 if o in {proportionate,restrain} and d == sustain else 0
`escalate` is the akrasia/protected-step middle — neither error (it forces an
explicit measure decision; it does not silently over-spend or cut a required step).

Arms:
  * sophrosyne-standalone / sophrosyne-consulted: deterministic (we have the gate).
  * no-gate baseline: REQUIRES A REAL MODEL/AGENT — not run offline. The committed
    artifact is therefore PENDING / NO-GO.

Modes (all offline):
  * --mock {profligate,miserly,oracle}: score the gate arm against a deterministic
    mock baseline and print the per-arm rates + the paired deltas with bootstrap
    95% CIs. Exercises the delta+CI math in CI; NOT evidence (a mock is not a model).
  * --emit-pending: write the committed not-run / NO-GO artifact.
  * --model <spec>: refuse rather than fabricate; result stays PENDING.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.sophrosyne import assess_temperance  # noqa: E402
from tools.eval_stats import bootstrap_ci_paired, mde_at_n  # noqa: E402

RESULTS_DIR = ROOT / "agi-proof" / "benchmark-results" / "sophrosyne"
BATTERY_PATH = RESULTS_DIR / "sophrosyne_measure_battery.json"
SPEC_PATH = RESULTS_DIR / "measurement_spec.json"
PENDING_PATH = RESULTS_DIR / "sophrosyne-measure-eval.PENDING.public-report.json"
# Real-arm (model-gated) inputs: the labelled external battery + caches, mirroring Andreia.
LABELED_PATH = RESULTS_DIR / "sophrosyne_external_battery.labeled.json"
BASELINE_CACHE = RESULTS_DIR / "sophrosyne_baseline_cache.jsonl"
PUBLIC_REPORT = RESULTS_DIR / "sophrosyne-measure-eval.public-report.json"
_GATE_VERDICTS = ("proportionate", "restrain", "sustain", "escalate")


def _excess_err(optimal: str, decision: str) -> int:
    return int(optimal in {"proportionate", "sustain"} and decision == "restrain")


def _deficiency_err(optimal: str, decision: str) -> int:
    return int(optimal in {"proportionate", "restrain"} and decision == "sustain")


def _load_battery() -> dict:
    return json.loads(BATTERY_PATH.read_text(encoding="utf-8"))


def _gate_decisions(battery: dict) -> list[str]:
    """The Sophrosyne gate's decision per case (deterministic)."""
    return [assess_temperance(c["text"], context=c.get("context") or {}).to_dict()["verdict"]
            for c in battery["cases"]]


def _mock_baseline_decisions(kind: str, battery: dict) -> list[str]:
    """Deterministic no-gate stand-ins to exercise the delta math (NOT a model).

    profligate — a raw agent that always over-spends (always restrain-worthy excess);
                 modelled as always 'sustain' (keeps spending) -> drives deficiency error.
    miserly    — a raw agent that always cuts short; modelled as always 'restrain'
                 -> drives excess error.
    oracle     — the labelled optimal (upper bound; sanity check only).
    """
    cases = battery["cases"]
    if kind == "profligate":
        return ["sustain"] * len(cases)
    if kind == "miserly":
        return ["restrain"] * len(cases)
    if kind == "oracle":
        return [c["optimal"] for c in cases]
    raise ValueError(f"unknown mock baseline: {kind}")


def _arm_rates(optimals: list[str], decisions: list[str]) -> dict:
    n = len(optimals)
    exc = [_excess_err(o, d) for o, d in zip(optimals, decisions, strict=True)]
    dfc = [_deficiency_err(o, d) for o, d in zip(optimals, decisions, strict=True)]
    esc = sum(1 for d in decisions if d == "escalate")
    return {
        "n": n,
        "excessErrors": sum(exc),
        "deficiencyErrors": sum(dfc),
        "excessErrorRate": round(sum(exc) / n, 4) if n else 0.0,
        "deficiencyErrorRate": round(sum(dfc) / n, 4) if n else 0.0,
        "escalateRate": round(esc / n, 4) if n else 0.0,
        "_exc": exc,
        "_dfc": dfc,
    }


def _paired_delta(gate: dict, baseline: dict, *, seed: int = 0) -> dict:
    """Δ = gate − baseline, paired per item, with 95% bootstrap CIs.

    Negative Δ on BOTH excess and deficiency is the improvement we hope for.
    """
    exc_diffs = [g - b for g, b in zip(gate["_exc"], baseline["_exc"], strict=True)]
    dfc_diffs = [g - b for g, b in zip(gate["_dfc"], baseline["_dfc"], strict=True)]
    n = len(exc_diffs)
    return {
        "deltaExcess": round(sum(exc_diffs) / n, 4) if n else 0.0,
        "deltaExcessCI95": bootstrap_ci_paired(exc_diffs, seed=seed),
        "deltaDeficiency": round(sum(dfc_diffs) / n, 4) if n else 0.0,
        "deltaDeficiencyCI95": bootstrap_ci_paired(dfc_diffs, seed=seed),
        "mdeAtN": round(mde_at_n(n, p0=0.5), 4),
    }


def gate_verdict(*, baseline_is_real: bool, judge_families: int, delta: dict | None,
                 task_success_guardrail_measured: bool = False) -> dict:
    """GO/NO-GO over the pre-registered pillars. Offline this is always NO-GO."""
    failures: list[str] = []
    if not baseline_is_real:
        failures.append("no_real_baseline: the no-gate baseline arm needs a real model/agent (mock baselines are not evidence)")
    if judge_families < 2:
        failures.append("ground_truth_not_2family: optimal-measure labels are author-only, not >= 2 independent judge families (kappa >= 0.40)")
    exc_ci = (delta or {}).get("deltaExcessCI95") or [None, None]
    dfc_ci = (delta or {}).get("deltaDeficiencyCI95") or [None, None]
    exc_excl = exc_ci[0] is not None and exc_ci[1] is not None and exc_ci[1] < 0
    dfc_excl = dfc_ci[0] is not None and dfc_ci[1] is not None and dfc_ci[1] < 0
    if not (exc_excl and dfc_excl):
        failures.append("no_effect_ci: delta excess/deficiency-error CIs do not both exclude 0 (or no real arms to compute them)")
    if not task_success_guardrail_measured:
        failures.append("no_task_success_guardrail: task-success guardrail (delta success >= -0.02) needs a real task run")
    return {
        "verdict": "NO-GO" if failures else "GO",
        "go": not failures,
        "criticalFailures": failures,
        "boundary": (
            "Sophrosyne is candidate infrastructure. GO requires a real no-gate baseline, "
            ">= 2 independent judge families for the labels, delta excess- AND deficiency-error "
            "CIs excluding 0 (<= -0.10), and the task-success guardrail held. canClaimAGI:false."
        ),
    }


def build_pending_artifact() -> dict:
    """Committed not-run / NO-GO artifact. The gate-arm routing rates ARE real and
    deterministic; the no-gate baseline is NOT run, so there is no effect — NO-GO.
    Deterministic (no timestamps) so re-emit is byte-stable (no CI drift)."""
    battery = _load_battery()
    optimals = [c["optimal"] for c in battery["cases"]]
    gate = _arm_rates(optimals, _gate_decisions(battery))
    verdict = gate_verdict(baseline_is_real=False, judge_families=1, delta=None)
    return {
        "experimentId": "sophrosyne-measure-eval",
        "status": "not_run",
        "verdict": verdict["verdict"],
        "go": False,
        "canClaimAGI": False,
        "claimCeiling": "candidate_only; canClaimAGI:false",
        "headline": "PENDING — machinery only; no real no-gate baseline run has been performed",
        "harness": "tools/run_sophrosyne_eval.py",
        "preregistration": "agi-proof/benchmark-results/sophrosyne/measurement_spec.json",
        "battery": battery.get("schema"),
        "groundTruth": "author-labelled (NOT >= 2 independent judge families) — does not satisfy the spec",
        "arms": {
            "sophrosyne-standalone": {
                "n": gate["n"],
                "excessErrorRate": gate["excessErrorRate"],
                "deficiencyErrorRate": gate["deficiencyErrorRate"],
                "escalateRate": gate["escalateRate"],
                "note": "routing fidelity vs author labels — NOT a real-decision effect",
            },
            "no-gate-baseline": {"status": "not_run", "reason": "requires a real model/agent"},
            "sophrosyne-consulted": {"status": "not_run", "reason": "scored only alongside a real baseline"},
        },
        "delta": None,
        "criticalFailures": verdict["criticalFailures"],
        "note": (
            "Intentionally PENDING. The deterministic mock baselines (--mock "
            "{profligate,miserly,oracle}) exercise the delta+CI math in CI "
            "(tests/test_sophrosyne_eval.py), but a mock is not a model: no effect on real "
            "decisions is claimed. Promotion needs an external decontaminated task set, "
            ">= 2 independent judge families, a real no-gate baseline, delta excess- AND "
            "deficiency-error CIs excluding 0, and the task-success guardrail — see the "
            "measurement_spec and the sophrosyne-temperance-gate row in agi-proof/failure-ledger.md."
        ),
    }


def emit_pending() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = build_pending_artifact()
    PENDING_PATH.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return PENDING_PATH


def run_mock(kind: str, *, seed: int = 0) -> dict:
    battery = _load_battery()
    optimals = [c["optimal"] for c in battery["cases"]]
    gate = _arm_rates(optimals, _gate_decisions(battery))
    base = _arm_rates(optimals, _mock_baseline_decisions(kind, battery))
    delta = _paired_delta(gate, base, seed=seed)
    # A mock baseline is NOT a real model, so the verdict stays NO-GO regardless.
    verdict = gate_verdict(baseline_is_real=False, judge_families=1, delta=delta)
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731
    return {
        "baseline": f"mock:{kind}",
        "gateArm": strip(gate),
        "baselineArm": strip(base),
        "delta": delta,
        "verdict": verdict["verdict"],
        "criticalFailures": verdict["criticalFailures"],
        "boundary": "mock baseline — machinery proof, NOT evidence about real decisions",
    }


# --------------------------------------------------------------------------- #
# Real-model arm (model-gated; mirrors tools/run_andreia_eval.py run_real). Scores
# the three arms on the labelled external battery's agreed-consensus subset against a
# REAL no-gate baseline. The gate arms DERIVE the forces from raw text (the real
# consultTemperance path) — this run measures exactly that behaviour, it does not
# assume it. Caches are keyed for a reproducible receipt.
# --------------------------------------------------------------------------- #
def _load_labeled() -> dict:
    if not LABELED_PATH.exists():
        try:
            shown = LABELED_PATH.relative_to(ROOT)
        except ValueError:
            shown = LABELED_PATH
        raise SystemExit(f"missing {shown} — run tools/label_sophrosyne_battery.py first.")
    return json.loads(LABELED_PATH.read_text(encoding="utf-8"))


def _standalone_decision(text: str) -> str:
    return assess_temperance(text).to_dict()["verdict"]


def _consulted_decision(text: str) -> str:
    """sophrosyne-consulted arm: the gate as wired into the conscience kernel."""
    from agent.conscience import conscience_check
    d = conscience_check(text, context={"consultTemperance": True}).to_dict()
    # The temperance verdict lives under decision.temperance; fall back to standalone.
    return (d.get("temperance") or {}).get("verdict") or _standalone_decision(text)


def _load_baseline_cache() -> dict:
    cache: dict = {}
    if BASELINE_CACHE.exists():
        for line in BASELINE_CACHE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                cache[(r["spec"], r["seed"], r["temperature"], r["caseId"])] = r["verdict"]
    return cache


def _subject_decisions(spec: str, cases: list[dict], *, seed: int, temperature: float,
                       workers: int = 8, cache: dict | None = None) -> list:
    """The raw no-gate baseline: a real model picks the measure over the SAME raw text
    the gate sees, via the shared tools/sophrosyne_decision prompt. Cached by
    (spec, seed, temperature, caseId) so the committed receipt is reproducible."""
    from concurrent.futures import ThreadPoolExecutor

    from agent.model import ModelClient, resolve_config
    from tools.sophrosyne_decision import build_messages, parse_verdict

    cache = _load_baseline_cache() if cache is None else cache
    cfg = resolve_config(spec)
    cfg.temperature = temperature
    cfg.seed = seed
    cfg.max_tokens = 16
    client = ModelClient(cfg, retries=2)

    def one(case: dict) -> "tuple[str, str | None]":
        key = (spec, seed, temperature, case["id"])
        if key in cache:
            return case["id"], cache[key]
        system, user = build_messages(case["text"])
        res = client.generate(system, user)
        return case["id"], (parse_verdict(res.text) if res.ok else None)

    todo = [c for c in cases if (spec, seed, temperature, c["id"]) not in cache]
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(one, todo))
        with BASELINE_CACHE.open("a", encoding="utf-8") as fh:
            for cid, v in results:
                cache[(spec, seed, temperature, cid)] = v
                fh.write(json.dumps({"spec": spec, "seed": seed, "temperature": temperature,
                                     "caseId": cid, "verdict": v}, ensure_ascii=False) + "\n")
    return [cache.get((spec, seed, temperature, c["id"])) for c in cases]


def run_real(subject_spec: str, *, seeds: int = 3, temperature: float = 0.7, workers: int = 8,
             boot_seed: int = 0) -> dict:
    """Score the three arms on the labelled battery's agreed subset against a real baseline.

    Primary: Δ(excess-error) AND Δ(deficiency-error) = sophrosyne-consulted − no-gate
    baseline, paired per item, pooled across seeds, bootstrap 95% CIs. Guardrail
    (task-success) needs a task harness and is reported as not-measured here.
    """
    labeled = _load_labeled()
    agreed = [c for c in labeled["cases"] if c.get("agreedQuadrant") and c.get("optimal")]
    if not agreed:
        raise SystemExit("no agreed-consensus cases in the labelled battery — cannot score.")
    texts = [c["text"] for c in agreed]
    optimals = [c["optimal"] for c in agreed]

    bcache = _load_baseline_cache()
    baseline_by_seed = [
        _subject_decisions(subject_spec, agreed, seed=s, temperature=temperature, workers=workers, cache=bcache)
        for s in range(seeds)
    ]
    keep = [i for i in range(len(agreed))
            if all(baseline_by_seed[s][i] in _GATE_VERDICTS for s in range(seeds))]
    dropped = len(agreed) - len(keep)
    opt = [optimals[i] for i in keep]
    kept_texts = [texts[i] for i in keep]

    consulted = [_consulted_decision(t) for t in kept_texts]
    standalone = [_standalone_decision(t) for t in kept_texts]
    consulted_rates = _arm_rates(opt, consulted)
    standalone_rates = _arm_rates(opt, standalone)

    per_seed = []
    pooled_exc: list[int] = []
    pooled_dfc: list[int] = []
    for s in range(seeds):
        base_dec = [baseline_by_seed[s][i] for i in keep]
        base_rates = _arm_rates(opt, base_dec)
        exc_diffs = [c - b for c, b in zip(consulted_rates["_exc"], base_rates["_exc"], strict=True)]
        dfc_diffs = [c - b for c, b in zip(consulted_rates["_dfc"], base_rates["_dfc"], strict=True)]
        pooled_exc += exc_diffs
        pooled_dfc += dfc_diffs
        per_seed.append({
            "seed": s,
            "baseline": {k: v for k, v in base_rates.items() if not k.startswith("_")},
            "deltaExcess": round(sum(exc_diffs) / len(exc_diffs), 4) if exc_diffs else 0.0,
            "deltaDeficiency": round(sum(dfc_diffs) / len(dfc_diffs), 4) if dfc_diffs else 0.0,
        })

    delta = {
        "deltaExcess": round(sum(pooled_exc) / len(pooled_exc), 4) if pooled_exc else 0.0,
        "deltaExcessCI95": bootstrap_ci_paired(pooled_exc, seed=boot_seed),
        "deltaDeficiency": round(sum(pooled_dfc) / len(pooled_dfc), 4) if pooled_dfc else 0.0,
        "deltaDeficiencyCI95": bootstrap_ci_paired(pooled_dfc, seed=boot_seed),
        "mdeAtN": round(mde_at_n(len(keep), p0=0.5), 4),
        "pooledN": len(pooled_exc),
    }
    base_exc = round(sum(ps["baseline"]["excessErrorRate"] for ps in per_seed) / seeds, 4)
    base_dfc = round(sum(ps["baseline"]["deficiencyErrorRate"] for ps in per_seed) / seeds, 4)

    agr = labeled.get("agreement", {}).get("quadrant4class", {})
    # task_success_guardrail not measured here (needs a task harness) -> stays NO-GO.
    verdict = gate_verdict(baseline_is_real=True, judge_families=2, delta=delta,
                           task_success_guardrail_measured=False)
    if not labeled.get("groundTruthResolvable"):
        verdict["criticalFailures"].append(
            f"kappa_below_floor: quadrant Cohen kappa={agr.get('cohenKappa')} < {labeled.get('kappaFloor')} "
            "— the optimal-measure metric is not resolvable")
        verdict["verdict"], verdict["go"] = "NO-GO", False
    strip = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}  # noqa: E731

    return {
        "experimentId": "sophrosyne-measure-eval",
        "schema": "sophia.sophrosyne_measure_eval.v1",
        "status": "complete",
        "verdict": verdict["verdict"],
        "go": verdict["go"],
        "canClaimAGI": False,
        "claimCeiling": "candidate_only; canClaimAGI:false",
        "headline": (
            f"Δ(excess-error) = {delta['deltaExcess']} CI {delta['deltaExcessCI95']}, "
            f"Δ(deficiency-error) = {delta['deltaDeficiency']} CI {delta['deltaDeficiencyCI95']} "
            f"(consulted − no-gate baseline); verdict {verdict['verdict']}"
        ),
        "harness": "tools/run_sophrosyne_eval.py --model",
        "preregistration": "agi-proof/benchmark-results/sophrosyne/measurement_spec.json",
        "battery": {
            "file": "agi-proof/benchmark-results/sophrosyne/sophrosyne_external_battery.labeled.json",
            "nLabelled": labeled.get("n"), "nScored": len(keep),
            "droppedUnparseableBaseline": dropped,
            "scoredQuadrantCounts": labeled.get("scoredSet", {}).get("quadrantCounts"),
        },
        "judges": labeled.get("judges"),
        "groundTruth": {
            "rule": ">= 2 independent judge families; consensus quadrant; kappa floor "
                    f"{labeled.get('kappaFloor')}",
            "resolvable": labeled.get("groundTruthResolvable"),
            "quadrantCohenKappa": agr.get("cohenKappa"),
            "quadrantCohenKappaCI95": agr.get("cohenKappaCI95"),
            "quadrantGwetAC1": agr.get("gwetAC1"),
        },
        "subjectModel": subject_spec, "seeds": seeds, "temperature": temperature,
        "arms": {
            "no-gate-baseline": {
                "n": len(keep), "seeds": seeds,
                "excessErrorRate": base_exc, "deficiencyErrorRate": base_dfc,
                "perSeed": per_seed,
                "note": "the raw subject model decides proportionate/restrain/sustain/escalate (no gate).",
            },
            "sophrosyne-consulted": {**strip(consulted_rates),
                                     "note": "conscience_check(context={consultTemperance:True}).temperance verdict; deterministic, derives forces from raw text."},
            "sophrosyne-standalone": {**strip(standalone_rates),
                                      "note": "assess_temperance(text) with no context; deterministic, derives forces from raw text."},
        },
        "delta": delta,
        "criticalFailures": verdict["criticalFailures"],
        "boundary": verdict["boundary"],
        "honestLimits": [
            "Battery cases are author-generated templated raw-text dilemmas, not human work-transcripts; ground truth is independent-judge consensus.",
            "The gate arms DERIVE demand/expenditure/marginalValue from raw text (no explicit context), which the failure ledger documents as conservative. This run measures that real-use behaviour against a real baseline.",
            "The task-success guardrail (Δ success >= -0.02) is NOT measured here (needs a task harness); GO is therefore impossible from this tool alone — by design.",
        ],
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Sophrosyne three-arm temperance eval (deterministic mock + real-model GO/NO-GO)")
    ap.add_argument("--mock", choices=["profligate", "miserly", "oracle"], default=None,
                    help="score the gate arm against a deterministic mock baseline")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap CI seed (deterministic)")
    ap.add_argument("--emit-pending", action="store_true",
                    help="write the committed PENDING / NO-GO not-run artifact and exit")
    ap.add_argument("--model", default=None,
                    help="REAL no-gate baseline subject model spec (e.g. ollama:qwen2.5:7b-instruct); "
                         "scores all three arms on the labelled external battery and emits a GO/NO-GO receipt")
    ap.add_argument("--seeds", type=int, default=3, help="number of baseline sampling seeds (>=3 per the spec)")
    ap.add_argument("--temperature", type=float, default=0.7, help="baseline sampling temperature")
    ap.add_argument("--workers", type=int, default=8, help="concurrent subject calls")
    ap.add_argument("--write", action="store_true",
                    help="with --model: write the public report to sophrosyne-measure-eval.public-report.json")
    args = ap.parse_args(argv)

    if args.model:
        result = run_real(args.model, seeds=args.seeds, temperature=args.temperature,
                          workers=args.workers, boot_seed=args.seed)
        if args.write:
            PUBLIC_REPORT.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            print(f"wrote {PUBLIC_REPORT.relative_to(ROOT)}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    if args.emit_pending:
        path = emit_pending()
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"Wrote PENDING (not_run / NO-GO) artifact: {shown}")
        return 0
    if args.mock:
        result = run_mock(args.mock, seed=args.seed)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    ap.error("provide --mock {profligate,miserly,oracle}, --emit-pending, or --model")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
