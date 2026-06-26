#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Third-party reproducer for Sophia's source-discipline uplift claim (SEIB-100).

One command. An outside reviewer runs this against their OWN model/API and gets a
PASS/FAIL verdict computed LIVE against a hash-pinned, pre-registered acceptance
threshold — trusting none of this repo's committed result artifacts.

Trust-minimization:
  1. Pins the eval pack by sha256 against the pre-registration (tamper check).
  2. Audits decontamination: no eval prompt may appear in the training corpus.
  3. Runs SEIB-100 raw-vs-sophia_full LIVE with the reviewer's model.
  4. RECOMPUTES the verdict from the live per-case rows against the pre-registered
     threshold (does NOT trust the runner's internal `ok`), including a paired
     bootstrap CI on the accuracy delta.
  5. Prints the pre-registration's own sha256 so the reviewer can confirm it
     predates the result, and emits an aggregate (no-prompts) report.

A PASS confirms the gate's provenance uplift ON THIS PACK ONLY. It is not an AGI
claim, not external generalization beyond this pack, not a hallucination guarantee.

    # real, validating run (reviewer's own model; >=3 runs):
    python3 tools/run_external_validation.py --model openrouter:openai/gpt-4o-mini --runs 3
    # offline plumbing smoke (clearly marked NON-VALIDATING):
    python3 tools/run_external_validation.py --allow-mock
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.dataset_guard import _load_jsonl, check_contamination, normalize  # noqa: E402
from tools.run_seib import run as run_seib  # noqa: E402

DEFAULT_PREREG = "agi-proof/external-validation/seib-uplift.preregistration.json"
# Committed, human-authored training sources that feed the (gitignored) MLX pack — so a
# clean clone can run the decontam audit without first rebuilding the pack.
DEFAULT_TRAINING_GLOBS = (
    "training/council/traces.jsonl",
    "training/local_sophia_v2/*.jsonl",
    "training/lora/*.jsonl",
    "training/corpus.jsonl",
    "training/moral_gate_sft.jsonl",
    "training/feedback/*.jsonl",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decontam_audit(pack_path: Path, training_globs: tuple[str, ...]) -> dict[str, Any]:
    """Independently check that no SEIB prompt leaks into the training corpus.

    Leaking eval prompts into training is the #1 way to fake uplift, so this runs
    even though the build-time guard also checks it — a reviewer should not have to
    trust that the build ran.
    """
    eval_prompts = {normalize(c["prompt"]) for c in _load_jsonl(pack_path) if c.get("prompt")}
    train_rows: list[dict] = []
    for g in training_globs:
        for p in sorted(ROOT.glob(g)):
            train_rows.extend(_load_jsonl(p))
    res = check_contamination(train_rows, eval_prompts)
    return {
        "trainingGlobs": list(training_globs),
        "nTrainRows": res["nTrain"],
        "nEvalPrompts": res["nEval"],
        "overlapCount": len(res["overlap"]),
        "clean": res["clean"],
        "evaluable": train_rows != [],
    }


def paired_delta_ci(rows: list[dict], *, cond_a: str = "raw", cond_b: str = "sophia_full",
                    bootstrap: int = 2000, seed: int = 12345) -> dict[str, Any] | None:
    """Paired bootstrap 95% CI on (acc[cond_b] - acc[cond_a]) over cases.

    Resamples CASE IDS (not condition rows) so raw and full are compared on the same
    case in every resample — the honest paired statistic for "does the gate help".
    """
    per_case: dict[str, dict[str, list[float]]] = defaultdict(lambda: {cond_a: [], cond_b: []})
    for r in rows:
        c = r.get("condition")
        if c in (cond_a, cond_b):
            per_case[r["id"]][c].append(1.0 if r.get("score", {}).get("correct") else 0.0)
    ids = [i for i, v in per_case.items() if v[cond_a] and v[cond_b]]
    if not ids:
        return None
    delta = {i: (sum(per_case[i][cond_b]) / len(per_case[i][cond_b]))
                - (sum(per_case[i][cond_a]) / len(per_case[i][cond_a])) for i in ids}
    rnd = random.Random(seed)
    n = len(ids)
    samples = []
    for _ in range(bootstrap):
        samples.append(sum(delta[ids[rnd.randrange(n)]] for _ in range(n)) / n)
    samples.sort()
    return {
        "point": round(sum(delta.values()) / n, 4),
        "ci95": [round(samples[int(0.025 * bootstrap)], 4),
                 round(samples[min(int(0.975 * bootstrap), bootstrap - 1)], 4)],
        "nCases": n,
        "bootstrap": bootstrap,
    }


def recompute_verdict(report: dict[str, Any], prereg: dict[str, Any], *, ci: dict[str, Any] | None,
                      decontam: dict[str, Any], pack_sha: str, allow_mock: bool) -> dict[str, Any]:
    """Apply the PRE-REGISTERED threshold to the LIVE metrics. Independent of report['ok']."""
    acc = prereg["acceptance"]
    full = report["byCondition"]["sophia_full"]
    deltas = report["deltas"]
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    add("packSha256Matches", pack_sha == prereg["evalPack"]["sha256"],
        f"{pack_sha} vs registered {prereg['evalPack']['sha256']}")
    add("minCases", report["nCases"] >= acc["minCases"], f"{report['nCases']} >= {acc['minCases']}")
    real = bool(report.get("realModelRun"))
    add("realModel", real or allow_mock, f"realModelRun={real} (allowMock={allow_mock})")
    add("minRuns", report.get("runs", 1) >= acc["minRuns"] or allow_mock,
        f"runs={report.get('runs', 1)} >= {acc['minRuns']}")
    add("falseAttributionRate", full["falseAttributionRate"] <= acc["maxFalseAttributionRate"],
        f"{full['falseAttributionRate']} <= {acc['maxFalseAttributionRate']}")
    add("fabricationRateOnContested", full["fabricationRateOnContested"] <= acc["maxFabricationRateOnContested"],
        f"{full['fabricationRateOnContested']} <= {acc['maxFabricationRateOnContested']}")
    add("accuracyDelta", deltas["raw_to_full_accuracy_delta"] > acc["minAccuracyDelta"],
        f"{deltas['raw_to_full_accuracy_delta']} > {acc['minAccuracyDelta']}")
    if acc.get("requireAccuracyDeltaCIExcludesZero"):
        ci_ok = ci is not None and ci["ci95"][0] > 0
        add("accuracyDeltaCIExcludesZero", ci_ok, f"ci95={ci['ci95'] if ci else None}")
    add("falsePositiveCost", full["falsePositiveCost"] <= acc["maxFalsePositiveCost"],
        f"{full['falsePositiveCost']} <= {acc['maxFalsePositiveCost']}")
    add("decontamination", decontam["evaluable"] and decontam["overlapCount"] <= acc["maxTrainingOverlap"],
        f"trainingOverlap={decontam['overlapCount']} <= {acc['maxTrainingOverlap']} "
        f"(audited {decontam['nTrainRows']} rows)" if decontam["evaluable"]
        else "NO training rows found — point --training-glob at the corpus (an empty audit is not 'clean')")

    passed = all(c["pass"] for c in checks)
    # "Validating" is stricter than "pass": a mock or single-run pass exercises the
    # plumbing but cannot count as evidence.
    validating = bool(real and report.get("runs", 1) >= acc["minRuns"]
                      and decontam["evaluable"] and decontam["overlapCount"] == 0
                      and pack_sha == prereg["evalPack"]["sha256"])
    verdict = "PASS" if passed else "FAIL"
    return {"verdict": verdict, "validating": validating, "checks": checks}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Third-party reproducer for the SEIB-100 uplift claim.")
    ap.add_argument("--preregistration", default=DEFAULT_PREREG)
    ap.add_argument("--model", default="mock", help="reviewer's model, e.g. openrouter:openai/gpt-4o-mini or mlx:Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=None, help="local MLX LoRA adapter dir (with --model mlx:<base>)")
    ap.add_argument("--runs", type=int, default=1, help="runs per case (the registration requires >=3 for a validating result)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--training-glob", action="append", default=[],
                    help="training packs to decontam-audit against (default: the MLX pack)")
    ap.add_argument("--allow-mock", action="store_true", help="permit a NON-VALIDATING offline plumbing smoke")
    ap.add_argument("--out", default="agi-proof/external-validation/seib-uplift.validation-report.json")
    args = ap.parse_args(argv)

    prereg_path = ROOT / args.preregistration
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    prereg_sha = sha256_file(prereg_path)
    pack_path = ROOT / prereg["evalPack"]["path"]
    pack_sha = sha256_file(pack_path)

    if args.model == "mock" and not args.allow_mock:
        raise SystemExit("refusing to run with --model mock without --allow-mock; pass a real --model for a validating result")

    training_globs = tuple(args.training_glob) or DEFAULT_TRAINING_GLOBS
    decontam = decontam_audit(pack_path, training_globs)

    # Live run. SEIB writes its own raw artifact (with per-case answers) to a TEMP file
    # outside the repo tree; we recompute the verdict ourselves from the returned rows.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        report = run_seib(str(pack_path), str(Path(tmp) / "seib-live.raw.json"),
                          real_model=(args.model != "mock"), model=args.model,
                          adapter=args.adapter, limit=0, runs=max(1, args.runs))

    ci = paired_delta_ci(report.get("rows", []), bootstrap=args.bootstrap)
    result = recompute_verdict(report, prereg, ci=ci, decontam=decontam, pack_sha=pack_sha,
                               allow_mock=args.allow_mock)

    out_report = {
        "schema": "sophia.external_validation.report.v1",
        "visibility": "public-aggregate-no-prompts",
        "preregistration": {"path": args.preregistration, "sha256": prereg_sha, "id": prereg.get("id")},
        "evalPack": {"path": prereg["evalPack"]["path"], "sha256": pack_sha,
                     "sha256Matches": pack_sha == prereg["evalPack"]["sha256"]},
        "model": args.model,
        "adapter": args.adapter,
        "runs": report.get("runs"),
        "realModelRun": bool(report.get("realModelRun")),
        "decontamination": decontam,
        "metrics": {
            "raw": report["byCondition"]["raw"],
            "sophia_full": report["byCondition"]["sophia_full"],
            "deltas": report["deltas"],
            "accuracyDeltaCI": ci,
        },
        "runnerOk": report.get("ok"),
        "recomputedVerdict": result["verdict"],
        "validating": result["validating"],
        "verdictAgreesWithRunner": (result["verdict"] == "PASS") == bool(report.get("ok")),
        "checks": result["checks"],
        "claimBoundary": prereg["claimBoundary"],
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"preregistration:  {args.preregistration}  sha256={prereg_sha}")
    print(f"eval pack:        {prereg['evalPack']['path']}  sha256={pack_sha}  matches={out_report['evalPack']['sha256Matches']}")
    print(f"model / runs:     {args.model}  runs={report.get('runs')}  realModel={out_report['realModelRun']}")
    print(f"decontamination:  overlap={decontam['overlapCount']}  (audited {decontam['nTrainRows']} train rows)")
    print(f"accuracy delta:   {report['deltas']['raw_to_full_accuracy_delta']:+.4f}  CI95={ci['ci95'] if ci else None}")
    print(f"false attr / fab: {report['byCondition']['sophia_full']['falseAttributionRate']} / {report['byCondition']['sophia_full']['fabricationRateOnContested']}  fpCost={report['byCondition']['sophia_full']['falsePositiveCost']}")
    for c in result["checks"]:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['check']}: {c['detail']}")
    label = result["verdict"] + ("" if result["validating"] else "  (NON-VALIDATING — mock/single-run/decontam/hash gap)")
    print(f"VERDICT:          {label}")
    print(f"wrote {args.out}")

    return 0 if result["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
