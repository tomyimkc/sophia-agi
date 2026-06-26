#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""SWE-bench Verified runner — Sophia produces patches; the OFFICIAL harness grades.

Hurdle 1 (external/independent validation). SWE-bench is the highest-value external
benchmark for this project because its verifier IS Sophia's verifier philosophy: a
task is resolved iff the project's real unit tests (FAIL_TO_PASS / PASS_TO_PASS) pass
after the patch. The grading is therefore EXTERNAL ground truth, run by the official
`swebench` harness — never by Sophia. This tool owns only the model-dependent and
bookkeeping halves:

  1. ``generate`` — for each instance, prompt the model (base / sophia-full / adapter)
     and extract a unified-diff patch, writing predictions in the official SWE-bench
     format ({instance_id, model_name_or_path, model_patch}).
  2. (hand off) ``python -m swebench.harness.run_evaluation`` grades the predictions in
     Docker against the real tests and writes its own report — the external oracle.
  3. ``parse`` — read that official report and write a no-overclaim repo artifact
     (resolved%, per-instance, decontamination note, claim boundary).

HONEST BOUNDS (state them in any report):
  - The committed solver is a MINIMAL scaffold (problem statement -> model -> diff). It
    has no repo navigation/retrieval, so resolved% reflects scaffold+model and is a
    FLOOR; improving the solver improves the number. The grading is what's defensible.
  - Grading needs Docker and the `swebench` package. The official harness historically
    targets x86_64; on Apple Silicon use arm64 images where available or run the
    EVALUATION step on a Linux/x86 host. Patch GENERATION runs anywhere.
  - SWE-bench may overlap a base model's pretraining; report the DELTA vs base (which
    cancels shared contamination), not just the absolute. Not validated until ≥3 runs
    + the base/base+tools ablation + the decontamination caveat are all stated.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLE = ROOT / "eval" / "external" / "swebench-style-sample.jsonl"
DEFAULT_OUT_DIR = ROOT / "agi-proof" / "external-benchmarks"

PROMPT_TEMPLATE = (
    "You are fixing a real software issue. Read the problem and return ONLY a unified "
    "diff patch (git diff format) that resolves it. Do not explain.\n\n"
    "Repository: {repo}\nBase commit: {base_commit}\n\n"
    "Problem statement:\n{problem_statement}\n\n"
    "{hints}"
    "Return the patch inside a ```diff code block."
)


def load_instances(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_prompt(instance: dict, *, max_problem_chars: int = 12000) -> str:
    problem = str(instance.get("problem_statement", ""))[:max_problem_chars]
    hints = instance.get("hints_text") or ""
    hints_block = f"Hints:\n{str(hints)[:2000]}\n\n" if hints else ""
    return PROMPT_TEMPLATE.format(
        repo=instance.get("repo", "?"),
        base_commit=instance.get("base_commit", "?"),
        problem_statement=problem,
        hints=hints_block,
    )


def extract_patch(text: str) -> str:
    """Pull a unified diff from model output. Prefers a ```diff fence, then a raw
    'diff --git' / '--- a/' block. Returns '' if none — an empty patch is a legitimate
    (unresolved) prediction, never a crash."""
    if not text:
        return ""
    fence = re.search(r"```(?:diff|patch)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        body = fence.group(1).strip("\n")
        if "diff --git" in body or body.lstrip().startswith(("--- ", "diff ")):
            return body + ("\n" if not body.endswith("\n") else "")
    idx = text.find("diff --git")
    if idx == -1:
        m = re.search(r"^--- a/", text, re.MULTILINE)
        idx = m.start() if m else -1
    if idx == -1:
        return ""
    body = text[idx:].rstrip()
    # strip a trailing code fence if present
    body = re.sub(r"\n```.*$", "", body, flags=re.DOTALL)
    return body + "\n"


def to_prediction(instance_id: str, model_name: str, patch: str) -> dict:
    return {"instance_id": instance_id, "model_name_or_path": model_name, "model_patch": patch}


def generate_predictions(
    instances: list[dict], solver: Callable[[dict], str], *, model_name: str
) -> list[dict]:
    preds = []
    for inst in instances:
        out = solver(inst)
        patch = extract_patch(out)
        preds.append(to_prediction(str(inst["instance_id"]), model_name, patch))
    return preds


def parse_official_report(report: dict, *, system: str, model: str, decontam_note: str | None = None) -> dict:
    """Turn the official swebench evaluation report into the repo's no-overclaim artifact.

    Accepts the official summary keys (``resolved_ids`` / ``unresolved_ids`` / counts).
    Resolved% is computed from the official numbers — Sophia never scores itself here.
    """
    resolved_ids = list(report.get("resolved_ids", []))
    unresolved_ids = list(report.get("unresolved_ids", []))
    error_ids = list(report.get("error_ids", []))
    empty_ids = list(report.get("empty_patch_ids", []))
    total = int(report.get("total_instances")
                or report.get("submitted_instances")
                or (len(resolved_ids) + len(unresolved_ids) + len(error_ids) + len(empty_ids)))
    resolved = int(report.get("resolved_instances", len(resolved_ids)))
    rate = round(resolved / total, 4) if total else 0.0
    return {
        "schema": "sophia.external_benchmark.v1",
        "benchmark": "swebench-verified",
        "system": system,
        "model": model,
        "gradedBy": "official swebench.harness.run_evaluation (Docker, real FAIL_TO_PASS/PASS_TO_PASS tests)",
        "oracle": "external — project unit tests, not the Sophia gate",
        "total": total,
        "resolved": resolved,
        "resolvedRate": rate,
        "resolvedIds": resolved_ids,
        "unresolvedIds": unresolved_ids,
        "errorIds": error_ids,
        "emptyPatchIds": empty_ids,
        "candidateOnly": True,
        "level3Evidence": False,
        "decontamination": decontam_note or ("SWE-bench may overlap base-model pretraining; report the "
                                             "DELTA vs base, which cancels shared contamination."),
        "claimBoundary": ("First-party EXECUTION of an external benchmark with external ground truth. "
                          "Not validated until >=3 runs + base/base+tools ablation + (for an independent "
                          "claim) third-party reproduction. Not an AGI claim."),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _mock_solver(instance: dict) -> str:
    # Offline plumbing only: echo the gold patch so the predictions pipeline is testable
    # without a model. Proves the harness, NOT the model.
    return f"```diff\n{instance.get('patch', '')}\n```"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", default=str(SAMPLE),
                    help="JSONL of SWE-bench instances (instance_id, repo, base_commit, problem_statement, ...)")
    ap.add_argument("--model", default=None, help="model spec for patch generation; omit for offline mock plumbing")
    ap.add_argument("--system", default="sophia-full", choices=["base", "sophia-full", "adapter"],
                    help="which system is producing patches (labels the artifact)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--predictions-out", default="agi-proof/external-benchmarks/swebench-predictions.jsonl")
    ap.add_argument("--report", default=None,
                    help="parse an OFFICIAL swebench evaluation report JSON into the repo artifact")
    ap.add_argument("--out", default=None, help="artifact path (defaults by system)")
    ap.add_argument("--dry-run", action="store_true", help="validate wiring; no model, no writes")
    args = ap.parse_args(argv)

    out = Path(args.out) if args.out else DEFAULT_OUT_DIR / f"swebench-verified-{args.system}.json"

    # PARSE MODE: official report -> repo artifact (the defensible number lands here).
    if args.report:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        artifact = parse_official_report(report, system=args.system, model=args.model or "unknown")
        _write_json(out, artifact)
        print(f"resolved {artifact['resolved']}/{artifact['total']} = {artifact['resolvedRate']:.1%}  ({args.system})")
        print(f"wrote {out}")
        return 0

    # GENERATE MODE: produce predictions in the official format.
    instances = load_instances(Path(args.instances))
    if args.limit:
        instances = instances[: args.limit]
    is_sample = Path(args.instances).resolve() == SAMPLE.resolve()

    if args.dry_run:
        for inst in instances:
            assert inst.get("instance_id"), "instance missing instance_id"
            build_prompt(inst)  # must not raise
        print(f"wiring OK :: {len(instances)} instance(s){'  [STYLE SAMPLE — not real SWE-bench]' if is_sample else ''}")
        print("next: --model <spec> to write predictions, then hand off to the official harness:")
        print("  python -m swebench.harness.run_evaluation --predictions_path <preds.jsonl> \\")
        print("    --dataset_name princeton-nlp/SWE-bench_Verified --run_id sophia-<system> --max_workers 4")
        print("  then: tools/run_swebench.py --report <official_report.json> --system <system>")
        return 0

    if args.model:
        from agent.model import default_client

        client = default_client(args.model)
        solver = lambda inst: getattr(client.generate("", build_prompt(inst)), "text", "") or ""
        model_name = f"sophia-{args.system}:{args.model}"
    else:
        solver, model_name = _mock_solver, "mock-plumbing"

    preds = generate_predictions(instances, solver, model_name=model_name)
    _write_jsonl(Path(args.predictions_out), preds)
    nonempty = sum(1 for p in preds if p["model_patch"].strip())
    print(f"wrote {len(preds)} prediction(s) ({nonempty} non-empty) -> {args.predictions_out}")
    if is_sample:
        print("NOTE: style sample — fetch princeton-nlp/SWE-bench_Verified for a citable run.")
    print("Grade with the official harness (Docker), then: "
          "tools/run_swebench.py --report <official_report.json> --system " + args.system)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
