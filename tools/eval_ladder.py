#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""The Sophia local-model evaluation ladder.

Compares the rungs the no-overclaim promotion rule needs:
    base · base+gate · adapter · adapter+gate
and records status so uplift is measured against a stored baseline, never vibes.

``--dry-run`` verifies wiring without loading weights. Real rungs require model
weights/dependencies on local hardware (Mac/MLX or GPU). Missing dependencies are
recorded as blockers rather than promoted as results.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HF_EVAL = "tools/eval_local_model.py"
MLX_EVAL = "tools/eval_mlx_model.py"
DOMAINS = ["philosophy", "psychology", "history", "religion"]
OUT = ROOT / "training" / "local_sophia_v2"
MANIFEST = OUT / "manifest.json"


def _slug(name: str) -> str:
    return name.replace("/", "-").replace(" ", "-").lower()


def _rungs(model: str, adapter: str | None, backend: str) -> list[tuple[str, list[str], str]]:
    py = sys.executable or "python3"
    eval_script = MLX_EVAL if backend == "mlx" else HF_EVAL
    base = [py, eval_script, "--model", model, "--domains", *DOMAINS]
    label = _slug(model)
    rungs = [("base", base, label), ("base+gate", base + ["--with-gate"], label)]
    if adapter:
        adapter_name = Path(adapter).name
        adapter_label = _slug(adapter_name)
        a = [py, eval_script, "--model", model, "--adapter", adapter, "--domains", *DOMAINS]
        rungs += [("adapter", a, adapter_label), ("adapter+gate", a + ["--with-gate"], adapter_label)]
    return rungs


def _domain_report_paths(label: str) -> list[Path]:
    return [ROOT / "benchmark" / "model_runs" / f"local-{label}-{domain}.report.json" for domain in DOMAINS]


def _load_reports(label: str) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in _domain_report_paths(label):
        if path.exists():
            try:
                reports.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
    return reports


def _channel_block(report: dict[str, Any]) -> dict[str, Any]:
    """Extract FORMAT / CONTENT / COMBINED headline block from a domain report."""
    total = int(report.get("total", 0))
    fmt = int(report.get("formatPassed", report.get("passed", 0)))
    content = int(report.get("contentPassed", report.get("passed", 0)))
    combined = int(report.get("passed", 0))
    pct = lambda n: round(100.0 * n / total, 1) if total else 0.0
    block = {
        "format": {"passed": fmt, "total": total, "score_pct": report.get("formatPct", pct(fmt))},
        "content": {"passed": content, "total": total, "score_pct": report.get("contentPct", pct(content))},
        "combined": {"passed": combined, "total": total, "score_pct": report.get("score_pct", pct(combined))},
        # Legacy combined fields for continuity
        "passed": combined,
        "total": total,
        "score_pct": report.get("score_pct", pct(combined)),
    }
    if "gateFailures" in report:
        block["gateFailures"] = report["gateFailures"]
    return block


def _summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not reports:
        return None
    domains = {r.get("domain", "unknown"): _channel_block(r) for r in reports}
    fmt_passed = sum(d["format"]["passed"] for d in domains.values())
    content_passed = sum(d["content"]["passed"] for d in domains.values())
    combined_passed = sum(d["combined"]["passed"] for d in domains.values())
    total = sum(d["total"] for d in domains.values())
    pct = lambda n: round(100.0 * n / total, 1) if total else 0.0
    gate_failures = sum(
        int(r.get("gateFailures", 0)) for r in reports if "gateFailures" in r
    )
    summary: dict[str, Any] = {
        "domains": domains,
        "channels": {
            "format": {"passed": fmt_passed, "total": total, "score_pct": pct(fmt_passed)},
            "content": {"passed": content_passed, "total": total, "score_pct": pct(content_passed)},
            "combined": {"passed": combined_passed, "total": total, "score_pct": pct(combined_passed)},
        },
        "passed": combined_passed,
        "total": total,
        "score_pct": pct(combined_passed),
    }
    if any("gateFailures" in r for r in reports):
        summary["gateFailures"] = gate_failures
    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _update_manifest_baseline(report: dict[str, Any]) -> None:
    if not MANIFEST.exists():
        return
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    manifest["baseline"] = report
    _write_json(MANIFEST, manifest)


def _run_rung(name: str, cmd: list[str], label: str) -> dict[str, Any]:
    print(f"[{name}] running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    reports = _load_reports(label)
    summary = _summarize_reports(reports)
    status = "complete" if summary else "blocked_or_failed"
    return {
        "rung": name,
        "command": cmd,
        "returncode": proc.returncode,
        "status": status,
        "summary": summary,
        "stdoutTail": proc.stdout[-1200:] if proc.stdout else "",
        "stderrTail": proc.stderr[-1200:] if proc.stderr else "",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--backend", choices=["hf", "mlx"], default="hf", help="Evaluation backend: hf/PEFT or mlx/MLX-LM")
    ap.add_argument("--dry-run", action="store_true", help="verify wiring only (no weights)")
    args = ap.parse_args(argv)

    rungs = _rungs(args.model, args.adapter, args.backend)
    print("Sophia eval ladder — promotion rule: improve provenance/citation at "
          "acceptable false-positive cost (no useful-correctness regression).\n")
    if args.dry_run:
        for name, cmd, _label in rungs:
            rc = subprocess.run(cmd + ["--dry-run"], cwd=ROOT).returncode
            print(f"[{name}] wiring {'OK' if rc == 0 else 'FAIL'} :: {' '.join(cmd)} --dry-run")
            if rc != 0:
                return 1
        print("\nAlso run on real weights: tools/run_seib.py --real-model --model <m>; "
              "run_all_phase_benchmarks.py; run_council_uplift.py; run_moral_public_standard_eval.py")
        return 0

    payload = {
        "schema": "sophia.eval_ladder.v2",
        "headline": "FORMAT / CONTENT / COMBINED per suite; protected gate uses CONTENT channel.",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "candidateOnly": True,
        "level3Evidence": False,
        "model": args.model,
        "adapter": args.adapter,
        "backend": args.backend,
        "claimBoundary": "Local-model ladder status/results; not AGI proof. External gates enforce correctness.",
        "promotionRule": "promote only if provenance/citation improves at acceptable false-positive cost.",
        "rungs": [],
    }
    for name, cmd, label in rungs:
        payload["rungs"].append(_run_rung(name, cmd, label))

    # Baseline is the no-adapter ladder. Adapter runs are separate reports.
    out_name = "eval_ladder_baseline.json" if not args.adapter else "eval_ladder_adapter.json"
    _write_json(OUT / out_name, payload)
    if not args.adapter:
        _update_manifest_baseline(payload)
        print(f"\nRecorded baseline into {MANIFEST}")
    print(f"Wrote {OUT / out_name}")
    # Do not fail merely because the model scored imperfectly; fail only if every rung lacked reports.
    return 0 if any(r["summary"] for r in payload["rungs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
