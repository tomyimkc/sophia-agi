#!/usr/bin/env python3
"""Generate the public RESULTS.md from the curated published-results.json.

RESULTS.md is the public face of the benchmark. It is GENERATED — never edit it
by hand; edit `agi-proof/benchmark-results/published-results.json` and re-run.
Only figures that cleared the no-overclaim gate appear as VALIDATED; everything
else is clearly marked illustrative.

    python tools/build_results_page.py            # write RESULTS.md
    python tools/build_results_page.py --check     # CI: fail if RESULTS.md is stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "agi-proof" / "benchmark-results" / "published-results.json"
OUT = ROOT / "RESULTS.md"


def _pct(x) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "—"


def render(doc: dict) -> str:
    L: list[str] = []
    L += [
        "# Provenance Delta — public results",
        "",
        "<!-- GENERATED from agi-proof/benchmark-results/published-results.json by",
        "     tools/build_results_page.py — do not edit by hand. -->",
        "",
        f"_Last updated: {doc.get('lastUpdated', 'n/a')}_",
        "",
        "**No-overclaim gate.** A number is **VALIDATED** only with "
        "≥2 independent judges in consensus (judge ≠ subject), reported inter-judge "
        "agreement, ≥3 runs, and confidence intervals. Everything else is "
        "**illustrative** and labelled. Hidden-eval prompts are never published — "
        "only aggregates. See [SECURITY.md](SECURITY.md) and "
        "[methodology](docs/11-Platform/Provenance-Delta.md).",
        "",
        "## Validated results",
        "",
    ]
    validated = doc.get("validated") or []
    if not validated:
        L += [
            "_None yet._ No run has cleared the gate (multi-judge consensus + "
            "agreement + ≥3 runs + CIs). This is intentional and honest: see the "
            "illustrative section and the audit below for why a single judge is not "
            "enough.",
            "",
        ]
    else:
        L += ["| Model | Judges | Agreement | Runs | Halluc. alone | Halluc. gated | Δ (95% CI) | FP cost | Coverage |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in validated:
            ci = r.get("deltaCI")
            d = _pct(r.get("delta")) + (f" [{_pct(ci[0])}, {_pct(ci[1])}]" if ci else "")
            L.append(
                f"| {r.get('model')} | {r.get('judge')} | {_pct(r.get('agreement'))} | "
                f"{r.get('runs')} | {_pct(r.get('hallucinationAlone'))} | "
                f"{_pct(r.get('hallucinationGated'))} | {d} | "
                f"{_pct(r.get('falsePositiveCost'))} | {_pct(r.get('coverage'))} |"
            )
        L.append("")

    L += ["## Illustrative only (not headline-grade)", ""]
    for r in doc.get("illustrative") or []:
        ci = r.get("deltaCI")
        d = _pct(r.get("delta")) + (f" [{_pct(ci[0])}, {_pct(ci[1])}]" if ci else "")
        L += [
            f"### `{r.get('model')}` — {r.get('modelNote', '')}",
            "",
            f"- Judge: **{r.get('judge')}** · runs: {r.get('runs')} · false cases: {r.get('falseCases')}",
            f"- Hallucination alone **{_pct(r.get('hallucinationAlone'))}** → gated "
            f"**{_pct(r.get('hallucinationGated'))}** · Δ **{d}**",
            f"- False-positive cost {_pct(r.get('falsePositiveCost'))} · gate coverage {_pct(r.get('coverage'))}",
            f"- ⚠ {r.get('caveat', '')}",
            "",
        ]

    external = doc.get("externalEvals") or []
    if external:
        L += [
            "## External-oracle evals (base-model accuracy via the harness)",
            "",
            "Scored by **exact-match against external gold** (no LLM judge). These "
            "report the **base model's** accuracy through Sophia's external-eval "
            "harness and validate the harness end-to-end — they are **not** claims "
            "about Sophia's provenance gate or any Sophia-specific capability.",
            "",
            "| Dataset | Model | N | Accuracy | Date |",
            "|---|---|---|---|---|",
        ]
        for e in external:
            L.append(
                f"| {e.get('dataset')} | `{e.get('model')}` | {e.get('n')} | "
                f"{_pct(e.get('accuracy'))} | {e.get('date', '—')} |"
            )
        L.append("")
        for e in external:
            if e.get("note"):
                L += [f"- _{e.get('dataset')}:_ {e['note']}"]
        L.append("")

    audit = doc.get("audit")
    if audit:
        L += ["## Judge audit (why the gate matters)", "", audit.get("note", ""), ""]
        if audit.get("robustRegardlessOfJudge"):
            L.append("Robust regardless of judge:")
            L += [f"- {x}" for x in audit["robustRegardlessOfJudge"]]
            L.append("")

    L += [
        "## Reproduce",
        "",
        "```bash",
        "python tools/run_provenance_delta.py --models mock            # offline plumbing",
        "python tools/run_provenance_delta.py --models <subject> \\",
        "    --judges <judgeA>,<judgeB> --runs 3                       # validated-grade run",
        "```",
        "",
        "Offline tests run in CI on every commit. Real-model numbers are produced "
        "locally by the maintainer and curated into "
        "`agi-proof/benchmark-results/published-results.json`.",
    ]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if RESULTS.md is stale")
    args = ap.parse_args(argv)

    doc = json.loads(SRC.read_text(encoding="utf-8"))
    rendered = render(doc)

    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != rendered:
            print("RESULTS.md is stale — run: python tools/build_results_page.py", file=sys.stderr)
            return 1
        print("RESULTS.md is up to date.")
        return 0

    OUT.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
