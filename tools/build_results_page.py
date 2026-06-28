#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
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

    verifier_evals = doc.get("verifierEvals") or []
    if verifier_evals:
        L += [
            "## Verifier evals (objective accuracy of a Sophia verifier)",
            "",
            "Scored by **exact-match against ground-truth labels** with a "
            "**deterministic verifier** (no LLM judge). Unlike the provenance-delta "
            "rows, these measure a **machine-checked gate's** accuracy directly, so "
            "they need no multi-judge consensus — but they are honestly bounded by "
            "small, constructed benchmarks and are **not** headline capability claims.",
            "",
            "| Verifier | Benchmark | N | Accuracy | Fabrication recall | False-alarm | Date |",
            "|---|---|---|---|---|---|---|",
        ]
        for e in verifier_evals:
            L.append(
                f"| `{e.get('verifier')}` | {e.get('benchmark')} | {e.get('n')} | "
                f"{_pct(e.get('accuracy'))} | {_pct(e.get('fabricationDetectionRecall'))} | "
                f"{_pct(e.get('falseAlarmRate'))} | {e.get('date', '—')} |"
            )
        L.append("")
        for e in verifier_evals:
            if e.get("note"):
                L += [f"- _{e.get('verifier')}:_ {e['note']}"]
        L.append("")

    calibration_evals = doc.get("calibrationEvals") or []
    if calibration_evals:
        L += [
            "## Calibration evals (abstention vs fabrication, deterministic)",
            "",
            "Scored by a **deterministic marker-based scorer** (no LLM judge) that rewards "
            "honest abstention on genuinely-unknown questions and scores a confident "
            "fabricated specific 0. Validated by **≥3 runs with a 95% CI excluding zero**. "
            "Honestly bounded: the scorer and pack are **self-authored** (internally valid "
            "cross-mode deltas; a third-party audit of the labels/markers — and human "
            "semantic review — would harden these to headline grade).",
            "",
            "| Method | Baseline | Pack (runs) | Calibration Δ (95% CI) | Fabrication reduction (95% CI) | Method fab-rate | Date |",
            "|---|---|---|---|---|---|---|",
        ]
        for e in calibration_evals:
            cd, fr = e.get("calibrationDelta", {}), e.get("fabricationReduction", {})
            L.append(
                f"| {e.get('method')} | {e.get('baseline')} | {e.get('pack')} ({e.get('runs')}) | "
                f"{_pct(cd.get('mean'))} [{_pct(cd.get('ciLow'))}, {_pct(cd.get('ciHigh'))}] | "
                f"{_pct(fr.get('mean'))} [{_pct(fr.get('ciLow'))}, {_pct(fr.get('ciHigh'))}] | "
                f"{_pct(e.get('methodFabricationRate'))} | {e.get('date', '—')} |"
            )
        L.append("")
        for e in calibration_evals:
            if e.get("note"):
                L += [f"- _{e.get('method')} vs {e.get('baseline')}:_ {e['note']}"]
        L.append("")

    ext_cal = doc.get("externalBenchmarkCalibration")
    if ext_cal and ext_cal.get("rows"):
        status = "**VALIDATED**" if ext_cal.get("validated") else "recorded but **not validated**"
        L += [
            "## External-benchmark calibration (selective prediction) — " + status,
            "",
            f"On a **public, human-authored, external** benchmark ({ext_cal.get('benchmark')}), "
            "graded by " + str(ext_cal.get("graders")) + ". The first Sophia calibration result "
            "validated on **non-self-authored** data — the selective-accuracy lift's 95% CI excludes "
            "zero on two independent subject models. A **calibration / selective-prediction** result, "
            "**not** an AGI claim.",
            "",
            "| Subject | Dataset | N (attempted) | Signal | AUROC | Selective-acc lift @20% cov (95% CI) | Inter-grader κ | Date |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in ext_cal["rows"]:
            lift = r.get("liftAt20Coverage", {})
            L.append(
                f"| {r.get('subject')} | {r.get('dataset')} | {r.get('n')} ({r.get('attempted')}) | "
                f"{r.get('signal')} | {r.get('auroc')} | "
                f"+{_pct(lift.get('mean'))} [{_pct(lift.get('ciLow'))}, {_pct(lift.get('ciHigh'))}] | "
                f"{r.get('kappa')} | {r.get('date', '—')} |"
            )
        L.append("")
        if ext_cal.get("note"):
            L += [f"- {ext_cal['note']}", ""]

    semantic = doc.get("semanticEvals")
    if semantic:
        _v = semantic.get("validated")
        status = ("**VALIDATED**" if _v is True
                  else "recorded but **not validated**" if _v is False
                  else "_None yet._")
        L += [
            "## Semantic evals (model-judged, gated)",
            "",
            "Judging whether a holding *supports* a proposition is a model call, so "
            "these are held to the no-overclaim gate (multi-judge + agreement + runs "
            "+ CIs). A single judge is illustrative, never a headline.",
            "",
            f"- Benchmark: {semantic.get('benchmark')}",
            f"- Gate: {semantic.get('gate')}",
        ]
        res = semantic.get("result")
        if _v is True and res:
            L.append(
                f"- Validated result: {status} — consensus accuracy "
                f"**{_pct(res.get('consensusAccuracy'))}** "
                f"(CI {res.get('ci')}), mean pairwise κ **{res.get('meanPairwiseKappa')}**, "
                f"N={res.get('n')}, {res.get('runs')} runs, families "
                f"{', '.join(res.get('families', []))} ({res.get('date', '—')})."
            )
        else:
            L.append(f"- Validated result: {status}")
        if semantic.get("note"):
            L.append(f"- {semantic['note']}")
        L.append("")

    audit = doc.get("audit")
    if audit:
        L += ["## Judge audit (why the gate matters)", "", audit.get("note", ""), ""]
        if audit.get("robustRegardlessOfJudge"):
            L.append("Robust regardless of judge:")
            L += [f"- {x}" for x in audit["robustRegardlessOfJudge"]]
            L.append("")

    cg = doc.get("continualGroundedEvals")
    if cg:
        g, r = cg.get("grounded", {}), cg.get("raw", {})
        gci, rci = g.get("ci", [None, None]), r.get("ci", [None, None])
        L += [
            "## Continual / grounded-answering (CANDIDATE — not a headline)",
            "",
            "Continual Provenance QA (CPQA): a frozen LLM answers either from the retrieved "
            "OKF/wiki source (`grounded`) or from parametric memory (`raw`), and a "
            "cross-provider judge panel scores both. Held to the no-overclaim gate and "
            "**candidate, not validated** — self-authored benchmark, keys held by one "
            "operator, no external replication.",
            "",
            f"- Benchmark: {cg.get('benchmark')}",
            f"- Answers: {cg.get('answerModel')} · Judges: {', '.join(cg.get('judges', []))} "
            f"({cg.get('gateway')}) · {cg.get('runs')} runs · N={cg.get('queryCount')}",
            f"- Overall consensus pass: grounded **{_pct(g.get('consensus'))}** "
            f"[{_pct(gci[0])}, {_pct(gci[1])}] vs raw **{_pct(r.get('consensus'))}** "
            f"[{_pct(rci[0])}, {_pct(rci[1])}]",
            f"- By expectation — **abstain/attribution-traps: grounded {_pct(g.get('abstain'))} "
            f"vs raw {_pct(r.get('abstain'))}**; recall: grounded {_pct(g.get('assert'))} "
            f"vs raw {_pct(r.get('assert'))} (a strong raw model already knows well-known facts; "
            "grounding's win is fail-closed abstention on traps)",
            f"- Inter-judge κ {cg.get('interJudgeKappa')} · percent-agreement "
            f"{_pct(cg.get('interJudgePercentAgreement'))}",
        ]
        hy = cg.get("hybrid")
        if hy:
            L += [
                f"- **Recall fix ({hy.get('system')}):** overall **{_pct(hy.get('consensus'))}** "
                f"[{_pct(hy.get('ci', [None, None])[0])}, {_pct(hy.get('ci', [None, None])[1])}], "
                f"recall **{_pct(hy.get('assert'))}** (up from strict {_pct(g.get('assert'))}), "
                f"traps {_pct(hy.get('abstain'))}; policy {hy.get('policyCounts')}",
                f"  - {hy.get('note', '')}",
            ]
        es = cg.get("enrichedStrict")
        if es:
            L += [
                f"- **Corpus enrichment ({es.get('system')}):** overall **{_pct(es.get('consensus'))}** "
                f"[{_pct(es.get('ci', [None, None])[0])}, {_pct(es.get('ci', [None, None])[1])}], "
                f"recall **{_pct(es.get('assert'))}** (up from strict {_pct(g.get('assert'))}), "
                f"traps {_pct(es.get('abstain'))} — pure grounding, no fallback",
                f"  - {es.get('note', '')}",
            ]
        tf = cg.get("threeFamily")
        if tf:
            gk = tf.get("meanPairwiseKappa", {})
            tg, tr = tf.get("grounded", {}), tf.get("raw", {})
            L += [
                f"- **3-family validation ({', '.join(tf.get('judges', []))}):** grounded "
                f"**{_pct(tg.get('consensus'))}** vs raw **{_pct(tr.get('consensus'))}**; "
                f"traps grounded **{_pct(tg.get('abstain'))}** vs raw {_pct(tr.get('abstain'))}; "
                f"mean pairwise κ {gk.get('grounded')}/{gk.get('raw')}; policy {tf.get('policyCounts')}",
                f"  - {tf.get('note', '')}",
            ]
        L += [f"- ⚠ {cg.get('note', '')}", ""]

    systems = doc.get("systemsBenchmarks") or []
    if systems:
        L += [
            "## Systems benchmarks (performance — candidate, host-dependent)",
            "",
            "Throughput/latency micro-benchmarks for the systems components. These are "
            "**candidate** engineering numbers (single host, vary by machine), not "
            "no-overclaim accuracy results — reproduce with the command in each note.",
            "",
        ]
        for s in systems:
            L += [f"### {s.get('name')} (`{s.get('component')}`)", "", s.get("description", ""), ""]
            if s.get("config"):
                L += [f"_Representative run — {s['config']}:_", ""]
            cols, rows = s.get("columns") or [], s.get("rows") or []
            if cols and rows:
                L.append("| " + " | ".join(cols) + " |")
                L.append("|" + "|".join(["---"] * len(cols)) + "|")
                for r in rows:
                    L.append("| " + " | ".join(str(c) for c in r) + " |")
                L.append("")
            if s.get("note"):
                L += [f"- {s['note']}", ""]

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
