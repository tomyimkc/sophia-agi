"""Emit the Provenance Delta report (machine-readable JSON + markdown table)."""

from __future__ import annotations

import json
from pathlib import Path

JUDGE_NOTE = (
    "Labels are external (Wikipedia/Wikidata + cited misattributions); the gate is "
    "the runtime treatment only; the judge shares no code with the gate. The default "
    "lexical judge is a screen — use an independent LLM-judge for headline claims."
)


def build_report(per_model: dict, *, run_at: str | None = None) -> dict:
    """``per_model``: {label: {"scores": {...}, "model": str, "onFail": str}}."""
    rows = []
    for label, payload in per_model.items():
        s = payload["scores"]
        rows.append(
            {
                "model": label,
                "modelSpec": payload.get("model", label),
                "onFail": payload.get("onFail", "repair"),
                "runs": payload.get("runs", 1),
                "hallucinationRateAlone": s["hallucinationRateAlone"],
                "hallucinationRateGated": s["hallucinationRateGated"],
                "delta": s["delta"],
                "ciDelta": s.get("ciDelta"),
                "perRunDelta": s.get("perRunDelta"),
                "falsePositiveCost": s["falsePositiveCost"],
                "coverageRecall": s["coverageRecall"],
                "falseCases": s.get("falseCases"),
                "trueCases": s.get("trueCases"),
            }
        )
    return {
        "benchmark": "provenance-delta",
        "runAt": run_at,
        "visibility": "public-aggregate",
        "nonCircularityContract": JUDGE_NOTE,
        "judgeMethod": next(iter(per_model.values()), {}).get("judgeMethod", "lexical") if per_model else "lexical",
        "rows": rows,
    }


def to_markdown(report: dict) -> str:
    lines = [
        "# Provenance Delta — results",
        "",
        f"_Run: {report.get('runAt') or 'n/a'} · judge: {report.get('judgeMethod')}_",
        "",
        "| Model | Runs | Halluc. alone | Halluc. gated | Δ (95% CI) | False-positive cost | Gate coverage |",
        "|---|---|---|---|---|---|---|",
    ]

    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    for r in report["rows"]:
        d = pct(r["delta"])
        if r.get("ciDelta"):
            d += f" [{pct(r['ciDelta'][0])}, {pct(r['ciDelta'][1])}]"
        lines.append(
            f"| {r['model']} | {r.get('runs', 1)} | {pct(r['hallucinationRateAlone'])} "
            f"| {pct(r['hallucinationRateGated'])} | {d} | {pct(r['falsePositiveCost'])} "
            f"| {pct(r['coverageRecall'])} |"
        )
    lines += ["", f"> {report['nonCircularityContract']}"]
    return "\n".join(lines)


def write_report(report: dict, json_path: Path, md_path: Path | None = None) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if md_path is not None:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(to_markdown(report) + "\n", encoding="utf-8")
