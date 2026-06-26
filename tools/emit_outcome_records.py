#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Emit canonical labeled-outcome records — the shared calibration substrate (Phase 0).

One JSONL schema feeds every calibration-shaped candidate (conformal abstention C1,
abstention-aware scoring C3, truth-probe C5)::

    {"id", "domain", "risk", "confidence", "nonconformity", "correct"?, "action"?}

  - ``confidence``    live provenance confidence in [0,1]
                      (``agent.grounded_confidence.grounded_source_confidence``).
  - ``nonconformity`` ``1 - confidence`` — the score the split-conformal gate thresholds.
  - ``correct``       OMITTED offline (we cannot judge truth without a model); produced
                      only on the ``--model`` path via a deterministic trap scorer.

Two honest paths, mirroring ``tools/calibrate_graded_thresholds.py``:

  (default, offline)  Over the OKF wiki, compute the live confidence/nonconformity for
                      every page. No ``correct`` label — this is the *bridge* that proves
                      the live-signal -> calibration loop is wired without fabricating
                      correctness. Ready to be labeled by a real run.

  --model SPEC        Over the abstain pack, run a real backend and label each row
                      ``correct`` by a deterministic fabrication-marker scorer (correct =
                      the answer names no fabricated author / abstains). Produces the
                      production calibration set the conformal policy is fitted on.

Nothing here mutates defaults; output is candidate data only.

    python tools/emit_outcome_records.py --out data/outcomes.okf.jsonl
    python tools/emit_outcome_records.py --source abstain-pack --model deepseek \
        --out data/outcomes.labeled.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ABSTAIN_PACK = ROOT / "agi-proof" / "baseline-ablation" / "abstain-pack-2026-06-22.json"

# OKF authorConfidence tiers that mark a claim as high-risk (weakly sourced). Mirrors
# tools/eval_graded_confidence.py so the two harnesses agree on the strong/weak split.
_WEAK_TIERS = ("disputed", "legendary", "anachronism_risk", "none_extant")


def _risk_for_tier(tier: "str | None") -> str:
    return "high" if (tier or "") in _WEAK_TIERS else "normal"


def emit_okf(*, hops: int = 1) -> list[dict]:
    """Offline bridge: live confidence/nonconformity per OKF page, no `correct` label."""
    from agent.config import WIKI_DIR
    from agent.grounded_confidence import grounded_source_confidence
    from okf.page import load_pages

    pages = load_pages(WIKI_DIR)
    rows: list[dict] = []
    for p in pages:
        conf = grounded_source_confidence(p.id, pages, hops=hops)
        if conf is None:
            continue
        tier = p.meta.get("authorConfidence")
        rows.append({
            "id": p.id,
            "domain": p.meta.get("domain") or "unspecified",
            "risk": _risk_for_tier(tier),
            "authorConfidence": tier,
            "confidence": round(float(conf), 6),
            "nonconformity": round(1.0 - float(conf), 6),
            # no `correct`: offline cannot judge truth (honest)
        })
    return rows


def _fabricated(answer: str, markers: list[str]) -> bool:
    """True if the answer trips any fabrication marker (named a forbidden author, etc.)."""
    text = answer or ""
    for m in markers or []:
        pat = m[3:] if m.startswith("re:") else re.escape(m)
        try:
            if re.search(pat, text, re.I):
                return True
        except re.error:
            if m.lower() in text.lower():
                return True
    return False


def emit_abstain_pack(*, model: str, samples: int = 5) -> list[dict]:
    """Labeled path: run a real backend over the abstain pack with self-consistency.

    For each abstain-labeled case we sample the model ``samples`` times and map each
    sample to a STANCE: ``answer`` if it trips the case's own fabrication markers (named a
    forbidden author), else ``abstain``. Then:

      - ``confidence`` = self-consistency agreement on the majority stance (a real,
        label-free confidence signal; ``agent.calibration.self_consistency``);
      - ``correct``    = the majority stance is ``abstain`` (the epistemically correct
        behaviour on an abstention trap) — a deterministic marker scorer, no LLM judge;
      - ``nonconformity`` = ``1 - confidence``.

    HONEST NOTE: on abstention traps confidence and correctness need not be positively
    correlated (a model can fabricate *consistently* = high confidence + wrong). The
    downstream report measures whether confidence predicts correctness rather than
    assuming it; abstain-traps are an imperfect conformal substrate (a positive-
    confidence QA pack is the real need) and this is flagged in the fit report.
    """
    from agent.calibration import self_consistency
    from agent.model import default_client

    cases = json.loads(ABSTAIN_PACK.read_text(encoding="utf-8"))
    if isinstance(cases, dict):
        cases = cases.get("cases") or list(cases.values())
    client = default_client(model)
    rows: list[dict] = []
    for case in cases:
        prompt = case.get("prompt", "")
        markers = case.get("fabricationMarkers", [])
        stances: list[str] = []
        for _ in range(max(1, samples)):
            res = client.generate("Answer with source discipline. If the answer is unknown or disputed, say so plainly and do not invent an author.", prompt)
            if not getattr(res, "ok", True):
                continue
            answer = getattr(res, "text", "") or ""
            stances.append("answer" if _fabricated(answer, markers) else "abstain")
        if not stances:
            continue
        majority, confidence = self_consistency(stances)
        rows.append({
            "id": case.get("id"),
            "domain": case.get("domain") or "unspecified",
            "risk": "high",
            "confidence": round(float(confidence), 6),
            "nonconformity": round(1.0 - float(confidence), 6),
            "correct": (majority == "abstain"),
            "action": majority,
            "nSamples": len(stances),
            "model": model,
        })
    return rows


def write_jsonl(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Emit canonical labeled-outcome records (Phase 0).")
    ap.add_argument("--source", choices=("okf-wiki", "abstain-pack"), default="okf-wiki")
    ap.add_argument("--model", default=None, help="backend spec for the labeled abstain-pack path")
    ap.add_argument("--samples", type=int, default=5, help="self-consistency samples per case (labeled path)")
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.source == "abstain-pack":
        if not args.model:
            ap.error("--source abstain-pack requires --model (labeling needs a real backend)")
        rows = emit_abstain_pack(model=args.model, samples=args.samples)
    else:
        rows = emit_okf(hops=args.hops)

    write_jsonl(rows, args.out)
    labeled = sum(1 for r in rows if "correct" in r)
    summary = {
        "source": args.source,
        "n": len(rows),
        "labeled": labeled,
        "out": str(args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out),
        "note": (
            "No `correct` labels offline (the OKF bridge cannot judge truth); point "
            "--source abstain-pack --model <backend> to produce labeled rows."
            if labeled == 0 else "Labeled via deterministic fabrication-marker scorer."
        ),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
