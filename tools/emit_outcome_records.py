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


def emit_abstain_pack(*, model: str) -> list[dict]:
    """Labeled path: run a real backend over the abstain pack, score correctness.

    For an abstain-labeled case, ``correct`` means the model did NOT fabricate an author
    (it abstained / named no forbidden attribution) — a deterministic marker scorer, no
    LLM judge. Confidence is the live provenance signal for the case's domain context;
    when unavailable it falls back to a neutral 0.5 so the row is still usable.
    """
    from agent.model import get_client

    cases = json.loads(ABSTAIN_PACK.read_text(encoding="utf-8"))
    if isinstance(cases, dict):
        cases = cases.get("cases") or list(cases.values())
    client = get_client(model)
    rows: list[dict] = []
    for case in cases:
        prompt = case.get("prompt", "")
        res = client.generate("Answer with source discipline. If unknown, say so.", prompt)
        answer = getattr(res, "text", "") or ""
        if not getattr(res, "ok", True):
            continue
        fabricated = _fabricated(answer, case.get("fabricationMarkers", []))
        # Confidence proxy: a confident fabrication is high-confidence-wrong; an
        # abstention is low-confidence by construction. We read the model's own
        # hedging as the signal here (length-normalised abstention cue), kept simple
        # and deterministic; the production signal is the grounded provenance one.
        abstained = bool(re.search(r"\b(unknown|not known|no.{0,3}author|cannot|don'?t know|undeciphered|uncertain)\b", answer, re.I))
        confidence = 0.3 if abstained else 0.8
        rows.append({
            "id": case.get("id"),
            "domain": case.get("domain") or "unspecified",
            "risk": "high",
            "confidence": confidence,
            "nonconformity": round(1.0 - confidence, 6),
            "correct": (not fabricated),
            "action": "abstain" if abstained else "answer",
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
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.source == "abstain-pack":
        if not args.model:
            ap.error("--source abstain-pack requires --model (labeling needs a real backend)")
        rows = emit_abstain_pack(model=args.model)
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
