#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""R3 — grounding phase-lock, honestly (the O3 redo without the label-token leak).

O3's weak separation was carried by [ENTAILS]/[CONTRADICTS] tokens the evidence restated.
Here we STRIP those markers and any verbatim claim-restatement, keep only factual content,
and re-run the claim<->evidence fixed-point with the SEMANTIC embedder. The honest question:
does phase-lock over *meaning-similarity* separate supported from unsupported claims, or does
it confirm the deeper limit — that supporting and refuting evidence are equally *topical*, so
you need entailment (NLI), not similarity?

Gate: on marker-free evidence, meanResidualSupported < meanResidualUnsupported with a real gap
(aurcByResidual materially below 0.5-equivalent), reproduced. Reported honestly either way.
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MARKER = re.compile(r"\[(ENTAILS|CONTRADICTS|SUPPORTS|REFUTES|NEUTRAL)\]", re.I)


def clean_evidence(ev_list, claim):
    """Strip entailment markers + drop segments that verbatim-restate the claim."""
    claim_low = re.sub(r"\s+", " ", claim.lower()).strip()
    out = []
    for e in ev_list:
        s = MARKER.sub("", str(e)).strip()
        s_low = re.sub(r"\s+", " ", s.lower()).strip()
        # drop a segment that is (near) the claim itself or trivially contains it verbatim
        if claim_low and claim_low in s_low and len(s_low) < len(claim_low) + 40:
            continue
        if s:
            out.append(s)
    return out


def build_rows():
    """C1 heldout claims x fixtures/live evidence, cleaned."""
    fx = json.load(open("eval/fact_check/fixtures_v1.json"))["claims"]
    ho = [json.loads(l) for l in open("eval/fact_check/heldout_v1.jsonl")]
    live = json.load(open("agi-proof/fact-check-live/fact-check-live-eval.LIVE-2026-06-24.json"))["cases"]

    def live_ev(claim):
        out = []
        for c in live:
            if c.get("claim") == claim:
                for cl in (c.get("claims") or []):
                    for l in (cl.get("layers") or []):
                        for e in (l.get("evidence") or []):
                            seg = " ".join(str(e.get(k, "")) for k in ("title", "publisher") if e.get(k))
                            if seg.strip():
                                out.append(seg.strip())
        return out

    dirty, clean = [], []
    for r in ho:
        claim = r["claim"]; ev = []
        if claim in fx:
            for e in fx[claim]:
                ev.append((str(e.get("title", "")) + " " + str(e.get("snippet", ""))).strip())
        ev += live_ev(claim)
        ev = [e for e in ev if e]
        if not ev:
            continue
        sup = (r["label"] == "true")
        dirty.append({"claim": claim, "evidence": ev, "supported": sup})
        clean.append({"claim": claim, "evidence": clean_evidence(ev, claim), "supported": sup})
    clean = [c for c in clean if c["evidence"]]
    return dirty, clean


def run_gate(rows, tag):
    import tools.fixedpoint_stability_gate as fp
    d = fp.run(rows)
    sep = d.get("separation", {})
    return {"tag": tag, "n": d.get("n"), "abstainRate": d.get("abstainRate"),
            "embedBackend": d.get("embedBackend"),
            "meanResidualSupported": sep.get("meanResidualSupported"),
            "meanResidualUnsupported": sep.get("meanResidualUnsupported"),
            "aurcByResidual": sep.get("aurcByResidual"),
            "gap": (round(sep["meanResidualUnsupported"] - sep["meanResidualSupported"], 4)
                    if sep.get("meanResidualUnsupported") is not None and sep.get("meanResidualSupported") is not None else None)}


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--output", default=None); a = ap.parse_args(argv)
    os.environ["OSC_EMBED_BACKEND"] = "minilm"; os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import importlib, tools.oscillator_core as oc
    oc._EMBED_CACHE.clear()
    dirty, clean = build_rows()
    print(f"rows: dirty={len(dirty)} clean(marker-free)={len(clean)}", file=sys.stderr)
    r_dirty = run_gate(dirty, "with-markers")
    r_clean = run_gate(clean, "marker-free (honest)")
    honest_gap = r_clean["gap"]
    report = {"schema": "sophia.reframe_r3.v1", "candidateOnly": True, "canClaimAGI": False,
              "reframe": "R3 grounding phase-lock (marker-free)",
              "withMarkers": r_dirty, "markerFree": r_clean,
              "leakDelta": (round(r_dirty["gap"] - r_clean["gap"], 4)
                            if r_dirty["gap"] is not None and r_clean["gap"] is not None else None),
              "verdict": ("grounding_lock_separates_on_content" if (honest_gap is not None and honest_gap >= 0.08)
                          else "separation_was_label_leak — similarity cannot capture entailment"),
              "note": ("If the marker-free gap collapses, phase-lock over similarity embeddings cannot "
                       "distinguish supporting from refuting evidence (both are topical) — the real fix is NLI/entailment, not coherence.")}
    txt = json.dumps(report, indent=2)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
