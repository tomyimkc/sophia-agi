#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""NLI-entailment grounding verifier vs the coherence baseline — the 'other way'.

Every coherence failure (O3/R3) traced to one root: similarity can't tell SUPPORTING from
REFUTING evidence (both are topical). NLI is built for exactly that. For a claim + retrieved
evidence, run textual entailment NLI(premise=evidence, hypothesis=claim):
  supportScore = max_e P(entail | e->claim) - max_e P(contradict | e->claim)   in [-1, 1]
High support => evidence entails the claim; low/negative => contradicts. On the SAME C1
claim+evidence data (marker-free) where the coherence fixed-point residual barely separated,
does entailment beat it?

Three verifiers compared, gold = (label == true):
  NLI       : dedicated cross-encoder (cross-encoder/nli-deberta-v3-base), pure entailment
  coherence : 1 - fixed-point residual (R3/oscillator, semantic embedder) — the baseline to beat
  llm_nli   : grok judges entails/contradicts/neutral (strong, uses reasoning) — optional

Gate: AUROC(NLI support vs supported) beats AUROC(coherence), paired-bootstrap CI excluding 0,
AND NLI AUROC CI excludes chance. Reported honestly. candidateOnly. No weights updated.
"""
from __future__ import annotations
import argparse, json, os, random, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import scipy.special as sp
from eval_o2_energy_hidden import auroc  # noqa

MARKER = re.compile(r"\[(ENTAILS|CONTRADICTS|SUPPORTS|REFUTES|NEUTRAL)\]", re.I)


def clean(seg, claim):
    s = MARKER.sub("", str(seg)).strip()
    cl = re.sub(r"\s+", " ", claim.lower()).strip()
    sl = re.sub(r"\s+", " ", s.lower()).strip()
    if cl and cl in sl and len(sl) < len(cl) + 40:   # drop verbatim claim restatement
        return None
    return s or None


def gather():
    """Real claim+evidence+label from C1 fixtures + live + factcheck packs, marker-free, deduped."""
    rows, seen = [], set()

    def add(claim, evs, supported):
        evs = [c for c in (clean(e, claim) for e in evs) if c]
        if claim in seen or not evs:
            return
        seen.add(claim); rows.append({"claim": claim, "evidence": evs, "supported": bool(supported)})

    # C1 heldout x fixtures
    fx = json.load(open("eval/fact_check/fixtures_v1.json"))["claims"]
    for r in (json.loads(l) for l in open("eval/fact_check/heldout_v1.jsonl")):
        c = r["claim"]
        if c in fx:
            add(c, [f"{e.get('title','')} {e.get('snippet','')}" for e in fx[c]], r["label"] == "true")

    # factcheck packs (title/publisher only; label true/false)
    for p in ["agi-proof/fact-check-live/fact-check-live-eval.LIVE-2026-06-24.json",
              "agi-proof/external-eval/factcheck-full-r1.json",
              "agi-proof/external-eval/factcheck-full-r2.json"]:
        try:
            for cs in json.load(open(p))["cases"]:
                if cs.get("label") not in ("true", "false"):
                    continue
                evs = [" ".join(str(e.get(k, "")) for k in ("title", "publisher") if e.get(k))
                       for cl in (cs.get("claims") or []) for l in (cl.get("layers") or []) for e in (l.get("evidence") or [])]
                add(cs["claim"], evs, cs["label"] == "true")
        except Exception:
            pass
    return rows


def nli_scores(rows, model_name):
    from sentence_transformers import CrossEncoder
    m = CrossEncoder(model_name)
    out = []
    for r in rows:
        pairs = [(e, r["claim"]) for e in r["evidence"]]        # premise=evidence, hyp=claim
        logits = m.predict(pairs)
        probs = sp.softmax(np.atleast_2d(logits), axis=1)        # cols: [contradiction, entailment, neutral]
        out.append(float(probs[:, 1].max() - probs[:, 0].max()))  # maxEntail - maxContradict
    return out


def coherence_scores(rows):
    os.environ["OSC_EMBED_BACKEND"] = "minilm"; os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import tools.oscillator_core as oc
    oc._EMBED_CACHE.clear()
    from tools.fixedpoint_stability_gate import iterate_fixedpoint
    return [1.0 - iterate_fixedpoint(r["claim"], r["evidence"], dim=64)["residual"] for r in rows]


def llm_nli_scores(rows):
    import agent.model as mdl
    out = []
    for r in rows:
        ev = " ; ".join(r["evidence"][:6])[:1500]
        prompt = (f"Evidence: {ev}\n\nClaim: {r['claim']}\n\nDoes the evidence ENTAIL (support), "
                  f"CONTRADICT, or is it NEUTRAL toward the claim? Reply one word: ENTAIL, CONTRADICT, or NEUTRAL.")
        try:
            o = (mdl.complete("You are a strict textual-entailment judge.", prompt, max_tokens=6, spec="grok") or "").upper()
        except Exception:
            o = ""
        out.append(1.0 if "ENTAIL" in o else (-1.0 if "CONTRAD" in o else 0.0))
    return out


def paired(a, b, labels, n_boot, seed):
    rng = random.Random(seed); n = len(labels); d = []
    ba, bb = auroc(a, labels), auroc(b, labels)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        la = [labels[i] for i in idx]
        if len(set(la)) < 2:
            continue
        xa, xb = auroc([a[i] for i in idx], la), auroc([b[i] for i in idx], la)
        if xa is not None and xb is not None:
            d.append(xa - xb)
    d.sort()
    return {"aurocA": round(ba, 4), "aurocB": round(bb, 4), "delta": round(ba - bb, 4),
            "ci95": [round(d[int(.025*len(d))], 4), round(d[int(.975*len(d))], 4)]}


def vs_chance(scores, labels, n_boot, seed):
    rng = random.Random(seed); n = len(labels); v = []
    base = auroc(scores, labels)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        la = [labels[i] for i in idx]
        if len(set(la)) < 2:
            continue
        a = auroc([scores[i] for i in idx], la)
        if a is not None:
            v.append(a)
    v.sort()
    return {"auroc": round(base, 4), "ci95": [round(v[int(.025*len(v))], 4), round(v[int(.975*len(v))], 4)]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--llm-nli", action="store_true", help="also run grok LLM-as-NLI (costs API calls)")
    ap.add_argument("--pack", default=None, help="pre-built {claim,evidence,supported} jsonl (else the C1/factcheck gather)")
    ap.add_argument("--n-boot", type=int, default=5000); ap.add_argument("--output", default=None)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(a.pack)] if a.pack else gather()
    y = [bool(r["supported"]) for r in rows]
    print(f"claim+evidence rows={len(rows)} supported={sum(y)} unsupported={len(y)-sum(y)}", file=sys.stderr)

    print("NLI cross-encoder ...", file=sys.stderr); nli = nli_scores(rows, a.nli_model)
    print("coherence baseline ...", file=sys.stderr); coh = coherence_scores(rows)
    report = {"schema": "sophia.nli_grounding.v1", "candidateOnly": True, "canClaimAGI": False,
              "reframe": "NLI-entailment grounding verifier vs coherence baseline",
              "nliModel": a.nli_model, "n": len(rows), "nSupported": sum(y), "seeds": {}}
    if a.llm_nli:
        print("LLM-as-NLI (grok) ...", file=sys.stderr); llm = llm_nli_scores(rows)
    for seed in (0, 1, 7):
        s = {"NLI_vs_chance": vs_chance(nli, y, a.n_boot, seed),
             "coherence_vs_chance": vs_chance(coh, y, a.n_boot, seed),
             "NLI_vs_coherence_paired": paired(nli, coh, y, a.n_boot, seed)}
        if a.llm_nli:
            s["llmNLI_vs_chance"] = vs_chance(llm, y, a.n_boot, seed)
            s["llmNLI_vs_coherence_paired"] = paired(llm, coh, y, a.n_boot, seed)
        report["seeds"][seed] = s
    nli_beats_chance = all(report["seeds"][s]["NLI_vs_chance"]["ci95"][0] > 0.5 for s in (0, 1, 7))
    nli_beats_coh = all(report["seeds"][s]["NLI_vs_coherence_paired"]["ci95"][0] > 0 for s in (0, 1, 7))
    report["NLI_beatsChance_allSeeds"] = nli_beats_chance
    report["NLI_beatsCoherence_allSeeds"] = nli_beats_coh
    report["verdict"] = ("NLI_beats_coherence_and_chance" if (nli_beats_chance and nli_beats_coh)
                         else "NLI_beats_chance_only" if nli_beats_chance else "NLI_no_better_than_chance")
    txt = json.dumps(report, indent=2)
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
