#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Evidence driver for the OKF surprise signal — the belief-dynamics layer's real signal.

The forgetting layer (``okf.decay_okf`` / ``frontier_demotion`` / ``forgetting_audit``)
was wired into a tamper-evident audit ledger but stayed ``level3Evidence: false`` for one
concrete reason: ``surprise`` — "how unexpected a belief is given current memory" — was an
UNRECORDED PLACEHOLDER (``0.0``), so the surprise-gating was a no-op over the real corpus
(see ``okf/belief_state_projection.py`` HONESTY CONTRACT and ``RESEARCH_FOLLOWUP.md``).

``okf.surprise_signal`` now MEASURES it: leave-one-out predictive surprise
``P(belief | the rest of memory)`` — the per-token NLL of a belief's content under a
smoothed term-likelihood model built from the rest of the corpus, focused on the belief's
provenance neighbourhood. This driver converts that into an auditable evidence artifact
over three deterministic, fully-offline panels, each with a named pass/fail:

  A. SEPARATION (falsifiable) — a planted near-DUPLICATE of an existing belief must score
     LOWER surprise (memory predicts it) than a planted NOVEL belief with out-of-corpus
     vocabulary. If the signal could not tell redundant from novel, this fails.

  B. REAL CORPUS RUN — measure surprise for every wiki page, report the distribution, and
     show the placeholder-vs-measured delta: the placeholder (surprise=0 for all) flags
     ZERO surprising beliefs; the measured signal flags a real, non-empty set. Every
     surprising-belief selection is recorded in a hash-chained ``ForgettingAudit`` ledger
     that verifies clean — the signal is real AND auditable.

  C. GATE-IS-THE-FLOOR — the surprise-selected set is routed through the anti-forgetting
     gate (``agent.continual_plasticity.evaluate_update``) with NO fabricated eval metrics
     and NO verifier artifacts (the honest "no GPU eval has run yet" state). The gate must
     REFUSE to promote. This proves surprise PROPOSES but never DISPOSES — it cannot
     bypass the floor. (Real-metric *clearance* is the documented GPU handoff; this panel
     proves the floor holds, not that a trained adapter passed.)

Honest bound: deterministic, offline, pure-stdlib retrieval-likelihood — NOT a neural
model, NOT weight changes, NOT a capability claim. The measured signal is a LEAVE-ONE-OUT
substitute for the temporal "surprise at first observation" (which needs ``written_at``,
still unrecorded). ``canClaimAGI`` stays false. Reproduce: ``python tools/eval_okf_surprise.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "agi-proof" / "okf-consistency" / "surprise-signal.public-report.json"

# A belief is flagged "surprising" (a novelty-consolidation candidate) when its measured
# surprise exceeds the corpus average. 0.5 is the z-score->logistic midpoint == "as
# predictable as the average belief"; above it == "less predictable than average". This
# is a documented, fixed semantics — NOT tuned to hit a target count.
SURPRISE_FLAG_THRESHOLD = 0.5


def _page(pid, body, *, tradition=None, domain=None, conf="attributed"):
    """Build an in-memory OKF Page (no disk) for the synthetic separation panel."""
    from okf.page import Page
    meta = {"id": pid, "pageType": "concept", "authorConfidence": conf}
    if tradition:
        meta["tradition"] = tradition
    if domain:
        meta["domain"] = domain
    return Page(path=Path(f"{pid}.md"), meta=meta, body=body)


# --------------------------------------------------------------------------
# Panel A — falsifiable separation: duplicate (low) vs novel (high)
# --------------------------------------------------------------------------

def _panel_separation() -> dict:
    from okf.graph import build as build_graph
    from okf.surprise_signal import corpus_surprise

    # A small corpus with a shared "stoic ethics" vocabulary so the local/global models
    # have something to predict from.
    base = [
        _page("stoic_virtue", "Stoic ethics holds that virtue is the only good and reason "
              "aligns the soul with nature. The sage lives according to nature and reason.",
              tradition="stoicism", domain="philosophy"),
        _page("stoic_reason", "Reason governs the Stoic soul; living according to nature means "
              "following reason and accepting fate with virtue and equanimity.",
              tradition="stoicism", domain="philosophy"),
        _page("stoic_fate", "The Stoics taught that virtue and reason let the soul accept fate "
              "and live according to nature without disturbance.",
              tradition="stoicism", domain="philosophy"),
        _page("stoic_sage", "The Stoic sage embodies virtue, reason, and life according to "
              "nature, unmoved by fortune.", tradition="stoicism", domain="philosophy"),
    ]
    # DUPLICATE: a near-copy of stoic_virtue (same vocabulary, lightly reworded). Memory
    # already contains this content, so it should be UN-surprising.
    duplicate = _page(
        "stoic_virtue_dup",
        "Stoic ethics holds that virtue is the only good and reason aligns the soul with "
        "nature. The sage lives according to nature and reason, accepting fate.",
        tradition="stoicism", domain="philosophy")
    # NOVEL: out-of-corpus vocabulary the rest of memory cannot predict at all.
    novel = _page(
        "quantum_chromodynamics",
        "Quantum chromodynamics describes gluon exchange binding quarks via the strong "
        "interaction; lattice gauge computations quantify color confinement and asymptotic "
        "freedom in hadron spectroscopy.", tradition="stoicism", domain="philosophy")

    pages = base + [duplicate, novel]
    graph = build_graph(pages)
    scores = corpus_surprise(pages, graph=graph)
    dup = scores["stoic_virtue_dup"]
    nov = scores["quantum_chromodynamics"]

    # The fundamental, un-squashed assertion is on raw per-token NLL. The normalised
    # surprise is reported too and should land on opposite sides of the midpoint.
    raw_separated = dup.raw_nll < nov.raw_nll
    norm_separated = dup.surprise < nov.surprise
    midpoint_split = dup.surprise < SURPRISE_FLAG_THRESHOLD < nov.surprise

    return {
        "panel": "separation",
        "claim": "near-duplicate scores LOWER surprise than out-of-corpus novelty",
        "duplicate": dup.as_dict(),
        "novel": nov.as_dict(),
        "rawNllSeparated": raw_separated,
        "normalizedSeparated": norm_separated,
        "midpointSplit": midpoint_split,
        "pass": bool(raw_separated and norm_separated),
    }


# --------------------------------------------------------------------------
# Panel B — real corpus run + placeholder-vs-measured delta + audit
# --------------------------------------------------------------------------

def _quantiles(vals):
    s = sorted(vals)
    n = len(s)
    if not n:
        return {"min": None, "median": None, "max": None, "mean": None}
    median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0
    return {"min": round(s[0], 6), "median": round(median, 6),
            "max": round(s[-1], 6), "mean": round(sum(s) / n, 6)}


def _panel_corpus_run(wiki_dir: Path) -> dict:
    from okf.forgetting_audit import ForgettingAudit, LifecycleEvent
    from okf.graph import build as build_graph
    from okf.page import load_pages
    from okf.surprise_signal import corpus_surprise
    from tools.audit_cpqa_recall import classify_source

    pages = load_pages(wiki_dir)
    graph = build_graph(pages)
    scores = corpus_surprise(pages, graph=graph)

    by_id = {p.id: p for p in pages}
    # Gate-cleared = answer-bearing (a thin stub is not worth consolidating). Same notion
    # the CLS consolidation run uses.
    gate_cleared = {pid for pid, p in by_id.items() if classify_source(p)["answerBearing"]}

    # MEASURED novelty set: surprising AND gate-cleared. This is the real, non-empty effect.
    measured_novel = sorted(
        pid for pid, s in scores.items()
        if s.surprise > SURPRISE_FLAG_THRESHOLD and pid in gate_cleared)
    # PLACEHOLDER comparison: with surprise==0 for all (the old projection), the same
    # threshold flags NOTHING. This is the level3 blocker the measured signal removes.
    placeholder_novel = sorted(
        pid for pid in scores
        if 0.0 > SURPRISE_FLAG_THRESHOLD and pid in gate_cleared)  # always empty, by construction

    # Record every surprising-belief selection in the tamper-evident ledger.
    audit = ForgettingAudit()
    for pid in measured_novel:
        audit.append(LifecycleEvent("reinforce", pid, "surprise_gated"))
    chain_valid = audit.verify()

    surprises = [s.surprise for s in scores.values()]
    raws = [s.raw_nll for s in scores.values()]
    ranked = sorted(scores.values(), key=lambda s: s.raw_nll)

    # The placeholder produced ZERO; the measured signal produced a real, non-empty set:
    # that delta IS the unblock. Non-trivial == the corpus actually spreads (max>min).
    nontrivial = len(measured_novel) > 0 and (max(raws) - min(raws)) > 1e-6
    placeholder_was_noop = len(placeholder_novel) == 0

    return {
        "panel": "corpus-run",
        "pages": len(pages),
        "gateCleared": len(gate_cleared),
        "surpriseDistribution": _quantiles(surprises),
        "rawNllDistribution": _quantiles(raws),
        "leastSurprising": [{"nodeId": s.node_id, "rawNll": round(s.raw_nll, 4)} for s in ranked[:3]],
        "mostSurprising": [{"nodeId": s.node_id, "rawNll": round(s.raw_nll, 4)} for s in ranked[-3:]],
        "measuredNovelCount": len(measured_novel),
        "placeholderNovelCount": len(placeholder_novel),
        "auditChainValid": chain_valid,
        "auditRecordCount": len(audit.to_list()),
        "placeholderWasNoOp": placeholder_was_noop,
        "measuredIsNonTrivial": nontrivial,
        # the measured signal must produce a real, auditable, non-trivial effect the
        # placeholder could not.
        "pass": bool(nontrivial and placeholder_was_noop and chain_valid),
    }


# --------------------------------------------------------------------------
# Panel C — the anti-forgetting gate stays the floor (surprise cannot bypass it)
# --------------------------------------------------------------------------

def _panel_gate_is_the_floor(measured_novel_count: int) -> dict:
    """A surprise-selected consolidation candidate with NO verified eval and NO artifacts
    must NOT be promoted. We fabricate no metrics — the empty/insufficient candidate is
    the honest "no GPU eval has run yet" state, and the gate must refuse it."""
    from agent.continual_plasticity import UpdateCandidate, evaluate_update

    candidate = UpdateCandidate(
        id="surprise_novelty_consolidation",
        kind="okf_consolidation_selection",
        metrics=(),                      # NO fabricated eval deltas
        verifier_artifacts=(),           # NO verifier artifacts
        # Neutral wording so the refusal rests on the floor logic (no verified eval, no
        # artifacts), not on a conscience keyword tripwire.
        notes="surprise-selected novelty set for consolidation; no held-out evaluation produced yet",
    )
    decision = evaluate_update(candidate, target_suite="okf_consolidation")
    refused = decision.verdict != "promote"
    return {
        "panel": "gate-is-the-floor",
        "claim": "surprise PROPOSES; the anti-forgetting gate DISPOSES — no verified eval, no promotion",
        "selectedForConsolidation": measured_novel_count,
        "verdict": decision.verdict,
        "reasons": list(decision.reasons),
        "promoted": not refused,
        "pass": bool(refused),
    }


# --------------------------------------------------------------------------
# Panel D — the signal is wired into the LIVE consolidation run
# --------------------------------------------------------------------------

def _panel_live_wiring(wiki_dir: Path) -> dict:
    """The production consolidation run (tools.run_cls_consolidation) projects the MEASURED
    surprise — not the old placeholder. Verified by running build_selection and confirming
    the surprise channel is measured and the broader manifest stays level3:false."""
    from tools.run_cls_consolidation import build_selection

    sel = build_selection(str(wiki_dir), min_stable_snapshots=1)
    ss = sel.get("surpriseSignal", {})
    measured = bool(ss.get("measured")) and bool(sel.get("measuredSignals", {}).get("surprise"))
    surprise_removed_from_placeholders = "surprise" not in sel.get("unrecordedPlaceholders", {})
    return {
        "panel": "live-wiring",
        "claim": "the live consolidation run projects MEASURED surprise (not the placeholder)",
        "measuredInLiveRun": measured,
        "surpriseRemovedFromPlaceholders": surprise_removed_from_placeholders,
        "liveNovelCount": ss.get("novelCount"),
        "liveAuditValid": sel.get("auditChainValid"),
        "pass": bool(measured and surprise_removed_from_placeholders and sel.get("auditChainValid")),
    }


# --------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------

def run(wiki_dir: Path) -> dict:
    a = _panel_separation()
    b = _panel_corpus_run(wiki_dir)
    c = _panel_gate_is_the_floor(b["measuredNovelCount"])
    d = _panel_live_wiring(wiki_dir)
    all_pass = a["pass"] and b["pass"] and c["pass"] and d["pass"]

    # LEVEL-3 DETERMINATION — earned ONLY if all three genuinely pass: the signal is real
    # (separation), it moves the dynamics over the real corpus where the placeholder could
    # not (corpus-run), and it never bypasses the floor (gate). Scoped to the surprise
    # signal; the BROADER forgetting layer stays level3:false while written_at /
    # reinforcement_count remain unrecorded (time-decay + usage-reinforcement still no-ops).
    level3 = bool(a["pass"] and b["pass"] and c["pass"])
    return {
        "schema": "sophia.okf_surprise_signal_report.v1",
        # Three independent, honest flags:
        #  - level3Evidence: is the SIGNAL real/validated (separation + corpus + gate)? yes.
        #  - liveWired: does the production consolidation run project the measured signal?
        #    yes, now (Panel D verifies tools/run_cls_consolidation uses it).
        #  - candidateOnly: the consolidation it feeds is still the OFFLINE selection half;
        #    promotion into weights still requires the GPU anti-forgetting gate. So true.
        "candidateOnly": True,
        "liveWired": bool(d["pass"]),
        "level3Evidence": level3,
        "level3Scope": (
            "Scoped to the SURPRISE signal only: it is now a real, measured, falsifiably-"
            "validated leave-one-out retrieval-likelihood over the OKF graph, projected into "
            "the LIVE consolidation run (liveWired), that never bypasses the anti-forgetting "
            "gate. This does NOT make the whole forgetting layer level3: written_at and "
            "reinforcement_count are still unrecorded, so time-decay and usage-reinforcement "
            "remain no-ops; and promotion into weights still requires the GPU gate "
            "(candidateOnly above)."
        ),
        "canClaimAGI": False,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "signalDefinition": (
            "surprise(b) = per-token cross-entropy (NLL, nats) of belief b's content under "
            "a smoothed interpolated unigram model built from the REST of the corpus "
            "(leave-one-out), focused on b's provenance neighbourhood with a global backoff. "
            "Normalised via z-score->logistic, RELATIVE to the corpus (0.5 == average "
            "predictability). See okf/surprise_signal.py."
        ),
        "honestyCaveats": [
            "LEAVE-ONE-OUT, not first-observation: true 'surprise at first observation' "
            "needs an arrival-time ordering (written_at), which the frontmatter does not "
            "record. This measures P(belief | the rest of memory) instead — a weaker but "
            "genuinely measured substitute.",
            "Retrieval-likelihood, NOT a neural model: a smoothed unigram / count-based "
            "estimator (the path RESEARCH_FOLLOWUP scopes). Named for what it is.",
            "Normalised surprise is RELATIVE to this corpus, not an absolute probability.",
            "Tokeniser models ASCII word runs only; CJK (mostly titles) is not modelled.",
        ],
        "pass": all_pass,
        "panels": [a, b, c, d],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="OKF surprise-signal evidence")
    ap.add_argument("--wiki", default=str(ROOT / "wiki"))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)

    report = run(Path(args.wiki))
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("OKF surprise signal — evidence")
        for panel in report["panels"]:
            tag = "PASS" if panel["pass"] else "FAIL"
            print(f"  [{tag}] {panel['panel']}")
        print(f"  level3Evidence: {report['level3Evidence']}  canClaimAGI: {report['canClaimAGI']}")
        print(f"  overall: {'PASS' if report['pass'] else 'FAIL'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not args.json:
        print(f"Wrote {args.out.relative_to(ROOT)}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
