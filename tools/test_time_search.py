#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Test-time compute via the provenance gate as a VERIFIER (best-of-n / self-consistency).

The repo's verifier is its highest-ROI asset at INFERENCE time, not just at training
time. Compute-optimal test-time search — sample N candidates from a base model and let
a *verifier* pick the winner — is known to let a small model match a much larger one
(Snell et al., "Scaling LLM Test-Time Compute Optimally...", 2024; Cobbe et al.,
"Training Verifiers to Solve Math Word Problems" / GSM8K, 2021; Wang et al.,
"Self-Consistency Improves Chain of Thought Reasoning", 2022 — all pre-cutoff classics).
Here the deterministic provenance gate IS the verifier, so we buy capability with no
extra training and no extra model.

For one prompt we:
  1. sample N candidates from agent.model.default_client(--model, default mock);
  2. score each with the INTRINSIC fail-closed gate — check_response(text,
     mode="advisor")["violations"] WITHOUT a question (passing a question would invoke
     the attribution TRAP-GRADER, which is for grading attribution targets, not for
     ranking free-form candidates — so we never use it here);
  3. add a light, deterministic quality heuristic (length floor, source-discipline
     framing, a 中文 summary, penalise degenerate/empty output) purely as a TIE-BREAK
     among already gate-equal candidates;
  4. SELECT the best.

CRITICAL fail-closed rule, enforced lexicographically in the sort key: a gate-CLEAN
candidate ALWAYS outranks a gate-VIOLATING one, regardless of heuristic quality. A
gate-clean ABSTENTION ("insufficient verified basis" / routes to a human) is therefore
an acceptable, even preferred, selection and can NEVER rank below a fluent answer that
fabricates a citation or figure. Abstention is a correct output, never a failure.

Modes:
  --mode best-of-n         : pick the single highest-scoring candidate.
  --mode self-consistency  : majority vote over a normalised answer signature, but the
                             vote is taken ONLY over gate-clean candidates (a fabricated
                             majority must never win); ties broken by the same heuristic.
                             If no candidate is gate-clean, fall back to the best-scoring
                             (which will be an abstention or the least-bad option).

    # offline plumbing (deterministic mock, writes nothing)
    python tools/test_time_search.py --model mock --dry-run
    python tools/test_time_search.py --model mock --prompt "Compute our runway." --n 4
    python tools/test_time_search.py --model mock --tasks data/council_tasks.json \
        --mode self-consistency --n 6
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

from agent.gate import check_response  # noqa: E402

TASKS = ROOT / "data" / "council_tasks.json"

SYSTEM = (
    "You are a source-disciplined advisor. Answer the question, stating each finding with "
    "a source where one is relied on, then give one decision. If you cannot verify a needed "
    "authority or figure, ABSTAIN and route to a human rather than guess. Label clearly as "
    "not professional advice; end with a 中文摘要."
)

# An answer that abstains is a CORRECT, selection-worthy output — never penalised.
ABSTAIN_MARKERS = (
    "abstain", "cannot verify", "can't verify", "insufficient verified basis", "insufficient",
    "unable to verify", "consult a", "seek a", "refer to a", "route to a", "escalate to a human",
    "需要核實", "無法核實", "諮詢專業",
)


def _is_abstention(text: str) -> bool:
    lowered = text.lower()
    return any(m in lowered for m in ABSTAIN_MARKERS)


def _gen(client, system: str, user: str) -> str:
    """Single completion as plain text; a broken/failed client yields '' rather than
    crashing (mirrors agent.council_deliberate._gen / tools/run_rft._gen)."""
    try:
        res = client.generate(system, user)
    except Exception:  # noqa: BLE001 - a broken client yields no content, not a crash
        return ""
    return (getattr(res, "text", "") or "").strip() if getattr(res, "ok", False) else ""


def _quality(text: str) -> float:
    """Light, deterministic TIE-BREAK heuristic in [0, 1]. Never overrides the gate:
    it only orders candidates that are already equal on the (clean > dirty) axis.

    Rewards: minimal substance (length floor), explicit source-discipline framing, a
    中文 summary. Penalises: degenerate repetition. An abstention scores fine — it is a
    legitimate, framed output — so a clean abstention is never demoted below a clean
    answer purely on heuristic grounds when that answer is also well-framed.
    """
    if not text.strip():
        return 0.0
    score = 0.0
    words = text.split()
    n = len(words)
    # length floor: reward up to a sane minimum, then flatten (no reward for padding).
    score += 0.30 * min(1.0, n / 40.0)
    lowered = text.lower()
    if any(m in lowered for m in ("source", "authority", "cite", "來源", "source discipline")):
        score += 0.25
    if re.search(r"[一-鿿]", text):  # has a 中文 section
        score += 0.20
    if "not professional advice" in lowered or "not legal advice" in lowered:
        score += 0.10
    # degeneracy penalty: a tiny unique-token ratio signals repetition/loops.
    if n >= 8:
        unique_ratio = len(set(words)) / n
        score += 0.15 * unique_ratio
    else:
        score += 0.15
    return round(min(1.0, score), 4)


def _sort_key(cand: dict) -> tuple:
    """Lexicographic ranking. FIRST axis is the gate (clean=1 outranks dirty=0); the
    heuristic only ever breaks ties WITHIN the same gate class. This is what guarantees
    a gate-clean abstention can never lose to a gate-violating answer."""
    return (1 if cand["gateClean"] else 0, cand["quality"])


def _signature(text: str) -> str:
    """Coarse normalised signature for self-consistency majority voting: lowercase,
    collapse whitespace, strip punctuation. Identical-intent answers cluster; this is a
    deterministic stand-in for semantic clustering (offline, no embeddings)."""
    norm = re.sub(r"[^\w一-鿿]+", " ", text.lower()).strip()
    norm = re.sub(r"\s+", " ", norm)
    return norm[:400]


def score_candidates(prompt: str, client, *, n: int) -> list[dict]:
    """Sample n candidates and score each with the intrinsic gate + heuristic. The gate
    is run WITHOUT a question (no trap-grader). Empty samples are kept as gate-dirty
    zero-quality rows so n is honest, but they sort last."""
    cands: list[dict] = []
    for i in range(n):
        text = _gen(client, SYSTEM, prompt)
        violations = check_response(text, mode="advisor")["violations"] if text.strip() else ["empty completion"]
        clean = not violations
        cands.append({
            "index": i,
            "text": text,
            "gateClean": clean,
            "violations": violations,
            "quality": _quality(text),
            "kind": "abstention" if (text.strip() and _is_abstention(text)) else ("answer" if text.strip() else "empty"),
            "signature": _signature(text) if text.strip() else "",
        })
    return cands


def _annotate(cands: list[dict], selected_index: int) -> dict:
    """Build the selected/rejected report with human-readable reasons."""
    selected = next(c for c in cands if c["index"] == selected_index)
    rejected = []
    for c in cands:
        if c["index"] == selected_index:
            continue
        if not c["text"].strip():
            reason = "empty completion"
        elif not c["gateClean"]:
            reason = "gate violation: " + "; ".join(c["violations"][:3])
        else:
            reason = f"gate-clean but lower quality ({c['quality']:.2f} <= {selected['quality']:.2f})"
        rejected.append({"index": c["index"], "kind": c["kind"], "gateClean": c["gateClean"],
                         "quality": c["quality"], "reason": reason})
    sel_reason = ("gate-clean abstention selected (fail-closed: a verified abstention "
                  "outranks any gate-violating answer)") if selected["kind"] == "abstention" and selected["gateClean"] \
        else ("highest-scoring gate-clean candidate" if selected["gateClean"]
              else "no gate-clean candidate; least-bad option selected (still surfaced for human review)")
    return {"selected": {"index": selected["index"], "kind": selected["kind"],
                         "gateClean": selected["gateClean"], "quality": selected["quality"],
                         "text": selected["text"], "reason": sel_reason},
            "rejected": rejected}


def best_of_n(prompt: str, client, *, n: int) -> dict:
    """Best-of-n: pick the top candidate under the gate-first lexicographic key."""
    cands = score_candidates(prompt, client, n=n)
    best = max(cands, key=_sort_key)
    out = _annotate(cands, best["index"])
    out["mode"] = "best-of-n"
    out["n"] = n
    out["cleanCount"] = sum(1 for c in cands if c["gateClean"])
    return out


def self_consistency(prompt: str, client, *, n: int) -> dict:
    """Self-consistency: majority vote, but ONLY over gate-clean candidates so a
    fabricated majority can never win. Ties (and the no-clean-candidate case) fall back
    to the gate-first best-of-n key."""
    cands = score_candidates(prompt, client, n=n)
    clean = [c for c in cands if c["gateClean"]]
    chosen: dict
    vote_note: str
    if clean:
        votes: dict[str, list[dict]] = {}
        for c in clean:
            votes.setdefault(c["signature"], []).append(c)
        # winning cluster = most votes; tie-break by best heuristic within the cluster.
        best_sig = max(votes, key=lambda s: (len(votes[s]), max(_quality(m["text"]) for m in votes[s])))
        cluster = votes[best_sig]
        chosen = max(cluster, key=_sort_key)
        vote_note = (f"majority over {len(clean)} gate-clean candidate(s): winning cluster "
                     f"had {len(cluster)} vote(s)")
    else:
        chosen = max(cands, key=_sort_key)
        vote_note = "no gate-clean candidate; fell back to gate-first best (abstention/least-bad)"
    out = _annotate(cands, chosen["index"])
    out["mode"] = "self-consistency"
    out["n"] = n
    out["cleanCount"] = len(clean)
    out["voteNote"] = vote_note
    return out


MODES = {"best-of-n": best_of_n, "self-consistency": self_consistency}


def run_tasks(tasks: list[dict], client, *, n: int, mode: str) -> list[dict]:
    fn = MODES[mode]
    results: list[dict] = []
    for t in tasks:
        res = fn(t["prompt"], client, n=n)
        res["taskId"] = t.get("id")
        res["prompt"] = t["prompt"]
        results.append(res)
    return results


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mock", help="model spec (default mock = offline plumbing)")
    ap.add_argument("--prompt", default=None, help="a single prompt to search over")
    ap.add_argument("--tasks", default=None, help="JSON file with {\"tasks\":[{id,prompt}]} (mutually exclusive with --prompt)")
    ap.add_argument("--n", type=int, default=4, help="candidates sampled per prompt (default 4)")
    ap.add_argument("--mode", choices=sorted(MODES), default="best-of-n")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; sample nothing, write nothing")
    args = ap.parse_args(argv)

    n = max(1, args.n)

    if args.prompt and args.tasks:
        print("error: pass --prompt OR --tasks, not both", flush=True)
        return 2

    if args.dry_run:
        if args.tasks:
            tasks_path = Path(args.tasks)
            if not tasks_path.is_absolute():
                tasks_path = ROOT / tasks_path
            count = len(json.loads(tasks_path.read_text("utf-8"))["tasks"])
            source = str(args.tasks)
        elif args.prompt:
            count, source = 1, "single --prompt"
        else:
            count, source = len(json.loads(TASKS.read_text("utf-8"))["tasks"]), str(TASKS.relative_to(ROOT))
        plan = {
            "model": args.model,
            "promptSource": source,
            "prompts": count,
            "n": n,
            "mode": args.mode,
            "verifier": "intrinsic gate (mode=advisor, no question) — fail-closed; no trap-grader",
            "selectionRule": "gate-clean ALWAYS outranks gate-violating; heuristic is tie-break only",
            "abstentionPolicy": "gate-clean abstention is an acceptable/preferred selection",
        }
        print("test-time search plan (dry-run, nothing sampled):", flush=True)
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return 0

    from agent.model import default_client
    client = default_client(args.model)

    if args.prompt:
        tasks = [{"id": "prompt", "prompt": args.prompt}]
    else:
        tasks_path = Path(args.tasks) if args.tasks else TASKS
        if not tasks_path.is_absolute():
            tasks_path = ROOT / tasks_path
        tasks = json.loads(tasks_path.read_text("utf-8"))["tasks"]

    results = run_tasks(tasks, client, n=n, mode=args.mode)
    summary = {
        "model": args.model,
        "mode": args.mode,
        "n": n,
        "prompts": len(results),
        "selectedClean": sum(1 for r in results if r["selected"]["gateClean"]),
        "selectedAbstentions": sum(1 for r in results if r["selected"]["kind"] == "abstention"),
    }
    print(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
