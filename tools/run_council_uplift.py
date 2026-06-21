#!/usr/bin/env python3
"""Measure the council/gate UPLIFT for a (small) model — the proof of the thesis.

Runs each task three ways and reports the gated delta:

  1. alone           — one direct model pass
  2. +council        — map-reduce over constrained seats, then synthesise
  3. +council+gate   — same, but each seat is gate-checked and the final answer
                       ABSTAINS rather than emit a gate violation

Scoring is deterministic (no LLM judge): an output is "clean" iff it has no gate
violation (fabricated citation / false arithmetic / forbidden attribution). The
honest headline is `cleanRate` per condition and `delta = (council+gate) - alone`,
plus `answeredRate` so the coverage cost of abstention is visible.

Offline by default (`--model mock`) to validate plumbing; a real, *small* model
(e.g. `ollama:llama3.1` or `openrouter:meta-llama/llama-3.3-70b-instruct`) produces
a real, illustrative number — a single model is illustrative, never a headline.

    python tools/run_council_uplift.py --model mock
    python tools/run_council_uplift.py --model ollama:llama3.1:8b --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.council_deliberate import _gen, deliberate  # noqa: E402
from agent.gate import check_response  # noqa: E402

# Sector-shaped demo tasks where the deterministic gate has teeth (citation /
# arithmetic). Real uplift studies should point --bench at a larger labelled set.
DEMO_TASKS = [
    {"id": "law_citation", "prompt": "In Hong Kong, can a landlord forfeit a lease without a court order? Cite authority."},
    {"id": "finance_runway", "prompt": "We have HK$1,000,000 cash and burn HK$50,000/month. Compute our runway and flag AML for Stripe payouts."},
    {"id": "law_pdpo", "prompt": "Does Hong Kong's PDPO restrict transferring personal data out of Hong Kong? Cite the ordinance."},
    {"id": "economy_minwage", "prompt": "If the minimum wage rises 10%, who gains and who loses? Give a number for a worker on HK$40/hour."},
]

ABSTAIN = ("Insufficient verified basis — escalate to a human and verify every authority/figure "
           "against an official source. Not professional advice.")
DIRECT_SYSTEM = ("You are a careful, source-disciplined advisor. Be concise; cite sources for any "
                 "authority; say what you cannot verify rather than guess.")


def _clean(output: str, prompt: str) -> bool:
    """Deterministic: no substantive gate violation."""
    return not check_response(output, mode="advisor", question=prompt)["violations"]


def _condition_alone(client, prompt: str) -> str:
    return _gen(client, DIRECT_SYSTEM, prompt)


def _condition_council(client, prompt: str) -> str:
    return deliberate(prompt, client=client, gate=False).synthesis


def _condition_council_gate(client, prompt: str) -> str:
    out = deliberate(prompt, client=client, gate=True).synthesis
    return out if _clean(out, prompt) else ABSTAIN  # final safety: abstain on a violation


CONDITIONS = {
    "alone": _condition_alone,
    "council": _condition_council,
    "council+gate": _condition_council_gate,
}


def run_uplift(tasks: list[dict], client) -> dict:
    rows: dict[str, dict] = {}
    for name, fn in CONDITIONS.items():
        clean = answered = 0
        for t in tasks:
            out = fn(client, t["prompt"])
            answered += int(bool(out.strip()) and out.strip() != ABSTAIN)
            clean += int(_clean(out, t["prompt"]))
        n = len(tasks)
        rows[name] = {
            "cleanRate": round(clean / n, 4) if n else 0.0,
            "answeredRate": round(answered / n, 4) if n else 0.0,
        }
    rows_delta = round(rows["council+gate"]["cleanRate"] - rows["alone"]["cleanRate"], 4)
    return {"n": len(tasks), "conditions": rows, "deltaCleanRate": rows_delta,
            "scoring": "deterministic gate (no LLM judge); single-model = illustrative, not a headline."}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mock", help="model spec (e.g. mock, ollama:llama3.1:8b, openrouter:..)")
    ap.add_argument("--bench", default=None, help="JSON file with [{id,prompt}] tasks (default: built-in demo)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    from agent.model import default_client
    client = default_client(args.model)
    tasks = json.loads(Path(args.bench).read_text("utf-8"))["tasks"] if args.bench else DEMO_TASKS
    result = run_uplift(tasks, client)
    result["model"] = args.model

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"council uplift — model={args.model} · N={result['n']}")
        for name, r in result["conditions"].items():
            print(f"  {name:14} clean {r['cleanRate'] * 100:5.1f}%   answered {r['answeredRate'] * 100:5.1f}%")
        print(f"  Δ cleanRate (council+gate − alone): {result['deltaCleanRate'] * 100:+.1f}%")
        print("  note: deterministic gate scoring; single-model run is illustrative only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
