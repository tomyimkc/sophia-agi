#!/usr/bin/env python3
"""Flag MISSTATED authorities — real citations used for propositions their holding
does not support (the *Ayinde* failure). Semantic tier above citation existence.

Needs an LLM judge (agent.model); without one configured it abstains on every pair
(fail-closed). Any number this prints from a single judge is ILLUSTRATIVE — a
headline requires the no-overclaim gate (≥2 independent judges + CIs). Not legal
advice; a human must still read every authority in full.

    python tools/check_legal_faithfulness.py --text "Obergefell, 576 U.S. 644, bars all immigration appeals."
    python tools/check_legal_faithfulness.py brief.docx --model anthropic:claude-sonnet-4-6
    python tools/check_legal_faithfulness.py --text "..." --require-support   # abstain = fail
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.legal_faithfulness import assess_text, make_llm_judge  # noqa: E402


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("path", nargs="?", help="document to scan (.txt/.md/.docx/.html/.pdf)")
    src.add_argument("--text", help="scan a raw text string instead of a file")
    ap.add_argument("--model", default=None, help="judge model spec (e.g. anthropic:claude-sonnet-4-6)")
    ap.add_argument("--require-support", action="store_true", help="treat abstentions as failures")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    if args.text is not None:
        text = args.text
    else:
        from agent.legal_docs import DocIngestError, extract_text
        try:
            text = extract_text(args.path)
        except DocIngestError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    result = assess_text(text, judge=make_llm_judge(args.model))
    contradicted = result["contradicted"]
    failed = bool(contradicted) or (args.require_support and bool(result["abstained"]))

    if args.json:
        print(json.dumps({**result, "passed": not failed}, ensure_ascii=False, indent=2))
    else:
        print(f"supported: {len(result['supported'])}  |  contradicted: {len(contradicted)}  |  "
              f"abstained: {len(result['abstained'])}")
        for c in contradicted:
            print(f"  ✗ MISSTATED {c['citation']}: {c.get('reason', '')}")
        for a in result["abstained"]:
            print(f"  ? unchecked {a['citation']}: {a.get('why', '')}")
        print("note: single-judge output is ILLUSTRATIVE; verify against the full text of each authority.")
        print("PASS — no authority found misstated" if not failed else "FAIL — misstated/unverifiable authority")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
