#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fill knowledge-gap stubs from trusted sources via the librarian (allowlisted + gated).

The final step of the self-correction loop: take the ``none_extant`` stubs that
`tools/close_gap_loop.py` materialized and promote any whose topic a **trusted source** covers
into a sourced page (`agent.source_fill`). Two boundaries enforce no-fabrication: the source
must be allowlisted (operator-curated `raw/` dir, or an authority-ranked id), and the extracted
page must pass the provenance gate. Filled pages land in the draft tier with ``needsReview``.

Extraction is LLM-gated, so a real fill needs an API key (the librarian's model). Dry-run still
needs extraction to show a verdict, so it too needs a client. **Dry-run by default**; ``--write``
to promote.

  python tools/fill_gap_stubs.py                      # dry-run over wiki stubs + raw/ sources
  python tools/fill_gap_stubs.py --write               # promote fillable stubs (gated)
  python tools/fill_gap_stubs.py --sources-dir DIR --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(*, sources_dir: Path, write: bool, min_trust: float, extractor=None) -> dict:
    from okf import load_pages

    from agent.config import WIKI_DIR
    from agent.source_fill import fill_gaps, load_trusted_sources, make_llm_extractor

    pages = load_pages(WIKI_DIR)
    sources = load_trusted_sources(sources_dir)
    extractor = extractor or make_llm_extractor()
    report = fill_gaps(pages, sources, extractor=extractor, write=write, min_trust=min_trust)
    report["sourcesDir"] = str(sources_dir)
    return report


def main(argv=None) -> int:
    from agent.config import RAW_DIR

    ap = argparse.ArgumentParser(description="Fill gap stubs from trusted sources (librarian)")
    ap.add_argument("--sources-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--write", action="store_true", help="promote fillable stubs (default: dry-run)")
    ap.add_argument("--min-trust", type=float, default=0.8)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    try:
        report = run(sources_dir=args.sources_dir, write=args.write, min_trust=args.min_trust)
    except Exception as exc:  # most likely: no API key for the librarian extractor
        print(f"fill aborted: {exc}\n(the librarian extractor needs an API key; see agent/model.py)")
        return 1

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(f"Stubs: {report['stubs']}  trusted sources: {report['trustedSources']}  "
          f"({'wrote' if report['wrote'] else 'dry-run'})")
    for r in report["results"]:
        if r.get("ok"):
            verb = "filled" if report["wrote"] else "would fill"
            print(f"  + {r['id']}  {verb} from {r.get('source')}  -> {r.get('authorConfidence')}")
        else:
            print(f"  - {r['id']}  skipped: {r.get('reason') or r.get('reasons')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
