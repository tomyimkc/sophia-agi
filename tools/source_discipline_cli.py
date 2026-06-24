#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Source-discipline checker CLI: run Sophia's provenance_faithful verifier over text.

Reads draft text on stdin (or ``--text``) and prints
``{"passed": bool, "reasons": [...], "violations": [...]}`` JSON. The OpenClaw
``before_agent_finalize`` plugin spawns this to gate agent replies against Sophia's
"never merge lineages" rule. Dependency-free, offline — a local-regex check, no model,
no network.

Exit code: ``0`` on a completed check (pass or fail is in the JSON); nonzero only on an
internal error, so the caller can distinguish "checked and violated" from "could not
check" (and fail open on the latter).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.verifiers import provenance_faithful  # noqa: E402

# Build the matcher once (loads the doNotAttributeTo records); ~one-time cost per process.
_VERIFY = provenance_faithful()


def check(text: str) -> dict:
    """Return ``{passed, reasons, violations}`` for ``text`` under the source-discipline rule."""
    result = _VERIFY(text or "", None, {})
    detail = result.get("detail") or {}
    return {
        "passed": bool(result.get("passed")),
        "reasons": list(result.get("reasons") or []),
        "violations": list(detail.get("violations") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check text against Sophia's source-discipline rule.")
    parser.add_argument("--text", help="text to check (default: read stdin)")
    args = parser.parse_args()
    text = args.text if args.text is not None else sys.stdin.read()
    print(json.dumps(check(text), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
