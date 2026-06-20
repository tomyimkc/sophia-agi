#!/usr/bin/env python3
"""Write salted public commitments for a private Sophia hidden eval pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def case_digest(case: dict[str, Any], salt: str) -> str:
    payload = {
        "salt": salt,
        "id": case["id"],
        "domain": case["domain"],
        "prompt": case["prompt"],
        "materials": case.get("materials", []),
        "scoring": case.get("scoring", {}),
        "requiresToolLog": bool(case.get("requiresToolLog", False)),
        "requiresMemoryDiff": bool(case.get("requiresMemoryDiff", False)),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_commitments(pack: dict[str, Any]) -> dict[str, Any]:
    salt = pack.get("salt")
    if not salt:
        raise ValueError("pack is missing salt; commitments require a private salt")
    return {
        "packId": pack["packId"],
        "createdAt": pack.get("createdAt"),
        "visibility": "public-commitment-only",
        "caseCount": len(pack.get("cases", [])),
        "domains": sorted({case["domain"] for case in pack.get("cases", [])}),
        "commitmentMethod": (
            "sha256(json({salt,id,domain,prompt,materials,scoring,"
            "requiresToolLog,requiresMemoryDiff}, sort_keys=True))"
        ),
        "saltStatus": "withheld until reveal",
        "cases": [
            {
                "id": case["id"],
                "domain": case["domain"],
                "sha256": case_digest(case, salt),
            }
            for case in pack.get("cases", [])
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build hidden eval public commitments")
    parser.add_argument("pack", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    commitments = build_commitments(load_json(args.pack))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(commitments, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
