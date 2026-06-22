"""Seal the held-out personality family → public SHA-256 commitments only.
The salt + unsealed prompts are written under gitignored private/; only the
per-case hashes are published. Reuses tools/hidden_eval_commitments.py."""
from __future__ import annotations

import argparse, json, secrets, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.personality_measure import load_bank          # noqa: E402
from agent.personality_behavioral import load_battery     # noqa: E402
from tools.hidden_eval_commitments import build_commitments  # noqa: E402

PACK_ID = "personality-heldout-2026-06-23"
COMMIT_OUT = ROOT / "agi-proof" / "hidden-reviewer-packs" / f"{PACK_ID}.commitments.json"
PRIVATE_OUT = ROOT / "private" / "hidden-evals" / f"{PACK_ID}.private.json"
HELDOUT_ITEMS = ROOT / "data" / "personality_items_heldout.json"
HELDOUT_BATTERY = ROOT / "data" / "behavioral_battery_heldout.json"


def build_private_pack(salt: "str | None" = None) -> dict:
    salt = salt or secrets.token_hex(32)
    cases = []
    for it in load_bank(HELDOUT_ITEMS)["items"]:
        cases.append({"id": f"ipip_{it['id']}", "domain": "personality",
                      "prompt": f"I {it['text']}.",
                      "scoring": {"channel": "self-report", "domain": it["domain"], "keyed": it["keyed"],
                                  "method": "ipip-likert-1-5"},
                      "requiresToolLog": False, "requiresMemoryDiff": False})
    for axis, prompts in load_battery(HELDOUT_BATTERY)["prompts"].items():
        for i, p in enumerate(prompts):
            cases.append({"id": f"batt_{axis}_{i}", "domain": "personality", "prompt": p,
                          "scoring": {"channel": "behavioral", "axis": axis, "method": "ocean-judge-0-100"},
                          "requiresToolLog": False, "requiresMemoryDiff": False})
    return {"packId": PACK_ID, "createdAt": datetime.now(timezone.utc).isoformat(),
            "salt": salt, "cases": cases}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--salt", default=None, help="hex salt (default: fresh random 256-bit)")
    args = ap.parse_args(argv)
    pack = build_private_pack(args.salt)
    commitments = build_commitments(pack)
    COMMIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    COMMIT_OUT.write_text(json.dumps(commitments, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    PRIVATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    PRIVATE_OUT.write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote public commitments {COMMIT_OUT} ({len(commitments['cases'])} cases)")
    print(f"wrote PRIVATE pack+salt {PRIVATE_OUT} (gitignored — store the salt securely)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
