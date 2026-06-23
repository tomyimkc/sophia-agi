#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import Gateway  # noqa: E402
from tools.sophia_skill_forge import forge_skill  # noqa: E402


EXAMPLES = [
    {"text": "delete db now", "label": True},
    {"text": "delete files now", "label": True},
    {"text": "delete logs now", "label": True},
    {"text": "delete cache now", "label": True},
    {"text": "read db now", "label": False},
    {"text": "read files now", "label": False},
    {"text": "read logs now", "label": False},
    {"text": "read cache now", "label": False},
]


def test_skill_forge_creates_files_and_registers() -> None:
    with tempfile.TemporaryDirectory() as d:
        gw = Gateway()
        out = forge_skill({"task_id": "danger-test", "examples": EXAMPLES}, out_root=Path(d), gateway=gw)
        assert out["created"] is True
        skill_dir = Path(out["skill_dir"])
        assert (skill_dir / "SKILL.md").exists()
        assert (skill_dir / "verifier.py").exists()
        assert (skill_dir / "eval_suite.jsonl").exists()
        assert gw.registry.get(out["skill_id"]) is not None
        assert out["eval"]["accuracy"] == 1.0


def main() -> int:
    test_skill_forge_creates_files_and_registers()
    print("test_skill_forge: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
