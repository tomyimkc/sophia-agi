#!/usr/bin/env python3
"""The HF Space UI functions are importable + correct without Gradio (offline)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("hf_app", ROOT / "huggingface" / "space" / "app.py")
app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(app)


def test_verify_accepts_sourced_and_holds_unsourced() -> None:
    assert "verdict      : accepted" in app.verify_claim_ui("Laozi", "wiki://dao", "UNCLASSIFIED")
    held = app.verify_claim_ui("a bare claim", "", "UNCLASSIFIED")
    assert "held" in held and "no_source" in held


def test_abstain_vs_fabricate() -> None:
    assert "abstained" in app.abstain_vs_fabricate_ui("The author is unknown.")
    assert "fabricated" in app.abstain_vs_fabricate_ui("It was written by Roger Bacon.")


def test_gateway_call_grounded_vs_withheld() -> None:
    assert "accepted" in app.gateway_call_ui("who wrote the dao de jing")
    assert "WITHHELD" in app.gateway_call_ui("something obscure")


def main() -> int:
    test_verify_accepts_sourced_and_holds_unsourced()
    test_abstain_vs_fabricate()
    test_gateway_call_grounded_vs_withheld()
    print("test_hf_space: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
