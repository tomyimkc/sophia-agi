#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for council-distillation trace generation + the gate-filter firewall (offline)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load():
    spec = importlib.util.spec_from_file_location("dct", ROOT / "tools" / "distill_council_traces.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _client(text: str):
    return SimpleNamespace(generate=lambda system, user: SimpleNamespace(ok=True, text=text))


FIN_TASKS = [{"id": "fin_runway", "prompt": "Model our runway and flag AML for Stripe payouts."}]


def test_clean_teacher_yields_a_trace() -> None:
    m = _load()
    rows, stats = m.generate_traces(FIN_TASKS, _client("Sound finance analysis, no figures asserted. 中文摘要：見上。"))
    assert stats["kept"] == 1 and rows
    msgs = rows[0]["messages"]
    assert msgs[0]["role"] == "system" and msgs[2]["role"] == "assistant"
    assert rows[0]["metadata"]["gatePassed"] is True


def test_gate_filter_drops_fabricated_citation() -> None:
    m = _load()
    # teacher emits a fabricated citation in every seat AND synthesis -> must be dropped
    bad = "Per Wong v Lee [2099] HKCFI 9999, proceed. 中文摘要：見上。"
    rows, stats = m.generate_traces(FIN_TASKS, _client(bad))
    # either every seat is gated out (abstention synthesis, which is clean) OR the
    # dirty synthesis is dropped — never a trace that contains the fabrication
    for r in rows:
        assert "9999" not in r["messages"][2]["content"]


def test_gate_filter_drops_false_arithmetic() -> None:
    m = _load()
    bad = "Runway: 100000 / 5000 = 25 months. Proceed. 中文摘要：見上。"
    rows, _ = m.generate_traces(FIN_TASKS, _client(bad))
    for r in rows:
        assert "= 25" not in r["messages"][2]["content"]  # false arithmetic never distilled


def test_abstention_cap() -> None:
    m = _load()
    # broken client -> all seats empty -> abstention synthesis; cap should limit how many we keep
    broken = SimpleNamespace(generate=lambda s, u: (_ for _ in ()).throw(RuntimeError("down")))
    tasks = [{"id": f"t{i}", "prompt": "Model our runway and flag AML for Stripe."} for i in range(8)]
    rows, stats = m.generate_traces(tasks, broken, abstain_cap=0.25)
    assert stats["abstentions"] <= 2  # 25% of 8


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_distill_council: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
