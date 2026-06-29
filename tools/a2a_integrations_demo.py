#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline demo of the REAL-machinery A2A integrations (agent/a2a_integrations.py).

Shows the three wirings on top of the A2A scaffolding, with no network/key/real model:
  ① the source-discipline gate blocking a forbidden lineage merge,
  ② calling the real provenance.validate / epistemic.gate_check skills by id,
  ③ the governance contract version advertised on the Agent Card.

Usage: python tools/a2a_integrations_demo.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import a2a  # noqa: E402
from agent import a2a_integrations as ai  # noqa: E402
from agent import model as m  # noqa: E402

TRAP = "Confucius wrote the Dao De Jing."
OK = "The Analects is associated with the Confucian tradition."


def main() -> int:
    client_model = m.ModelClient(m.resolve_config("mock"))
    server = ai.make_sophia_a2a_server("http://spark.local:8080", client=client_model)
    client = a2a.A2AClient(a2a.StubA2ATransport(server), gate=ai.sophia_gate)

    print("=== ③ CONTRACT HANDSHAKE on the Agent Card ===")
    xs = client.get_card()["x-sophia"]
    print(f"  provenanceDiscipline={xs['provenanceDiscipline']}  contractVersion={xs.get('contractVersion')!r}")

    print("\n=== ① SOURCE-DISCIPLINE GATE (real verifier) ===")
    for text in (TRAP, OK):
        v = ai.sophia_gate(text)
        print(f"  {v.label:6} | {text!r}  -> {v.reason}")

    print("\n=== ② REAL CARD SKILLS by id ===")
    for skill in ("provenance.validate", "epistemic.gate_check"):
        t = ai.invoke_skill(client, skill, TRAP)
        print(f"  {skill:22} on trap -> {t.answer()[:90]}")
    t_ok = ai.invoke_skill(client, "provenance.validate", OK)
    print(f"  provenance.validate    on benign -> {t_ok.answer()[:90]}")

    print("\n=== default route (no @skill) -> swarm ===")
    t = client.send_task("summarise the provenance discipline")
    print(f"  swarm state={t.state} estCostSteps={t.est_cost_steps}")

    print("\nOK — real source-discipline + skills + contract, offline, fail-closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
