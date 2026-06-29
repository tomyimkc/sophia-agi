#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline A2A demo — two Sophia agents discovering + delegating to each other.

Runs the whole agent-to-agent path with NO network, key, or real model (the ``mock`` model
client). Shows the three building blocks from docs/09-Agent/A2A-Sophia-Interop.md:

  1. Discovery   — a client fetches a peer's Agent Card.
  2. Delegation  — the client sends a task; the peer routes it through the swarm and returns
                   a fail-closed answer over the A2A task lifecycle.
  3. Trust       — the client runs the peer's answer through ITS OWN gate before trusting it.
  4. MCP-as-agent — the same peer reached as a gated MCP tool via gateway.federation.

Usage:
    python tools/a2a_demo.py
    python tools/a2a_demo.py --task "validate: Confucius wrote the Dao De Jing"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import a2a  # noqa: E402
from agent import model as m  # noqa: E402


def _client():
    return m.ModelClient(m.resolve_config("mock"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline A2A agent-to-agent demo")
    ap.add_argument("--task", default="Summarise the provenance discipline in two lines.")
    args = ap.parse_args()

    # --- Peer "Spark-Sophia": Sophia served as an A2A agent --------------------
    card = a2a.sophia_agent_card("http://spark.local:8080", name="Spark-Sophia")
    ok, problems = card.validate()
    print(f"[peer] Agent Card valid: {ok} {problems if problems else ''}")
    server = a2a.A2AServer(card, client=_client())

    # --- Caller: another agent talking to the peer over a stub transport --------
    client = a2a.A2AClient(a2a.StubA2ATransport(server))

    print("\n=== 1. DISCOVERY (agent/getCard) ===")
    fetched = client.get_card()
    print(f"  name   : {fetched['name']}")
    print(f"  skills : {[s['id'] for s in fetched['skills']]}")
    print(f"  gated  : {fetched['x-sophia']['provenanceDiscipline']}  reduce={fetched['x-sophia']['reduce']}")

    print("\n=== 2. DELEGATION (message/send) ===")
    task = client.send_task(args.task)
    print(f"  task {task.id}  state={task.state}  estCostSteps={task.est_cost_steps}")
    print(f"  gate : {task.gate.to_dict() if task.gate else None}")
    print(f"  answer: {task.answer()[:200]}")

    print("\n=== 3. TRUST BOUNDARY (peer answer through OUR gate) ===")
    verdict = client.delegate_to_peer(args.task)
    print(f"  verdict: accept={verdict.accept} label={verdict.label} reason={verdict.reason!r}")

    print("\n=== 3b. FAIL-CLOSED on an empty task ===")
    empty = client.send_task("   ")
    print(f"  state={empty.state}  error={empty.error!r}  (no swarm spawned on noise)")

    print("\n=== 4. MCP-AS-AGENT (remote agent as a gated MCP tool) ===")
    bridge = a2a.A2AMcpTransport(client)
    out = bridge.call("sophia.delegate", {"task": args.task})
    print(f"  sophia.delegate -> accept={out['accept']} sources={out['sources']}")
    print(f"  text: {out['text'][:160]}")

    print("\nOK — full agent-to-agent round trip, offline, fail-closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
