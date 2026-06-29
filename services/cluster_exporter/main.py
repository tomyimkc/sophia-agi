# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophia cluster Prometheus exporter — stdlib HTTP server (R3).

Exposes:

* ``GET /metrics`` — Prometheus text-format fleet + MTTR metrics
* ``GET /health``  — liveness/readiness JSON

Metrics are computed on scrape from a ``FleetProvider`` sweep and the incident ledger.
Default source is the offline ``MockProvider`` so the exporter runs with no GPUs and no
keys; set ``SOPHIA_CLUSTER_SOURCE=runpod`` (with ``RUNPOD_API_KEY``) for live inventory.

    python3 -m services.cluster_exporter.main --port 9881
    curl localhost:9881/metrics

The Prometheus text rendering (``render_metrics``) is a pure function over a sweep +
ledger stats, so it is fully unit-testable without binding a socket.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cluster import ledger as ledger_mod  # noqa: E402
from agent.cluster.acceptance import accept_node, mock_benchmark_runner  # noqa: E402
from agent.cluster.health import NodeMetrics, Verdict, evaluate_node  # noqa: E402
from agent.cluster.provider import get_provider, sweep  # noqa: E402


def _esc(label_value: str) -> str:
    return str(label_value).replace("\\", "\\\\").replace('"', '\\"')


def _metric(lines: list[str], name: str, help_text: str, mtype: str,
            samples: list[tuple[dict, float]]) -> None:
    lines.append(f"# HELP {name} {help_text}")
    lines.append(f"# TYPE {name} {mtype}")
    for labels, value in samples:
        if labels:
            lbl = ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items())
            lines.append(f"{name}{{{lbl}}} {value}")
        else:
            lines.append(f"{name} {value}")


def render_metrics(nodes: list[NodeMetrics], mttr: dict, *,
                   acceptance: dict[str, bool] | None = None) -> str:
    """Render a fleet sweep + MTTR stats as Prometheus text format."""

    healths = [evaluate_node(n) for n in nodes]
    lines: list[str] = []

    # Per-node verdict (0=PASS,1=WARN,2=FAIL) and reachability.
    _metric(lines, "sophia_node_health", "Node health verdict (0=PASS,1=WARN,2=FAIL)",
            "gauge", [({"node": h.node_id}, int(h.verdict)) for h in healths])
    _metric(lines, "sophia_node_reachable", "Node reachable (1) or not (0)", "gauge",
            [({"node": n.node_id}, 1 if n.reachable else 0) for n in nodes])

    # GPU telemetry (only emit where measured — absent ≠ zero).
    _metric(lines, "sophia_gpu_temp_celsius", "GPU temperature in Celsius", "gauge",
            [({"node": n.node_id}, n.gpu_temp_c) for n in nodes if n.gpu_temp_c is not None])
    _metric(lines, "sophia_gpu_ecc_uncorrectable", "Uncorrectable ECC error count", "gauge",
            [({"node": n.node_id}, n.ecc_uncorrectable) for n in nodes if n.ecc_uncorrectable is not None])
    _metric(lines, "sophia_rdma_links_down", "Count of down RDMA/IB links", "gauge",
            [({"node": n.node_id}, n.rdma_link_down) for n in nodes if n.rdma_link_down is not None])
    _metric(lines, "sophia_nvlink_down", "Count of down NVLink lanes", "gauge",
            [({"node": n.node_id}, n.nvlink_down) for n in nodes if n.nvlink_down is not None])

    # Fleet rollups.
    fail = sum(1 for h in healths if h.verdict == Verdict.FAIL)
    warn = sum(1 for h in healths if h.verdict == Verdict.WARN)
    _metric(lines, "sophia_fleet_nodes_total", "Total nodes swept", "gauge",
            [({}, len(nodes))])
    _metric(lines, "sophia_fleet_nodes_failing", "Nodes with FAIL verdict", "gauge",
            [({}, fail)])
    _metric(lines, "sophia_fleet_nodes_warning", "Nodes with WARN verdict", "gauge",
            [({}, warn)])

    # Acceptance (R2) — 1 accepted, 0 rejected — when provided.
    if acceptance:
        _metric(lines, "sophia_node_acceptance_pass", "Bring-up acceptance (1=accepted)",
                "gauge", [({"node": k}, 1 if v else 0) for k, v in sorted(acceptance.items())])

    # MTTR + self-heal (R1/R4).
    _metric(lines, "sophia_incidents_total", "Incidents recorded in the ledger", "gauge",
            [({}, mttr.get("total", 0))])
    _metric(lines, "sophia_incidents_open", "Currently-open incidents", "gauge",
            [({}, mttr.get("open", 0))])
    _metric(lines, "sophia_job_mttr_seconds", "Mean time to recovery (seconds)", "gauge",
            [({}, mttr.get("mttr_seconds_mean", 0.0))])
    _metric(lines, "sophia_selfheal_ratio", "Auto-healed / recovered incident ratio", "gauge",
            [({}, mttr.get("self_heal_ratio", 0.0))])

    return "\n".join(lines) + "\n"


def collect(source: str | None = None, ledger: Path | None = None,
            *, with_acceptance: bool = False) -> str:
    source = source or os.environ.get("SOPHIA_CLUSTER_SOURCE", "mock")
    ledger = ledger or ledger_mod.DEFAULT_LEDGER
    nodes = sweep(get_provider(source))
    mttr = ledger_mod.mttr_stats(ledger)
    acceptance = None
    if with_acceptance:
        acceptance = {}
        for n in nodes:
            res = accept_node(n.node_id, n.gpu_model, mock_benchmark_runner)
            acceptance[n.node_id] = res.accepted
    return render_metrics(nodes, mttr, acceptance=acceptance)


class _Handler(BaseHTTPRequestHandler):
    source = "mock"
    ledger_path: Path | None = None
    with_acceptance = False

    def log_message(self, *args):  # quiet by default
        pass

    def _send(self, code: int, body: str, content_type: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802 (stdlib signature)
        if self.path.rstrip("/") in ("/metrics", ""):
            try:
                body = collect(self.source, self.ledger_path,
                               with_acceptance=self.with_acceptance)
                self._send(200, body, "text/plain; version=0.0.4; charset=utf-8")
            except Exception as exc:  # exporter must never crash the scrape
                self._send(500, f"# exporter error: {exc}\n", "text/plain; charset=utf-8")
        elif self.path.rstrip("/") == "/health":
            self._send(200, json.dumps({"status": "ok", "source": self.source}),
                       "application/json")
        else:
            self._send(404, "not found\n", "text/plain; charset=utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Sophia cluster Prometheus exporter (R3).")
    ap.add_argument("--port", type=int, default=9881)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--source", default=os.environ.get("SOPHIA_CLUSTER_SOURCE", "mock"))
    ap.add_argument("--acceptance", action="store_true",
                    help="also run bring-up acceptance per node (heavier scrape)")
    args = ap.parse_args(argv)

    _Handler.source = args.source
    _Handler.with_acceptance = args.acceptance
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"sophia cluster exporter on http://{args.host}:{args.port}/metrics (source={args.source})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
