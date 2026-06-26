# AI Compute-Cluster Reliability & Performance Engineering — Roadmap

**Purpose.** This document maps the Sophia repo against the responsibilities of an
*AI 算力集群性能与可靠性工程师* (AI compute-cluster performance & reliability engineer,
DeepSeek-style infra/运维 role) and proposes a concrete, repo-grounded development
path so the project demonstrates the skills that role demands.

It is a **portfolio/skills roadmap**, not a claim that Sophia is a production
cluster-management system. As with everything in this repo, every number a future
build emits must clear the no-overclaim measurement gate
([RESULTS.md](../../RESULTS.md), [VISION.md](../../VISION.md)).

---

## 1. Where the gap is — honest assessment

Sophia's center of gravity is the **trust layer** of reasoning: provenance,
verification, calibration, fail-closed gating. The infra role's center of gravity is
the **reliability layer** of GPU compute: MTTR, observability, bring-up, stress
testing, self-healing.

These are different disciplines, but Sophia already touches GPU infrastructure in
exactly the places this role cares about:

| Existing asset | What it already proves | Gap vs. the role |
|----------------|------------------------|------------------|
| `tools/runpod_rlvr.py` — create pod → poll SSH → run → copy artifacts → **always delete** | Full GPU node lifecycle automation over an API + SSH | No health-gating, no baseline check before "投入生产", no MTTR loop |
| `tools/runpod_train.py`, `runpod_speedup.py` | Real CUDA training + a timing micro-benchmark on rented GPUs | Not a reusable **stress/perf-baseline harness**; no acceptance criteria |
| `tools/estimate_runpod_eta.py` | Calibrated wall-clock model from *observed real runs* | Heuristic only; not fed by live telemetry |
| `runpod` MCP server (`list-gpu-types`, `get-pod`, `endpoint-health`, …) | Programmatic fleet inspection already wired | Inspection only — no monitoring/alerting/auto-remediation built on it |
| `agent/security/`, `agent/dataflow/firewall.py`, `sophia_mcp/audit.py` | Fail-closed gating + audit discipline | Not yet applied to *infra* actions (node drain, reboot, job kill) |

**The strategic move:** don't bolt on a generic SRE demo. Wrap Sophia's *existing*
GPU usage in a reliability/observability layer, and reuse the repo's signature
discipline — **fail-closed gating, provenance, calibrated confidence** — as the
differentiator. A reliability tool that *abstains and escalates when unsure* instead
of blindly rebooting a node is a uniquely on-brand contribution.

---

## 2. Roadmap mapped to the job's four responsibilities

### R1 — Daily ops, fault localization, shorten MTTR
**职责:** 巡检、维修、故障定位与生命周期管理，缩短 MTTR。

- **`tools/cluster/inspect_fleet.py` — fleet 巡检 (inspection) sweep.** Build on the
  `runpod` MCP `list-pods`/`get-pod`/`endpoint-health` tools. For each node, collect
  GPU temp, ECC error counts, throttle state, XID errors, disk/mem pressure, NCCL/RDMA
  link health. Emit a structured health record per node with a pass/warn/fail verdict.
- **Incident model + MTTR ledger (`agi-proof/`-style).** A JSONL ledger of incidents
  (detected_at → diagnosed_at → recovered_at) so MTTR is *measured*, not asserted —
  the same evidence discipline the repo already uses for benchmark claims.
- **Fault-localization playbook.** A decision tree (GPU fell off bus / XID 79 / ECC
  storm / NCCL timeout / RDMA flap) → root-cause hypothesis → remediation, with each
  branch citing the signal that triggered it (provenance for ops).

### R2 — Fast delivery, baseline check, performance validation before production
**职责:** 新资源快速交付上线，基线检查与性能调优、验证。

- **`tools/cluster/bringup.py` — node acceptance gate.** Reuse the
  `runpod_rlvr.py` lifecycle to provision a node, then run an **acceptance suite**
  before declaring it production-ready: `nvidia-smi` topology, `dcgmi diag`,
  single-GPU GEMM/FLOPs, HBM bandwidth, NVLink/`p2pBandwidthLatencyTest`,
  NCCL all-reduce bus bandwidth, RDMA `ib_write_bw`. Node passes only if every metric
  clears a committed baseline — a **fail-closed bring-up gate**, exactly mirroring
  `agent/gate.py`'s philosophy applied to hardware.
- **Performance-baseline registry.** Commit reference baselines per GPU type (extend
  the `list-gpu-types` data) so regressions are detected against a real, versioned
  reference — the same pattern as the repo's seal/holdout protocol.

### R3 — Monitoring, alerting, observability
**职责:** 建设监控、告警与可观测性体系，实时感知集群健康。

- **Prometheus exporter (`services/cluster_exporter/`).** Expose the R1 inspection
  records as Prometheus metrics (`sophia_gpu_temp_celsius`, `sophia_gpu_ecc_errors_total`,
  `sophia_node_acceptance_pass`, `sophia_job_mttr_seconds`, …). Mirror the existing
  `services/rag_api` service layout so it fits the repo's conventions.
- **Grafana dashboards (as code).** Commit dashboard JSON + alert rules under
  `docs/11-Platform/` or a new `observability/` dir: fleet health, GPU saturation,
  RDMA/NCCL link errors, MTTR trend, acceptance pass-rate.
- **Alert rules with calibrated thresholds.** Reuse `agent/calibration.py`'s ECE /
  risk-coverage machinery to set alert thresholds that are *calibrated* against real
  incident outcomes — minimizing false-page rate. This is the repo's most defensible
  cross-over contribution: calibrated, low-false-positive alerting.

### R4 — Automation toolchain, raise self-heal rate, lower manual cost
**职责:** 自动化运维工具链，提升故障自愈率与运维效率。

- **`tools/cluster/heal.py` — gated auto-remediation.** Map symptoms → actions
  (reset GPU, drain+reboot node, requeue job, cordon). **Critical differentiator:**
  route every remediation through the repo's `agent/guarded.py` + `sophia_mcp/audit.py`
  fail-closed gate. Low-risk + high-confidence → auto-heal; high-stakes or low-confidence
  → human-in-the-loop escalation. This is VISION.md's "risk-proportional
  human-in-the-loop" applied to ops, and it directly addresses 自愈率 while staying safe.
- **Self-heal metrics.** Track auto-resolved vs. escalated ratio over time so the
  自愈率 improvement is measured against the MTTR ledger.

---

## 3. Suggested build order (smallest credible slice first)

1. **`tools/cluster/inspect_fleet.py` + incident/MTTR ledger** — pure read path over the
   already-wired `runpod` MCP tools. No new infra, immediate signal. (R1)
2. **`services/cluster_exporter/` Prometheus endpoint** over those records, plus one
   committed Grafana dashboard JSON. (R3) — gives a visible, demo-able artifact.
3. **`tools/cluster/bringup.py` acceptance gate** reusing `runpod_rlvr.py`'s lifecycle
   + a committed per-GPU baseline. (R2)
4. **`tools/cluster/heal.py`** behind the existing fail-closed gate, with the self-heal
   ratio feeding the MTTR ledger. (R4)
5. **Calibrated alert thresholds** via `agent/calibration.py`, once enough incident
   data exists to fit them. (R3/R4)

Each slice is independently shippable, reuses existing repo machinery, and produces a
measurable artifact rather than a claim.

### Implementation status — all five slices landed (offline-first, tested)

A first working cut of every slice now ships. Core logic lives in `agent/cluster/`
(offline-testable, injectable providers); CLIs in `tools/cluster/`; the exporter in
`services/cluster_exporter/`; observability-as-code in `observability/`.

| Slice | Modules | CLI / service | Tests |
|-------|---------|---------------|-------|
| R1 inspect + MTTR ledger | `agent/cluster/health.py`, `provider.py`, `ssh_provider.py` (live SSH telemetry), `playbook.py`, `ledger.py` | `tools/cluster/inspect_fleet.py` | `tests/test_cluster_health.py`, `test_cluster_ledger.py`, `test_cluster_ssh_provider.py` |
| R3 observability | `services/cluster_exporter/main.py` (pure-stdlib Prometheus exporter) | `python -m services.cluster_exporter.main` | `tests/test_cluster_exporter_calibrate.py` |
| R3 dashboards/alerts | — | `observability/grafana/*.json`, `observability/prometheus/alerts.yml` | (JSON/YAML validated) |
| R2 bring-up gate | `agent/cluster/acceptance.py`, `config/cluster_baselines.json` | `tools/cluster/bringup.py` | `tests/test_cluster_acceptance.py` |
| R4 gated self-heal | `agent/cluster/heal.py` (audited, fail-closed) | `tools/cluster/heal.py` | `tests/test_cluster_heal.py` |
| R3/R4 calibrated thresholds | `agent/cluster/calibrate.py` (reuses `agent/calibration.py`) | `tools/cluster/calibrate_alerts.py` | `tests/test_cluster_exporter_calibrate.py` |

**Live telemetry — `SSHProvider` (shipped).** `agent/cluster/ssh_provider.py` fills the
DCGM-level fields by running a single probe script per node over SSH and parsing real
`nvidia-smi` / `dmesg` (XID) / `nvidia-smi nvlink` / InfiniBand `state` output into
`NodeMetrics`. It reuses the repo's trusted SSH lifecycle (`PodConnection`, `_ssh_base`,
and `_api_request` for RunPod discovery from `tools/runpod_rlvr.py`), sweeps the fleet
in parallel, and the parser (`parse_probe`) is a pure function so it is fully tested
offline with canned output (`tests/test_cluster_ssh_provider.py`). Targets come from a
JSON inventory (`config/cluster_inventory.example.json` /
`SOPHIA_CLUSTER_INVENTORY`) or live RunPod discovery; the key is
`SOPHIA_CLUSTER_SSH_KEY`. Drive it with `--source ssh`:

    SOPHIA_CLUSTER_SSH_KEY=~/.ssh/id_ed25519 \
      python3 tools/cluster/inspect_fleet.py --source ssh \
      --inventory config/cluster_inventory.example.json --diagnose

**Honest bounds.** Telemetry defaults to a deterministic `MockProvider` (no GPUs, no
keys, no cost). The `RunPodProvider` maps the live `GET /pods` inventory (status/shape
only) and leaves DCGM-level fields `None` — the fail-closed evaluator treats those as
"unknown, can't clear", so use `--source ssh` for real GPU health. A probe section that
is absent or unparseable (no IB fabric, ECC reported `N/A`) also stays `None` → WARN,
never a false green. The bring-up baselines are conservative reference numbers and must
be re-measured against the operator's own reference node before production use.
Auto-remediation is **dry-run by default** and never executes without explicit opt-in
(`SOPHIA_CLUSTER_HEAL=1` + a wired executor). `heal.py` ships only a no-op executor.

## 4. Skills this demonstrates for the role

- **Linux/cluster ops & fault analysis** → R1 inspection + fault-localization playbook.
- **GPU/RDMA/NCCL infrastructure** → bring-up acceptance suite, RDMA/NCCL bandwidth gates.
- **Shell/Python/LLM automation** → the entire `tools/cluster/` toolchain, LLM-assisted
  root-cause via Sophia's reasoning core.
- **Prometheus/Grafana observability** → `cluster_exporter` + dashboards-as-code.
- **Bring-up / acceptance / perf-baseline** → R2 gate + baseline registry.
- **The differentiator** → calibrated, fail-closed, audited remediation: reliability
  engineering that *knows when to abstain*, carried over from Sophia's trust layer.

---

*Bound to [VISION.md](../../VISION.md): every metric this roadmap produces must be
reproducible and clear the no-overclaim gate. This is a skills/portfolio roadmap, not
a production cluster-management claim.*
