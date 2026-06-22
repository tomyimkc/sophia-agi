# Sophia — Methodology & Results (arXiv-style writeup)

*A reproducible, citable summary of what Sophia is, how it is measured, and what the
evidence does and does not show. For third-party reviewers and replicators.*

## Abstract

Sophia is a **provenance-aware, verifier-gated reasoning system**: every claim is
recorded with its sources, checked by a fail-closed gate, and **only published when the
gate accepts** — otherwise the system abstains. The central empirical result is that, on
genuinely-unknown-answer questions, the gated system **fabricates 0%** where the raw base
model fabricates 17–25%, a difference invisible to keyword scoring but measurable under a
calibration scorer and corroborated by two independent LLM-judge families. We make no
claim of AGI; we pre-register thresholds, measure against them honestly, and report every
number through a no-overclaim gate.

## 1. System

- **Belief graph (OKF):** claims carry `derivesFrom` provenance; supports counterfactual
  removal, retraction, and revision-cascade. (`okf/`)
- **Governance contract:** `record_claim → verify_claim → {accepted|rejected|superseded|held}`,
  Bell-LaPadula classification (no-read-up/no-write-down), budget caps, kill switch,
  idempotency, audit log. Versioned (semver) with golden-vector conformance. (`sophia_contract/`, `CONTRACT.md`)
- **Self-extending flywheel:** abstain → synthesize a verifier → validate on held-out →
  improve via verified reward → coverage rises, with an anti-gaming held-out check. (`selfextend/`)
- **Gateway:** a super-MCP proxy that gates any tool/skill through the contract (firewall,
  federation, verifiable skills, reliability registry, verified consensus). (`gateway/`, `docs/11-Platform/Sophia-Gateway.md`)

## 2. The no-overclaim measurement gate

A number is **VALIDATED** only with: ≥2 independent judge families in consensus
(judge ≠ subject), reported inter-judge agreement (Cohen's κ), ≥3 runs, and a 95% CI.
Deterministic verifiers (machine-checked, no LLM judge) are reported in their own,
honestly-bounded tier. Everything else is labelled **illustrative**. Hidden-eval prompts
are never published — only aggregates. (`RESULTS.md`, `SECURITY.md`)

## 3. Headline results (see `RESULTS.md` for the live, generated table)

| Result | Method | Value |
|---|---|---|
| Hallucination reduction (gate) | 2 judge families, 3 runs, N=290 | Δ **12.5%** [+5.6%, +19.4%], 0% FP-cost |
| Fabrication on unknown-answer items | deterministic calibration scorer, 3 runs | sophia-full **0%** vs raw **17–25%**; calibration Δ +22.0% [14.5%, 29.6%] |
| ↳ corroboration | 2 distinct judge families (GPT-4o + Claude) | inter-judge κ **0.74**; both rank sophia-full lowest fabrication |
| Cross-entity grounding | deterministic | false-positive **100% → 0%** at full recall on KB-covered entities |
| Self-extending loop | offline, deterministic | closes on a held-out domain (policy 0.5→1.0; fail-closed on unlearnable) |

## 4. Limitations (honest)

- The calibration packs and epistemic labels are **self-authored** — internally valid;
  full independence needs a third-party pack + human semantic review.
- **Live RLVR** (a GPU weight update) and **live grounding** (retrieval) are built as
  interfaces but not run as capability claims.
- Long-horizon autonomy is measured as a curve / mechanism, not a real multi-day run.
- These gaps are tracked, openly, in `agi-proof/failure-ledger.md`.

## 5. Reproduce

```bash
python scripts/demo_gate.py            # the gate, 30s, offline, no key
python tools/run_selfextend_loop.py    # the self-extending loop closing
python tools/run_grounding_gate.py     # cross-entity FP 100%->0%
python tools/run_calibration_judge.py agi-proof/baseline-ablation/abstain-pack-2026-06-22.json \
    /tmp/calib-runs/private-*.json --judge openrouter:openai/gpt-4o --judge openrouter:anthropic/claude-3.5-sonnet
```
Everything in CI (`.github/workflows/ci.yml`) is deterministic and offline; LLM-judge
re-runs need an inference key (OpenRouter recommended for judge-family diversity).

## 6. How to cite / verify independently

See `agi-proof/REPLICATION.md` for the third-party replication checklist, the OSF-style
pre-registration (`agi-proof/PRE-REGISTRATION.md`), and the data/code locations.
