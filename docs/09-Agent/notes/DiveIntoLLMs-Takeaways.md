# Takeaways from Lordog/dive-into-llms — and what we ported

**Source:** <https://github.com/Lordog/dive-into-llms> (SJTU graduate teaching lab; 11 chapters
of slides + Jupyter notebooks, Chinese-language). **Date reviewed:** 2026-06-29.

## Honest framing

`dive-into-llms` is *pedagogical breadth*, not a frontier agent system. Sophia is
architecturally past it on trust / provenance / verification / calibration. Its value to us is
narrow and real: **four chapters map onto open items in the failure ledger and the unbuilt
"more-capable half" of the harness roadmap.** We treat it as a lab-bench source for specific
primitives and external eval methodology, never as a strategy or an AGI blueprint.

## Chapter → Sophia component map

| Chapter | Maps onto | What we took |
|---|---|---|
| 9. GUI agents (Qwen2-VL-7B + OS-Kairos + LLaMa-Factory; action vocab `CLICK/TYPE/SCROLL/PRESS_BACK/PRESS_HOME/ENTER/IMPOSSIBLE`; per-action **confidence 1–5**) | `agent/harness.py`, Harness-Roadmap Build 3 (ultra-long-horizon, unbuilt) | The per-action self-confidence is the reusable idea → ported as `agent/gui_action_gate.py` (confidence-gated, fail-closed action decision). |
| 10. Agent safety (R-Judge: multi-turn records, binary safety label + risk description; F1/Recall/Specificity) | `agent/ssil_*` battery, beside the G9D eval-awareness tripwire | The *methodology* (not the data) → `agent/ssil_risk_awareness.py` (gate **G10R**) + an author-written, decontam-safe case bank. |
| 6. Jailbreak attacks | `agent/auto_approval_breaker.py`, `tests/test_server_gateway_live.py` | Attack templates to extend the gateway red-team corpus (not yet ported — see next steps). |
| 7. LLM steganography | `agent/dataflow/taint.py`, `sophia_mcp/audit.py` | Covert-channel exfil detection — a real gap for a confidentiality-first system (thesis T4). |
| 3. Knowledge editing (ROME/MEMIT-style) | `okf/graph.py`, `agent/belief_revision_scaling.py` | Used as the *anti-pattern* baseline: parametric weight-editing is unauditable; we retract claims in the provenance graph instead. |
| 11. RLHF (PPO) | `provenance_bench/rl_reward.py`, `agent/verified_trace_rlvr.py` | Nothing new — RLVR is already ahead of vanilla PPO-RLHF. Skipped. |

## What shipped in this pass

1. **`agent/ssil_risk_awareness.py` (G10R)** — open-agent risk-awareness eval adapting R-Judge's
   methodology. Scores a candidate's safe/unsafe judgments over multi-turn records with
   F1/Recall/Specificity. Fail-closed: a **missed risk** (low unsafe-class recall) is a hard
   `reject`; an **alarmist** (low specificity, over-flagging benign records) is `quarantine`d via
   a specificity guardrail (the dual of the dikaiosyne false-equivalence guardrail). Standalone
   gate, NOT folded into the G8 danger-aggregate (the sign is inverted: here higher recall =
   better). Tests: `tests/test_ssil_risk_awareness.py` (8).
2. **`eval/risk_awareness/rjudge_adapted_v1.jsonl`** — 12 author-written, decontam-safe records
   across R-Judge's domains (OS, finance, web, IoT, software, health, applications, IoT),
   including 3 benign safe-controls to pin the specificity guardrail. Labels are single-source
   (author) until ≥2 judge families confirm — stated in the gate boundary.
3. **`agent/gui_action_gate.py`** — chapter-9's per-action confidence (1–5) ported into Sophia's
   idiom: a side-effecting action runs only after clearing **risk class → confidence floor →
   precondition verifier**, else `escalate`/`abstain`/`block`. Confidence cannot buy past a
   high-risk/irreversible action (always human-in-the-loop). Tests:
   `tests/test_gui_action_gate.py` (10).
4. **`agi-proof/benchmark-results/calibration-verifier/measurement_spec.json`** — the T3 thesis,
   pre-registered (status `preregistration_only`, `go:false`, `canClaimAGI:false`).

## Research theses (specificity for "best agents targeting ASI")

Our edge is *scale that carries its own proof*. The ASI-relevant move is making verification
scale **with** capability so the agent stays auditable as it gets stronger. Priority order:

- **T3 — Calibration that scales with capability** *(highest leverage; pre-registered here).*
  Train a separate trace-feature verifier (semantic entropy + corroboration + author-confidence,
  no answer access) to predict correctness, and show calibration (ECE) holds/improves across
  base sizes rather than degrading. The failure ledger already shows the provenance prior is a
  weak (~0.52 balanced-acc), non-monotonic correctness predictor — this is the bar to beat. A
  NULL is a valid, publishable outcome. An ASI you cannot calibrate is one you cannot deploy.
- **T1 — Verifier-gated computer-use** *(seeded here by `gui_action_gate.py`).* Long-horizon
  GUI/computer-use where every *action* (not just every output) is a claim with a precondition
  verifier. Open question: does per-action verification cost grow sub-linearly with horizon if
  verified sub-goal invariants are cached against the KV-stable prefix (`agent/context_manager.py`)?
  Becomes Harness-Roadmap Build 3.
- **T2 — Adversarial self-modeling / sandbagging.** Extend `ssil_eval_awareness.py` (G9D) + the
  new G10R probes to measure whether the agent behaves differently when it detects evaluation.
  Frontier-safety-relevant (scheming/sandbagging); we have the SSIL scaffold to measure it under
  the no-overclaim gates.
- **T4 — Covert-channel-resistant multi-agent.** Combine chapter-7 steganography with the
  council/A2A layer (`agent/a2a.py`, `agent/sector_council.py`): can a council member exfiltrate
  via a steganographic side-channel, and does taint-tracking catch it? Frames multi-agent
  orchestration as an information-flow-security problem (Bell-LaPadula wheelhouse).

## Next steps (not done in this pass)

- Build the T3 harness `tools/run_calibration_verifier_eval.py` + its decontaminated trace
  corpus + ≥2 judge families (GPU work goes through the GitHub Actions cost-guarded path).
- Port chapter-6 jailbreak templates into `tests/test_server_gateway_live.py`.
- Open a failure-ledger item for G10R (author-only labels → needs ≥2 judge families before any
  cited claim) and for T3.
- Wire `gui_action_gate.gate_action` into `agent/harness.py` as the Build-3 action admission step.

## Honest limits

- The G10R case bank is small (12) and author-written; it pins the gate's behaviour, it is NOT
  evidence about open-deployment safety. Labels need ≥2 independent judge families before any
  number is cited in `RESULTS.md`.
- `gui_action_gate.py` is a *decision* gate — it does not perceive a UI or execute anything; it
  trusts the caller's risk tags and precondition verifier.
- All four theses are candidate-only; `canClaimAGI` stays false.
