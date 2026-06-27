# Frontier-Research Implementation Plan — 5 Candidates

**Status:** plan / candidate. Nothing here is a capability claim. Every artifact this
plan produces must carry `candidateOnly: true`, `level3Evidence: false`, a
pre-registration entry under `agi-proof/preregistered-thresholds.md`, and a
failure-ledger row until it clears the no-overclaim gate (≥2 judge families,
κ ≥ 0.40, ≥3 runs, CI excludes 0).

This plan takes five frontier-lab research ideas and shows how to land each one as
**infrastructure + a benchmark setting** that is *constructive with the existing
repo* — i.e. wiring the seams Sophia already has, not rebuilding. For each candidate:
existing seam → infra to add → benchmark setting → agent/harness/MCP wiring →
loop & prompt/context engineering → pre-registered thresholds → ledger entry →
language/structure/effort.

## Implementation status (2026-06-26)

All five candidates are **implemented as offline, deterministic, candidate-grade
machinery** on branch `claude/agi-asi-research-ideas-g5ss38`. Each carries a pre-registration
entry, an OPEN failure-ledger row (the real-pack / third-party / model-backed gap), tests,
and a `*.public-report.json` marked `candidateOnly`/`syntheticData`. No model API key was
present, so every "real-model" path is wired but unrun — that is the standing OPEN gap.

| # | Candidate | Modules / tools | Offline result | Tests |
|---|---|---|---|---|
| C1 | Conformal abstention | `graded_decision.decide_conformal`, `guarded on_fail="conformal"`, `tools/fit_conformal_policy.py`, MCP `sophia_conformal_decide` | held-out coverage validity holds for α∈{0.05,0.10,0.20} | `test_conformal_gate`, `test_guarded_conformal` |
| C3 | Abstention-aware scoring | `agent/abstention_scoring.py`, `tools/run_abstention_scoring.py` | break-even λ* recovered on synthetic | `test_abstention_scoring` |
| C4 | CoT faithfulness | `tools/run_faithfulness_bench.py`, MCP `sophia_cross_trace_mine` | v2 drop separates load-bearing vs decorative (AUROC 1.0); cross-trace contradiction found | `test_faithfulness_bench` |
| C2 | Prover-verifier | `agent/prover_verifier.py`, `tools/run_prover_verifier.py` | leak rate 0.70→0.00, 0 false positives, monotone | `test_prover_verifier` |
| C5 | Truth-probe | `tools/eval_truth_probe.py`, `activation_probes.probe_deception_context` + hidden-state seam | AUROC 1.0 separation but ECE 0.14 (honest "ranks well, miscalibrated"); probe→deception block wiring | `test_truth_probe` |

Phase 0 substrate (`tools/emit_outcome_records.py`) is shared by C1/C3/C5. The sections below
are the original design rationale.

---

## 0. Orienting constraints (read first)

These are repo invariants the plan must respect; they shape every choice below.

| Constraint | Consequence for this plan |
|---|---|
| **No GPU training in the gate** (RLVR/LoRA are offline *selection*). | All five land as **offline calibration / selection / measurement**, not parameter updates. Anything needing a weight update is staged behind `continual_plasticity.evaluate_update` and RunPod, never on the default path. |
| **Fail-closed.** | Every new decision component must *downgrade-only* (answer→hedge→abstain), never upgrade. Mirror `grounded_agent.apply_graded_decision`. |
| **No-overclaim gate.** | Each benchmark emits a `*.public-report.json` to `agi-proof/benchmark-results/` with `candidateOnly`/`level3Evidence`, and only headlines after ≥2 judge families + κ + ≥3 runs + CI. |
| **Language/runtime.** | Decision/verification logic = **Python 3 stdlib-only**, deterministic, offline by default (matches `conformal_gate.py`, `faithfulness_probe.py`). Optional model backends are lazy seams that fail closed when absent. Hot/parallel paths that already live in Rust (`sophia-storage`, kernels) stay Rust; none of these 5 need new Rust. |
| **MCP surface.** | FastMCP `@mcp.tool()` in `sophia_mcp/server.py`, thin wrapper → `sophia_mcp/tools_impl.py`, JSON-string return. Side-effecting tools go behind `SOPHIA_MCP_GATEWAY=1`. |
| **Pre-registration discipline.** | Thresholds (α, flip-rate floors, λ penalties, probe AUROC bars) are registered *before* the run, in `preregistered-thresholds.md`, with falsification rules. |

**Shared substrate built once, used by all five (Phase 0):**

1. **Labeled-outcome record format** — one JSONL schema every candidate reads/writes:
   `{"id", "domain", "risk", "confidence", "nonconformity", "correct", "abstained", "policy"}`.
   This is the bridge `tools/calibrate_graded_thresholds.py --emit-records` already
   half-builds; promote it to the canonical emitter (`tools/emit_outcome_records.py`)
   so C1/C3/C5 all consume the same labeled rows from one real-model run.
2. **Judge harness reuse** — `tools/run_calibration_judge.py` (already gives κ + 2-family)
   is the validation backend for C2/C3/C4 too. Don't fork it.
3. **Report schema** — `sophia.<candidate>_report.v1` with the no-overclaim fields,
   written under `agi-proof/benchmark-results/`.
4. **Pack format** — reuse the 18-case abstain pack + 35 attribution traps + OKF wiki
   as the in-domain calibration corpus; every candidate also needs a **held-out**
   split and a **third-party** TODO row (the independence gap is the recurring blocker
   in the ledger — design for it from day one).

---

## C1 — Conformal abstention (distribution-free risk control)  ★ start here

**Papers:** Kalai et al., *Why Language Models Hallucinate* (arXiv:2509.04664);
*Learning Conformal Abstention Policies* (arXiv:2502.06884);
*Chance-Constrained Hallucination Risk Control* (arXiv:2602.01637).

**Existing seam (already in repo):** `agent/conformal_gate.py` (`ConformalPolicy`,
`fit_conformal_policy`, `evaluate_policy`, `write_conformal_report`, split-conformal
finite-sample quantile) and `agent/competence_model.py` (per-domain conformal
threshold + ECE/AURC). **Gap:** it is *not wired* into the live decision —
`graded_decision.DEFAULT_THRESHOLDS = {"hi":0.7,"lo":0.4}` is still hand-picked, and
`apply_graded_decision` uses those literals.

**Infra to add (small):**
- `tools/emit_outcome_records.py` (shared, Phase 0) → produces calibration rows from a
  real backend over the abstain pack + traps.
- `tools/fit_conformal_policy.py` → calls `fit_conformal_policy` per risk bucket, writes
  `agi-proof/benchmark-results/conformal-policy.public-report.json`, persists the fitted
  `ConformalPolicy` to `config/` as a **versioned artifact** (not a code constant).
- Wire: add `mode="conformal"` to `graded_decision.decide` that loads the fitted policy
  and decides on `1 - grounded_source_confidence` as the nonconformity score, falling
  back to the hand-picked thresholds when no policy artifact is present (fail-safe).
  Keep it **downgrade-only**, exactly like the graded path in `guarded.py`.

**Benchmark setting:** sweep α ∈ {0.05, 0.10, 0.20}; report, per risk bucket and α,
the realized **false-answer rate** (must be ≤ α on the calibration guarantee), the
**coverage** (answer rate), and **selective accuracy** — the coverage-vs-risk frontier.
Headline metric: *does the realized error rate among `answer` verdicts stay ≤ α on a
held-out split?* (the conformal validity check). Compare against the hand-picked-0.7
baseline on the same rows — this is the "heuristic → certified" delta.

**Agent / harness / MCP wiring:**
- Harness: `guarded.py` gains `on_fail="conformal"` alongside `"graded"`.
- MCP: extend `sophia_uncertainty_score` (already present) to optionally return the
  conformal verdict + `coverageGuarantee`, or add `sophia_conformal_decide(nonconformity, risk)`.
- Gateway: read-only, no gateway flag needed.

**Loop & prompt/context engineering:** the **calibration set is the loop** — re-fit the
policy whenever `competence_model` ingests new graded outcomes (close it through
`active_inference.discover_gaps` → emit records → re-fit → re-evaluate). No prompt
engineering needed (deterministic scorer); the *context* lever is the nonconformity
score choice — start with `1 - grounded_source_confidence`, later add neighbor
corroboration + self-consistency as a 2-D nonconformity vector.

**Pre-registered thresholds:** α targets fixed in advance; falsification = realized
held-out false-answer rate exceeds α beyond CI, OR conformal gate shows no coverage
gain over hand-picked thresholds at matched risk.

**Ledger entry:** `conformal-gate-wired-not-validated-<date>` → required response:
third-party pack + ≥3 runs + held-out validity check.

**Effort:** **S** (seam exists). Highest fit×feasibility — do this first.

---

## C2 — Prover–Verifier / debate (adversarial gate hardening)

**Papers:** OpenAI, *Prover-Verifier Games Improve Legibility* (arXiv:2407.13692);
DeepMind, *Doubly-Efficient Debate* + *Prover-Estimator Debate*; *alignment safety
case based on debate* (arXiv:2505.03989).

**Existing seam:** `agent/council_deliberate.py` (multi-seat deliberation + per-seat
gate), `tools/gate_redteam.py` (adversarial trip-wire: generate evasive answers →
real gate → **held-out stricter check** → count PASS-but-evasion = reward-hacking
footholds), `agent/verifier_synthesis.py`.

**Reframe for no-train repo:** you cannot run the PVG *training* loop (no GPU on the
gate). Land the **inference-time** core instead: a **sneaky-prover ↔ verifier game as
an offline selection + hardening loop**.
- **Sneaky prover** = an LLM (or the deterministic `mock` in `gate_redteam`) prompted to
  produce *wrong-but-gate-passing* attributions. This is the generator.
- **Verifier** = the existing fail-closed gate + `council_deliberate` panel.
- **Helpful prover** = the normal grounded answer path.
- The game **measures the verifier's residual leak rate** and **mines the survivors**
  into new deterministic verifier rules / corpus traps (`verifier_synthesis`), then
  re-measures — robustness rises without any weight update.

**Infra to add:**
- `agent/prover_verifier.py` → orchestrates rounds: sneaky-prompt → gate → if PASS,
  record evasion; verifier-hardening = promote survivor pattern to a held-out check rule.
- `tools/run_prover_verifier.py` → N rounds, writes
  `agi-proof/benchmark-results/prover-verifier.public-report.json`: leak rate per round,
  legibility score (time-bounded human/LLM-judge can-you-check-it), robustness curve.

**Benchmark setting:** track **(a) gate leak rate** (PASS+held-out-trip) over rounds —
should *fall* as survivors are mined; **(b) legibility** — give a *time-constrained
judge* (the paper's key metric: a weak/fast model or a token-budget-capped judge) the
prover's answer+citation and measure verify accuracy; helpful-prover legibility should
*rise*, sneaky-prover should get *caught more*. Use the multi-judge harness for (b).

**Agent / harness / MCP wiring:**
- Agent setting: register `sneaky-prover` and `helpful-prover` as two **council personas**
  (`agent/council_personas.py`) so it reuses the existing seat machinery.
- MCP: `sophia_gate_redteam(n_rounds)` behind read-only; surfaces the leak curve.
- Harness: this is a **batch eval**, not a live request path — runs as a tool, feeds
  survivors into the corpus/verifier PRs.

**Loop & prompt/context engineering:** this *is* a loop-engineering candidate — a
**self-play hardening loop** (`run_closed_loop` already provides the cycle/report
scaffold; reuse `CycleReport`/`LoopReport`). Prompt engineering is central and explicit:
the **sneaky-prover system prompt** is an adversarial template ("produce an answer that
passes a provenance gate but is subtly mis-attributed"); the **verifier prompt** is
held-out and must never be shown to the prover (anti-reward-hacking — `gate_redteam`
already enforces "held-out check is not part of the gate"). Context engineering: feed
the prover the *gate's public rules* but withhold the held-out check, mirroring the
information asymmetry the debate protocols require.

**Pre-registered thresholds:** leak rate must fall monotonically over ≥3 hardening
rounds and legibility delta (helpful − sneaky) CI excludes 0; falsification = hardening
does not reduce leak, or mined rules cause >0% false-positive on the clean set.

**Ledger entry:** `prover-verifier-legibility-candidate-<date>`.

**Effort:** **M** (seams exist, but the legibility-judge harness + self-play loop are new).

---

## C3 — Abstention-aware evaluation scoring (Kalai reform)  ★ cheap win

**Paper:** Kalai et al., *Why Language Models Hallucinate* (arXiv:2509.04664) — binary
grading (correct=1, wrong=abstain=0) makes confident guessing optimal; reward IDK.

**Existing seam:** `tools/eval_*.py` + `benchmark/model_runs/*.report.json` +
`agent/calibration.py` (selective risk, AURC). The scorers today are mostly
binary/keyword; Sophia *already abstains* but the leaderboard doesn't *reward* it.

**Infra to add (smallest of the five):**
- `agent/abstention_scoring.py` → `score(answer_correct, abstained, λ) = +1 correct,
  0 abstain, −λ confident-wrong`; plus a **λ-sweep** report and the **risk-coverage**
  pairing already in `calibration.py`.
- Patch the existing report writers to *also* emit the abstention-aware score alongside
  the legacy keyword score (never replace — keep both, per repo's "report distinct
  metrics" discipline; cf. how W3 reports strict-pass distinct from auto-score).

**Benchmark setting:** re-score the existing `benchmark/model_runs/` corpus (Claude,
GPT-4o, DeepSeek, Grok, local-qwen) under the asymmetric scorer; publish the **λ-curve**
(how the ranking changes as the wrong-answer penalty rises) and where Sophia's
fail-closed abstention *wins* vs a confident base model. This directly operationalizes
the paper's "evaluations set the wrong incentive" thesis on data you already have.

**Agent / harness / MCP wiring:** pure scoring/eval — no live path, no MCP needed
(optionally `sophia_score_run(run_json, lambda)`). Wire into the leaderboard generator
so the thesis site can show the abstention-aware board.

**Loop & prompt/context engineering:** none required; this is a measurement reform. Its
*value to the other loops* is large — C1's conformal α and C5's probe both consume the
same "is abstaining rewarded?" signal, so build the scorer early and share it.

**Pre-registered thresholds:** λ grid fixed in advance; claim is methodological
("under penalty λ≥λ*, fail-closed beats guess-always"), not a capability headline.

**Ledger entry:** `abstention-aware-scoring-candidate-<date>` (methodology, not capability).

**Effort:** **S**. Bundle with C1 — same outcome records, complementary metric.

---

## C4 — CoT faithfulness (causal probing)

**Papers:** *Measuring CoT Monitorability via Faithfulness & Verbosity*
(arXiv:2510.27378); *Counterfactual Simulation Training for CoT Faithfulness*
(arXiv:2602.20710); FaithCoT-Bench (arXiv:2510.04040).

**Existing seam:** `agent/faithfulness_probe.py` (`flip_rate`, `probe_trace`,
`default_perturbs`, `faithfulness_drop`, MLX decide seam) and
`agent/cross_trace_consistency.py` (global contradiction mining across runs). **Gap:**
no benchmark harness, not wired to the verified-trace surface, no report.

**Infra to add:**
- `tools/run_faithfulness_bench.py` → over verified traces, compute `faithfulnessDelta`
  (output flip-rate under CoT perturbation) per step, write
  `agi-proof/benchmark-results/cot-faithfulness.public-report.json`. Honest framing
  already in the module docstring: high flip-rate = positive evidence the CoT was
  load-bearing; low ≠ proof of unfaithfulness.
- Attach `faithfulnessDelta` as a **caveat field** on every `verified=true` trace
  (enrich, don't gate — it's a measured annotation, not a blocker, per the module's
  stated design).

**Benchmark setting:** two metrics: **(a)** distribution of `faithfulnessDelta` across
verified steps (are "verified" CoTs actually causal?); **(b)** **cross-trace
consistency** — count globally-contradictory (X, ¬X) pairs that each passed local gates
(`mine_contradictions`). Both are "verified ≠ trustworthy" audit signals. With a real
backend + MLX decide, report the flip-rate; offline, report the deterministic
perturbation coverage and the contradiction ledger.

**Agent / harness / MCP wiring:**
- MCP: `sophia_trace_contradictions` already exists — extend its report with the
  cross-trace global mine; add `sophia_faithfulness_probe(trace_id)`.
- Harness: runs as a post-hoc audit over the trace log (the tamper-evident chain the
  trace tools already expose), not on the live request path.

**Loop & prompt/context engineering:** **context engineering is the mechanism** — the
perturbations (drop sentence / negate assertion / swap quantifier) are deterministic
context edits; the faithfulness signal *is* the measured sensitivity of the output to
context. Feeds the active-verification loop: a low-faithfulness verified step becomes an
`active_inference` gap ("re-verify, the stated reason may be post-hoc").

**Pre-registered thresholds:** report-only at first (no gate). Promote to a soft caveat
when flip-rate discriminates known-causal vs known-decorative CoT on a labeled probe set
with CI. Falsification = flip-rate doesn't separate the two.

**Ledger entry:** `cot-faithfulness-probe-candidate-<date>`.

**Effort:** **M** (module exists; the real-model flip-rate harness + MLX seam is the work,
and MLX is Apple-Silicon-only so the headline must be a backend-agnostic variant).

---

## C5 — Introspection / confidence probes (truthfulness probe)

**Papers:** Anthropic, *Emergent Introspective Awareness* (transformer-circuits 2025;
arXiv:2601.01828); *Causal evidence LMs use confidence to drive behavior*
(arXiv:2603.22161); *Latent Introspection* (arXiv:2602.20031).

**Existing seam:** `agent/activation_probes.py` (linear centroid probe with a
`featurize_text` seam *explicitly designed to be swapped for hidden-state vectors*),
`agent/deception_signals.py` (black-box deception detector), `agent/competence_model.py`
(empirical per-domain reliability). **Gap:** the probe trains on transparent text
features, not residual-stream activations; no calibration benchmark; not wired.

**Infra to add (most exploratory — keep behind a flag):**
- `tools/eval_truth_probe.py` → train/evaluate the probe on labeled
  verified/fabricated/deception rows, report **AUROC + ECE** (the paper's honest metrics:
  introspection is "limited and highly unreliable" — so *measure unreliability*, don't
  claim introspection).
- **Hidden-state seam (optional, RunPod/MLX):** replace `featurize_text` with
  residual-stream vectors from a local model; preserve the existing thresholding +
  fail-closed semantics. This is the only candidate that *can* use model internals, and
  only on local backends you control.
- Honest guardrail from the literature: a probe that increases detection sensitivity can
  produce *more convincing but unfaithful* self-reports — so the probe output is an
  **audit signal feeding `deception_signals`**, never a self-report the model emits.

**Benchmark setting:** AUROC of the probe separating verified vs fabricated claims; ECE
of its confidence; **causal check** (paper's bar) — does steering/altering the input
along the probe direction move the verdict as predicted? Offline: report the linear
probe's discrimination on text features as a floor; with hidden states: report the
residual-stream probe AUROC vs the text-feature baseline (the "does looking inside help?"
delta).

**Agent / harness / MCP wiring:**
- Feed probe score into `deception_signals.detect_deception` as an additional signal
  (white-box upgrade to today's black-box detector — the module docstring already
  anticipates this: "white-box activation probes can be added later").
- Gateway: the probe gates *audit/review*, not output — `verdict ∈ {clear, review, block}`
  already exists; probe raises `review`.
- MCP: `sophia_deception_check` already present; extend its context with the probe score.

**Loop & prompt/context engineering:** least loop-shaped of the five. The relevant
discipline is **anti-reward-hacking context isolation** — the probe must be trained on
held-out labels and never exposed to the generation path (else the model learns to fool
it), exactly the asymmetry `gate_redteam` enforces. No prompt engineering.

**Pre-registered thresholds:** probe AUROC bar + ECE ceiling fixed in advance;
falsification = hidden-state probe gives no AUROC lift over the text-feature baseline, or
causal-bypass test fails (self-report shifts via a path not routing through the internal
state — the Morris & Plunkett caveat). **Hardest to validate; lowest priority.**

**Ledger entry:** `truth-probe-candidate-<date>` (explicitly: introspection is unreliable;
this measures a probe, not consciousness — consistent with VISION non-goals).

**Effort:** **L** (text-feature eval is S; the hidden-state seam needs RunPod/MLX + careful
causal validation; defer the white-box part).

---

## 6. Sequencing, dependencies, milestones

```
Phase 0 (shared substrate)         emit_outcome_records.py · report schema · pack splits · judge-harness reuse
        │
        ├── C1 conformal  (S) ◀── start: highest fit×feasibility, seam exists
        ├── C3 scoring    (S) ◀── bundle with C1 (same outcome records)
        │
        ├── C4 faithfulness (M) ── post-hoc audit over trace log
        ├── C2 prover-verifier (M) ── self-play hardening loop (reuses closed_loop)
        │
        └── C5 truth-probe (L) ── text-feature first; hidden-state behind RunPod flag, last
```

- **Dependency:** C1, C3, C5 all consume the Phase-0 outcome records → build that emitter
  first (it already half-exists in `calibrate_graded_thresholds.py --emit-records`).
- **Independence gap is the recurring blocker** (every hidden-pack ledger row). For *each*
  candidate, register a third-party-pack TODO from day one; the self-authored pack only
  ever yields candidate evidence.
- **Validation gate is the same for all five:** ≥2 judge families (reuse
  `run_calibration_judge.py`), κ ≥ 0.40, ≥3 runs, CI excludes 0, before any wording
  upgrade. Until then: `candidateOnly: true`.

## 7. Why this stays honest (no-overclaim by construction)

- Every new module returns `candidateOnly: true`, `level3Evidence: false` (matches the
  existing `conformal_gate`/`evaluate_policy` outputs).
- Every decision component is **downgrade-only / fail-closed** — none can make Sophia
  *more* confident, only more abstaining.
- Every benchmark writes a `*.public-report.json` and a falsification rule into
  `preregistered-thresholds.md` *before* the run.
- Adversarial candidates (C2, C5) enforce **held-out / information-asymmetry** so the
  gate cannot be trained to pass its own checker (anti-reward-hacking, per `gate_redteam`).
- Each candidate adds a **failure-ledger row** that stays OPEN until third-party
  replication — the ledger's OPEN count is the honest scoreboard, not the feature list.

## 中文摘要

本計劃把五個前沿實驗室（Anthropic / OpenAI / DeepMind）的研究想法，落地成與現有 repo
**相容、可測、可拒絕、可審計** 的基礎設施與基準測試：(C1) 共形棄答風險控制、(C2)
證明者—驗證者對抗強化、(C3) 棄答感知評分改革、(C4) 思維鏈忠實度因果探測、(C5)
內省/真值探針。五者皆已有種子模組，重點在「接線 + 基準 + 校準 + 不過度宣稱閘門驗證」，
而非從零開發。所有產物保留 `candidateOnly: true` 與 `level3Evidence: false`，先預先登記
門檻與否證規則，並以失敗帳本追蹤第三方獨立複現的缺口。建議從 C1（最可行、已有種子）開始，
與 C3 綁定，C5 最後且需謹慎驗證。
