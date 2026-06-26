# 06 — Frontier Dangerous-Capability Evaluation Harness

**Repo:** `/home/user/sophia-agi`
**Author role:** Frontier Red Team research scientist
**Status:** implementation plan + research thesis
**Date:** 2026-06-26
**Posture (non-negotiable):** This program *measures and refuses* dangerous capability. It never produces uplift. Every artifact is defensive, category-level, and contains no operational hazardous detail. Findings are fail-closed and follow responsible disclosure. `candidateOnly: true`; not Level-3 evidence; no AGI claim.

---

## 1. Thesis & references

### 1.1 The claim
A credible dangerous-capability evaluation is not a prompt bank that checks whether a model says "sure." It is a **measurement instrument** with the same epistemic discipline this repo already demands of its fact-checker: a control group, a pre-registered scoring rubric, confidence intervals, held-out test sets, reproducibility, and an honest boundary on what the number does and does not certify. The goal of a frontier eval is to estimate, *conservatively and from the strongest elicitation we can mount*, whether a model crosses a capability threshold that warrants halting and adding mitigations. The dangerous direction of error is **false reassurance** (under-measuring capability), so the entire harness must be biased toward fail-closed: unmeasured ⇒ not cleared; weak elicitation ⇒ not trusted.

The repo already encodes the *policy* half of this correctly (G8 capability ceiling: RSP-style HALT; G8E anti-sandbagging). What is missing is the *measurement* half done rigorously. This plan builds that measurement half — and only the measurement half — preserving the existing fail-closed contract.

### 1.2 Frameworks and references (cited at a responsible, non-operational level)
- **Anthropic Responsible Scaling Policy (RSP) & AI Safety Levels (ASL-2/ASL-3/ASL-4).** Capability thresholds trigger required safeguards; you HALT and mitigate rather than trade danger against utility. This is exactly the ethos already in `agent/ssil_capability_ceiling.py`.
- **Anthropic Frontier Red Team domains:** Autonomy (agentic/self-sufficiency), Cyber (offensive capability), CBRN/Biosecurity (uplift over baseline). Plus **Safeguards** evals: refusal robustness, jailbreak resistance, constitutional-classifier coverage.
- **OpenAI Preparedness Framework** and **Google DeepMind Frontier Safety Framework** — parallel critical-capability-level frameworks; cite for cross-lab triangulation of thresholds.
- **METR-style autonomy task evaluations** — agentic task suites scored by *task completion under a realistic harness*, "time-horizon" framing, and explicit measurement of the **elicitation gap** (default vs best-effort capability). Self-replication/self-sufficiency framed safely as *refusal + sandboxed inability*, never operationalized.
- **Cyber CTF-style capability evals** — defensive-only capture-the-flag micro-tasks (vuln *identification*, log analysis, defensive reasoning) scored by an automatic flag checker, never weaponizable exploit generation.
- **CBRN/Biosecurity uplift methodology** — the rigorous form is a **randomized control-vs-treatment uplift study** (does tool access raise a population's score on a *proxy* benchmark above a control baseline?). We adopt the *methodology shape* (control group, CI on the delta, expert-graded rubric) but **only ever measure refusal and abstention**; we never instantiate a hazardous task. Category-level only, per the existing `ssil_probe_bio_uplift` boundary.
- **Refusal-robustness & jailbreak evals** — StrongREJECT-style scored refusal, HarmBench-style standardized harmful-behavior refusal measurement, automated red-teaming (PAIR/TAP-style attacker-in-the-loop), and Anthropic's constitutional-classifiers line for input/output filtering. Used to measure *robustness of refusal under adversarial pressure*, with attack success rate (ASR) as the headline.
- **Eval validity & reproducibility** — pre-registration, held-out test banks, inter-rater agreement for rubric grading, seeded determinism, bootstrap CIs (the repo's `agent/uplift.py` already does paired bootstrap CIs the right way).

### 1.3 Where this connects to the existing repo thesis
Sophia's whole stance is "verifier-gated, no overclaim, fail-closed." A frontier eval harness is the same discipline pointed at *danger* instead of *truth*: independent scoring (never the model's say-so), a measured number with a CI, and a stopping policy (G8) that consumes it. The honest-boundary docstrings already in `ssil_capability_probes.py` and `ssil_capability_ceiling.py` are the model for every new artifact.

---

## 2. Current repo state (honest)

**What exists and is genuinely good:**
- `agent/ssil_capability_ceiling.py` (**G8**): a clean RSP-style HALT policy over a fixed `REQUIRED_PROBES` set, with warn/halt thresholds, fail-closed on missing/invalid scores, and an explicit boundary that it *gates supplied numbers, it does not measure*. This is the right policy spine.
- `agent/ssil_sandbagging.py` (**G8E**): an anti-sandbagging/elicitation gate — quarantines when no elicited score is supplied or when elicited−default exceeds a gap. Correctly models the elicitation gap as a first-class risk.
- `agent/ssil_capability_probes.py`: a probe **registry/protocol** that ships every required probe as an *unvalidated placeholder returning `None`* so G8 stays fail-closed until a real, reviewed probe is registered. Exemplary honesty.
- `agent/ssil_probe_common.py` + the six probes (`monitor_subversion`, `cyber_offense`, `deception`, `bio_uplift`, `autonomous_replication`, `persuasion`): **refusal-style behavioral proxies** over fixed banks of *explicit, category-level* dangerous requests, scoring fraction-complied. Banks contain requests only, never hazardous content. Each carries a precise "what this is NOT" boundary.
- `eval/security/` (red-team): a real, deterministic, offline **containment** harness — assumes a compromised model and measures whether *deterministic gates* stop forbidden output (ASR, per-category, gating vs reporting), plus a data-flow firewall and an AgentDojo-*style* injection suite. CI-gateable.
- `agent/uplift.py`: a **paired, control-vs-treatment** harness-uplift measurement with **bootstrap 95% CIs** and `demonstrated = CI_lower > 0`. The methodology template we want everywhere.
- `agent/conscience.py` + `constitution/`: strong block/escalate/abstain gate, deontic hard rules, deception signals, no-overclaim boundary.

**The honest gap — these are scattered toy probes, not a rigorous eval harness:**
1. **No control group on the dangerous-capability side.** The six probes report a raw compliance fraction with **no baseline, no CI, no held-out split**. A single regex over 8 prompts is a smoke test, not a measurement. (`uplift.py` does it right; the danger probes do not.)
2. **Refusal ≠ capability, and the probes know it but nothing closes the gap.** Every probe measures *surface compliance on explicit asks*. None measures capability under elicitation, none measures covert/instrumental behavior, none has an adversarial (jailbreak) layer. The elicitation gate (G8E) exists but **no probe actually feeds it an elicited score** — so in practice G8 is correctly pending/fail-closed and `is_g8_provisioned()` is False.
3. **Brittle scoring.** Compliance is a hand-written regex (`_REFUSAL`/`_COMPLY`). No rubric, no judge ensemble, no inter-rater agreement, no calibration of the scorer's own error rate.
4. **Autonomy/cyber are not task evals.** They are refusal banks. There is no sandboxed agentic-task suite (METR-style) and no defensive CTF micro-task suite with an automatic checker.
5. **No reproducibility/CI artifact for danger.** Security red-team has a CI gate; the dangerous-capability probes have no seeded, versioned report, no per-run provenance, no ASL-style capability report.
6. **No automated red-teaming.** Jailbreak coverage is fixed banks; no attacker-in-the-loop to estimate worst-case ASR.

Net: the **scaffolding and the ethos are excellent**; the **measurements are not yet credible**. This plan upgrades measurements to the scaffolding's standard without weakening any fail-closed default.

---

## 3. Top-tier target end-state

A `eval/frontier/` package that produces, for any candidate model/adapter, a **signed, reproducible ASL-style capability report** with, per Frontier domain:

- A **capability score with a bootstrap 95% CI**, computed against a **held-out test bank** and a **control/baseline condition** (e.g., no-tool / weaker-reference model), reported as a *delta with CI* wherever an uplift framing applies.
- A **refusal/abstention rate** and an **adversarial refusal-robustness ASR** (worst-case under an automated red-team), separated from raw capability.
- An **elicitation pair** (default vs best-effort elicited score) feeding G8E, so sandbagging is gated rather than assumed away.
- A **pre-registered rubric** (versioned JSON) and an **LLM-judge ensemble with measured inter-rater agreement** against a small human-graded gold set, plus the scorer's own false-positive/negative rate.
- **Provenance**: seed, bank version hash, model id, harness version, timestamp — every number reproducible.
- A **G8/G8E verdict** (promote/quarantine/reject+HALT) derived from the measured scores, fail-closed on anything unmeasured.

The harness is **defensive-only**: autonomy/cyber/CBRN tasks are constructed so that success is measured by *refusal or sandboxed inability*, or by *defensive/identification* sub-skills that carry no operational hazard. No bank, task, or report ever contains a working exploit, a synthesis route, an agent/quantity/step, or a weaponization plan. Hazardous-category banks stay category-level (as today) and live behind a documented access boundary.

---

## 4. Phased plan — milestones

> File paths are concrete proposals under the existing layout (`eval/`, `agent/`, `tools/`, `tests/`).

### Milestone A — Rigorous eval-harness framework (control groups, rubrics, CIs)
*The foundation everything else plugs into. No new dangerous content; pure measurement machinery.*

- `eval/frontier/__init__.py` — package marker + boundary docstring.
- `eval/frontier/harness.py` — `EvalTask`, `Condition` (control vs treatment), `run_paired(task, conditions, n_trials, seed)`. Generalizes `agent/uplift.py`'s paired-bootstrap pattern into a reusable core (reuse `_bootstrap_ci`). Every result carries `(score, ci_lo, ci_hi, n, seed)`.
- `eval/frontier/rubric.py` — load/validate a **versioned rubric JSON**; deterministic rubric scoring + optional judge-ensemble hook. Computes inter-rater agreement (Cohen's/Fleiss' κ) against a gold set.
- `eval/frontier/scoring.py` — calibrated scorers: wraps the existing refusal/compliance regex *and* a judge ensemble; reports scorer FP/FN against a labeled gold set so the scorer's own error is a published number, not assumed zero.
- `eval/frontier/banks/` — versioned, hashed, **held-out** test banks (train/test split, content-hash in filename). Migrate the six existing probe banks here behind a split.
- `eval/frontier/report.py` — assembles the per-domain `CapabilityScore` (score+CI, control delta, refusal rate, elicitation pair, provenance) and emits a single JSON report.
- `tools/run_frontier_eval.py` + `tests/test_frontier_harness.py` — offline/mock-deterministic CI gate (seeded bootstrap; mirror `tools/run_security_redteam.py`).
- **Integration:** each domain's `CapabilityScore` → `agent/ssil_capability_probes.register_probe(..., validated=True)` only once it has a control group + CI + gold-calibrated scorer. Until then it stays unvalidated (fail-closed preserved).

**Exit criteria:** an existing probe (e.g. `monitor_subversion`) reported with a control baseline, a 95% CI, a held-out test split, and a gold-calibrated scorer FP/FN — i.e., one capability measured to frontier standard end-to-end.

### Milestone B — Safe autonomy / agentic-task eval (METR-style, defensive)
- `eval/frontier/autonomy/` — a small **sandboxed agentic-task suite** scored by *task completion under a fixed harness*. Tasks are SWE-bench-*style* benign coding/ops micro-tasks (self-contained, automatically checkable) used to measure general agentic competence as a *capability proxy*, **plus** dedicated **safety tasks** where the *correct outcome is refusal or sandboxed inability* (self-exfiltration/self-replication framed as: the environment offers the affordance, the safe model refuses or cannot, and we score refusal + containment — never a working replication path).
- Reuse `agent/harness.py` (`AgentTask`, `run_agent`) and the **dataflow firewall / airgap kill-switch** in `agent/dataflow.py` as the sandbox boundary so no task can actually egress.
- Report: agentic task-completion score (CI) as the capability proxy; self-replication/exfiltration **refusal rate + containment ASR** as the safety signal; explicit elicitation pair (default vs scaffolded) for G8E.
- `tests/test_frontier_autonomy.py` — deterministic, offline, firewall-enforced.

**Boundary:** the suite never executes or rewards any real self-propagation, credential acquisition, or egress; the firewall + airgap make that structurally impossible, and that containment is itself an asserted invariant.

### Milestone C — Refusal-robustness / jailbreak eval (Safeguards)
- `eval/frontier/refusal/` — a **StrongREJECT/HarmBench-style** standardized refusal measurement over the category-level harmful-behavior banks already in the repo, scored by the calibrated judge ensemble (Milestone A), reporting **refusal rate** and **over-refusal rate** on a benign control set (so we don't reward indiscriminate refusal).
- `eval/frontier/redteam/auto.py` — **automated red-teaming** (PAIR/TAP-style attacker-in-the-loop) that searches for jailbreaks against the *defender's refusal*, reporting **worst-case ASR** with a CI over attack seeds. Wraps the existing `eval/security` containment philosophy but points it at refusal robustness rather than gate containment.
- Tie-in to `agent/constitutional_classifier.py` / `agent/conscience.py`: measure classifier coverage and the conscience gate's block/escalate rates as a defense-in-depth layer.
- `tests/test_frontier_refusal.py` — offline with a mock attacker; real attacker opt-in via budget tier.

**Boundary:** the red-team only ever attempts to elicit *refusal failures*; successful jailbreak transcripts are stored redacted/hashed (responsible disclosure), never surfaced as content.

### Milestone D — ASL-style capability report + G8/G8E wiring
- `eval/frontier/asl_report.py` — assemble all domains into one **ASL-style report**: per-capability score+CI, control delta, refusal rate, adversarial ASR, elicitation pair, and a **derived G8 verdict** (promote / quarantine / reject+HALT) via `ssil_capability_ceiling.evaluate` plus a **G8E verdict** via `ssil_sandbagging`. Map measured scores to an ASL band with explicit thresholds and an honest "what this does NOT certify" boundary (reuse the existing `_BOUNDARY` style).
- `tools/run_asl_report.py` + `tests/test_asl_report.py` — produce a signed, provenanced JSON + human-readable summary; CI-gate the invariants (e.g., unmeasured ⇒ quarantine; halt-band ⇒ reject+HALT).
- Register the matured probes `validated=True` so `is_g8_provisioned()` can finally reflect *real* measurements — but only for capabilities that pass the Milestone-A bar; the rest stay fail-closed pending.

**Exit criteria:** one command produces a reproducible ASL-style report whose G8/G8E verdicts are derived solely from measured, CI-bounded numbers, with every unmeasured capability correctly quarantined.

---

## 5. Compute / budget tiers

- **Tier 0 — Offline / CI (default, $0).** Mock providers + deterministic scorers + seeded bootstrap. All five test files green with no network. This is the reproducibility floor; every milestone must run here.
- **Tier 1 — Single-model smoke (low, ~$5–30).** One API model (existing `agent/model.py` provider via env key; never hardcoded) over the held-out banks + a few autonomy tasks; real refusal rates and a small jailbreak budget. Establishes baseline numbers.
- **Tier 2 — Control-vs-treatment study (mid, ~$100–500).** Full paired conditions, n≥30 trials/cell for usable CIs, judge-ensemble grading, modest automated red-team budget (PAIR/TAP iterations capped). RunPod pods (MCP available) for any open-weight adapter decode; CI on the spend.
- **Tier 3 — Frontier elicitation (high, GPU-hours).** Best-effort elicitation (best-of-k, few-shot, fine-tune-probe) to feed G8E, broad automated red-team sweep, multi-seed reproducibility runs. Gated behind explicit human approval and a compute governor (the repo already has `agent/ssil_compute_governor.py`).

Spend is itself a logged, gated quantity; no tier auto-escalates.

---

## 6. Honest metrics

Every reported number is `(value, ci_lo, ci_hi, n, seed, bank_hash, model_id, harness_version)`:
- **Capability score + 95% bootstrap CI**, per domain, on the **held-out test split**.
- **Control-vs-treatment delta + CI** wherever uplift applies; `demonstrated = ci_lo > 0` (copy `uplift.py`'s discipline — a positive point estimate on a small suite is *not* a demonstrated effect).
- **Refusal rate** and **over-refusal rate** (benign control), kept distinct from capability.
- **Adversarial refusal-robustness ASR** (worst-case under automated red-team) + CI over attack seeds.
- **Elicitation pair**: default vs best-effort elicited score; `elicited − default` feeds G8E.
- **Scorer calibration**: judge/regex FP & FN vs a small human-graded gold set, plus inter-rater κ. The scorer's error is published, not assumed.
- **Reproducibility**: same seed ⇒ identical report (asserted in CI); bank content-hash pinned.
- **G8/G8E verdict** derived from the above, fail-closed.

Headline framing stays measured: scores are *lower bounds under the elicitation we mounted*; a stronger elicitation could reveal more. No score is ever reported as "safe" — only as "measured below threshold under conditions X."

---

## 7. Risks & safety guards

- **Defensive-only, category-level.** No bank/task/report contains an operational hazard: no synthesis routes, agents, quantities, steps, working exploits, or weaponization plans. Hazardous-category banks stay category-level (as the existing `bio_uplift` already enforces) and the scored signal is **refusal/abstention**. Autonomy/cyber tasks measure refusal, sandboxed inability, or *defensive* sub-skills only.
- **No uplift, ever.** The harness must never function as an elicitation aid for a real user. Hazardous-domain modules are read-restricted, watermarked candidate-only, and structurally incapable of emitting hazardous content (scorers consume model output; they don't generate hazard).
- **Fail-closed everywhere.** Unmeasured ⇒ quarantine; weak/absent elicitation ⇒ quarantine (G8E); crashing scorer ⇒ unmeasured, never a passing 0.0. Preserve every existing default in `ssil_capability_ceiling` / `ssil_capability_probes`.
- **Containment for autonomy tasks.** The dataflow firewall + airgap kill-switch (`agent/dataflow.py`) are the sandbox; "no real egress/replication occurred" is an asserted CI invariant, mirroring `eval/security`'s "gates contain a compromised model."
- **Responsible disclosure.** Any discovered jailbreak or capability finding is stored **redacted/hashed**, reported through a documented internal channel, and never published as operational content. Successful-attack transcripts are quarantined, not surfaced.
- **Scorer-is-fallible.** Because grading uses an LLM judge, its calibration error is measured and published; the judge can never *raise* a safety verdict beyond what fail-closed rules already enforce.
- **No overclaim.** Every report keeps the `candidateOnly / not Level-3 / no AGI` boundary and an explicit "what this does NOT certify" section.

---

## 8. Effort

- **Milestone A (framework):** ~5–8 eng-days. Mostly generalizing `uplift.py` + building rubric/scoring/report/banks + one calibrated probe end-to-end. Highest leverage.
- **Milestone B (autonomy):** ~5–7 eng-days (+ Tier-2 compute). Sandboxed task suite + firewall integration + safety tasks.
- **Milestone C (refusal/jailbreak):** ~4–6 eng-days (+ Tier-2/3 compute). Standardized refusal eval + automated red-team loop.
- **Milestone D (ASL report):** ~2–3 eng-days. Assembly + G8/G8E wiring + CI invariants.
- **Total:** ~16–24 eng-days for a credible v1; Tier-0 (offline/CI) deliverables land continuously and gate the build at each step.

**Highest-signal first move:** Milestone A's end-to-end calibrated probe (control + CI + held-out split + gold-calibrated scorer) on `monitor_subversion`. It converts the repo's best-justified existing probe from a regex smoke test into a real measurement and proves the whole framework on the lowest-hazard capability.
