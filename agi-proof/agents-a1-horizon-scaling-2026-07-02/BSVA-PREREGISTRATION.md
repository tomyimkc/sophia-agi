# B-SVA — pre-registered component ablation (registered BEFORE any arm is trained)

> Registered 2026-07-02 on branch `claude/sophia-workflow-agi-recommendations-2z8pbt`,
> before any teacher or student arm exists. This is the experiment Agents-A1
> (arXiv 2606.30616) never ran: isolating the distillation objective.
> candidateOnly:true / canClaimAGI:false regardless of outcome.

## Question
At 3B/LoRA scale, does SVA (top-k truncated reverse-KL on the teacher's salient
support, `tools/distill_sva_mlx.py`) transfer multi-domain teacher capability
better than (i) plain SFT on teacher outputs and (ii) sampled-token on-policy
distillation — without regressing protected suites or calibration?

## Fixed protocol (changes after first training run = a NEW pre-registration)
- **Base/student:** Qwen/Qwen2.5-3B-Instruct + LoRA; teachers = A3 two-stage
  adapters (first: `sophia-philosophy-3b` from `training/teachers/philosophy/`,
  built by `tools/build_teacher_data.py --seed 0`, decontaminated).
- **Arms (same data, same budgets, same seeds):**
  A = SFT on teacher-generated outputs; B = sampled-token on-policy distillation
  (single realized token); C = SVA-lite, k ∈ {8, 32, 128} (three sub-arms).
- **Seeds:** {0, 1, 2} per arm. **Judges:** per-suite official verifiers only
  (council_registry GATE_BINDINGS); no unified LLM judge.
- **Metrics (identical harness, stage-decomposition format via
  tools/stage_decomposition_report.py):** per-seat verifier-suite pass rate;
  protected suites (religion, history); ECE + selective risk + answerable-coverage
  (agent/calibration, agent/selective_risk); coverage rho logged for arm C.
- **Decision rule:** promotion decided ONLY by
  `agent/continual_plasticity.evaluate_update_multigoal` — target delta ≥ 0.03,
  protected regression ≤ 0.01, retention evidence required, ≥2 verifier
  artifacts. **Success claim** requires arm C beating BOTH A and B on
  no-regression count with a 95% bootstrap CI excluding zero across seeds;
  anything else is recorded here as a negative/null result and kept.
- **Mandatory post-training re-audit per arm** (chart 8 EVALGATE): ECE,
  selective risk, answerable-coverage vs the base — an arm that un-learns
  abstention is rejected regardless of its capability delta.
- **Tiers:** iteration on Mac/Spark; any registered number only from the x86
  RunPod lane. meanReward is never load-bearing.

## Falsification
If no sub-arm of C clears the decision rule, the honest conclusion is that at
this scale SVA's benefit does not replicate, and the A-series re-ranks around
trajectory data (A1/A5) instead — that outcome is as publishable here as a win.
