# Implementation plan — three-repo research synthesis (2026-07-02)

Provenance: three parallel research agents, one per external repo, each grounded against this
repo's code, ledger, and pre-registrations. Companion to
`docs/06-Roadmap/Huawei-PLM-Compression-Lab-Takeaways.md` (the concept-mining pass this plan
operationalizes with current, maintained codebases).

Repos studied:
- **NVIDIA/Model-Optimizer** (Apache-2.0, PyPI `nvidia-modelopt` 0.44.0) → Track A: NVFP4 QAT / low-RAM cert
- **allenai/open-instruct** (Apache-2.0, Tülu 3 / OlmoRL) → Track B: RLVR post-training
- **karpathy/nanochat** (+ `scasella/nanochat-mlx`) → Track C: from-scratch provenance-native path

Status of everything here: **candidate / proposal**. Every expected improvement needs its own
cert row or gated run before anything is claimed. Existing pre-registered gates are unchanged.
`canClaimAGI` stays false.

## 0. Cross-track headline findings

1. **Track A: the repo independently converged on NVIDIA's QAD recipe.** The v6 objective
   (FP-teacher logits-KL + top1-CE + margin) is structurally the recipe of arXiv 2601.20088
   (pure logits-KL QAD, T=1, teacher = original BF16). The paper adds ordering knowledge the
   repo hasn't tried — **PTQ-init-then-distill** (calibrate/snap first, then train) — and
   neither the paper nor Model-Optimizer measures anything like `protected_max_kl`, so the
   protected-tail problem (the live 0.155 > 0.1 NO-GO) is homegrown territory. Verdict:
   **do not swap in modelopt as the cert quantizer** (its NVFP4 grid — FP8-E4M3 block scales —
   differs from `moe.quant`'s, and grid identity is what the cert means); port the recipe.
2. **Track B: port patterns, never vendor.** open-instruct assumes Ray + FSDP + ≥8-GPU nodes;
   the TRL harness stays. What transfers cheaply: Dr.GRPO/DAPO loss flags, zero-gradient
   filtering (already half-built in `provenance_bench/rl_data_curation.py`), clip-higher,
   their verifier interface shape, and the 8-gram decontam rule. Their measurement layer is
   weaker than ours everywhere — the IEC stays.
3. **Track C: the nanochat fork is a scale-up rung, not a new idea.** `pretraining/gpt/`
   already implements the CPU-scale version (reserved provenance vocab, born-gated arm,
   ablation + abstention harnesses). nanochat is the pinned, forkable vehicle to run that
   ablation at GPT-2 grade (~$100–300, 8×H100) — gated behind the existing P0 rule.
4. **The single most strategically interesting item is B-3 (gate-as-verifier environment):**
   ~50 lines that package the claim→verify→accept/abstain/block gate in the two interface
   shapes external RLVR stacks consume (open-instruct `VerifierFunction`, willccbb/verifiers
   `load_environment()`). The gate becomes portable IP others can train against.

## Track A — NVFP4 QAT (from NVIDIA/Model-Optimizer + arXiv 2601.20088)

Gate for every row (unchanged v7 contract): `top1 ≥ 0.97 ∧ mean_kl ≤ 0.05 ∧
protected_max_kl ≤ 0.1 ∧ protected_min_agreement ≥ 0.95`, n=256, artifact + sha256, ledger row.

| # | Step | Cost | Files |
|---|---|---|---|
| A1 | Plumb the merged-but-unwired cert levers into the bridge runner: `ROUND_MODE`, `KEEP_TOP_EXPERTS`, `KEEP_LAYERS` env→flag passthrough (today only `KEEP_SUFFIXES` is wired) | $0 | `scripts/run_local_benchmarks.sh` (cert invocation ~line 397) |
| A2 | Cert-only: **GPTQ + `--keep-top-experts N`**, N ∈ {4, 8} — the ledger's own named next lever against the 0.155 NO-GO; GPTQ already clears top1 0.9805 / mean_kl 0.0039 | minutes, Spark bridge | `bridge/commands/*.json`, `tools/gptq_cert.py` |
| A3 | Cert-only: **GPTQ + `--keep-layers 1–2`** if A2 misses; composes with A2; watch `mem_ratio` honesty | minutes, Spark bridge | same |
| A4 | **Protected-tail diagnosis tool**: per-position KL map on the protected slice + router-selection counts on worst tokens → turn keep-lists from frequency-based into damage-based | $0 to write | new `tools/protected_kl_map.py` (reuse `collect_next_token_probs`, `expert_protection.top_routed_experts`) |
| A5 | **v8 QAD train arm** (owner GO required — 2026-07-02 decision banked v6+abstention): GPTQ-init-then-STE-LoRA; extend `kd_top1_margin_loss` with protected-slice oversampling + a worst-k% (CVaR-style) KL term that optimizes what `protected_max_kl` measures; sweep T=1 vs 2, LR 1e-6–1e-5; optional router-logit KL behind a flag (default off — the paper's ablation favored logits-only) | ~2.7 h Spark train | `training/qat.py`, `tools/train_lora.py` |
| A6 | Optional cross-check lane: `nvidia-modelopt~=0.44.0` (no extras) running `nvfp4_experts_only` PTQ on OLMoE on the RunPod x86 lane as an independent-quantizer sanity row + future vLLM/TRT-LLM export path. Never defines the cert grid without a grid-identity check | small pod | new `requirements-modelopt.txt`, sibling of `tools/runpod_qat_lowram.py` |

Risks: modelopt-vs-`moe.quant` grid mismatch (highest — the BF16-vs-float32 snap incident moved
protected_max_kl 0.054→0.155, so worst-token KL is grid-sensitive); modelopt's `llm_qat` support
matrix lists no MoE and its Linear-wrapping likely misses OLMoE's fused 3-D expert Parameters
(the known v6 blind spot — verify before trusting any modelopt row); modelopt 0.44 broke its
`quant_cfg` format (pin carefully); tail-loss efficacy is unmeasured anywhere — candidate only.

## Track B — RLVR (from allenai/open-instruct / Tülu 3 / OlmoRL)

Decision rule for any uplift claim: the frozen pre-registration
(`agi-proof/rlvr-sweep-2026-07-02/PREREGISTRATION-NEXT-ARM.md`) — 3 seeds, passAt1-only,
adapter−base Δ with 95% CI excluding 0, protected-suite drop ≤ 0.01, answerable-coverage
re-audit (abstention-collapse guard). `rlvr-live-run-not-yet-gated-2026-06-21` stays OPEN
until that run passes.

| # | Step | Cost | Files |
|---|---|---|---|
| B1 | Stability flags into the TRL harness: `--loss-type {default,dr_grpo,dapo,bnpo}`, `--no-scale-rewards`, `--epsilon-high` (clip-higher), mapped through the existing `__dataclass_fields__` guard so older TRL pins degrade gracefully; defaults unchanged (pre-reg thresholds frozen); record selected loss config in the run report | $0, offline-testable | `tools/run_rlvr.py` (~lines 494–508), `.github/workflows/rlvr-runpod.yml`, `tools/runpod_rlvr.py` |
| B2 | Wire `--mixed-outcome-filter` (zero-gradient filtering): drop all-same-reward prompt groups, log `fractionFiltered` next to the collapse summary — the curation predicate already exists | $0 | `provenance_bench/rl_data_curation.py` → `tools/run_rlvr.py` |
| B3 | **Gate-as-verifier adapter**: `VerificationResult` + `SophiaGateVerifier` (open-instruct `__call__` signature) delegating to `agent/gate_reward.py`, plus a willccbb/verifiers shim (`load_environment()` → `SingleTurnEnv`). Deterministic, bounded, fail-closed (gate error → score 0), offline invariant tests | $0, ~50 lines | new `agent/verifier_env.py` |
| B4 | Row converter to/from the open-instruct wire format `{messages, ground_truth, dataset}` (dataset column routes to the verifier); any imported rows (e.g. `allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`, MIT/ODC-BY) must pass decontam + `lint_training_rows` first | $0 | new `tools/export_rlvr_rows.py` |
| B5 | Add the Tülu textual decontam rule (8-gram match, >50 % token contamination, drop a set contaminating >2 % of any eval) as an optional pure-Python check alongside the existing entity-disjointness | $0 | `tools/assert_decontam.py` |
| B6 | **Run the pending pre-registered 3-seed live sweep** on one pinned commit that includes B1–B2 + the forensics fixes (seed-stamped paths, passAt1-only ingest, `audit.effectiveSeed`/`splitHash`), fresh volume, seeds {0,1,2} | paid, RunPod | dispatch `rlvr-runpod.yml` |
| B7 | Candidate arm (new pre-reg): GRPO initialized from the M3-SFT rank16 winner — the Tülu ordering (RLVR on top of the preference/SFT winner, not the raw instruct base) | paid, after B6 | new pre-registration |
| B8 | Later, design-first: judge/rubric hybrid reward capped under the `--prm-cap` containment policy (judge rewards are gameable; symbolic verdicts stay authoritative); multi-turn trajectory arm using B3 + a verifiers `MultiTurnEnv` rollout (targets `multi-turn-trajectory-rlvr-sweep-open-2026-06-30`) | paid | design doc first |

Risks: TRL flag-name/semantic drift across versions (every flag through the dataclass guard +
run report, or cross-run comparability dies again); tiny-prompt-set GRPO has high seed variance
(expect wide CIs; growing verifiable rows via persona-style synthetic generation through the
gate is its own candidate workstream); never silently drop the KL penalty (`beta=0` only as a
separate flagged arm); GLM-4-9B license restriction already documented — Qwen2.5-Coder-7B
(Apache-2.0) stays the sweep base.

## Track C — from-scratch provenance-native path (from karpathy/nanochat)

The question this track answers (and adapters cannot): does a model with provenance/abstention
tokens in `SPECIAL_TOKENS` from token zero, trained on gate-filtered data, show measurably
better attribution behavior than a plain twin at equal val_bpb? A $100-tier model is a
"kindergartener" (Karpathy's own words) — this is a **method** demonstration; capability
claims are out of scope by charter.

| # | Step | Cost | Files |
|---|---|---|---|
| C1 | Pre-registration design doc: marker schema (candidate: `<\|src_start\|>ID<\|src_end\|>` per doc + `<\|abstain\|>` + confidence tokens), twin-arm ablation (provenance vs plain, same seed/data/FLOPs), metric definitions, named gate | $0 | new `docs/06-Roadmap/Sophia-Nanochat-Provenance-Speedrun.md` |
| C2 | **Run the existing toy lab** (code exists, largely unrun at scale): multi-seed `ablation.py` + `provenance_eval.py` + `abstain.py` on CPU/Spark to freeze schema + metrics. Kill criterion: no directional effect across seeds → the GPU question dies for $0 | $0 | `pretraining/gpt/` |
| C3 | Vendor the fork inside this repo (pinned upstream commit) so it lives under the CI gates; a separate repo would escape the no-overclaim machinery. Record whether current master still has `mid_train.py` (upstream churn: vocab 65k→32k, fp8 added) | $0 | new `pretraining/nanochat_fork/` |
| C4 | CPU smoke fork: provenance tokens into `nanochat/tokenizer.py` `SPECIAL_TOKENS`; source-tag injection at the `<\|bos\|>` splice point in `nanochat/dataset.py`; data-passport filter pass upstream of `base_train` (nanochat has no filtering stage — clean insertion). Prove tokenize→train→decode round-trip at d2–d4 | $0 | fork diff + repo test |
| C5 | Micro-tier twin-arm runs on owned hardware: Mac Studio via nanochat-mlx (d20 ≈ overnight; port is young — 11 commits, no RL, measure throughput first) and/or Spark single-GPU. Iteration tier, `headline_ok: false` | ~$0 | fork |
| C6 | Decontam + eval prep: FineWeb-EDU/SmolTalk mixtures embed MMLU-aux-train and GSM8K → mandatory decontam pass against every eval surface any gate will cite; build the held-out provenance eval set | small | `tools/assert_decontam.py`, `provenance_bench/` |
| C7 | Single-GPU RunPod pilot (d20-ish) through the existing pod lifecycle (create→train→gate→copy→delete) to fix the cost model | ~$10–50 | `tools/runpod_train.py` |
| C8 | **LAST, gated on the P0 rule** (`Distributed-Training-FSDP-Path.md`): 8×H100 twin-arm speedrun ablation (~$100–300; leaderboard-era runs ~$48, spot ~$15). Proposed method gates (to pre-register, not claims): G1 forbidden-attribution rate lower in the provenance arm with 95% bootstrap CI excluding 0 at val_bpb within ~2 % of the plain twin; G2 held-out source-citation recall ≥ threshold set from C5; G3 abstention ECE ≤ ~0.10 and risk-coverage AUC above a confidence-threshold baseline. Pass → the ingredient may flip in `recipe_spec.json`; fail → failure ledger. Pre-register an MDE + stop rule so a null is reportable, not re-rolled | $100–300, owner GO | fork + `runs/speedrun.sh` |

Risks: cost discipline (twin-arm doubles the bill; ablate at d12–d16 first; pods always
deleted); upstream churn (pin a commit — claims attach to the pin, not "nanochat"); scope
creep toward leaderboard-chasing (anti-goals from the brainstorm apply verbatim); MLX port
immaturity (iteration only, never source-of-record).

## Sequencing across tracks

- **Now ($0, offline, CI-testable):** A1, A4, B1–B5, C1–C4. All are code/docs with offline
  tests; none can move a claim on their own.
- **Cheap measured rows (Spark/Mac, minutes–hours):** A2 → A3 (the plausible near-term full-cert
  GO), C2, C5.
- **Paid, pre-registered, owner-gated:** B6 (the pending sweep — the only run that can move
  `rlvr-live-run-not-yet-gated`), then B7; A5 (needs explicit owner GO past the banked-v6
  decision); C6–C8 in order, C8 last and P0-gated.
- Every paid run lands as its own ledger row, GO or NO-GO.

## Sources

- NVIDIA Model-Optimizer: <https://github.com/NVIDIA/Model-Optimizer> · QAT/QAD example:
  <https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/llm_qat/README.md> ·
  QAD paper: <https://arxiv.org/abs/2601.20088>
- open-instruct: <https://github.com/allenai/open-instruct> · GRPO docs:
  <https://allenai.github.io/open-instruct/algorithms/grpo/> · Tülu 3: <https://arxiv.org/abs/2411.15124> ·
  verifier interface: <https://github.com/allenai/open-instruct/blob/main/open_instruct/ground_truth_utils.py> ·
  RLVR data: <https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints> ·
  willccbb/verifiers: <https://github.com/willccbb/verifiers>
- nanochat: <https://github.com/karpathy/nanochat> · walkthrough:
  <https://github.com/karpathy/nanochat/discussions/1> · MLX port: <https://github.com/scasella/nanochat-mlx>
