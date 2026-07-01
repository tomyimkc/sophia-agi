# DISPATCH — 3 pre-registered live runs (cloud → Mac/farm session)

**Branch:** `claude/sophia-w1-w5-live-g04zs1` (fetch + checkout it). **Why you and not the cloud:** the
cloud session has NO API keys and cannot reach the local farm — any run there hits the mock backend,
and *a metric from mock is a fabrication*. You have the Spark vLLM + Mac mlx servers + the keys. The
cloud verified the baseline is green (14/14 offline tests, `make claim-check` GO/OK) and the three
instruments are real and dispatch-ready.

## Cardinal rules (do not bypass)
- **Fail closed.** No backend / mock / degenerate input → an environment artifact (`status:not_run`,
  `ok:false`) or non-zero exit, NEVER a fabricated metric.
- **Real backend, not mock.** `agent.model` returns `"mock"` with no key and mock `.generate()`
  fabricates text with `ok=True`. Assert `kind != "mock"` before trusting any number; any judge
  response containing `[mock:` voids that run.
- **A run is not a result.** Flip a ledger row ONLY when its pre-registered gate is met, with the
  artifact + sha256 under `agi-proof/benchmark-results/`. Otherwise the row stays **Open** and you
  report the number you got + why it didn't clear the gate. Do NOT relax any bar.
- Before any commit/push to g04zs1: `python tools/lint_claims.py` + `make claim-check` + the
  `ci-artifact-drift` skill. `canClaimAGI` stays **false** everywhere.

---

## ① m3-3family judge — DO FIRST (closest to VALIDATED)
The 2-family run already found BOTH families favor the adapter across 3 seeds; only inter-judge
κ=0.236 < 0.40 blocks VALIDATED (prevalence-deflation, Feinstein-Cicchetti — not real disagreement).
Adding a 3rd vendor family (deepseek) + forced-choice is the honest attempt to clear or confirm.

**Serve first:**
- Spark: `vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000` (aarch64+sm_100 build)
- Mac: `mlx_lm.server --model mlx-community/Llama-3.3-70B-Instruct-4bit --port 8080`
  (reach via **openai:** transport, NOT `mlx:` — the local loader returns empty/all-TIE)
- Export `OPENROUTER_API_KEY` for the deepseek family.

**Run (seeds 1,2,3):**
```
OPENROUTER_API_KEY=... python3 tools/judge_pilot_answers.py \
  --answers agi-proof/benchmark-results/wisdom-market/M3-pilot-answers-seed{1,2,3}.json \
  --judges vllm:Qwen/Qwen2.5-7B-Instruct@http://SPARK_HOST:8000/v1,openai:mlx-community/Llama-3.3-70B-Instruct-4bit@http://MAC_HOST:8080/v1,openrouter:deepseek/deepseek-chat \
  --forced-choice \
  --out agi-proof/benchmark-results/wisdom-market/m3-3family-judge/judge-seed{1,2,3}.json
```
(Subject = gemma-3-4b-it; the 3 judge families qwen / mlx-community / deepseek are all ≠ subject —
CI-verified. `tests/test_mac_judge_3family_config.py` covers the config.)

**Pre-registered gates (`.../m3-3family-judge/measurement_spec.json`):**
- **VALIDATED:** all 3 families' adapter win-rate Wilson95 CI-excludes-0.5 across ≥3 seeds **AND**
  (≥1 pairwise κ≥0.40 **OR** forced-choice majority-consensus reliability ≥0.40).
- **CANDIDATE (deflation-bound):** all 3 families significant across ≥3 seeds BUT every pairwise +
  consensus κ<0.40 → real edge, but κ is the wrong instrument for a high-prevalence task. **This is
  itself a valid, reportable finding — do NOT relax the κ bar to force VALIDATED.**
- **NO-GO:** any family's win-rate CI includes 0.5, or a judge is dropped/mock, or the adapter loses.
- On result: replace the `.PENDING.` artifact with the real report + **sha256**; update ledger row
  **m3-sft-2family-judge-not-validated-2026-06-29**.

---

## ③ source-contamination rigor pass — DO SECOND
2026-06-28 caught 42/43 (97.7%) but single-run, answer==judge, curated refs. Harness already has
`--runs N`+bootstrap CI, `--answer-spec`/`--judge-spec` (enforced-to-differ), `--retrieve` (open-world)
— no code change (`tests/test_source_contamination_rigor.py` covers the offline aggregation).

**Run — ≥2 DISTINCT family pairs, answer≠judge, ≥3 runs each, with `--retrieve`:**
```
OPENROUTER_API_KEY=... OPENAI_API_KEY=... python3 tools/run_source_contamination_bench.py \
  --relay --answer-spec deepseek:deepseek-chat --judge-spec openai:gpt-4o-mini --runs 3 --retrieve
# then a 2nd distinct pair, e.g.:
OPENROUTER_API_KEY=... python3 tools/run_source_contamination_bench.py \
  --relay --answer-spec openrouter:qwen/qwen-2.5-72b-instruct --judge-spec deepseek:deepseek-chat --runs 3 --retrieve
```
- `--answer-spec` and `--judge-spec` MUST differ (harness exits 2 if equal). `--fake` is CI plumbing
  only — **never a result.**
- **Gate → VALIDATED:** ≥3 runs/family with answer≠judge AND a `--retrieve` arm — `caught_rate` CI
  **lower** bound stays high AND `clean_over_blocked_rate` CI **upper** bound stays low, on **≥2
  families**. On result: replace `rigor-multifamily.PENDING.public-report.json` with the real report +
  **sha256**; update ledger **source-contamination-live-multifamily-2026-06-28**.

---

## ② realtime C1 Phase-1 — LARGEST BUILD (do last / in parallel)
Seed committed (`eval/fact_check/phase1_dated_seed_v1.jsonl`, N=28 real dated facts = ~7% of target —
a SEED, not a result).
- **Bottleneck:** grow to **N ≥ 393** (`eval_stats.required_n_for_mde(0.10)`; spec brackets up to ~785
  paired at rho=0) with **real dated live sources**. Do NOT synthesize/paraphrase to inflate N (that is
  contamination). Keep `sourceTimestamp`/`validFrom` on every item; temporally clean vs the 2026-07-01
  cutoff.
- **Run:** `python3 tools/run_realtime_benchmark.py --online` (keyless LiveFactBackend; **assert
  kind!=mock**). Report precision/recall/fabricationRate with 95% CIs; extend decontam to time; add ≥2
  independent judge families (judge ≠ verifier, κ≥0.40); gate any slow-loop LoRA delta through
  `tools/claim_gate.py` before merge. On result: replace `phase1-online.PENDING.public-report.json`.

---

## Also: the lost W1–W5 instruments (verify, don't reconstruct)
The original W1–W5 untapped-training instruments (`tools/{train_calibration_objective,
distill_process_reward_model,provenance_weighted_training,adversarial_gate_selfplay,
probe_representation_training}.py` + tests + `agi-proof/untapped-training-2026-07-01/`) were untracked
in a prior container and are absent from every ref + disk. **First check the Spark / Mac Studio / any
other checkout for the untracked files and `git add` them if they exist** (honest recovery). If they
are truly gone, LEAVE the honest ledger row as-is. **Do NOT reconstruct them from scratch** unless the
owner explicitly authorizes it — new code must be labeled as fresh/unvetted, never as "the advisor's
tested instruments." (Owner has NOT authorized reconstruction as of this dispatch.)

## Report back
Write outcomes (verdict + numbers + which gate was/ wasn't met) to `handoff/from-mac/STATUS.json` and
the relevant ledger rows. Each row flips ONLY with a real (non-mock) run meeting its pre-registered
gate + committed artifact + sha256. `make claim-check` green + `canClaimAGI:false` before every push to
`claude/sophia-w1-w5-live-g04zs1`.

## Note for the cloud
The `.claude/skills/**` files are git-crypt-encrypted (public repo) and git-crypt is LOCKED in the
cloud session — the cloud must NOT commit them (would write plaintext + leak secret module names). If
they need tracking, commit them ENCRYPTED from a key-holding box (`git-crypt unlock` first).
