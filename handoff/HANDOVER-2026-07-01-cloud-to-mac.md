# HANDOVER — cloud → Mac Studio local bench (2026-07-01)

You are the **Mac Studio operator** for `tomyimkc/sophia-agi`. The cloud session has been driving the
v5/v6 NVFP4 low-RAM cert work and needs you to execute the parts only a box with the GPU + both git
trees can do. **Read `handoff/to-mac/STATUS.json` (this branch, commit `5133fed7`) as the authoritative
machine-readable detail** — this doc is the human summary of the same thing.

Branches: work = `claude/sophia-positioning-gaps-84kb0v` (HEAD **`c69c5ce8`**); bridge = `spark-bridge`
(the Hermes idle-poller runs from a checkout of it); coordination = `mac-handoff` (from-mac / to-mac).

---

## State the cloud established (all pushed)

- **Work branch `c69c5ce8` is the single v6+hardened source of truth.**
- **v5 @ NVFP4 genuinely FAILS both cert bars** at n=1024: top1 **0.8975** < 0.97, mean_kl **0.0551** >
  0.05. Fully merged, `incomplete_merge=false` (the peft monkeypatch works; this is the honest number,
  not the old 64/96 half-merge artifact). Confirmed by both the Mac and the cloud re-cert.
- **But v5 + conformal abstention SHIPS.** 95%-robust operating point: coverage **0.6426** / answered
  **0.9970** / floor **0.9831** (abstain on the ~36% lowest-margin tokens, answer the rest at ~99.7%
  with a 95%-confident floor above 0.97). Built end-to-end:
  - `serving/quant_abstention.py` — the coverage/answered-top1 frontier.
  - `serving/abstention_serve.py` — `policy_from_cert(cert, confidence=0.95)` selects the max-coverage
    point whose **Wilson lower bound** clears the target (never the razor-thin point estimate); raises
    if none is robust (never falls back to answering everything).
  - `serving/abstaining_decoder.py` — enforces the policy in the token loop (confident → argmax;
    low-margin → `ABSTAIN_TOKEN`) with a `stop_on_coverage_below` circuit-breaker.
  - `tests/test_abstention_serve.py` — 8/8.
- `down_proj`-bf16 lever (n=1024): +6pts robust coverage AND fixes mean_kl, but costs ~42% of the
  compression (3.30x → 1.90x) → a **frontier-improver, not a v6 substitute**.
- Recipe `conformal-abstention-serve` promoted `candidate → validated` (real v5 n=1024 evidence);
  **`adopted` stays false — owner-gated.**

---

## TASK 1 — deploy the v6 code path to spark-bridge + fire the v6 train

**Owner gave explicit GO on the v6 `--run-train` (2026-07-01).** The cloud did the analysis + branch
convergence but the byte-exact file movement is a **git job for you** — cloud `create_or_update_file`
replaces whole files, so pasting the 360–1191-line runner/trainer risks a typo that breaks the
allowlisted bridge runner or crashes the hours-long run. Do it with git:

1. **Byte-exact file deploy (git, NOT hand-merge):** from a `spark-bridge` worktree —
   `git checkout claude/sophia-positioning-gaps-84kb0v -- training/qat.py tools/train_lora.py` then
   commit to `spark-bridge`. The work-branch versions are **verified supersets** (qat 15/15;
   `train_lora.py` carries the Spark `device_map={"":0}` GPU-pinning that prevents the bf16 meta-offload
   crash "Tensor on device meta is not on the expected device cuda:0").

2. **Surgical runner merge (do NOT clobber `run_local_benchmarks.sh`):** the work-branch runner LACKS
   spark-bridge's `--target-modules attn-mlp` crash-fix (+ judge `/v1` + seeds). Edit spark-bridge's
   `scripts/run_local_benchmarks.sh` B1 `train_lora.py` invocation **in place** — after
   `--qat-lambda "${QAT_LAMBDA}" \` add:
   ```
   --qat-kd-weight "${QAT_KD_WEIGHT:-0}" \
   --qat-top1-weight "${QAT_TOP1_WEIGHT:-0}" \
   --qat-temp "${QAT_TEMP:-2.0}" \
   --qat-margin "${QAT_MARGIN:-0}" \
   ```
   **KEEP its `--target-modules "${QAT_TARGET_MODULES:-attn-mlp}"` and `--lora-dropout 0` lines.**

3. **Poller allowlist:** `tools/github_bridge_poll.py` `ENV_ALLOWLIST` — add `QAT_KD_WEIGHT`,
   `QAT_TOP1_WEIGHT`, `QAT_TEMP`, `QAT_MARGIN` (next to `QAT_EPOCHS`/`QAT_LAMBDA`). Mirror into
   `tools/spark_bridge.ENV_ALLOWLIST` if that file exists on your tree.

4. **VERIFY before GPU:** `python training/qat.py` (expect **15/15**, incl the 4 v6 checks). Then
   `scripts/run_local_benchmarks.sh --bench-b --run-train` (**dry-run**, no `--execute`) and confirm the
   printed B1 command contains **BOTH** `--target-modules ... attn-mlp` **AND** the four `--qat-*` v6
   flags. Only if both appear → proceed. (A wiring error here costs zero GPU.)

5. **FIRE v6** — args `--bench-b --run-train --execute`, approvedBy = owner GO 2026-07-01, env:
   ```
   QAT_BASE=allenai/OLMoE-1B-7B-0924-Instruct
   QAT_ADAPTER=training/lora/checkpoints/olmoe-qat-spark-v6
   QAT_DATA=training/lora/train.jsonl
   QAT_KD_WEIGHT=1.0 QAT_TOP1_WEIGHT=0.5 QAT_TEMP=2.0 QAT_MARGIN=0
   QAT_LAMBDA=0.0005 QAT_EPOCHS=5
   CERT_NEVAL=1024 CERT_OUT=agi-proof/benchmark-results/nvfp4-v6-cert.json
   ```

6. **Report:** write the v6 cert + abstention frontier to `handoff/from-mac/STATUS.json` + a ledger row.
   The v6 cert @ n=1024 is **Condition 2 of the pre-registered T8 adoption bar**
   (`docs/06-Roadmap/Spark-Theory-Test-Forecast.md`): adopt `conformal-abstention-serve` only if the
   95%-robust operating point shows coverage ≥ 0.60 AND floor ≥ 0.97 on **≥2 independent adapters**
   (v5 = met) **+ owner sign-off**. **Do NOT flip `adopted` yourself.** Pre-registered v6 forecast: raw
   top1 0.95–0.98, GO ~45%; a v6 that misses raw 0.97 but still serves ≥0.60 robust coverage still
   satisfies T8 (the hedge is a serving claim, not a never-flip claim).

---

## TASK 2 — real-model world-model ablation (the other high-value track)

The only path to promote any of the 6 world-model prototypes `candidate → adopted` is real-model
evidence on your 70B:

- **A)** `tools/replication_pack.py` on the 70B (raw) for its ACTUAL trap fabrication rate, then apply
  `agent/verifiability_model.decide()` to the same traps → real fabrication reduction + real
  control-over-abstain rate.
- **B)** `tools/counterfactual_traps.py` PAIRS on the 70B → does the real model's OWN abstention FLIP
  with the counterfactual edit (report intervention-consistency of the real model).
- **C)** Construct 5 traps in a NEW knowability family (not in {fictional, future-date, unfalsifiable})
  → expect `agent/verifiability_model` to FAIL on novel types; report honestly to bound #1's scope.
- Real delta at low control cost → propose the `recipe_spec` promotion + run `tools/lint_recipe.py`;
  **the owner approves the actual flip.** No transfer is a valid result → keep candidate, log it.

Sequence TASK 1 (long GPU) and TASK 2 (cheaper) however your GPU schedule prefers; hold the
one-GPU-job invariant (a stray heavy load can kill an in-flight cert/train).

---

## Guardrails (do not bypass)

- `canClaimAGI = false` throughout. v5 ships via a **measured** abstention hedge (~64% coverage), not a
  capability claim. Religion/history are PROTECTED.
- **Merge, don't clobber:** `train_lora.py`/`qat.py` via `git checkout` are byte-exact; the runner is a
  surgical edit that KEEPS `--target-modules`.
- **Dry-run BOTH flags before any `--execute`.** v6 `--run-train` is authorized ONCE by the owner.
- The **owner** flips any `adopted` (T8) — never you, never on a gate pass alone.
- `python tools/lint_claims.py` + `python tools/lint_recipe.py` must pass before any receipt commit;
  run the `ci-artifact-drift` skill (`make claim-check`) before commit/push.
- `spark-bridge` is a branch of a **PUBLIC** repo — no plaintext secrets on it, ever.
- Report status back via `handoff/from-mac/STATUS.json`.
