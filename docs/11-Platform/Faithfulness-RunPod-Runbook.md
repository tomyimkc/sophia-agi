# Faithfulness GRPO on RunPod — turnkey runbook

How to launch the retrieval-faithfulness GRPO run (reasoning-core design) on a rented
RunPod GPU through GitHub Actions, read the result, and turn it into a promotable claim.

**Scope ceiling:** every artifact this produces is `candidate_only; canClaimAGI:false;
narrow corpus-bound feasibility`. A run does **not** by itself establish an uplift — only
`tools/claim_gate.py` against the pre-registered spec does. This runbook augments (does not
replace) the `wisdom-gpu-prebaked` cost-guard runbook; read that first.

Related: `agi-proof/reasoning-core-design.md` (design + the four-stage recipe),
`agi-proof/benchmark-results/faithfulness/measurement_spec.json` (pre-registration),
`provenance_bench/faithfulness_{rollout,grpo,eval,seams}.py` + `retrieval_faithfulness.py`
(the implementation), `tools/run_rlvr.py --task faithfulness`, `tools/eval_faithfulness.py`.

---

## 0. One-time prerequisites (operator only — these spend money / hold secrets)

1. **Rotate any API key that was ever pasted into a chat.** Treat it as exposed.
2. **Repo Actions secrets** (Settings → Secrets and variables → Actions):
   - `RUNPOD_API_KEY` — required; the account must have GPU credit.
   - `DEEPSEEK_API_KEY` **and/or** `LLMHUB_API_KEY` — only if you want the live entailment
     verifier (`entailment_provider` = `deepseek` / `llmhub`). Omit to use the deterministic
     `lexical` placeholder (no external calls).
   - Optional repo **variable** `LLMHUB_BASE_URL` (default `https://api.llmhub.com.cn`).
   The launcher forwards these into the pod env **only when set**; they are never printed or
   committed (`private/secrets/` is gitignored; the workflow reads from secrets).
3. The workflow must also exist on `main` for `workflow_dispatch` registration (it does once
   this branch merges; until then dispatch against the feature branch where it exists).

---

## 1. Cheap validation FIRST (≈ a few minutes, ~$0.30)

Always do an `offline` + `limit=24` dispatch before any full run — it surfaces train/eval/
retrieval/entailment bugs for cents instead of a pod-hour. Dispatch **rlvr-runpod** with:

| input | value | why |
|---|---|---|
| `confirm` | `RUN` | required acknowledgement of GPU cost |
| `remote_mode` | `offline` | cheap remote smoke; skips the live GRPO |
| `task` | `faithfulness` | selects the retrieve-then-reason loop |
| `entailment_provider` | `deepseek` (or blank) | live verifier vs lexical placeholder |
| `limit` | `24` | cap training cases for the validation |
| `reward` | `verifier` | required field; **ignored** by the faithfulness task |
| `model` | `Qwen/Qwen2.5-7B-Instruct` | an **instruct** model (not the Coder default) |
| `seed` | `0` | canonical |
| rest | defaults | gpu_type / interruptible / network_volume_id |

The `offline` mode runs the reward-wiring + instrument invariants on the pod and copies back
`*.rlvr.offline-report.json` — proving the path end-to-end without a GRPO spend.

---

## 2. The live run

Re-dispatch with `remote_mode = live` (keep `limit=24` for a first real run; raise it once a
small live run is clean). For the faithfulness task the launcher automatically uses
`--vllm none` (the loop is a custom GRPO, no vLLM) and forwards `--entailment-provider` +
`--limit`. On the pod the job: clones the branch → installs `requirements-rl` → trains the LoRA
(`tools/run_rlvr.py --task faithfulness …` → `faithfulness_grpo.train`) → runs the **on-pod
base-vs-adapter** held-out eval (`eval_faithfulness.py --compare`, local-HF policy seam) → tars
the adapter → self-deletes.

### Watch (do not walk away for the first ~6 minutes)

- **Restart loop** = ≥2 `pod heartbeat` / repeated `self-reported result` commits ~45 s apart,
  or repeated identical artifacts. If you see it, **cancel the run and delete the pod now**
  (don't rely on the auto-abort). Cause is usually a setup death (pull-auth, weight-download
  filling the volume) — re-diagnose on a `limit=24` validation.
- **Zero leaked pods at the end** (the anti-wastage contract):
  ```
  curl -sS https://rest.runpod.io/v1/pods -H "Authorization: Bearer $RUNPOD_API_KEY"
  ```
  Expect `[]`. Delete anything named `sophia-rlvr-*` left from this effort; leave other
  efforts' pods alone.
- **Actions "success" ≠ a result.** Confirm the artifacts actually came back (next section).

---

## 3. Read the result

Artifacts land under `agi-proof/benchmark-results/runpod-rlvr/` (also in the run's uploaded
artifacts). The faithfulness-specific one:

- `*.faithful.compare-eval.json` — the **base-vs-adapter** contrast:
  - `baseGroundingRate` / `adapterGroundingRate` — counterfactual grounding rate (fraction of
    knowledge claims that are supported **and** disappear when their source chunk is dropped).
  - `meanDiff` + `pairedBootstrapCI95` — the per-case paired uplift and its 95% CI.
  - `nPaired`, `caveats` — read these. At `limit=24` the N is far below the pre-registered
    `requiredN` (377), so the CI is wide and the point estimate is **illustrative**, not a result.
- `*.sophia-rlvr-v1.tar.gz` — the trained LoRA adapter (+ `.sha256`).
- `*.rlvr.public-report.json`, `*.repo-head.txt` — run provenance.

What "good" looks like at this stage: `adapterGroundingRate > baseGroundingRate` with a paired
CI clear of 0 — but **only** as a candidate signal. A wide CI or an unstable rate across seeds
means "underpowered," not "worked."

---

## 4. From a run to a promotable claim (the honest last mile)

The compare-eval is `candidate`, never a GO. To promote, satisfy the pre-registered
`measurement_spec.json` and let the gate decide:

1. **Power it.** Raise `limit` (and run multiple `seed`s) until the eval yields ≥ `requiredN`
   knowledge claims; report the paired bootstrap CI **and** the anytime-valid confidence
   sequence (the loop iterates, so fixed-n CIs alone are not enough).
2. **≥2 independent judge families**, judge ≠ subject (the trained policy): e.g. DeepSeek +
   an LLMHub non-DeepSeek family; report inter-judge agreement (κ ≥ 0.40 or Gwet AC1 + CI).
3. **Decontaminate** (content-shingle) and reserve a **private held-out split** never used in
   any tuning/selection loop.
4. **Run the gate:** `tools/claim_gate.py` against the spec → GO/NO-GO. On GO, promote in
   `published-results.json` and regenerate `RESULTS.md`. On NO-GO, record an honest negative.

Until all four hold, the wording stays `candidate_only; canClaimAGI:false`.

---

## 5. Quick reference

```
# cheap validation (offline, ~$0.30): rlvr-runpod dispatch
#   confirm=RUN remote_mode=offline task=faithfulness entailment_provider=deepseek
#   limit=24 model=Qwen/Qwen2.5-7B-Instruct
# then live:
#   confirm=RUN remote_mode=live  task=faithfulness entailment_provider=deepseek limit=24
# end every effort:
curl -sS https://rest.runpod.io/v1/pods -H "Authorization: Bearer $RUNPOD_API_KEY"   # expect []
# read:
#   agi-proof/benchmark-results/runpod-rlvr/<pod>.faithful.compare-eval.json
# offline instrument check (no GPU, anytime):
python tools/eval_faithfulness.py --mock
```
