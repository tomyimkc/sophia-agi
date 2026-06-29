# M1 on DGX Spark — Local Runbook & Agent Prompt

**Status:** candidate infrastructure. `candidateOnly: true`, `canClaimAGI: false`. This runs the
**M1 acceptance gate** (Thesis D: multi-axis reward prevents reward collapse) from
[`World-Model-And-Self-Scaffolding-Program.md`](./World-Model-And-Self-Scaffolding-Program.md)
**locally on a DGX Spark** — no RunPod, no pod, no cloud cost.

## Why local
On-demand A100 80GB was out of capacity on RunPod; spot got reclaimed mid-run. The Spark
(Grace Blackwell, aarch64, 128 GB unified memory) runs the same `tools/run_rlvr.py` GRPO trainer
directly. Use `--quant bf16 --vllm none` to dodge the aarch64 wheel blockers (bitsandbytes /
vLLM-colocate / flash-attn); 128 GB unified memory makes bf16 feasible where rented pods needed 4-bit.

## What M1 proves (pre-registered, falsifiable)
The single-axis gate reward (`agent.gate_reward`, values in {-1, 0.5, 1}) **collapses** under GRPO —
within-group reward std → 0, so advantages vanish and the policy stops learning. The dense multi-axis
reward (`agent.multiaxis_reward`, Thesis D) keeps within-group std > 0.
**Acceptance gate:** `multiaxis.collapse.finalGroupStd > gate.collapse.finalGroupStd`, with the gate
arm trending toward ~0 and multiaxis staying clearly positive — at no protected-axis regression.

---

## COPY-PASTE AGENT PROMPT (run this on the DGX Spark)

> **Task: run the M1 reward-collapse comparison locally on this DGX Spark.**
>
> Repo: `tomyimkc/sophia-agi`, branch `claude/llm-training-research-brainstorm-fmeyiq`.
> Hardware: DGX Spark (Grace Blackwell, aarch64, CUDA, 128 GB unified memory). Use bf16, no vLLM,
> no bitsandbytes (aarch64 wheels are flaky). Do NOT rent any cloud GPU — everything runs on-box.
>
> **Step 0 — checkout.**
> ```bash
> git fetch origin claude/llm-training-research-brainstorm-fmeyiq
> git checkout claude/llm-training-research-brainstorm-fmeyiq
> ```
>
> **Step 1 — GPU-free sanity (must all pass before training).** These prove the harness with no GPU:
> ```bash
> python tests/test_epistemic_bench.py          # Phase 0 sealed 6-axis scorer
> python tests/test_multiaxis_reward.py          # Thesis D reward invariants (anti-collapse)
> python -m agent.multiaxis_reward               # prints the reward self-check
> python tools/run_rlvr.py --model mock --reward multiaxis --task provenance   # reward-wiring check
> ```
> Expect: all tests pass; the reward self-check shows `distinctMultiAxisValues > distinctSingleAxisValues`
> and `RLVR REWARD WIRING VERIFIED ✓`.
>
> **Step 2 — install RL deps (bf16 path; skip bitsandbytes + vllm).**
> ```bash
> pip install "torch>=2.3" "transformers>=4.53,<4.54" "trl>=0.19,<0.20" \
>             "peft>=0.13" "datasets>=3.6,<4" "accelerate>=1.0"
> # NOTE: do NOT install bitsandbytes or vllm on aarch64 — bf16 + --vllm none need neither.
> # If GLM-4-9B is gated for you, run: huggingface-cli login   (or export HF_TOKEN=...)
> ```
> The transformers/trl pins match the versions the cloud path validated (vllm-free here).
>
> **Step 3 — M1 arm A (control: single-axis gate reward).**
> ```bash
> python tools/run_rlvr.py \
>   --task provenance --reward gate \
>   --model zai-org/glm-4-9b-chat-hf --quant bf16 --vllm none \
>   --epochs 1 --seed 0 \
>   --out reports/m1-gate.json \
>   --output ckpt/m1-gate
> ```
>
> **Step 4 — M1 arm B (treatment: dense multi-axis reward). Same seed — only the reward differs.**
> ```bash
> python tools/run_rlvr.py \
>   --task provenance --reward multiaxis \
>   --model zai-org/glm-4-9b-chat-hf --quant bf16 --vllm none \
>   --epochs 1 --seed 0 \
>   --out reports/m1-multiaxis.json \
>   --output ckpt/m1-multiaxis
> ```
>
> **Step 5 — compare (the M1 result).**
> ```bash
> python - <<'PY'
> import json
> g = json.load(open("reports/m1-gate.json"))
> m = json.load(open("reports/m1-multiaxis.json"))
> gc, mc = g.get("collapse", {}), m.get("collapse", {})
> print("gate     :", gc)
> print("multiaxis:", mc)
> gate_final = gc.get("finalGroupStd")
> mx_final   = mc.get("finalGroupStd")
> passed = (gate_final is not None and mx_final is not None
>           and mx_final > gate_final and (gc.get("collapsed") is True or gate_final < 1e-2))
> print("M1 PASS" if passed else "M1 NOT MET",
>       "-> multiaxis keeps within-group reward variance where single-axis collapses")
> PY
> ```
> **M1 passes** iff `multiaxis.finalGroupStd > gate.finalGroupStd` and the gate arm collapsed
> (finalGroupStd ≈ 0). Record both `reports/*.json` (they carry `rewardSelected`, `collapse`,
> `meanReward`) and commit them under `agi-proof/benchmark-results/`.
>
> **Step 6 (optional) — sealed epistemic-bench eval.** Generate completions from the before/after
> adapters on `eval/epistemic_bench/data/cases.jsonl` and score:
> ```bash
> python -m eval.epistemic_bench.score \
>   --cases eval/epistemic_bench/data/cases.jsonl \
>   --completions reports/<model>-completions.jsonl --wiki wiki/
> ```
>
> **Notes / gotchas:**
> - `--vllm none` uses native HF generation (slower, fine for the small provenance set).
> - The trainer is wired to GLM module names (`q_proj/k_proj/v_proj/o_proj/gate_up_proj/down_proj`);
>   keep `--model zai-org/glm-4-9b-chat-hf` (a different architecture needs different `GLM_TARGET_MODULES`
>   in `tools/run_rlvr.py:69`).
> - If 9B is too heavy for a quick smoke, lower `--num-generations` / `--epochs`, not the model.
> - Equivalent one-liner via the wrapper (no pod): `python tools/runpod_rlvr.py --local --yes
>   --reward multiaxis --task provenance --quant bf16 --vllm none --model zai-org/glm-4-9b-chat-hf`.

---

## After M1: the rest of the methodology
M1 is step 1 of 6. The full plan (value-ranked, with acceptance gates) is in
[`World-Model-And-Self-Scaffolding-Program.md`](./World-Model-And-Self-Scaffolding-Program.md):
**D (done here) → B (epistemic world model) → A (self-scaffolding evidence harness) → C (calibration
warm-up) → E (learned council topology) → F (fabricator-vs-gate self-play).** B is the headline bet
(an adversarial epistemic world model for Sim-RL) and is the natural next Spark target once M1 is green.

## Cloud cleanup (do once)
The earlier RunPod attempts: offline smoke (succeeded, pod gone), one spot arm (cancelled — **check the
RunPod console for a lingering `sophia-rlvr-*` pod and delete it** if present; I could not verify from
the dev box, the API key is a GH secret), one on-demand attempt (failed at create → no pod). Also note a
**pre-existing** issue unrelated to M1: a stale committed `agi-proof/benchmark-results/runpod-rlvr/
mr9sr03clgpk5g.rlvr.adapter-eval.json` makes the workflow's ingest step "promote" stale numbers even when
no pod runs — worth cleaning up separately.
