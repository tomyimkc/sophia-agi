# DGX Spark Runbook — M1 GPU runs (Live RL + Interpretability)

Turnkey, copy-paste recipes for the real GPU work that can't run from the cloud
session (no `ssh`/secret values there). Run these **on the DGX Spark** (Blackwell,
~128GB unified — Qwen2.5-7B fits comfortably). Each writes a report JSON you paste
back; the cloud session then aggregates, gates, and updates the failure ledger.

> **Secrets:** set them in the shell that runs these (NOT in the repo). Prefer
> `export VAR=…` in a non-committed `~/.sophia.env` you `source`. **Rotate any key
> ever pasted into chat.** Qwen2.5 is ungated, so `HF_TOKEN` is optional.

```bash
cd ~/sophia-agi && git fetch origin && git checkout claude/anthropic-skills-gap-research-ni1zl2 && git pull
python -m venv .venv && source .venv/bin/activate        # or your existing env
```

---

## A) Live RL — M1: first gated GRPO weight update (provenance arm)

Pre-registration: `05-live-rl-M1-prereg.md`. Closes ledger
`rlvr-live-run-not-yet-gated-2026-06-21`. ~6 GPU-hr.

```bash
pip install -r requirements-rl.txt        # trl, peft, vllm, bitsandbytes, etc.

# Judge keys for the multi-judge eval ONLY (training reward is judge-free).
export ANTHROPIC_API_KEY=…                 # family 1 (Claude direct)
export OPENROUTER_API_KEY=…                # family 2 (route an OpenAI/Google model ≠ Qwen)
# (Optional) export HF_TOKEN=…

# 0) Offline contract — must be green before spending (already green in CI):
python tools/run_rlvr.py --model mock
python tools/eval_rlvr_adapter.py --mode mock

# 1) Train 3 seeds — GRPO, gate reward (abstention-positive), Qwen2.5-7B, bf16+vLLM colocate:
for S in 0 1 2; do
  python tools/run_rlvr.py \
    --model Qwen/Qwen2.5-7B-Instruct --task provenance --reward gate \
    --quant bf16 --vllm colocate \
    --epochs 3 --num-generations 8 --beta 0.04 --lr 1e-5 --seed $S \
    --output training/rlvr/checkpoints/prov-qwen-s$S
done

# 2) Eval each adapter on the entity-disjoint held-out split (multi-judge semantic axis):
for S in 0 1 2; do
  python tools/eval_rlvr_adapter.py --mode real --task provenance \
    --model Qwen/Qwen2.5-7B-Instruct \
    --adapter training/rlvr/checkpoints/prov-qwen-s$S --seed $S
done
```

**Paste back:** `agi-proof/benchmark-results/rlvr.public-report.json`,
`rlvr.adapter-eval.json` (all seeds), and the per-step curves. The cloud session
runs `provenance_bench/aggregate.py::_is_validated` (≥2 judge families, κ≥0.40,
≥3 runs, CI excludes 0, no FP-regression) and updates the ledger PASS/NULL.

**Gate (pre-registered):** mean Δ>0 across 3 seeds, 95% CI excludes 0,
`_is_validated` true, no FP-regression, no abstention collapse, contamination-free.
A null is a logged outcome — do **not** weaken a gate to manufacture a pass.

> If the QLoRA(4-bit)+vLLM-colocate combo errors (trl#4973), use `--quant bf16`
> (80GB) or `--vllm server` on a second device. `run_rlvr.py` refuses the bad combo.

---

## B) Interpretability — M1→M2: harvest activations + train first SAE

Plan: `03-interpretability.md`. Offline M0 core is already green in CI; this is the
real Qwen2.5-7B run. ~$30–80-equivalent of GPU time.

```bash
pip install -r requirements-interp.txt    # torch, transformer_lens, sae_lens, nnsight

# M1 — harvest residual-stream activations at a mid layer (≈ L16/28) to sharded safetensors:
python tools/harvest_activations.py \
  --model Qwen/Qwen2.5-7B-Instruct --layer 16 --point resid_post \
  --corpus training/corpus.jsonl --tokens 50_000_000 \
  --out data/interp/qwen7b-L16            # (M1 tool lands next; see plan §M1)

# M2 — train a 16k-dict TopK SAE on the harvested activations; track L0/FVU/CE-recovered/dead-%:
python tools/train_sae.py \
  --activations data/interp/qwen7b-L16 --dict-mult 8 --k 32 \
  --out agi-proof/interp/sae-L16          # (M2 tool lands next; see plan §M2)
```

**Paste back:** `agi-proof/interp/sae-L16*.json` (metrics). **Pre-registered M2
acceptance gate:** L0 in band (~20–80), **CE-recovered ≥ ~0.9**, dead-% under
ceiling — else the SAE is reported *not-yet-usable* and we iterate (an honest
"did not meet bar" is a valid result).

> `harvest_activations.py` / `train_sae.py` are the M1/M2 entrypoints described in
> the plan; if not yet on the branch, the cloud session ships them next — the
> offline core (`interp/sae/*`, `tools/run_interp.py --mode mock`) is already in.

---

## Notes
- **Teardown / cost:** these run on hardware you own; no RunPod spend. If you instead
  use RunPod, `tools/runpod_rlvr.py` auto-deletes the pod in `finally` + watchdog.
- **Determinism:** seeds are fixed; same seed ⇒ same report. Keep the printed config
  hashes — they go in the checkpoint registry.
- **Rearm:** after pasting results back, the cloud session updates
  `agi-proof/failure-ledger.md` and the roadmap status.
