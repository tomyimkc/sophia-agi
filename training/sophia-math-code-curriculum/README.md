# Sophia math-code curriculum pack (Stage 2 data → Stage 3 QLoRA)

Sympy/exec-verified synthetic SFT rows for the pre-registered experiment
`sophia-math-code-curriculum`. Training-oracle passes are **not** benchmark proof;
`canClaimAGI` stays **False**.

| File | Rows | Notes |
|------|------|-------|
| `sft_all.jsonl` | 144 | Combined math + code (use for SFT) |
| `sft_math.jsonl` | 126 | Math only |
| `sft_code.jsonl` | 18 | Code only |
| `manifest.json` | — | Counts, contamination guard, tier ladder |

Pre-registration: `agi-proof/sophia-math-code-curriculum/preregistration.json`  
Held-out seal: `agi-proof/sophia-math-code-curriculum/heldout-seal.manifest.json`

## Regenerate data (Stage 2)

```bash
python tools/generate_math_code_curriculum.py
python tools/build_local_sophia_dataset.py --check
python tools/seal_math_code_heldout.py --check
```

## Stage 3 GPU — QLoRA 4-bit (local CUDA)

Completion-only loss (`--mask-prompt` is default). Base model per manifest:
`Qwen/Qwen2.5-7B-Instruct`. Run **3 seeds** (0, 1, 2) for citable numbers.

```bash
pip install -r requirements-lora.txt

for SEED in 0 1 2; do
  python tools/train_lora.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data training/sophia-math-code-curriculum/sft_all.jsonl \
    --4bit \
    --epochs 2 \
    --seed "$SEED" \
    --output "training/sophia-math-code-curriculum/checkpoints/seed${SEED}"
done
```

`--data` also accepts the pack directory (resolves `manifest.json` → `sft_all.jsonl`):

```bash
python tools/train_lora.py --dry-run --data training/sophia-math-code-curriculum/
```

Do **not** pass `--scaffold` / `--guard` for this pack (oracle-verified math/code, not
provenance-corpus SFT).

## Stage 3 GPU — RunPod (3 seeds)

**Policy:** RunPod jobs for this experiment must run via **GitHub Actions** only — not from
a Cursor agent shell or local SSH (outbound SSH to RunPod mapped ports times out from agent
egress; see `agi-proof/sophia-math-code-curriculum/stage3-runpod-blocker.public-report.json`).

1. Ensure repo secret `RUNPOD_API_KEY` is set (Settings → Secrets and variables → Actions).
2. Actions → **sophia-math-code-sft-runpod** → Run workflow → set `confirm` to `RUN`.
3. Optional inputs: single `seed` (0–2), `branch`, `epochs`, `interruptible`.

Artifacts land under `agi-proof/benchmark-results/runpod-train/` and as a workflow artifact
(`sophia-math-code-sft-artifacts`). The shell launcher below is what GHA invokes; do not run
it locally unless you have working RunPod SSH egress:

```bash
bash agi-proof/sophia-math-code-curriculum/runpod-sft-3seed.sh
```

## Optional — MLX split (Apple Silicon)

PEFT/CUDA path reads `sft_all.jsonl` directly. For `--backend mlx`, materialize a
chat-data directory (no duplicate committed in git):

```bash
python tools/prepare_math_code_mlx.py
python tools/train_lora.py --backend mlx \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data training/sophia-math-code-curriculum/mlx/train.jsonl \
  --epochs 2 --seed 0 \
  --output training/sophia-math-code-curriculum/checkpoints/mlx-seed0
```

## After training (Stage 4+, not prep)

- Record Qwen2.5-7B **base** on sealed held-out oracles before citing adapter Δ.
- Run evidence-oracle eval on held-out splits (≥3 seeds, 95% CI excludes 0).
- Run `tools/promote_adapter.py` protected-floor gate (religion/history must not regress).
