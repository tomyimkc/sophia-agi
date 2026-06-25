# local_sophia_7b — Qwen2.5-7B training pack

Verifier-gated wisdom-model data for `claude/sophia-7b-train-verify`.
**NOT AGI** — behavioral discipline only.

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Build: `python tools/build_local_sophia_dataset.py --out training/local_sophia_7b --base-model Qwen/Qwen2.5-7B-Instruct`
- Guard: `python tools/build_local_sophia_dataset.py --check --out training/local_sophia_7b`
- Holdout seal: `python tools/seal_sophia_7b_holdout.py --check`
- CUDA train: `python tools/train_lora.py --model Qwen/Qwen2.5-7B-Instruct --4bit --mask-prompt --epochs 2`
- RunPod (3 seeds): `agi-proof/sophia-7b-train-verify/runpod-sft-3seed.sh`

Adapter weights are gitignored; commit manifests and eval reports only.
