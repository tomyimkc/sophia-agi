# Local Sophia Wisdom Model — training plan (honest, verified)

**Goal:** a small/medium local LLM fine-tuned to follow Sophia's source discipline,
uncertainty, abstention, council reasoning, moral routing, and tool habits — wrapped by
Sophia's external MCP/verifier gates. **Not AGI.** The weights learn *habits*; the external
gate enforces *truth discipline*. This realises **W1/W4** of `AGI-Substrate-Plan.md`.

> Every command below was verified to exist with the flags shown. Fine-tuning itself needs
> your hardware (Mac/MLX or a cloud GPU) — it does not run in CI or on the maintainer's
> shared sandbox.

## The offline scaffolding (built; runs anywhere, no GPU)

```bash
python tools/export_training_jsonl.py                 # training/corpus.jsonl (528)
python tools/wiki_to_training.py                      # wiki_provenance_{sft,dpo}.jsonl
python tools/mine_hard_negatives.py                   # hard_negatives_dpo.jsonl
python tools/prepare_lora_dataset.py                  # training/lora/{train,holdout}.jsonl (439/89)
python tools/build_local_sophia_dataset.py            # assemble + DECONTAMINATE + manifest
```

`build_local_sophia_dataset.py` writes `training/local_sophia_v2/` with role-split packs
(`sft_source_discipline`, `sft_wiki_provenance`, `sft_council_traces`, `sft_moral_gate`,
`general_instruct`, `dpo_hard_negatives`, `dpo_wiki_provenance`, `holdout`) and a
`manifest.json`. It **fails closed** if any training prompt overlaps the held-out `eval/**`
sets or the holdout — and it **decontaminates** (drops the overlap) first, recording
`droppedForDecontamination` in the manifest. (`--check` runs the guard with no writes;
it's wired into CI via `tests/test_local_sophia_dataset.py`.) The build also writes
`training/local_sophia_v2/mlx/{train,valid}.jsonl` for MLX-LM chat training.

## The 6 corrections baked into this plan

1. **General instruction-retention is required.** The current v2 pack includes a small
   license-clean external `OpenAssistant/oasst1` Apache-2.0 slice at
   `training/local_sophia_v2/general_instruct.jsonl`; verify the license before release or
   commercial reuse. It is not synthesized from the Sophia corpus.
2. **One adapter first, not six.** Prove uplift on a single `sophia-v2` adapter at acceptable
   false-positive cost *before* splitting into per-role adapters (source/moral/tool/council).
3. **DPO data exists; a DPO trainer does not.** The implemented RL path is **RLVR/GRPO**
   (`tools/run_rlvr.py`). Treat the DPO packs as future, or add a TRL-DPO cloud script.
4. **Train/eval disjointness is enforced in code**, not a comment — the contamination guard
   above (`provenance_bench/dataset_guard.py`).
5. **Baseline before training.** Record the base-model numbers (`manifest.baseline`) via the
   eval ladder *before* you fine-tune; "uplift" is always vs that stored baseline.
6. **`moral_gate_sft` is generated from `moral_corpus/`.** Run
   `python tools/build_moral_gate_sft.py`; it covers all seven verbs
   (`allow|revise|retrieve|clarify|escalate|abstain|block`) and remains candidate-only
   routing supervision, not a moral-consciousness claim.

## Train (your hardware)

**Mac / Apple Silicon — MLX-LM** (not bitsandbytes, which is CUDA-only):
```bash
pip install mlx-lm
mlx_lm.convert --hf-path Qwen/Qwen2.5-3B-Instruct -q --q-bits 4
python3 -m mlx_lm lora --train --model Qwen/Qwen2.5-3B-Instruct \
  --data training/local_sophia_v2/mlx --iters 500 --batch-size 4 --mask-prompt \
  --adapter-path training/mlx_adapters/sophia-v2
```
`--mask-prompt` = learn the *answer* behavior, not memorize prompts.

**Cloud GPU — repo QLoRA / RLVR:**
```bash
pip install -r requirements-lora.txt
python tools/train_lora.py --model Qwen/Qwen2.5-3B-Instruct --4bit --epochs 2 \
  --output training/lora/checkpoints/sophia-v2
# verifier-as-reward (GPU): tools/run_rlvr.py --model <m> --vllm none   (offline: --model mock --dry-run)
```

## Evaluate — the ladder (promotion rule)

```bash
python tools/eval_ladder.py --backend mlx --dry-run     # verify MLX wiring (CI-safe, no weights)
python tools/eval_ladder.py --backend mlx --model Qwen/Qwen2.5-3B-Instruct
python tools/eval_ladder.py --backend mlx --model Qwen/Qwen2.5-3B-Instruct \
  --adapter training/mlx_adapters/sophia-v2
# CUDA/PEFT path remains: python tools/eval_ladder.py --adapter training/lora/checkpoints/sophia-v2
```
Rungs: `base · base+gate · adapter · adapter+gate`, plus real-weight
`run_seib.py --real-model`, `run_all_phase_benchmarks.py`, `run_council_uplift.py`,
`run_moral_public_standard_eval.py`.

**Promotion rule:** promote only if provenance/citation improves at **acceptable
false-positive cost** — *fewer hallucinations at acceptable over-abstention*, not abstention
maximised. No result is promoted without clearing the no-overclaim gate.

## v2 first MLX run (2026-06-24)

One local MLX adapter was trained and measured as candidate evidence only:

| Rung | Score | Gate flags |
|---|---:|---:|
| MLX base | 16/32 (50.0%) | 0 |
| MLX base + gate | 16/32 (50.0%) | 29 |
| MLX adapter | 20/32 (62.5%) | 0 |
| MLX adapter + gate | 20/32 (62.5%) | 28 |

Promotion status: **not promoted**. Aggregate score improved, but religion regressed
(1/6 → 0/6), gate flags remain high, and a SEIB smoke run did not clear the promotion rule.
See `training/local_sophia_v2/eval_ladder_adapter.json` and the failure-ledger entry
`local-sophia-v2-mlx-trained-not-promoted-2026-06-24`.

## Honest bound
This trains **behavioral discipline**, not general intelligence. Fine-tuning does not give
grounded world models, long-horizon autonomy, continual safe learning, or general transfer.
Honest headline: *"a local, verifier-gated wisdom model — external gates enforce correctness."*
