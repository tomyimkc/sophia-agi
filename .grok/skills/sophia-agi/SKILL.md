---
name: sophia-agi
description: >
  Operate the Sophia AGI repo: provenance corpus (validate attributions, run the
  epistemic gate, score benchmarks) AND the local verifier-gated training substrate
  (build/decontaminate packs, W2 promotion gate, continual feedback loop). Use when the
  user mentions Sophia AGI, source discipline, provenance, attribution traps, corpus
  validation, benchmark scoring, local training, adapter promotion, RSI/continual
  learning, or runs /sophia-agi. Prefer MCP tools (sophia_validate, sophia_gate_check,
  sophia_benchmark_*) when the sophia-agi MCP server is enabled.
metadata:
  short-description: "Sophia AGI corpus + epistemic gate + training-substrate operator"
---

# Sophia AGI skill

**Wisdom before intelligence.** Provenance-aware reasoning across philosophy, psychology,
history, and religion — and an honest, verifier-gated local-training substrate. **Not AGI:**
the model learns *habits*; external gates enforce *truth*.

## Read first

1. `AGENTS.md` and `README.md`
2. `data/attributions.json` for any cited text; `docs/04-Disputes/` when authorship is contested
3. `docs/09-Agent/MCP-Server.md` for MCP tool wiring
4. Training/RSI/continual work: `docs/06-Roadmap/Training-RSI-Continual-Convergence.md`,
   `docs/11-Platform/Local-Sophia-Training.md`, `agi-proof/failure-ledger.md`,
   `training/feedback/README.md`
5. Portable skill (any repo): `skills/portable/sophia-source-discipline/` — install via
   `python tools/install_skills.py --all`

## Guardrails (this repo enforces them — do not bypass)

- **No overclaiming.** `python tools/lint_claims.py` must print OK before every commit. Never
  claim AGI, validated uplift, "0 hallucination", or that an adapter is promoted unless the
  promotion gate says so.
- **The promotion gate is the authority, not your judgment.** `tools/promote_adapter.py`
  (the W2 `agent/continual_plasticity.py` gate) decides promote/quarantine/reject. `religion`
  and `history` are PROTECTED suites — a regression there forces reject. Do not lower its
  thresholds to force a pass.
- **Contamination guard must stay CLEAN.** Never put eval/holdout prompts into training.
  Author NEW examples; `tools/build_local_sophia_dataset.py` decontaminates and fails closed.
- **Failures are evidence.** Record every run (including failures) in `agi-proof/failure-ledger.md`
  with the numbers and what is NOT yet proven (single seed, first-party pack, no third-party
  validation).
- **Corpus discipline.** Never attribute a text to a figure in `doNotAttributeTo`; treat
  `authorConfidence: compiled | legendary | none_extant` as uncertain; keep traditions separate
  (儒家 vs 道家) unless evidence links them.
- **Process.** Assistant outputs: English + canonical Chinese terms + concise 中文 summary.
  Ask before multi-file writes; commit only when asked; develop on the branch the user named.

## MCP tools (preferred when available)

| Tool | Use when |
|------|----------|
| `sophia_validate` | Before PR, after corpus edits |
| `sophia_gate_check` | Checking a draft answer for attribution traps |
| `sophia_benchmark_list` / `sophia_benchmark_score` | Listing / scoring eval cases |
| `sophia_corpus_stats` | Version / example counts for release notes |
| `sophia_export_corpus` | Regenerate `training/corpus.jsonl` |
| `sophia_get_attribution` / `sophia_get_record` | Lookup philosophy textId / domain record |
| `sophia_list_disputes` / `sophia_read_dispute` | Dispute notes |

## Corpus CLI (fallback when MCP is off)

```bash
python tools/validate_attribution.py
python tools/sophia_agent.py advisor "..."          # epistemic decisions
python tools/sophia_agent.py repo "..." --execute --approve
python tools/run_external_models.py --domain philosophy --providers claude-sonnet
python tools/score_benchmark.py benchmark/model_runs/MODEL.json --domain philosophy
python tools/export_training_jsonl.py               # corpus.jsonl
```

## Training substrate & continual loop (the current path)

The local-Sophia program is the **`training/local_sophia_v2/`** pack + a 4-rung eval ladder +
the W2 promotion gate. Plan: `docs/06-Roadmap/Training-RSI-Continual-Convergence.md` (C1/C3/C4
done; C2/C5 are hardware-bound). Fine-tuning needs your hardware — it does NOT run in CI.

### Build / refresh the pack (no GPU)
```bash
python tools/export_training_jsonl.py && python tools/wiki_to_training.py \
  && python tools/mine_hard_negatives.py && python tools/prepare_lora_dataset.py
python tools/build_local_sophia_dataset.py          # assemble + DECONTAMINATE + token-fit + manifest
python tools/build_local_sophia_dataset.py --check  # CI guard, no writes
```
Rows are token-fit to `MLX_MAX_TOKENS` so nothing is silently truncated
(`tools/split_long_training_rows.py <file> --dry-run` to inspect).

### Train one adapter, then evaluate (your hardware)
```bash
python tools/eval_ladder.py --backend mlx --model Qwen/Qwen2.5-3B-Instruct          # baseline FIRST
python3 -m mlx_lm lora --train --model Qwen/Qwen2.5-3B-Instruct \
  --data training/local_sophia_v2/mlx --iters 500 --batch-size 4 --mask-prompt \
  --adapter-path training/mlx_adapters/<name> --max-seq-length 1024
python tools/eval_ladder.py --backend mlx --model Qwen/Qwen2.5-3B-Instruct --adapter training/mlx_adapters/<name>
python tools/run_seib.py --real-model --model mlx:Qwen/Qwen2.5-3B-Instruct --adapter training/mlx_adapters/<name> --out <artifact.json>
```

### Decide promotion — the gate, not you
```bash
python tools/promote_adapter.py --candidate-id <id>   # reads eval ladders → PromotionDecision artifact
```
Promote only if provenance/citation improves at acceptable false-positive cost, **no protected
regression**. Record the verdict in the failure ledger either way.

### Continual loop: mined misses → next pack (no GPU; non-circular)
```bash
python tools/feedback_to_training.py mine <run_case_results.jsonl>   # → pending queue (promoted:false)
python tools/feedback_to_training.py approve <rid> --reviewer me --note "..."  # human gate (default-deny)
python tools/feedback_to_training.py build-sft                       # promoted only → sft_from_feedback.jsonl
python tools/build_local_sophia_dataset.py                           # ingest (decontaminated)
python tools/run_learning_shift.py <spec.json> --backend adapter     # post-test: post>pre, protected stable
```

### Live RLVR (open ledger item; GPU)
```bash
python tools/run_rlvr.py --model mock --dry-run     # offline invariant check (CI-safe)
# live: entity-disjoint split, ≥2 judge families, κ≥0.40, ≥3 runs, CI excludes 0 + manual review
```

## Quick workflows

- **Add corpus data:** edit `data/attributions.json` (+ dispute note) → add
  `training/examples/NNN-slug.json` → `sophia_validate` / `validate_attribution.py` →
  `export_training_jsonl.py`.
- **Check an answer before publish:** `sophia_gate_check` (question + response) → fix traps,
  add 中文 summary, re-check.
- **Benchmark a model:** `sophia_benchmark_list` → collect responses → `sophia_benchmark_score`
  → on failures `python tools/run_correction_loop.py --generate`.

## Agent paths

| Path | Command |
|------|---------|
| Advisor | `python tools/sophia_agent.py advisor "..."` |
| Repo | `python tools/sophia_agent.py repo "..."` |
| Life | `python tools/sophia_agent.py life "..."` |

> Legacy distillation lab (`tools/claude_model_lab.py`, `tools/claude_teacher.py`) predates the
> `local_sophia_v2` substrate above — use the substrate path for new training work; the lab
> remains for teacher-pair generation only.
