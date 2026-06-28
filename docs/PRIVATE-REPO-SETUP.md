# Open-core migration — protect the future recipe & algorithm

> **Goal:** keep Sophia's *results* open and verifiable while keeping the *next-generation
> training recipe and algorithm* private. This is the **open-core** model.
>
> **Honest premise (read first):** the v1–v3 training recipe and the current algorithm are
> **already public and irrevocable** — committed under Apache-2.0, mirrored in git history
> and any forks. Nothing below un-publishes them, and this guide does not pretend to. What
> you *can* protect is everything you build **from here on**. Plan accordingly.

## What is already permanently public (accept, then leverage)

A repo audit (2026-06) found these crown-jewel artifacts already committed:

- **Exact hyperparameters** — `training/local_sophia_v2/training_run_mlx_sophia_v2.json` and
  `training/mlx_adapters/sophia-v2/adapter_config.json` expose the full command line,
  `learning_rate 1e-05`, LoRA `rank 8 / scale 20 / dropout 0`, `adam`, `iters 500`,
  `batch 4`, `max_seq 1024`, base model `Qwen2.5-3B-Instruct`.
- **Trained weights** — `training/mlx_adapters/sophia-v2,v3/adapters.safetensors`.
- **Training data** — `training/local_sophia_7b/*.jsonl` (SFT + DPO sets).
- **Pipelines** — `.github/workflows/runpod-sophia-7b-sft.yml`, `rlvr-runpod.yml`, etc.
- **Algorithm** — `okf/`, `reasoning/consequence/`, `selfextend/`, `agent/verified_trace_rlvr.py`,
  the ConsequenceGate.

**Reframe, don't mourn:** these are now your **defensive-publication priority record** (the
DOI [10.5281/zenodo.20930874](https://doi.org/10.5281/zenodo.20930874) timestamps them as
yours). Leave them public as *proof the method is real*. The moat moves forward, not back.

## Part A — File-by-file: KEEP PUBLIC vs. GO PRIVATE

### 🟢 KEEP PUBLIC (proof layer — builds trust; already out anyway)
| Path | Why public |
|---|---|
| `RESULTS.md`, `agi-proof/**` reports, `agi-proof/failure-ledger.md` | Verifiable claims + honesty signal |
| `eval/`, `benchmark/`, `provenance_bench/` | The measurement harness — lets skeptics check you |
| `okf/`, `reasoning/`, `selfextend/`, `agent/` (current) | Already public; your evidence the gate works |
| `training/examples/**` (the 528-corpus) | Already on Hugging Face |
| `training/local_sophia_v2,v3/`, `training/mlx_adapters/sophia-v2,v3/` | Already public — historical proof, leave as-is |
| `scripts/demo_gate.py`, `schema/`, `CITATION.cff`, `paper/` | Interfaces + citation + whitepaper |
| `configs/*.sample.json` | Sanitized shapes (no real hyperparameters) |

### 🔴 GO PRIVATE (build in the private repo; stop committing here)
| What | Where it would otherwise land | Action |
|---|---|---|
| vNext **training-run configs** (v4+) | `training/**/training_run_*.json` | Private repo only |
| vNext **adapter weights + configs** | `training/mlx_adapters/sophia-v4+/` | Already ignored by `training/mlx_adapters/**`; keep it that way |
| **Data-mixing / curriculum** ratios | `pretraining/data_mixing/`, `synthetic_scaling/`, `optimizer_probe/` | Commit only *abstracted* notes; real ratios private |
| **RL reward shaping** internals | `config/reward_surface.*.json`, `selfextend/verified_reward.py` (future) | New tuning → private |
| New **algorithm internals** (pre-publication) | new modules under `agent/`, `okf/`, `reasoning/` | Develop private; publish only after you've abstracted into the whitepaper |
| **Training pipelines** baking in the recipe | `.github/workflows/*runpod*` | Keep workflow generic; move recipe params into private secrets/inputs |
| **Personal setup / keys** | `.env`, `secret/**`, RunPod IDs | Private; see `.gitignore` rules added |

### 🟡 ABSTRACT (publish the idea, not the implementation)
For each genuinely novel vNext step: write a **conceptual** description in
`paper/sophia-whitepaper.md` (proves priority, dates the idea) while **withholding the
production tuning** that makes it work. Publish *what* and *that it works*; keep *how-exactly*.

## Part B — Private-repo setup

1. **Create** a private repo, e.g. `tomyimkc/sophia-core-private` (GitHub → New → Private).
2. **Structure it** as the engine behind a clean interface:
   ```
   sophia-core-private/
     recipe/        # real training_run + adapter configs, curricula, data-mix ratios
     algorithm/     # vNext method internals, pre-publication
     data/          # curated/private datasets
     README.md      # all-rights-reserved (NOT Apache-2.0)
   ```
3. **License it** *all rights reserved* (a one-line proprietary notice). Do **not** copy the
   public repo's Apache-2.0 into it.
4. **Interface boundary:** the public repo should depend only on a *documented interface*
   (function signatures / a small reference implementation), never the private internals.
   Options:
   - **Cleanest:** keep them fully separate — public publishes results/artifacts the private
     engine produced; no code link.
   - **If you must link:** add the private repo as a **git submodule** that only *you* can
     clone (collaborators without access still build the public repo against the reference
     impl). Do **not** use git-crypt as the boundary — encrypted blobs are still visible and
     one key leak exposes all history.
5. **Move work, don't copy secrets into history:** start vNext *in* the private repo. Never
   commit it to public "temporarily" — public git history is forever.

## Part C — Going-forward hygiene (enforced by `.gitignore`)
The committed `.gitignore` rules now keep these out of the public repo automatically:
- `private/` — your local vNext working dir
- `**/*.private.*` and `.env.private` — the explicit-private naming convention
- `training/**/training_run_v[4-9]*.json`, `training/**/*.recipe.json` — new recipe dumps
- `configs/*.json` except `*.sample.json` — real configs stay out; commit sanitized samples
- `secret/**` — personal setup (run `git rm -r --cached secret/` once if it has tracked files)

**Rule of thumb:** before every push, ask "does this let a competitor *rebuild* the method,
or only *verify the result*?" Rebuild → private. Verify → public.

## Part D — Relabel honestly (recommended README change)
Holding back vNext while the README still says *"100% public … forever"* would be an
overclaim — and overclaiming is the one thing this project's brand cannot afford. Recommended
edit to the "Dual License & Trademark Protection" section:

- **Before:** "Sophia stays **100% public and Apache-2.0-licensed forever**…"
- **After:** "Sophia is **open-core**: all published code, benchmarks, proofs, and the
  evaluation harness are **Apache-2.0 and fully verifiable**; the next-generation training
  recipe and curated data are **proprietary** (developed in a private repo). What we publish,
  we publish honestly and reproducibly — we simply don't publish *everything*."

This keeps the credibility you actually rely on ("honest, verifiable measurement") while
being upfront that full from-scratch reproduction is not the promise.

## Part E — The unavoidable trade-off (state it plainly)
With the recipe hidden, a skeptic can verify your **results are real** (released outputs +
public eval harness) but **cannot verify the hidden recipe produced them** — that link rests
on your word. That is normal and acceptable for open-core, *provided you never claim full
reproducibility*. Credibility kept: honest measurement. Credibility spent: from-scratch
reproducibility. Choose the trade knowingly.
