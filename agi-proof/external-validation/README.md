# External validation — reproduce Sophia's source-discipline uplift yourself

This directory lets an **independent reviewer** confirm or refute Sophia's core
provenance claim **without trusting any committed result artifact**. You run one
command against **your own model/API**; the harness recomputes the verdict live from
your run against a **hash-pinned, pre-registered threshold**.

## The claim under test

> Sophia's source-discipline pipeline (`sophia_full`) reduces false attribution and
> contested fabrication versus the raw base model, at bounded false-positive cost, on
> the held-out external-labeled **SEIB-100** provenance set.

This is **not** an AGI claim, **not** external generalization beyond this pack, and
**not** a hallucination guarantee. A PASS confirms the uplift *on this pack only*; a
FAIL refutes it.

## What you need

- Python 3.11+ and this repo, cloned fresh.
- A model you control: an OpenRouter/OpenAI-style key, or a local MLX adapter.
  (No key? Run the offline plumbing smoke with `--allow-mock` — it is clearly marked
  **NON-VALIDATING**.)

## One command

```bash
# validating run — your own model, >=3 runs (the registration's bar):
python3 tools/run_external_validation.py --model openrouter:openai/gpt-4o-mini --runs 3

# or a local adapter:
python3 tools/run_external_validation.py --model mlx:Qwen/Qwen2.5-3B-Instruct \
  --adapter training/mlx_adapters/sophia-v3 --runs 3

# offline plumbing smoke (NON-VALIDATING):
python3 tools/run_external_validation.py --allow-mock
```

Exit code is `0` on PASS, `2` on FAIL. The aggregate report (no prompts) is written to
`agi-proof/external-validation/seib-uplift.validation-report.json`.

## What the harness does (trust-minimization)

1. **Pins the eval pack** by `sha256` against
   [`seib-uplift.preregistration.json`](./seib-uplift.preregistration.json) — a tampered
   or swapped pack fails immediately.
2. **Audits decontamination** itself: it loads the committed training sources and checks
   that **no** SEIB prompt appears in them. An empty audit (0 rows found) is treated as a
   **FAIL**, not a silent "clean".
3. **Runs SEIB-100 live** in `raw` vs `sophia_full` conditions with *your* model. Labels
   are external (provenance citations / Wikidata snapshot); the scorer is independent of
   the runtime gate (the gate is treatment only — non-circular).
4. **Recomputes the verdict** from your run's per-case rows against the pre-registered
   acceptance threshold — it does **not** read the runner's internal `ok`, and reports
   whether the two agree. Includes a **paired bootstrap 95% CI** on the accuracy delta.
5. **Prints the pre-registration's own `sha256`** so you can confirm (via git history)
   that the threshold predates any result it judges.

## Acceptance threshold (pre-registered, not editable post-hoc)

From `seib-uplift.preregistration.json` — a result is **validating PASS** only if all hold:

| Criterion | Bar |
|---|---|
| cases | = 100 |
| runs | ≥ 3, real model (not mock) |
| `sophia_full` false-attribution rate | = 0.0 |
| `sophia_full` contested-fabrication rate | = 0.0 |
| raw→full accuracy delta | > 0 **and** 95% bootstrap CI excludes 0 |
| `sophia_full` false-positive cost | ≤ 0.10 |
| training overlap | = 0 (audit must be non-empty) |
| eval-pack `sha256` | matches the registration |

The matching falsification rules (when the claim must be reported as **refuted**) are
listed in the registration's `falsification` field.

## Honest bounds

- This validates **one pack, one capability** (provenance/source-discipline). It says
  nothing about generality, long-horizon autonomy, or any AGI-level claim.
- A single run or a `--allow-mock` run is **never** validating, by construction.
- For the current local adapter the direct SEIB run did **not** clear this bar (see the
  failure ledger entry `local-sophia-v3-mlx-promoted-by-w2-but-not-validated`). This
  harness exists so that when a model *does* clear it, an outsider — not the authors —
  is the one who confirms it.
