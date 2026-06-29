# Sophia chem-bio curriculum pack (Stage 2 data → Stage 3 QLoRA)

Oracle-verified synthetic chemistry + biology SFT rows for the pre-registered experiment
`sophia-chem-bio-curriculum`. Every row's gold answer is produced by a **deterministic
oracle** (`agent/chem_verifier.py`, `agent/bio_verifier.py`) and re-verified through that
oracle before it is kept — the chem/bio analogue of the sympy/exec gating in the
math-code pack. **Training-oracle passes are NOT benchmark proof; `canClaimAGI` stays
False.**

| File | Rows | Notes |
|------|------|-------|
| `sft_all.jsonl` | 116 | Combined chemistry + biology (+ abstention) — use for SFT |
| `sft_chem.jsonl` | 62 | Chemistry rows (incl. chem abstention) |
| `sft_bio.jsonl`  | 54 | Biology rows (incl. bio abstention) |
| `manifest.json`  | —  | Counts, oracle map, tier ladder, contamination guard |

Held-out tier3 (**never trained**): `eval/chem_bio_capability/heldout_v1.jsonl` (16 items).
Living under `eval/**`, it is automatically a held-out surface for the contamination guard
(`provenance_bench.dataset_guard`), so any train/eval prompt overlap fails closed.

Pre-registration: `agi-proof/sophia-chem-bio-curriculum/preregistration.json`

## Tier ladder

- **tier0 — facts & parsing:** molar mass, atom counts, GC%, reverse-complement, transcription.
- **tier1 — single-step quantitative + abstention:** equation balancing, mole↔gram,
  codon translation, Hardy–Weinberg; **calibrated-abstention** rows (unanswerable prompts
  whose gold target is a refusal, verified to carry a hedge marker and no fabricated value).
- **tier2 — multi-step:** percent-by-mass, mass-to-mass stoichiometry, Punnett ratios,
  expected offspring counts.
- **tier3 (held-out eval only):** percent yield, molarity, ORF-to-stop translation, dihybrid
  recessive counts, carrier frequency.

## Regenerate data (Stage 2)

```bash
python tools/generate_chem_bio_curriculum.py            # writes pack + held-out eval
python tools/generate_chem_bio_curriculum.py --check    # validate, no writes (CI-safe)
```

The generator is deterministic (`seed = 20260629`); `tests/test_generate_chem_bio_curriculum.py`
asserts the committed `sft_all.jsonl` matches a fresh generation (drift guard).

## Stage 3 GPU — QLoRA 4-bit (3 seeds)

Completion-only loss (`--mask-prompt` is default). Base model per manifest:
`Qwen/Qwen2.5-7B-Instruct`. `--data` accepts the pack directory (resolves
`manifest.json` → `sft_all.jsonl`).

```bash
pip install -r requirements-lora.txt
# optional oracle extras for eval-time grading of free-text answers:
pip install -r requirements-chem.txt   # rdkit, pint  (chem_verifier abstains without them)

for SEED in 0 1 2; do
  python tools/train_lora.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --data training/sophia-chem-bio-curriculum/ \
    --4bit --epochs 2 --seed "$SEED" \
    --output training/sophia-chem-bio-curriculum/checkpoints/seed${SEED}
done
```

Do **not** pass `--scaffold`/`--guard` (oracle-verified curriculum, not provenance-corpus SFT).
Per repo policy, RunPod jobs run via **GitHub Actions only**, never local SSH.

## After training (Stage 4+, not prep)

- Record Qwen2.5-7B **base** on the sealed held-out oracles before citing any adapter Δ.
- Score adapter vs base on `eval/chem_bio_capability/heldout_v1.jsonl` (oracle markers) +
  a ≥2-family judge panel + an external anchor (MMLU college_chemistry/biology, GPQA chem/bio),
  ≥3 seeds, 95% CI excluding 0 (see the pre-registration).
- Gate the **abstention** behavior and the **dual-use hazard floor** (WMDP-bio/chem ≤ base)
  as GO/NO-GO in `tools/promote_adapter.py`; keep the protected floor (religion/history).
- Until all of that clears: results are **candidate**, `canClaimAGI: false`.
