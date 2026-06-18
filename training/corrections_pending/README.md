# Pending corrections (Phase 4 loop)

Draft training fixes from failed benchmark evals. Human review before `--promote`.

## Proof run (v0.6.0)

| File | Source failure | Status |
|------|----------------|--------|
| `correction-stockholm-every-kidnapping.json` | `local-sophia-v1-psychology` / `stockholm_every_kidnapping` | Reviewed — passes scorer |

Promote duplicates benchmark holdout `513-bench-stockholm-pop-myth-v2.json`; keep here as **pipeline evidence**.

```bash
python tools/run_correction_loop.py --dry-run
python tests/test_correction_loop.py
python tools/run_correction_loop.py --promote   # only new non-duplicate corrections
```