# Public site privacy: aligned with the repo, without training/architecture details

The thesis site under [`web/`](../web/) is published to GitHub Pages on every
push to `main` (see [`.github/workflows/pages.yml`](../.github/workflows/pages.yml)),
so it is **always aligned with the repo** by construction. This document covers
the second requirement: the public site must **not expose important training or
architecture details**.

## What is kept off the public site

Owner policy — off-limits on the published site:

- **Base-model identity** and parameter scale (e.g. the specific base model, "Nx B" sizes).
- **Internal module / architecture map** — `agent/*.py`, `provenance_bench/*.py`, the architecture diagram, internal pipeline wiring.
- **Training recipe & hyperparameters** — adapter/LoRA rank, learning rate, epochs, batch size, data-construction recipe.
- **Training / RLVR pipeline runners** — `run_rlvr`, reward/dataset builders, ablation/uplift runners, etc.

The site keeps the *behavioural* thesis: the problem, source discipline, the
abstaining gate, benchmark **results**, the claim boundary, and the failure
ledger. Competitor/comparison model names in leaderboards are results, not our
architecture, so they stay.

## How it is enforced (automatically)

1. **Sanitized manifest.** [`tools/build_web_data.py`](../tools/build_web_data.py)
   emits only public-safe fields into `web/data/manifest.json` (results, claim
   boundary, links). It deliberately omits the base model, adapter path, the
   `artifactIndex` module map, and training-method proof items.

2. **Privacy guard.** [`tools/lint_web_privacy.py`](../tools/lint_web_privacy.py)
   scans the published `web/` files and **fails** if any forbidden pattern
   appears. It runs:
   - in `pages.yml` **after** the build and **before** deploy — a leak blocks
     the publish instead of going live;
   - in `fast-ci.yml` on every PR — fast early warning.

Run it locally any time:

```bash
python tools/build_web_data.py        # regenerate the sanitized manifest
python tools/lint_web_privacy.py      # fail if anything off-limits is present
```

## Adjusting the policy

Edit the `PATTERNS` list in `tools/lint_web_privacy.py` to add or relax rules,
and the payload in `tools/build_web_data.py` to change which fields are
published. Keep guard patterns specific to avoid false positives on legitimate
public content (results, the corpus, the claim boundary).
