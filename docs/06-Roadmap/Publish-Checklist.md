# Publish & external-verification checklist

The concrete steps to get the repo in front of real humans for independent verification.
Ordered by leverage / effort.

## 1. Make it public + demoable (zero cost)
- [ ] Flip the GitHub repo **public** (CI badge, `GOOD_FIRST_ISSUES.md`, `RESULTS.md` are review-ready).
- [ ] Deploy the **Hugging Face Space** (`huggingface/space/`) — one-click interactive demo of the gate.
- [ ] Publish the **dataset card** (`huggingface/DATASET_CARD.md`) as a HF Dataset.

## 2. Strengthen the judging (needs an inference key — OpenRouter recommended)
- [ ] Set `OPENROUTER_API_KEY`; re-run the multi-family corroboration across 3–4 distinct
      families (`tools/run_calibration_judge.py … --judge openrouter:…`). Diversity of
      families is exactly what the no-overclaim gate rewards.

## 3. Human verification (closes the independence gap)
- [ ] Commission a **third-party-authored** abstain/hidden pack (not the existing one).
- [ ] Recruit annotators (**Prolific / Surge AI / MTurk**) for a 2-pass semantic review;
      report human-vs-scorer κ. Closes `calibration-self-authored-pack` + `hidden-manual-review-not-complete`.

## 4. Formal credibility
- [ ] Mirror `agi-proof/PRE-REGISTRATION.md` to **OSF** for a timestamp/DOI.
- [ ] Post `docs/11-Platform/Methodology.md` as an **arXiv** preprint; link code on **Papers With Code**.
- [ ] Archive a release on **Zenodo** for a citable DOI.

## 5. Adversarial eyeballs
- [ ] **Show HN / r/MachineLearning / r/LocalLLaMA** for general critique.
- [ ] **LessWrong / AlignmentForum** for the no-overclaim / AGI-claims-discipline angle.
- [ ] **huntr / HackerOne** to stress-test the MCP firewall + fail-closed gate.

## What NOT to publish
- Hidden-eval prompts (only aggregates) — see `SECURITY.md`.
- Any API key (all keys used in development were env-only; rotate after use).
