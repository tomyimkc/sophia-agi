# Security & the public/private boundary

Sophia AGI is a public, transparency-first project: code, methodology, and
**audited** result *aggregates* are open. But "open" is not "everything." Three
categories must **never** be published, because publishing them either leaks a
secret or *destroys the validity of the very results we publish*.

## Never make public

1. **Secrets** — API keys, tokens, `.env` files. These are git-ignored (`.env`,
   `.env.*`, `*.key`, `*.pem`). Never paste a key into an issue, PR, commit, or
   chat. If a key is exposed anywhere, **rotate it immediately** — exposure, not
   misuse, is the trigger.
2. **Hidden-evaluation prompts/answers** — everything under `private/`
   (`private/hidden-evals/`, …) is git-ignored on purpose. A held-out test set is
   only meaningful while it is secret; publishing the prompts lets any model be
   tuned to them and makes every future score meaningless. Publish only
   **aggregate scores + hashed commitments** of the held-out set (see
   `tools/hidden_eval_commitments.py`), never the items.
3. **Un-validated numbers presented as headlines** — see the gate below.

## What IS public

- All source, the benchmark **methodology**, and the non-circularity contract.
- Offline CI test results (deterministic, no model calls).
- Benchmark **aggregates** (rates, confidence intervals, judge agreement) — but
  only with the honesty labels below.

## The no-overclaim gate (before any number is a "result")

A number may be published as **VALIDATED** only if it clears all of:

- **≥2 independent judges** in consensus (`--judges a,b,c`), judges ≠ the model
  under test;
- judges from **≥2 distinct provider families**;
- **inter-judge agreement reported**, with chance-corrected **Cohen's κ ≥ 0.40**
  (a single judge was ~2× off in our own audit — see
  `docs/11-Platform/Provenance-Delta.md`);
- **≥3 runs** with a **confidence interval that excludes zero**;
- a label of `illustrative` vs `validated` on every figure.

Anything not meeting this is published as **illustrative** only, clearly marked.
This rule is enforced in `provenance_bench/aggregate.py` (`validated` flag) and
surfaced by `tools/build_results_page.py`.

## If you find a vulnerability

Open a minimal private report to the maintainer rather than a public issue with
exploit detail. For leaked secrets, rotate first, then report.

## Quick self-audit

```bash
git check-ignore .env private/          # both should print (ignored)
git grep -nE "sk-[a-z0-9]{20,}" $(git rev-list --all)   # must be empty
```
