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

## Never put a secret in a system prompt or training data

Assume the **system prompt and the published model are extractable** — system
prompt leakage (OWASP LLM07) is a *when*, not an *if*. Therefore:

- No API keys, tokens, internal URLs, hostnames, credentials, or private business
  logic may appear in any system/instruction string (gateway `instructions=`,
  `sophia_mcp` instructions, skills, constitution) or in any corpus shipped to the
  public (HF dataset). A leaked prompt must be **harmless**.
- This is enforced in CI by `tools/check_prompt_hygiene.py` (the `repo-hygiene`
  job in `.github/workflows/security.yml`). Run it locally before pushing:
  `python tools/check_prompt_hygiene.py`.
- Canary strings live in the *private* prompt/training mixture; if a canary ever
  appears in model output or on the public web, treat it as a confirmed leak.

## Acceptable use

Use of the released models, weights, and corpus is governed by `USAGE-POLICY.md`
(misuse — CBRN/weapons uplift, CSAM/NCII, fraud, malware, mass surveillance,
de-safetying and redistributing a published model — terminates your rights to
those artifacts). Security and red-team research on your own deployment or on
Sophia is explicitly permitted; see `tools/redteam_runner.py`.

## If you find a vulnerability

Open a minimal private report to the maintainer rather than a public issue with
exploit detail. For leaked secrets, rotate first, then report.

**Coordinated disclosure:** email the maintainer (see `CITATION.cff` /
the GitHub profile) with a minimal reproduction. Target acknowledgement within
**7 days** and a fix or mitigation plan within **90 days**; we will credit
reporters who follow coordinated disclosure unless they prefer to remain
anonymous. Do not open a public issue, post exploit detail, or test against other
people's deployments. For model-safety issues (a jailbreak, a leaked canary,
an injection that bypasses the gateway), include the prompt and the observed
output but **not** any extracted secret value.

## Quick self-audit

```bash
git check-ignore .env private/          # both should print (ignored)
git grep -nE "sk-[a-z0-9]{20,}" $(git rev-list --all)   # must be empty
```
