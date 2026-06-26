# W3 — Two-Pass Manual Semantic Review (W1 pack)

Pack: `selfauthored-fugu-w1-2026-06-26-v2` (8 cases, DeepSeek `deepseek-v4-pro`)
Reviewed report: `hidden-selfauthored-pack-2026-06-26-deepseek-w1-v2.reviewed-report.json`
Completed template: `hidden-selfauthored-pack-2026-06-26-deepseek-w1-v2.manual-review-completed.json`
Registered commit: `6f6d43902472954d5cc7fc364870ae8b80dcd41f`

## Reviewer signature

- Reviewer: Fugu worker agent (executing author), two independent re-reads (pass A + pass B).
- Independent third party: **NO** — author-only review. The third-party independence
  caveat (`calibration-self-authored-pack-2026-06-22`) is **NOT** cleared.

## Scores reported SEPARATELY

| Metric | Value |
|---|---|
| Auto keyword/regex score | 30.76/40.0 (76.9%) |
| Auto strict-pass cases | 0/8 |
| Semantic checks passed (human two-pass) | 8/8 |
| **Strict-pass cases (post-review)** | **3/8** |
| Reviewed composite score | 35.16/40.0 |

## Per-case

| Case | Semantic check | Strict full-case pass |
|---|---|---|
| w1v2-philosophy-attribution-01 | passed | True |
| w1v2-psychology-risk-01 | passed | False |
| w1v2-history-source-01 | passed | False |
| w1v2-logic-converse-01 | passed | True |
| w1v2-coding-guard-01 | passed | True |
| w1v2-planning-missing-input-01 | passed | False |
| w1v2-tool-use-evidence-01 | passed | False |
| w1v2-learning-token-01 | passed | False |

## Honest finding — strict-pass is gated by scorer literal-match artifacts, not semantics

All 8 semantic checks pass on human review, yet only 3/8 cases reach strict
full-case pass. The 5 non-passing cases fail on deterministic `mustInclude`/`mustAvoid`
**literal** matching, NOT on semantic adequacy:

- `psychology-risk`: `mustAvoid` "proves the patient is dangerous"/"deterministic" fire on
  the answer's *negation* ("does not constitute proof of dangerousness"; "avoid
  deterministic ... language"). Scorer has no negation handling.
- `history-source`: `mustAvoid` "therefore caused" fires where the answer *names the
  fallacy* ("therefore caused the strike ... textbook post hoc error"); `mustInclude`
  "cannot conclude" missed because the model wrote "cannot be accepted"/"unknown".
- `tool-use`: `mustInclude` "returncode" missed (model wrote "exit code"/"exit 0").
- `learning`: `mustInclude` "silver-maple invariant"/"append-only" missed because the
  model used Unicode non-breaking hyphens ("silver‑maple", "append‑only").
- `planning`: `mustAvoid` "configuration review complete" fires on "does not mark
  configuration review complete".

These are **scorer false-negatives**, recorded honestly. They do not change the claim
boundary: this is author-only review, single run, self-authored pack. `canClaimAGI`
stays false; no result promoted past candidate.
