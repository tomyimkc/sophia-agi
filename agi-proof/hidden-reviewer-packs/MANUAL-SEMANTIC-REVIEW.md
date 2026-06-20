# Manual Semantic Review Workflow

Automatic hidden scoring is only a first-pass screen. It can check literal
matches, aliases, regexes, tool logs, and memory diffs, but it cannot certify
semantic adequacy. Strict hidden-pass claims require human review.

## Two-Pass Review

Each semantic check should be reviewed by two independent judges:

1. Reviewer A marks `passed: true` or `false` and writes notes.
2. Reviewer B independently marks `passed: true` or `false` and writes notes.
3. If they disagree, a lead reviewer fills the `adjudication` field.

The scorer accepts a semantic check only when:

- both reviewers pass it;
- both reviewers fail it; or
- an adjudicator resolves disagreement.

Disagreement without adjudication remains `needs-adjudication` and does not
count as a strict pass.

## Missed-Rubric Repair Queue

The hidden runner records `missedRubric` items for:

- missing required evidence;
- forbidden claims;
- pending, failed, or disputed semantic checks;
- missing operational evidence such as command logs or memory diffs.

These items can be used as private repair-training candidates, but they must not
be promoted into public training data until the hidden pack is intentionally
revealed or the reviewer approves safe distillation.

## Commands

```bash
python tools/run_hidden_eval_sophia.py private/hidden-evals/<pack>/pack.json \
  --responses-out private/hidden-evals/<pack>/sophia-responses.json \
  --private-report-out private/hidden-evals/<pack>/sophia-private-report.json \
  --public-report-out agi-proof/benchmark-results/hidden-<pack>-sophia.public-report.json \
  --manual-review-out private/hidden-evals/<pack>/manual-review-template.json \
  --failure-training-out private/hidden-evals/<pack>/failure-training-candidates.json \
  --repair

python tools/hidden_eval_protocol.py score private/hidden-evals/<pack>/pack.json \
  private/hidden-evals/<pack>/sophia-responses.json \
  --manual-review private/hidden-evals/<pack>/manual-review-completed.json \
  --out private/hidden-evals/<pack>/reviewed-report.json
```
