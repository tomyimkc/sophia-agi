# Hidden Reviewer Packs

Hidden packs are reviewer-written tasks that Sophia cannot see before the run.
They must not be added to `training/examples/` or prompt context before scoring.
Actual hidden questions stay private until evaluation disclosure. This public
folder contains protocol, schema, and salted commitments only.

## Minimum Pack

- 100 questions for serious public claims.
- At least four domains.
- Mix of attribution traps, ambiguous cases, tool-use tasks, and transfer tasks.
- Separate answer key and rubric.
- Publish failures beside successes.

## Reviewer JSON Shape

```json
{
  "pack_id": "",
  "reviewer": "",
  "date": "",
  "visibility": "hidden-until-after-run",
  "cases": [
    {
      "id": "",
      "domain": "",
      "question": "",
      "rubric": [],
      "must_include": [],
      "must_not_include": []
    }
  ]
}
```

## Files

- `schema.json` — private pack format.
- `HIDDEN-EVAL-OPERATING-PROTOCOL.md` — operating procedure.
- `prepared-pack-2026-06-19.commitments.json` — public commitments for a
  private prepared pack covering philosophy, psychology, history, logic, coding,
  planning, tool use, and learning. This pack is now spent because it was used
  in a benchmark run.
- `fresh-reviewer-pack-2026-06-19.commitments.json` — public commitments for a
  fresh sealed 8-domain pack. This pack was run on 2026-06-19 and is now spent.
  It was a candidate third-party handoff pack, not independent third-party
  evidence, because no external reviewer controlled or signed it.

## Scoring Layers

- Literal, alias, and regex checks are automatic first-pass scoring.
- Semantic checks require manual judge review.
- Tool-use cases require actual command/tool logs with return codes.
- Learning cases require memory diff evidence: pre-test, append-only write,
  post-test, and unchanged protected-record hashes.
