# Hidden Reviewer Packs

Hidden packs are reviewer-written tasks that Sophia cannot see before the run.
They must not be added to `training/examples/` or prompt context before scoring.

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
