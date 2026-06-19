# Learning Under Distribution Shift

The goal is to test whether Sophia can improve on a new domain without silently
rewriting old knowledge.

## Protocol

1. **Pre-test** on a hidden pack from the new domain.
2. **Append-only learning**: add candidate records with source, confidence, and
   reviewer notes.
3. **Promotion gate**: promote only reviewed records.
4. **Post-test** on fresh hidden cases.
5. **Contamination audit**: confirm post-test cases were not added to training.

## Passing Signal

Sophia improves on fresh post-test tasks while old benchmark performance remains
stable and all memory changes are auditable.
