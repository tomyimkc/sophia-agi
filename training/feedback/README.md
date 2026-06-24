# Gate-feedback → training queue (C4)

This directory is the **return path** that makes Sophia's training loop *continual*: gate
MISSES (a hallucination the LLM judge caught but the deterministic gate let through) become
candidate `doNotAttributeTo` records, a human reviews them, and only **promoted** candidates
become training rows for the next pack.

The `*.jsonl` working files here are generated and git-ignored; this README is the durable
record of the workflow.

## Why it's non-circular

The runtime gate never trains on its own unreviewed output:

1. **Separate file.** Candidates land in `pending_candidates.jsonl`, never in the frozen
   runtime records (`agent`/`provenance_bench` gate records are untouched).
2. **Default deny.** Every candidate starts `promoted: false`. `build-sft` emits **nothing**
   until a human flips it.
3. **Same decontamination.** Promoted rows go through `build_local_sophia_dataset.py`'s
   contamination guard like any source, so they cannot leak eval/holdout prompts.

## Workflow

```bash
# 1. Mine misses from a run's case results into the pending queue (deduped)
python tools/feedback_to_training.py mine <run_case_results.jsonl>

# 2. Review, then promote specific candidates (the human gate)
python tools/feedback_to_training.py status
python tools/feedback_to_training.py approve <rid> --reviewer me --note "verified: ..."

# 3. Build SFT rows from ONLY promoted candidates
python tools/feedback_to_training.py build-sft        # -> sft_from_feedback.jsonl

# 4. Ingest into the next pack (decontaminated automatically)
python tools/build_local_sophia_dataset.py

# 5. Validate the gain with the learning-under-shift protocol (needs a model backend):
#    pre-test -> append-only learning -> post-test -> old-benchmark stability -> contamination audit
python tools/run_learning_shift.py <spec.json> --backend adapter
```

## Files (generated)

- `pending_candidates.jsonl` — flat, reviewable candidates (`promoted: false` until reviewed).
- `sft_from_feedback.jsonl` — promoted candidates as source-discipline SFT rows; an optional
  source in `tools/build_local_sophia_dataset.py`.
- `promoted_records.jsonl` — promoted candidates as `{rid: {...}}` gate records, for a future
  gate-adoption step (kept separate from the live frozen records on purpose).

Not an AGI claim: this trains the refuse-unsupported-attribution **habit** from reviewed
evidence; external gates still enforce correctness at runtime.
