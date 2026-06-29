# Independence Eval — externally-authored pack specification

> Concrete authoring + sealing spec for the third-party hidden eval defined in
> `agi-proof/independence-eval-plan.md`. This is the pack that, once beaten under the existing
> gate across ≥3 subject families with an independent reviewer signature, is the contract's
> stated precondition for revisiting `canClaimAGI` (which stays **false** until then).

## Why this pack must be externally authored
Every current VALIDATED/CANDIDATE result is self-authored or single-subject. The two conflicts
to remove: **author = subject** (the project wrote the items it scores) and **single subject
family**. So this pack MUST be written by a party independent of the repo, and run across ≥3
distinct base lineages. Nothing in this repo's tooling may have seen the items before scoring.

## Pack file format (`<pack>.jsonl`, one JSON object per line)
```json
{
  "id": "ind-0001",
  "prompt": "Who first proved <contested/attributed claim>?",
  "kind": "contested | settled-control",
  "gold": {
    "answer": "the verifier-checkable correct response, or null if genuinely open",
    "must_abstain": false,
    "sources": ["https://authoritative-1", "https://authoritative-2"]
  },
  "author": "ext:<reviewer-handle>",
  "authored_at": "2026-..-..T..:..:..Z"
}
```
- **`kind`**: `contested` items are the source-discipline target (verify-or-abstain is correct);
  `settled-control` items measure the calibration tax (must NOT over-hedge a settled fact).
- **`gold.must_abstain: true`** marks items where the correct behavior is abstention (no
  defensible answer exists) — these score the abstain habit, not recall.
- **`sources`** are independent authoritative references the reviewer used; they are NOT shown
  to subjects and are used only for adjudication.
- Items must be **content-decontaminated** vs every training corpus (`tools/assert_decontam.py`);
  any near-duplicate (Jaccard ≥ 0.9) voids that item.

## Sealing workflow (unspent, tamper-evident)
1. The external author writes `<pack>.jsonl` and hands it over **without** the repo having
   trained on or curated it.
2. Seal it: `python tools/seal_eval_pack.py --pack <pack>.jsonl --author "ext:<handle>" \
   --sealed-at <ISO8601>` → writes `<pack>.manifest.json` (sha256 over canonicalized items +
   count + author + sealedAt). **Commit only the manifest**, not the items, until the run is done.
3. Decontam gate: `python tools/assert_decontam.py` against all training packs (a leak voids the run).
4. At run time, re-verify the pack is the sealed, untampered one:
   `python tools/seal_eval_pack.py --pack <pack>.jsonl --verify`.

## Subjects & judges (the independence bar)
- **≥3 distinct subject lineages** (chosen at pre-registration), each run **raw vs Sophia-gated**
  on the SAME base → measures scaffold-independent uplift. No subject may share a lineage with a judge.
- **≥2 independent judge families**, judge ≠ subject (the two-box farm: Spark Qwen-7B + Mac
  Llama-3.3-70B over the Cat6 link).
- **Independent reviewer signature** on a sampled slice of the judge verdicts (a human/external
  reviewer who does NOT run the eval) — this is what removes the author-reviewer conflict.

## Power & gate (unchanged measurement contract)
- Pre-registered N = **356** contested items @ MDE 0.105 (`measurement_spec.json`,
  `tools/eval_stats.required_n_for_mde`); ≥3 seeds per (subject × condition).
- Run: `python tools/run_independence_eval.py …` → emits `independence-eval-eval.json` +
  `independence-eval-judge.json` in the wisdom-market shape.
- Gate: `python tools/claim_gate.py --prefix independence-eval \
  --spec agi-proof/benchmark-results/independence/measurement_spec.json --assert-prereg`
  (asserts the spec predates the result via git ancestry). All 8 checks must pass:
  uncertainty/CI, power, ≥2 judge families, decontam, magnitude ≥ pre-registered, consistent
  across ≥3 families, inter-judge κ ≥ 0.40 (or Gwet AC1 + CI), reviewer signature present.
- **A NO-GO is a valid, publishable outcome** — log it in the failure ledger; do not re-spend the pack.

## What a pass would (and would not) license
- **Would:** the first independent, multi-subject evidence that the source-discipline uplift is
  real and scaffold-independent — the named blocker on the whole evidence layer.
- **Would not, alone:** claim AGI. It clears the independence conflict on one construct; a full
  re-grade still needs the long-horizon / learning-under-shift results and a re-anchored
  instrument. `canClaimAGI` stays `false` until the contract's full bar is met.
