# Third-party verifiable-domain intake protocol

**Status:** SCAFFOLD READY / loop-closed N = 0 (honest, pre-registered — not a failure).

This is the *intake* side of the third-party held-out effort (see [PROTOCOL.md](PROTOCOL.md)
for the *sealing* side). The verifier-as-reward loop is starved of **externally-authored
verifiable domains**: the committed [`third-party.commitments.json`](third-party.commitments.json)
is EMPTY (`caseCount: 0`), so the count of external domains on which the loop actually closes
is **N = 0**. This protocol defines what a valid external manifest looks like, the
decontamination requirement, the two-counter rule, and the exact GO condition.

Tool: [`tools/third_party_intake.py`](../../tools/third_party_intake.py).
Pre-registration: [`intake_measurement_spec.json`](intake_measurement_spec.json).

`canClaimAGI = false` regardless of outcome.

## 1. What a valid external domain manifest looks like

A manifest is JSONL, one item per line. Each item MUST carry:

| field           | requirement |
|-----------------|-------------|
| `itemId`        | non-empty, stable id (kebab or SCREAMING is fine) |
| `domain`        | the verifiable domain, e.g. `math`, `code`, `legal-citation` |
| `externalSource`| provenance of the item (mathlib slice, a GitHub CI suite, a legal-citation corpus). MUST be a party with **no access** to Sophia's train/eval prompts. |
| `author`        | who authored it (must be **not** the party that writes Sophia training data) |
| `prompt`        | the task text |
| `scoring`       | a **machine-checkable** oracle (see §3) |
| `decontamProof` | the author's decontamination warrant + method (see §2) |

Items whose `domain` is a **protected domain** (religion, history) are out of scope for this
intake — verifiable-reward domains are math / code / citation-checking, never a
same-lineage/impersonation claim in a protected domain.

The repo ships a clearly-synthetic illustrative manifest at
[`eval/third_party_intake/sample_manifest.jsonl`](../../eval/third_party_intake/sample_manifest.jsonl).
It is **not** a real external corpus; it exists only to exercise the pipeline (it contains one
deliberately-contaminated item and one non-machine-checkable item so the reject paths are
observable).

## 2. Decontamination requirement (HARD gate)

Every item's `prompt` is checked against the **committed eval prompt surface** using the SAME
primitives as [`tools/assert_decontam.py`](../../tools/assert_decontam.py) (`normalize`,
`_shingles`, `_jaccard`), via a direct import — intake and CI therefore agree by construction.
Two layers:

1. **Exact / normalized overlap.** If the item's normalized prompt equals any eval prompt, it
   is **refused**.
2. **Content k-shingle near-duplicate.** If the item's prompt has Jaccard ≥ threshold (default
   0.9, k = 5) against any eval prompt, it is **refused**.

The gate is **fail-closed**: an item that fails decontam **never** enters the admitted set. A
gate that always passes would be worse than no gate. (The intake's eval baseline excludes the
manifest file itself so a manifest that lives under `eval/` is not spuriously matched against
its own prompts — but a prompt that ALSO appears in another eval file is still a true collision
and is still refused.)

An item is also refused at the **validity** stage before decontam if it is missing a
machine-checkable oracle, an `itemId`, a `decontamProof`, or has an empty prompt.

## 3. Machine-checkable oracle requirement

Only objective, ungameable oracles are admitted to the headline intake:

- `scoring.kind = "sympy"` with a non-empty `scoring.gold` (sympy canonical equivalence), or
- `scoring.kind = "exec"` with a non-empty `scoring.test` (hidden asserts checked by execution).

`llm-judge` and any other non-deterministic oracle are **not** admitted here — a judge is a
separate, labelled family, not part of the verifiable-reward headline.

## 4. The two-counter rule (never conflate)

The pipeline keeps **two strictly separate** counters:

- **`admittedCount`** — items that passed validity **and** decontam and entered the pool. This
  measures **intake capacity** only.
- **`loopClosedCount`** — items for which the verifier **admitted** the item **AND** a held-out
  gain was **actually measured** on it (verifier-admitted **AND** held-out gain).

Admitting an item does **NOT** close the loop. `loopClosedCount` is never derived from
`admittedCount`; the invariant `loopClosedCount ≤ admittedCount` is asserted, and
`loopClosedCount` stays **0** until real verifier + gain data exist. Reporting `admittedCount`
as if it were closed-loop N would be an overclaim.

## 5. GO condition

**GO requires real N ≥ 1**: at least one externally-authored verifiable domain that

1. passes `assert_decontam` (decontam-clean against the committed eval surface), **and**
2. is **admitted** by the verifier, **and**
3. yields a **measured held-out gain** clearing the no-overclaim bar (≥ 3 seeds, 95% CI
   excludes 0, per the run's measurement spec).

Until then `loopClosedCount = 0`. That zero is the **pre-registered honest starting state**
(no external corpus is committed and there is no in-session network to fetch one), **not** a
NEGATIVE result and **not** a pipeline failure. `go = false`, `canClaimAGI = false`.
