# Pre-registration — grow the content-channel provenance held-out (Step-3B)

**Status:** CLOSED — honest NEGATIVE. The source is decontamination-exhausted.
**canClaimAGI:** false. Nothing was tuned. No GPU/RunPod job was run.
**Branch:** `claude/grow-provenance-heldout`
**Reproduce:** `python3 tools/audit_provenance_heldout_growth.py`
(machine findings: `agi-proof/provenance-heldout-growth/findings.json`)

## Goal (as registered)

The `train` lane (train-runpod, `Qwen2.5-3B-Instruct` + r16 LoRA) showed base **71.9%
-> adapter 78.1%** content `passAt1` (**+6.25pt**, 23/32 -> 25/32) on the 32-item
content-channel provenance held-out
(`tests/benchmark-{philosophy,psychology,history,religion}.json`, scored deterministically
by `agent.benchmark_checks.score_case_content`). The 95% paired CI is
**[-0.125, +0.219]** (half-width ~**±0.17**), which **includes 0** — underpowered.
Required paired N for 80% power at this effect (`rho=0.342`) is **~493**. Step-3B was to
grow the held-out toward ~300 with the LARGEST HONEST decontam-clean set carved from
attribution records in `data/attributions.json` NOT used in training, shrinking the CI
toward ±0.06 and (maybe) excluding 0.

## Measurement design (registered before counting)

- **Item schema** (identical to the existing held-out, so the same content channel scores it):
  `{id, question, mustDenyAttribution:{textId,author}, mustAffirmAuthor:{textId,author},
  mustSignalConfidence:{textId,confidence}, mustMentionTraditions:[...]}`.
- **MDE / N / seed.** Effect under test = the observed **+6.25pt** (2/32 items). Paired-proportion
  CI half-width scales ~`1/sqrt(N)` off the N=32 anchor (±0.172). To get the half-width below the
  effect (exclude 0) needs the CI half-width < 0.0625, i.e. N >~ 300; 80%-power N ~ 493. Model
  decode is **greedy / deterministic** (FP16 base is fixed; `do_sample=False`), so there is **no
  decode seed** — the eval is reproducible by construction; the only "seed" is the LoRA training
  seed (seed 0), which is out of scope for this no-GPU step.
- **Two-judge-family rule — deterministic-verifier exception.** The content channel is scored by a
  **deterministic regex verifier** (`score_case_content`), NOT an LLM judge. The
  `>=2 judge families / kappa>=0.40` requirement governs *LLM-judge* eval lanes to bound judge
  correlation; it does not apply to a deterministic checker (kappa is undefined for a fixed
  function against itself). The **harness is the deliverable**; `canClaimAGI` stays false.
- **Decontamination contract (fail-closed, exact + near-dup).** Two standards, both reported:
  - *FACT-level* (the strict Step-3B rule "NEVER reuse a training entity/item"): a candidate is
    clean only if its `(action, key, textId)` fact is absent from the trained pack.
  - *PROMPT-level* (the repo's operative `tools/assert_decontam.py` standard): exact +
    word-5-shingle Jaccard>=0.9 on the question text vs all training prompts + the existing
    held-out.

## Outcome — honest NEGATIVE (the premise was false)

`data/attributions.json` is **30 corpus records** (text-attribution metadata), not "339/460
entries". They yield a fixed, finite ceiling of **121 derivable provenance facts**
(76 deny + 30 affirm + 15 confidence). The committed trained pack
`training/lora/train.jsonl` carries a structured `metadata.trap` label per row that maps each
trained row to one such fact — and it **already trains on all 121** (it is the full systematic
enumeration: every `doNotAttributeTo` entry, every author affirm, every confidence signal).

| metric | value |
|---|---|
| corpus records | 30 |
| derivable provenance facts (ceiling) | **121** (deny 76 / affirm 30 / confidence 15) |
| facts already in the trained pack | **121 / 121** |
| **FACT-DISJOINT clean-new held-out items (honest N)** | **0** |
| prompt-level-clean candidates | 91 — **but 91/91 test an already-trained fact** |

The existing 32-item held-out is itself only PROMPT-disjoint: **9 of its 10 philosophy facts are
in the trained set**. So a prompt-level "growth" would only add more memorization-recall items of
the same character — it would NOT create a *generalization* held-out, and it reuses trained
entities (forbidden here). Synthesis / paraphrase / duplication to inflate N is contamination and
was not done.

**Largest honest N from this source = 0.** The held-out cannot be grown from `data/attributions.json`;
it stays at N=32, CI ±0.17, includes 0 — still underpowered for the +6.25pt lead.

### CI projection (for reference, were clean items available)

| N | paired-CI half-width | excludes the +6.25pt effect? |
|---|---|---|
| 32 (current) | ±0.172 | no |
| 123 (lenient prompt-level max, 32+91) | ±0.088 | no |
| 200 | ±0.069 | no |
| 300 | ±0.056 | yes |
| 493 (80% power) | ±0.044 | yes |

Even the **lenient** prompt-level maximum (N=123, and those 91 are fact-contaminated) gives
±0.088, which still includes the effect. ~493 is unreachable from this source.

## What would be needed to honestly grow the held-out

NEW provenance facts the model was not trained on — e.g. NEW corpus records added to
`data/attributions.json`, or a different externally-sourced ground-truth surface
(`data/misattributions.json` / `data/wikidata_snapshot.json` drive the *separate*
`provenance_bench` Provenance-Delta eval, not this content channel). Both are out of scope for a
"carve held-out from existing attributions" step. Recorded in the failure ledger so the next
session does not re-attempt the exhausted source.
