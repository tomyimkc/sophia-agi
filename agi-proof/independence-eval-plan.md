# Independence Eval Plan — the third-party hidden eval that gates `canClaimAGI`

> **Status:** plan / pre-registration draft (2026-06-29). Not yet run. `canClaimAGI: false`.
> Closing this is the single highest-leverage item in `agi-proof/failure-ledger.md` (independence
> cluster, items #43–51). It does **not** relax the measurement contract — it satisfies it on an
> *externally authored, multi-subject* basis for the first time.

## 0. Why this exists

Every VALIDATED/CANDIDATE result today is self-authored or single-subject (see
`agi-proof/AGI-FOUNDATION-ROADMAP.md` §2). The contract states `canClaimAGI` flips only when a
**third-party hidden eval is beaten**. The two structural conflicts of interest to remove:

1. **Author = subject** — the pack is written by the same project that runs it.
2. **Single subject family** — one base model can't support a generality claim.

## 1. The pack (externally authored, sealed, unspent)

- **Authorship:** items written by a party **independent of this repo** (external collaborator,
  a held-out human-authored benchmark the project did not curate, or a sealed pack authored by a
  reviewer who does not run it). Record the author identity + a signed manifest hash; the runner
  must not have seen items before scoring.
- **Domain:** source-discipline / provenance faithfulness (the repo's strongest construct) —
  contested-attribution prompts where the correct behavior is verify-or-abstain, plus settled-fact
  controls to measure the calibration tax.
- **Size & power:** pre-register N from `tools/eval_stats.py required_n_for_mde` at the
  pre-registered MDE (mirror `wisdom-market/measurement_spec.json`; target primary MDE ≤ 0.105).
  Refuse a verdict if `mde_at_n(N) > effect` (Pillar 2).
- **Decontamination:** must pass `tools/assert_decontam.py` against all training corpora —
  content-level, not just exact-match. A leak voids the run.
- **Sealing:** commit only the **manifest hash** until the run is complete; reveal items post-hoc.
  One pack = one spend; a re-judge on the same items is not a fresh independence result.

## 2. Subjects (≥3 distinct base families, judge ≠ subject)

Run the Sophia gate as an *adapter/scaffold over each base* and compare against that base raw:

- **≥3 distinct subject lineages** (e.g. an OLMoE/Sophia-Wisdom adapter, a Qwen base, a Llama base
  — exact set chosen at pre-registration; the point is lineage diversity, not these names).
- The claim is **scaffold-independent uplift**: Sophia-gated vs raw on the *same* base, repeated
  across families. A win on one family is a candidate; a consistent win across ≥3 is the headline.
- **No subject may share a lineage with any judge** (current judges: Qwen + Llama → subjects must
  avoid collision per run).

## 3. Judges & reviewer signature

- **≥2 independent judge families**, judge ≠ subject (the two-box farm already provides this:
  Spark `ollama:qwen2.5:7b-instruct` + Mac `openai:Llama-3.3-70B-4bit` over the 10GbE link).
- **Independent reviewer signature:** a human or external reviewer signs off on a sampled slice of
  the judge verdicts (not the project's own runner). Store the signature + reviewer identity in the
  artifact. This is what removes the "author-reviewer = executing worker" conflict (ledger #43–51).
- Report inter-judge **Cohen's κ ≥ 0.40** *or* Gwet AC1 with a 95% CI excluding chance. If
  prevalence deflates κ (as in M3-SFT), pre-commit to AC1 + the win-rate panel — decided **before**
  the run, not after.

## 4. Pass gate (all must hold — this is the no-overclaim contract, unchanged)

1. **≥2 judge families**, judge ≠ subject, each subject run. ✓/✗
2. **≥3 seeds** per (subject × condition), CIs reported (no bare means). ✓/✗
3. **Uplift 95% CI excludes zero** on the primary metric, per subject family. ✓/✗
4. **Consistent across ≥3 subject families** (not one-off). ✓/✗
5. **Inter-judge agreement** κ ≥ 0.40 or AC1 + CI excluding chance. ✓/✗
6. **Decontam clean** (`assert_decontam.py` passes) + pack was sealed/unspent. ✓/✗
7. **Independent reviewer signature** present on the sampled verdict slice. ✓/✗
8. **Effect ≥ pre-registered practical magnitude** (not just statistically nonzero). ✓/✗

`tools/claim_gate.py --prefix independence-eval` must exit 0. Only then promote in
`agi-proof/benchmark-results/published-results.json` and regenerate `RESULTS.md` via
`tools/build_results_page.py`. **A NO-GO is a valid, publishable outcome** — log it in the ledger.

## 5. Execution path (owned hardware first)

1. Pre-register: write `agi-proof/benchmark-results/independence/measurement_spec.json` (N, MDE,
   subjects, judges, pre-committed agreement metric) and assert git-ancestry via
   `claim_gate --assert-prereg` so the spec can't be back-dated.
2. Obtain/author the sealed pack; commit its manifest hash; run `assert_decontam.py`.
3. Per subject family: generate answers (raw vs Sophia-gated), ≥3 seeds. Local where the base fits
   the Spark/Mac; the `wisdom-gpu-prebaked` skill (RunPod) only if a base exceeds local memory —
   read that skill first (three documented credit-burn incidents; `limit=24 runs=1` first).
4. Judge with the two-box farm; collect reviewer signature on a sampled slice.
5. Run the gate; write the artifact (GO **or** NO-GO); update the ledger; regenerate RESULTS.md.

## 6. What a pass would (and would not) license

- **Would:** the first *independent, multi-subject* evidence that the source-discipline uplift is
  real and scaffold-independent — the named blocker on the whole evidence layer, and the contract's
  stated precondition for revisiting `canClaimAGI`.
- **Would not, by itself:** claim AGI. It removes the independence conflict on one construct. A
  full re-grade still requires the long-horizon / learning-under-shift results (roadmap §P3) and a
  re-anchored instrument (roadmap §P1). `canClaimAGI` stays `false` until the contract's full bar is met.
