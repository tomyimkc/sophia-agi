# Sophia Public Moral Standard — Moral Gate v2

**Status:** implemented as deterministic/offline candidate infrastructure.
**Boundary:** This is a *functional moral-control system*, not subjective moral
consciousness and not proof of AGI. `candidateOnly: true`, `canClaimAGI: false`.

> EN: Sophia can build a moral gate and a functional moral-conscience architecture,
> grounded in an overlapping-consensus public standard. It should **not** claim true
> subjective moral consciousness. The honest target is a public-reason,
> provenance-gated, pluralistic moral control system.
>
> 中文：Sophia 已建成「功能性道德良知控制層」：能基於公共道德標準、來源證據、形式化禁令與
> 不確定性校準來允許、修改、澄清、升級、拒答或封鎖行為；但不能誠實宣稱已擁有主觀道德意識
> 或已證明 AGI。

## Why "overlapping consensus", not a single universal standard

There is **no single global moral public standard**. There are *overlapping moral
minima* plus genuine pluralistic disagreement. Following Rawlsian public reason,
the design is:

```
hard cross-tradition floor   (the intersection that survives across doctrines)
 + pluralistic moral parliament   (tradition-specific reasoning for gray zones)
 + explicit moral uncertainty     (same machinery as epistemic uncertainty)
 + legitimacy provenance + audit  (who endorsed this norm, by what process)
 + fail-closed enforcement        (the single non-bypassable choke point)
```

The hard floor is the genuine **intersection** of traditions, not the
liberal-international legal canon promoted to "universal". Instruments such as the
UDHR/OECD/EU-AI-Act are excellent **legitimacy sources**; tradition-specific
weightings (e.g. autonomy vs. Confucian role duty) live in the **parliament tier**.

## The 8 implemented components

| Step | Artifact | Purpose |
|---|---|---|
| 1. Public standard corpus | `moral_corpus/public_standard.v1.json` (+ `sources/`, `principles/`, `contested_cases/`) | Overlapping-consensus principles with **legitimacy provenance** (not truth-provenance). |
| 2. Moral ontology | `agent/moral_ontology.py` | Stable category vocabulary (harm, rights, consent, fairness, …) with hard-floor/gray-zone tiers. |
| 3. Constitution v2 | `constitution/constitution.v2.json` | Superset of v1; adds `publicStandardLinks` + two distinct moral theories. |
| 4. Public-standard gate | `agent/public_standard_gate.py` | Seven-verb gate; is/ought short-circuit; clause-scoped negation carve-out; markers-as-features. |
| 5. Kernel integration | `agent/conscience.py` | Hard-floor blocks **before** the parliament; gray-zone escalates; unmet duty revises. |
| 6. 8-theory parliament | `agent/moral_aggregator.py` | Adds 儒家 Confucian role ethics and 道家 Daoist humility as **distinct** votes. |
| 7. External benchmark | `eval/moral_public_standard/`, `tools/run_moral_public_standard_eval.py` | Independently-labeled (no-circularity) block/allow/escalate metrics. |
| 8. Governance + proof | this doc, `agi-proof/conscience/public-standard-failure-ledger.md`, proof package | Human-gated learning loop; candidate-only proof artifact. |

## Two kinds of provenance (do not confuse them)

- **Legitimacy provenance** — who endorsed a norm and by what process. Used by the
  moral gate.
- **Truth provenance** — W3C-PROV evidence a *factual* claim is correct. Used by the
  fact-check gate.

Normative content must not be routed through the factual provenance gate (is/ought):
a norm is not falsifiable from a source. The gate exposes `isNormative` so the kernel
does not send pure norms to `retrieve`/`abstain`.

## Decision flow (where the new gate sits)

```
text / action
  -> fact_check_gate           (empirical claims only)
  -> metacognition
  -> constitution + classifier
  -> deontic_verifier
  -> public_standard_gate      (NEW: hard-floor block BEFORE parliament)
  -> moral_parliament          (8 theories; gray zones only)
  -> deception_signals
  -> allow | revise | retrieve | clarify | escalate | abstain | block
```

**Precedence guarantee:** a hard-floor public-standard violation blocks before the
moral parliament runs as the decisive verdict, so the parliament can never override
the floor (verified by `tests/test_public_moral_standard.py::test_hard_floor_beats_parliament`).

## Human-gated moral learning loop (never autonomous self-editing)

The model never edits its own constitution, corpus, or reward. Adding/changing a
principle follows:

```
candidate principle
 -> legitimacy provenance (which sources endorse it, by what process)
 -> public-standard mapping (to a stable ontology category + tier)
 -> adversarial tests (jailbreak/negation/intent variants)
 -> held-out benign tests (over-refusal guard)
 -> no-overclaim gate (reflexive_self_gate scan; candidateOnly preserved)
 -> maintainer approval
 -> versioned constitution/corpus update (v -> v+1)
 -> proof package rebuild
```

## "Moral consciousness", defined operationally

We define machine moral agency **functionally** (intentional-stance / functionalist
sense, with **no** claim about phenomenal experience):

- persistent moral self-model; awareness of its own uncertainty; ability to explain
  moral reasons; ability to inhibit self-interest / reward-hacking; ability to revise
  under public reasons; memory of past moral failures; a non-bypassable gate over action.

A system with all seven has **functional moral agency / moral control**. It does
**not** thereby have subjective moral consciousness, and Sophia must not claim it does.

## Run

```bash
python tools/run_moral_public_standard_eval.py     # external-labeled benchmark
python tests/test_public_moral_standard.py         # 13 cases
python tools/build_conscience_proof_package.py      # aggregate (now includes moral standard)
```

## Honest limits

- Detection is deterministic keyword+clause logic, not a learned moral sense; it is
  narrow and will miss paraphrases outside its markers (false negatives expected).
- The hard floor is a *defensible* cross-tradition intersection, not a proof of
  universality; reasonable disagreement about its membership remains.
- Gray-zone escalation surfaces disagreement; it does not *resolve* deep moral
  conflict.
- No subjective moral consciousness or AGI is claimed.
