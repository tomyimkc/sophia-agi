# Semantic Grounding & Compositional Grammar — research note + benchmark plan

**Status:** proposal / candidate infrastructure. `candidateOnly: true`,
`level3Evidence: false`, `canClaimAGI: false`. This document proposes a *measurement
harness* and a hypothesis. It claims **no** capability. A NO-GO is a valid, publishable
outcome. Nothing here changes the AGI claim.

> **The question this answers (user's words):** *"How does AI know what a prompt means
> when I type in English? Right now it's transformer probability. Can we instead let AI
> match wordings to dictionary meaning — using the OKF / LLM-wiki to match definitions —
> and then train it to learn grammar so it composes meaning like a human?"*

---

## 1. The thesis, stated honestly

### 1.1 What a transformer actually does (and the real gap)

A transformer maps your prompt to meaning through the **distributional hypothesis**
("you shall know a word by the company it keeps", Firth 1957; Harris 1954): each token
becomes a vector whose position is learned from co-occurrence statistics over a huge
corpus, and "meaning" is the geometry of those vectors plus attention over them. It is
*probability over learned form*.

Two well-known critiques frame the user's intuition precisely:

- **Symbol grounding problem** (Harnad 1990): symbols defined only by other symbols are
  ungrounded — a dictionary that defines every word using other words it never grounds.
- **Form vs. meaning** (Bender & Koller 2020, *"Climbing towards NLU"*; the "octopus
  test"): a system trained on form alone learns *distribution*, not *communicative
  intent / reference*. It can be fluent and still not "mean" anything.

So the user's instinct — *anchor words to explicit definitions instead of pure
probability* — is the **grounding** move, and the second instinct — *learn grammar so
meaning composes from parts* — is the **compositionality** move:

- **Compositionality / systematicity** (Fodor & Pylyshyn 1988; Montague semantics;
  CCG, Steedman): the meaning of a whole is a *function of the meanings of its parts and
  the rules (grammar) that combine them*. Transformers are known to **fail systematic
  compositional generalization** on controlled tests — SCAN (Lake & Baroni 2018), COGS
  (Kim & Linzen 2020), CFQ (Keysers 2020). "Learn grammar" = learn the *combination
  rules*, not just the parts.

### 1.2 The honest correction to the idea

Wholesale *replacing* probability with dictionary lookup will not work, and it is worth
saying why up front so the experiment is designed correctly:

- **Polysemy / word-sense:** "bank", "spring", "seat" — a dictionary has many senses;
  picking the right one **needs context**, i.e. needs the distributional model. (This is
  classic Word-Sense Disambiguation: GlossBERT, Huang 2019.)
- **Coverage:** no dictionary covers every phrasing, idiom, or new term. Symbolic-only
  systems are brittle; that brittleness is why statistical NLP won in the 1990s.
- **Definitions are themselves text** (Harnad's circularity) — grounding has to bottom
  out somewhere (reference, provenance, or use), not in more glosses.

The productive, literature-supported framing is therefore **neuro-symbolic, hybrid**:
keep the transformer for surface/lexical understanding, and **add** a layer that (a)
links spans to explicit definitions, (b) **composes** them via a small grammar/logic,
and (c) **verifies / abstains**. This improves *faithfulness and compositional
verifiability* — not necessarily raw fluency — and that improvement must be **measured,
not asserted**.

### 1.3 Why this repo is the right place — it is *already half-built*

Sophia already contains a working neuro-symbolic split. The user's "OFK/LLM-wiki" is this
machinery:

| The user's intuition | What already exists in the repo |
|---|---|
| "match wordings to dictionary meaning" | `wiki/concept/*`, `wiki/figure/*` — provenance-native **definition pages** (gloss + tradition + `authorConfidence`), the `okf/` package that loads them into a typed belief graph |
| "look up the definition" | `agent/lexical_embed.py` (offline deterministic vectors), `agent/hybrid_retrieval.py` (dense+sparse RRF), `agent/retrieval.py` (chunks **carry** `tradition` / `author_confidence` / `do_not_attribute_to`) |
| "grammar so meaning composes" | OKF concept-TBox edges `subClassOf` / `disjointWith` / `scopedAnalogy` (`okf/schema.py`, `schemas/ontology_edge.schema.json`) — a tiny **categorial grammar of concepts** |
| "compose / reason like a human" | `agent/datalog_engine.py` — a stdlib least-fixed-point Datalog (the **symbolic** half), fed by `agent/datalog_provenance.py` (the **neural/lexical** half). The module docstring literally calls this "the symbolic half of a neuro-symbolic split." |
| "train it to learn" | `tools/wiki_to_training.py` (mine glosses → SFT/DPO), `tools/concept_edge_cli.py` + the concept-edge RLVR reward, the PEFT/LoRA training stack |
| "don't overclaim the result" | `make claim-check`, `tools/lint_claims.py`, `tools/eval_stats.py`, `tools/claim_gate.py`, the failure ledger |

**The contribution this proposal makes** is to extend that machinery from its current
narrow job (provenance / attribution discipline) to the user's *general* question —
**lexical-semantic grounding + compositional generalization** — and to build the
deterministic benchmark that decides whether grounding+grammar actually beats a pure
transformer, at equal model size, without overclaiming.

The governing contract is **`docs/11-Platform/Ontology-Claim-Boundary.md`**: a grounding
gate is **not** a truth oracle, closed-world, abstain-on-absence. Everything below honours
its five rules.

---

## 2. Brainstorm — the creative idea space (cheap → ambitious)

Ordered by confidence/cost. The recommended program is **I1 + I3 + I7** (a definition
probe, a compositional-derivation probe, and the 3-arm ablation that *is* the user's
question). I6 is the training extension if the ablation shows signal.

**I1 — OKF gloss-grounded Word-Sense / Definition-Faithfulness probe.** *(cheapest, highest
confidence)* Use the wiki concept/figure pages as a **sense inventory**. Task: given a
prompt using a term, does the model select and justify the OKF gloss appropriate to the
*tradition/context*, instead of merging senses? Deterministic scoring reuses
`hybrid_retrieval` + the provenance metadata + `agent/verifiers.py`. Directly tests "match
dictionary meaning."

**I2 — Definition-modeling reverse probe.** Ask the model to *generate* a definition for a
concept; check it entails the OKF gloss (genus–differentia overlap) **and** does not leak a
`doNotAttributeTo` / merge a `doNotMergeWith`. Tests "knows the meaning" by reconstruction
(cf. definition modeling, Noraset 2017).

**I3 — Grammar-as-typed-composition over OKF edges.** *(headline neuro-symbolic idea)* Treat
the TBox edges as a **categorial grammar of concepts**: a multi-hop claim ("X is a kind of
Y, Y is disjoint from Z, therefore …") must be **derived edge-by-edge** by
`datalog_engine`, not vibed. The neural layer *proposes* edges; the symbolic layer produces
a **verifiable parse/derivation** and abstains when the closed world is silent. Measures
whether grounded composition cuts hallucinated multi-hop inference vs. free-form
chain-of-thought. Extends `tools/concept_edge_cli.py` / `agent/datalog_ontology.py`.

**I4 — SCAN/COGS-style systematic-generalization harness.** Port a controlled
compositional-generalization split (or a Sophia-native one over concept edges) into `eval/`,
so the "grammar helps" claim has a *deterministic, falsifiable* test that the field already
trusts. Parse-then-compose vs. vanilla decoding at equal model.

**I5 — Grammar-constrained decoding for definitional faithfulness.** Constrain generation to
parse into the `ontology_edge` schema (CFG/typed constraint), so outputs are always
checkable against OKF. Cheap inference-time intervention with a strong honesty story.

**I6 — Wiki→training flywheel for sense-grounding.** *(the "train it to learn" path)* Extend
`tools/wiki_to_training.py`: chosen = definition-faithful, rejected = sense-merge. Train a
small LoRA, then RLVR with the concept-edge reward. Re-measure on I1/I3. Uses the existing
GitHub-Actions→RunPod training path (never local SSH; read `wisdom-gpu-prebaked` first).

**I7 — The 3-arm grounding ablation.** *(this is literally the user's question, made
falsifiable)* Same base model, three arms:
- **A0** pure transformer (distributional only);
- **A1** transformer **+ OKF definition retrieval** (hybrid_retrieval, provenance-carrying);
- **A2** transformer + retrieval **+ symbolic composition/abstain** (datalog_ontology + gate).
Whichever arm wins, *and by how much with CIs*, is the answer.

**I8 — Differentiable / induced grammar.** *(speculative, future work)* Instead of
hand-writing TBox edges, *induce* candidate edges and gate them through the ontology
verifier — a "grammar learner" whose proposals are admitted/quarantined, never trusted blind.

**I9 — Pictographic compositionality (bilingual angle).** *(speculative, future work)* The
corpus is Zh/En. Chinese characters compose meaning from radicals — a *built-in*
compositionality testbed: does sub-character structure predict semantic composition? Creative
and unique to this corpus.

---

## 3. Implementation plan — benchmark it with this repo's tools

Methodology mirrors `eval/epistemic_bench` and the Cardinal-Virtue GO/NO-GO protocol:
**build the harness first, claim nothing, decontaminate, pre-register the effect size, run
≥3 seeds with bootstrap CIs, promote only if the CI excludes zero AND a task-success
guardrail holds.**

### Phase 0 — build the sealed benchmark (no model, no claims)

Create `eval/semantic_grounding/` mirroring `eval/epistemic_bench/`
(`README.md`, `data/`, `score.py`, `candidateOnly: true`):

1. **Sense inventory & datasets** — generate from `wiki/concept/*` + `wiki/figure/*` via the
   `okf` loader:
   - **D1 (definition-faithfulness, I1):** prompt uses a term; gold = the correct OKF gloss
     + tradition; distractors = sibling senses / cross-tradition merges.
   - **D2 (compositional derivation, I3/I4):** multi-hop claims over `subClassOf` /
     `disjointWith` / `scopedAnalogy`; gold = the **Datalog derivation** (or `abstain` when
     the closed world is silent), checkable by `agent/datalog_engine.py`.
2. **Deterministic scorer** (`score.py`) — hard checks where possible, reusing
   `okf.graph.contradiction_ledger`, `datalog_engine`, `agent/verifiers.py`
   (`provenance_faithful`, `citation_faithful`). Use ≥2 judge families **only** for the
   prose-quality slice, with Cohen κ ≥ 0.40 (the RESULTS.md rule).
3. **Decontaminate & pre-register** — adapt `tools/assert_decontam.py` (0 overlap vs the
   training texts), compute and freeze an MDE (target ≤ 0.10, via `tools/eval_stats.py
   mde_at_n`), and freeze the split by git-ancestry — exactly as
   `tools/build_sophrosyne_external_battery.py` does.
4. **Commit the measurement spec first** — write
   `agi-proof/benchmark-results/semantic_grounding/measurement_spec.json` (constructs,
   `primaryMetric`, `primaryN`, `primaryMDE`, guardrails, `claimCeiling:
   "candidate_only; canClaimAGI:false"`) and commit it **before** any result, mirroring
   `agi-proof/benchmark-results/wisdom-market/measurement_spec.json`. This is the
   pre-registration the claim gate later checks.

*Exit Phase 0:* the harness runs offline and is green under `make claim-check`; still
`canClaimAGI: false`.

### Phase 1 — the experiment (I7, the 3-arm ablation)

Run **A0 / A1 / A2** (§2-I7) on the **same** base model over the farm via the bridge
(Spark + Mac + RunPod; one-GPU-job invariant — see `spark-cluster-ops`). Metrics per arm:

| Metric | Source |
|---|---|
| definition-faithfulness (D1) | `score.py` deterministic |
| compositional-generalization accuracy (D2) | `datalog_engine` derivation match |
| hallucinated-multi-hop rate | `verifiers.provenance_faithful` (multi-hop variant) |
| abstention correctness | `agent.gate_reward.is_abstention` |
| calibration (ECE, risk-coverage AUC) | `agent.calibration.calibration_report` |

Follow the Cardinal-Virtue GO-path template (`docs/11-Platform/Cardinal-Virtue-Benchmarks.md`):
label any prose slice with **≥2 independent judge families** (κ ≥ 0.40, via
`tools/eval_stats.py cohen_kappa` / `gwet_ac1`), then score the arms over **≥3 seeds** with
`bootstrap_ci_paired`. Convert the run into a GO/NO-GO receipt with
`python tools/claim_gate.py --prefix semantic-grounding --spec <measurement_spec.json>`
(writes `…/semantic-grounding.gate.json`). **Promotion rule:** A1 (or A2) beats A0 only if
the metric's Δ 95% CI excludes 0 **and** the claim gate returns GO **and** a task-success
guardrail (fluency / answer-rate) does not regress. **A NO-GO is logged in the failure
ledger, not tuned away.**

### Phase 2 — train, only if Phase 1 shows signal (I6)

*(Implemented:* `eval/semantic_grounding/` now carries a deterministic train/eval **fold**
on every case (`build_dataset.fold_of`), `tools/wiki_to_sense_training.py` emits SFT/DPO from
the **train** fold only, and the uplift is measured on the **eval** fold
(`run_semantic_grounding_eval.py --fold eval`) — so train/eval are disjoint by construction.
The OKF concept corpus was grown to enlarge D1; grow it further via the librarian to scale N.)*

Train a small LoRA on the train-fold SFT/DPO via
**GitHub Actions → RunPod** — the `.github/workflows/rlvr-runpod.yml` dispatch /
`tools/runpod_rlvr.py --task concept --reward verifier` path (read `wisdom-gpu-prebaked`
first; never local SSH; `--dry-run` before any paid run; the pod is always deleted in the
`finally` guard). Ingest results through `tools/ingest_rlvr_eval.py`, then re-measure on the
**frozen** Phase-0 benchmark. Log the measured GO/NO-GO in the failure ledger; do not chase
the number.

### Gates & hygiene (every commit/push)

- `make claim-check` (lint_claims, lint_training_rows, assert_decontam, eval_stats, claim_gate).
- Run the **`ci-artifact-drift`** skill (regen RESULTS/ledger/index) and **`git-discipline`**
  (stale-snapshot avoidance) before any commit/push.
- Add an **OPEN** row to `agi-proof/failure-ledger.md` for the unproven
  "grounding+grammar beats distributional baseline" claim, with claim-impact and the required
  measured response. `canClaimAGI` stays **false** until the gate — not human judgment — says
  otherwise.

---

## 4. What may and may not be said about the result

| MAY claim (if the CI supports it) | May NOT claim |
|---|---|
| "OKF-definition grounding lifts definition-faithfulness by Δ … [CI]" | "the model now *understands* meaning" |
| "symbolic composition reduces hallucinated multi-hop inference vs. CoT" | "we solved compositional generalization / it reasons like a human" |
| "abstains when the closed world is silent" | "knows when it does not know" |
| "a measured neuro-symbolic faithfulness result" | "an AGI / grounded-cognition unlock" |

---

## 5. References (entry points for the literature)

- Firth (1957); Harris (1954) — distributional hypothesis.
- Harnad (1990) — the symbol grounding problem.
- Bender & Koller (2020) — *Climbing towards NLU* (form vs. meaning, the octopus test).
- Fodor & Pylyshyn (1988) — systematicity critique of connectionism.
- Montague (1970); Steedman (CCG) — compositional / categorial semantics.
- Lake & Baroni (2018, SCAN); Kim & Linzen (2020, COGS); Keysers et al. (2020, CFQ) —
  compositional-generalization benchmarks.
- Noraset et al. (2017) — definition modeling. Huang et al. (2019, GlossBERT) — gloss-based WSD.
- In-repo: `docs/11-Platform/Ontology-Claim-Boundary.md`, `docs/11-Platform/OKF-Wiki.md`,
  `eval/epistemic_bench/README.md`, `agent/datalog_engine.py`.
