# Semantic Grounding & Compositional Grammar — benchmark (Phase 0)

**Status:** candidate infrastructure. `candidateOnly: true`, `level3Evidence: false`,
`canClaimAGI: false`. This is a **measurement harness**, not a capability claim. It is the
build-first prerequisite for
[`docs/06-Roadmap/Semantic-Grounding-And-Compositional-Grammar-Program.md`](../../docs/06-Roadmap/Semantic-Grounding-And-Compositional-Grammar-Program.md).

It makes one question falsifiable: *does grounding word meaning in explicit OKF definitions,
and composing meaning with a small concept grammar, behave better than ungrounded
distributional guessing?* — measured, with confidence intervals, never asserted.

## The two task families (each scored deterministically — no LLM judge in the scorer)

| Task | Measures | Source of truth | Reuses |
|---|---|---|---|
| **D1** definition-faithfulness | did the answer pick the OKF gloss that defines the term (word-sense) **and** avoid a forbidden attribution? | `wiki/concept/*` glosses + `doNotAttributeTo` | `agent.lexical_embed.rank`, `agent.verifiers.provenance_faithful` |
| **D2** compositional derivation | is a multi-hop claim *entailed* / *violation* / *abstain* under a closed world of concept-TBox axioms? | a frozen, sourced `subClassOf`/`disjointWith` seed | `agent.datalog_engine` (least-fixed-point transitive closure) |

D2's gold verdict is a **derivable theorem** (`score.reference_verdict` runs the Datalog
engine), not a hand label — the scorer re-derives it and fails closed (`D2_dataset_valid`) if
a committed gold ever drifts from the engine.

## Layout

```
eval/semantic_grounding/
  build_dataset.py   # projects D1 from wiki/concept + generates D2 from the AXIOM_WORLDS seed
  score.py           # deterministic scorer + Datalog reference reasoner + --self-test
  data/
    d1_definition_faithfulness.jsonl   # 26 cases (generated; CI checks for drift)
    d2_compositional_derivation.jsonl  # 14 cases (generated; gold = engine-derived)
```

## Run

```bash
# Regenerate / verify the sealed datasets (the source of truth is wiki/ + the seed)
python -m eval.semantic_grounding.build_dataset --emit
python -m eval.semantic_grounding.build_dataset --check     # CI drift gate

# Score a run (completions keyed by case id)
python -m eval.semantic_grounding.score \
  --cases eval/semantic_grounding/data/d1_definition_faithfulness.jsonl \
  --completions runs/arm.jsonl

# Model-free seam check (CI)
python -m eval.semantic_grounding.score --self-test
```

The 3-arm ablation (A0 base · A1 +OKF-definition-retrieval · A2 +symbolic-compose/abstain)
is driven by `tools/run_semantic_grounding_eval.py` (offline `--mock` self-test here; real
runs are model-farm-gated). See that tool and the program doc for the methodology.

## Completion schema

Each completion is a JSON object keyed by case `id`:

- **D1:** `{"id", "completion": "<prose>", "selected": "<conceptId>"}` — `selected` optional;
  if absent the scorer falls back to an offline lexical match of the prose to the candidate
  glosses.
- **D2:** `{"id", "completion": "<prose>", "verdict": "entailed|violation|abstain"}` —
  `verdict` optional; if absent it is parsed from the prose.

## Honest limits (pre-registered)

- **N is small** (D1=26, D2=14). This is a candidate seam, not a powered result; report the
  MDE (`tools/eval_stats.mde_at_n`) and treat single-run deltas as coarse. Phase 1 must scale
  the item count and run ≥3 seeds before any claim.
- D1 distractors are lexically-near sibling glosses, not adversarial paraphrases — an easy
  floor, not a stress test.
- D2 lives in a **closed world**: a verdict is grounding relative to the axioms we wrote down,
  **not** a truth claim (see `docs/11-Platform/Ontology-Claim-Boundary.md`). `abstain` is the
  only honest answer when the world is silent.
- The scorer measures **faithfulness/composition**, never "understanding". No result here may
  be read as an AGI claim; `canClaimAGI` stays `false`.
