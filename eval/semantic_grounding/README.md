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
    d1_definition_faithfulness.jsonl   # 34 cases (generated; CI checks for drift)
    d2_compositional_derivation.jsonl  # 41 cases over 10 worlds (gold = engine-derived)
```

Every case carries a deterministic `fold` (`train` / `eval`, ~70/30 via
`build_dataset.fold_of`). The Phase-2 training generator
(`tools/wiki_to_sense_training.py`) draws the **train** fold only; a Phase-2 uplift is
measured on the **eval** fold (`run_semantic_grounding_eval.py --fold eval`). The two
folds are disjoint **by construction**, so there is no train/eval leakage to argue about
— `tools/assert_semantic_grounding_decontam.py` also asserts it.

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

The 3-arm ablation (A0 closed-book base · A1 +retrieved OKF definition · A2 +provenance/CoT
scaffold) is driven by `tools/run_semantic_grounding_eval.py` (offline `--mock` self-test here;
real runs via `--model <spec> --seeds N`). See that tool and the program doc for the methodology.

## Completion schema

Each completion is a JSON object keyed by case `id`:

- **D1:** `{"id", "completion": "<prose>", "selected": "<conceptId>"}` — `selected` optional;
  if absent the scorer falls back to an offline lexical match of the prose to the candidate
  glosses.
- **D2:** `{"id", "completion": "<prose>", "verdict": "entailed|violation|abstain"}` —
  `verdict` optional; if absent it is parsed from the prose.

## Honest limits (pre-registered)

- **N is still small** (D1=34, D2=41; eval fold smaller still). This is a candidate seam, not a
  powered result; report the MDE (`tools/eval_stats.mde_at_n`) and treat single-run deltas as
  coarse. Phase 1 must scale the item count and run ≥3 seeds before any claim. Grow the OKF
  concept corpus (via the wiki librarian) to enlarge D1 further.
- **D1 is closed-book** (A0 defines from parametric knowledge; the candidate glosses are a
  scoring-only inventory). The **sense** axis is therefore LOW-VALIDITY — it lexically matches the
  model's free-form definition to provenance-focused OKF glosses, so a valid definition can be
  mis-scored. The robust D1 axis is **faithfulness** (does the model assert a forbidden author?).
  An LLM-judge sense grader is a future upgrade.
- The **A0 vs A1** (does retrieved grounding help?) and **A0 vs A2** (does the provenance /
  reasoning scaffold help?) deltas are the load-bearing comparisons. A2-D2 is a real
  chain-of-thought arm — the engine verdict is never injected.
- D2 lives in a **closed world**: a verdict is grounding relative to the axioms we wrote down,
  **not** a truth claim (see `docs/11-Platform/Ontology-Claim-Boundary.md`). `abstain` is the
  only honest answer when the world is silent.
- The scorer measures **faithfulness/composition**, never "understanding". No result here may
  be read as an AGI claim; `canClaimAGI` stays `false`.
