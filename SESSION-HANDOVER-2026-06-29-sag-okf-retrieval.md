# Session Handover — 2026-06-29 (SAG vs OKF: provenance-aware graph retrieval)

Compared the external **Zleap-AI/SAG** graph-RAG repo against Sophia's **OKF/LLM-Wiki**,
then built the OKF answer to it: an entity-graph recall layer that **carries provenance
through every hop**. `canClaimAGI` stays **false** — the validated contribution is
provenance *faithfulness*, NOT a recall capability claim; all third-party numbers are
gated and unrun.

## Branch

`claude/sag-llmwiki-comparison-sp13ek` (exclusive to this worktree; NOT merged to `main`).
4 commits: `f10a0a6` → `e4b8096` → `663d35b` → `fab8232` (+ a 5th with the NER backend +
this handover). No PR opened (user did not ask).

## The thesis (one line)

SAG is a *recall engine* (optimizes Recall@K); OKF is a *belief-discipline engine*
(optimizes epistemic honesty). The move that is ours: run SAG-style multi-hop entity
recall **through** Sophia's confidence-propagation machinery, so a multi-hop answer is
floored by its weakest-provenance hop and **cannot launder confidence**. SAG has no
provenance and cannot do this.

## Shipped (on the branch, all tested + gated)

- **`okf/extract.py`** — provenance-tainted extraction + provenance-aware recall.
  - `extract_events()`: wiki bodies → `(event, entities)` units, each stamped with the
    page's EFFECTIVE (min-over-`derivesFrom`) rank. Folds in frontmatter edges
    (`links`/typed/`attributedAuthor`/`tradition`) — the real wiki authors edges in
    frontmatter, NOT inline `[[wikilinks]]` — and EXCLUDES negative signals
    (`doNotAttributeTo`).
  - `multi_hop_recall()`: shared-entity expansion, `provenanceFloor` = min over the path;
    `capped=True` when a path rests on weak provenance. Stopword filter on query tokens;
    **inverse-frequency (`1/df`) bridge weighting** so over-connected tradition hubs don't
    flood top-k (this was a real bug: naive expansion DEGRADED recall before the fix).
- **`okf/trace.py`** — provenance-colored retrieval trace (`trace_records`/`format_trace`);
  CLI `python tools/eval_okf_recall.py --trace "<query>"`.
- **`tools/eval_okf_recall.py`** — first-party decontaminated benchmark over `wiki/`
  (10 self-authored probes). **Result: provenance-faithfulness = 1.0, R@3=R@5=1.0,
  decontam CLEAN.** Honest caveat below.
- **`tools/eval_okf_multihop_qa.py`** — third-party harness (HotpotQA/2Wiki/MuSiQue):
  `vector_only` (lexical_embed) vs `graph_multihop` arms over the same pool; Recall@{2,5,10}
  + paired lift; `--check-decontam` (fail-closed, `vacuous`-aware) + `--out`. **NER backend
  is WIRED:** `--ner-backend <model-id>` calls `agent.llm` for real NER, fail-closed
  (exit 2) without a key, per-paragraph parse-fallback to the floor.
- **`.github/workflows/okf-multihop-recall.yml`** — manual `workflow_dispatch`, CPU-only.
  Downloads HotpotQA dev-distractor (fixed URL constant; injection-safe, no free-form
  inputs), runs both arms with the fail-closed decontam gate, uploads the report artifact.
  Does NOT auto-commit.
- **Pre-registration** `agi-proof/benchmark-results/okf-multihop/`: `measurement_spec.json`
  (GO criteria), byte-stable `okf-multihop-qa.PENDING.public-report.json` (status
  `not_run`, `canClaimAGI:false`), synthetic fixture.
- **Doc** `docs/11-Platform/SAG-vs-OKF-Retrieval.md` (comparison + roadmap + §4a/§4b
  results). **Failure-ledger row** `okf-multihop-recall-lift-not-shown-firstparty-2026-06-29`.
- **Tests**: `tests/test_okf_extract.py` (8), `tests/test_okf_multihop_qa.py` (8). Existing
  `tests/test_okf.py` unchanged + green.

## Honest status — what is and is NOT proven

- **VALIDATED (first-party):** provenance flooring works. Faithfulness 1.0 — every
  retrieved gold's `capped` verdict matches the corpus's ground-truth provenance. The
  necessity case (strong page reachable only via a weak bridge → floored) is unit-tested.
- **NOT proven:** the multi-hop **recall LIFT**. On `wiki/`, direct lexical recall already
  saturates (ablation: direct R@5 1.0 = 2-hop R@5 1.0, `multiHopReach 0.0`), so graph
  expansion isn't load-bearing for recall there. On HotpotQA/2Wiki/MuSiQue: **unrun**
  (datasets not committed, no in-session network).
- **No overclaim:** do NOT quote R@5=1.0 as a capability headline (first-party, lexically
  saturated). `canClaimAGI:false` everywhere.

## Next session — the one remaining step is operational

1. **Run the gated benchmark on the farm** (where training corpora are unlocked so decontam
   is non-vacuous): Actions → dispatch `okf-multihop-recall` (HotpotQA floor run), or
   locally `python tools/eval_okf_multihop_qa.py --data <dev.json> --dataset hotpot
   --check-decontam --out <report> --k 2 5 10`. For the LLM ceiling add
   `--ner-backend <model-id>` + an API key.
2. **GO criteria** (in the spec): paired Recall@2 lift > 0 with 95% CI excluding 0 over ≥3
   slices/seeds, decontam CLEAN (non-vacuous), no high-K regression. Report BOTH the
   deterministic floor and the LLM ceiling — the lift should hold at the floor.
3. **If GO:** promote the PENDING report → measured public-report, update the ledger row.
   **If NO-GO / null:** that's a valid outcome — log the measured numbers, do not tune.
4. **Watch-outs:** the decontam gate fail-closes (exit 2) on a `vacuous` scan — that's
   intended; ensure the corpora are present/unlocked before reading a number as certified.
   The CMU HotpotQA URL can be flaky; a HF mirror is the documented fallback.

## Reproduce locally (offline, no key)

```bash
python tests/test_okf_extract.py && python tests/test_okf_multihop_qa.py
python tools/eval_okf_recall.py                          # first-party: faithfulness 1.0
python tools/eval_okf_recall.py --trace "daoist scripture of the dao"
python tools/eval_okf_multihop_qa.py                     # fixture wiring self-test (not a result)
```
