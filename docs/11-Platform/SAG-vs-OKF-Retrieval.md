# SAG (Zleap-AI) vs. OKF/LLM-Wiki — retrieval comparison and roadmap

Status: analysis + spike. The spike (`okf/extract.py`, `tests/test_okf_extract.py`) is
runnable and tested but **not** a validated benchmark. No capability claim here is
gated; see [`agi-proof/failure-ledger.md`](../../agi-proof/failure-ledger.md) for what
remains unproven.

Source compared: <https://github.com/Zleap-AI/SAG> (README + architecture, read
2026-06-29).

## 1. The two systems optimize for opposite things

They look alike (markdown → graph → query) but the objective functions diverge:

| | **Zleap-AI/SAG** | **OKF / LLM-Wiki (this repo)** |
|---|---|---|
| Optimizes for | Recall@K on multi-hop QA | epistemic honesty (no overclaiming) |
| Graph nodes | machine-**extracted** events + entities | human-**curated** typed pages |
| Edges | entity co-occurrence, cheap to rebuild | hand-authored typed edges (`contradicts`, `supersedes`, `subClassOf`, `derivesFrom`) |
| Retrieval unit | one **event** per chunk + entity index | whole page / RAG chunk |
| Signature feature | multi-hop entity traversal, served | confidence-laundering + contradiction ledger (`okf/graph.py`) |
| Provenance | none | the entire point |
| Shape | one product: TS, Postgres+pgvector, React UI, MCP | dependency-free Python lib + committed numpy index, in a research monorepo |

SAG is a **retrieval engine**. OKF is a **belief-discipline engine**. The correct move
is not to converge on SAG but to put SAG-style recall *underneath* the OKF provenance
layer. SAG has no way to say a claim is `legendary`, anachronistic, or laundered — that
is our moat.

## 2. What SAG does better (worth borrowing)

1. **Event-as-retrieval-unit.** SAG extracts *one complete event + N entities* per
   chunk — finer-grained and more semantically coherent than our page/chunk granularity.
2. **An extracted entity index for multi-hop expansion.** Our graph edges are
   hand-authored frontmatter — high precision, but sparse (hence `orphans()` /
   `dangling_links()` in `okf/linker.py`). SAG gets cross-document traversal for free
   from shared entity mentions.
3. **Search-trace visualization** — it shows *why* each result was retrieved. We compute
   a trace in `agent/ai_search.py` but never surface it.
4. **A public, falsifiable retrieval number** — "Recall@2 68.14 → 79.30 vs HippoRAG 2"
   on HotpotQA / 2WikiMultiHop / MuSiQue. We have no standard multi-hop retrieval bench.
5. **Cheap-rebuild discipline** — events stay as semantic units, entities are just an
   index; no heavyweight KG rebuild. Our `okf.graph.build()` reconstructs the typed
   graph in memory per call (fine at ~96 pages, a wall later).

**Not worth copying:** the Postgres/Fastify/React serving shell. That is productization,
off-thesis; our committed-index + MCP approach is right for research.

## 3. What OKF already does that SAG cannot

- Provenance-native frontmatter (`authorConfidence`, `doNotAttributeTo`,
  `doNotMergeWith`, `tradition`).
- Confidence-laundering detection via min-over-chain propagation
  (`okf.graph.propagate_confidence`).
- Contradiction ledger: self-merges, supersede cycles, TBox subclass cycles, disjointness
  violations, cross-tradition unscoped mappings (`okf.graph.contradiction_ledger`).
- Counterfactual retraction / belief revision / abstention (`okf/counterfactual.py`,
  `okf/revision.py`).
- No-overclaim gates, decontamination, claim linting (`tools/lint_claims.py`,
  `tools/claim_gate.py`).

We also already have hybrid dense+sparse RRF retrieval (`agent/hybrid_retrieval.py`) and
multi-hop sub-query fan-out + rerank (`agent/ai_search.py`). SAG's retrieval is partly
something we have — just not productized as an extracted event/entity index.

## 4. Roadmap — thesis-aligned, ranked

1. **Provenance-tainted event/entity extraction (foundation).** Extract `(event,
   entities)` units from wiki bodies, each stamped with the page's *effective*
   confidence rank. → shipped as a spike: `okf/extract.py::extract_events`.
2. **Provenance-aware multi-hop recall (the actual contribution).** Traverse shared
   entities like SAG, but floor each path by the weakest page it touches, so a confident
   event reached through a `legendary`/`anachronism_risk` bridge surfaces with a low
   `provenanceFloor` and `capped=True`. SAG can report Recall@K; we can report Recall@K
   **and** a provenance-faithfulness score. → spike:
   `okf/extract.py::multi_hop_recall`.
3. **Auto-extracted entity index to close the orphan/dangling gap.** Keep hand-authored
   typed edges as the *trusted* layer; add the extracted entity-mention index as a
   *candidate* layer that must earn promotion through the existing gate — never
   auto-merged. → entity index shipped (`build_entity_index`); promotion path = TODO.
4. **Surface the retrieval trace with per-hop provenance.** Emit each unit + why it
   matched + its effective rank + the hop that contributed it. SAG's search-trace, but
   provenance-colored. → **shipped:** `okf/trace.py` (`trace_records` / `format_trace`),
   CLI `python tools/eval_okf_recall.py --trace "<query>"`.
5. **Decontaminated multi-hop QA benchmark.** → **first-party shipped:**
   `tools/eval_okf_recall.py` over 10 self-authored, decontaminated probes on `wiki/`
   (faithfulness validated; see §4a). **Third-party scaffolded, gated:**
   `tools/eval_okf_multihop_qa.py` runs OKF entity-graph recall vs a vector-only
   baseline on HotpotQA / 2Wiki / MuSiQue, with HotpotQA/2Wiki/MuSiQue normalizers, a
   deterministic entity floor (real NER/LLM is the `--ner-backend` farm seam), and a
   synthetic wiring fixture. Pre-registered (`agi-proof/benchmark-results/okf-multihop/`
   `measurement_spec.json` + byte-stable `…PENDING.public-report.json`, status
   `not_run`). The datasets aren't committed and there's no in-session network, so the
   real run is **farm-only**. See §4b.
6. **(Optional, off critical path) Provenance-colored graph viz.** Color nodes by
   `effectiveConfidenceRank`; draw contradiction-ledger edges. SAG literally cannot
   render this — it has no provenance.

## 4a. Benchmark result (first-party, honest)

`python tools/eval_okf_recall.py` (10 decontaminated probes over `wiki/`, fully offline):

| metric | value | reading |
|---|---|---|
| decontam | CLEAN | every probe query is shingle-disjoint from its gold page body |
| Recall@3 / @5 | 1.0 / 1.0 | gold answer page is retrieved |
| MRR | 0.75 | gold is usually near the top |
| **provenance-faithfulness** | **1.0** | the `capped` verdict matches the corpus's ground-truth provenance on every retrieved gold |
| ablation R@5 | direct 1.0 → 2-hop 1.0 | multi-hop is **neutral** here |
| multi-hop reach | 0.0 | every gold was found by direct lexical match |

**What this proves and what it does not.** The validated contribution is the
provenance **flooring** (faithfulness 1.0) — extraction carries the effective rank
through, and a path is correctly capped by its weakest hop. The multi-hop **recall
lift** is *not* demonstrated: on this small, templated corpus direct lexical recall
already saturates, so graph expansion is not load-bearing for recall. An earlier naive
expansion actually *degraded* recall (2-hop 0.8 < direct 1.0) by flooding top-k with
tradition-hub siblings; that was fixed with inverse-frequency (`1/df`) bridge weighting
(hub bridges add ~0) — an honest fix, not a tuned number. The case where multi-hop is
*necessary* (a strong page reachable only through a weak bridge) is proven in the unit
tests, not yet at benchmark scale. Logged in the failure ledger
(`okf-multihop-recall-lift-not-shown-firstparty-2026-06-29`). Do not quote R@5=1.0 as a
capability headline — it is first-party and lexically saturated.

## 4b. Third-party benchmark scaffold (gated, farm-only)

`tools/eval_okf_multihop_qa.py` is the SAG-comparable harness: same candidate pool,
two arms — `vector_only` (`agent.lexical_embed` cosine) vs `graph_multihop`
(`okf.extract.multi_hop_recall`) — reporting supporting-paragraph Recall@{2,5,10} and
the paired graph-minus-vector lift, with a decontam pre-check.

Two honesty constraints are baked in:
- **No provenance on web text.** HotpotQA/2Wiki/MuSiQue paragraphs are unlabeled, so the
  provenance-faithfulness metric (§4a) is *not* computable there — this harness isolates
  the **recall lift** only. Faithfulness stays a `wiki/`-only result.
- **Nothing claimed from the fixture.** With no `--data`, it runs a 3-item synthetic
  fixture purely to validate wiring; the output is banner-flagged "not a result." The
  real datasets are downloaded and run on the farm (`--data … --dataset hotpot`).

GO criteria (in the spec): paired Recall@2 lift > 0 with a 95% CI excluding 0 over ≥3
slices/seeds, decontam clean, no regression at higher K. Until the farm run clears them,
no third-party number is claimed.

**One dispatch away.** `.github/workflows/okf-multihop-recall.yml` (manual
`workflow_dispatch`, CPU-only, no GPU) downloads the canonical public HotpotQA
dev-distractor set, runs both arms with a **fail-closed decontam gate**, and uploads the
report as an artifact. It deliberately does *not* auto-commit — an operator reviews the
artifact and promotes the PENDING report by hand only if the GO criteria hold. The
decontam check flags a `vacuous` scan (corpora absent / git-crypt-locked) and fails
closed, so a number can't be certified without the training corpora present. 2Wiki /
MuSiQue run by editing the `DATA_URL`/`DATASET` constants (the workflow takes no
free-form inputs, by the repo's injection-safe convention).

## 5. One-line takeaway

Steal SAG's **event/entity extraction + multi-hop entity index**; ignore its serving
stack. The move that is *ours* is to run that recall **through the confidence-propagation
and contradiction machinery** — retrieval that refuses to launder provenance like nothing
else does. Items 1–4 are shipped (`okf/extract.py`, `okf/trace.py`,
`tools/eval_okf_recall.py`); the provenance-flooring is validated first-party, and the
multi-hop recall-lift + third-party datasets stay gated TODOs.

## 6. Try the spike

```bash
python tests/test_okf_extract.py        # provenance-floor property tests
python - <<'PY'
from okf import page, extract
events = extract.extract_events(page.load_pages("wiki"))
for h in extract.multi_hop_recall("penicillin discovery", events, top_k=5):
    print(f"{h.event.page_id:24} hops={h.hops} floor={h.provenance_floor} capped={h.capped}")
PY
```
