# Provenance Delta — what you still need to do

The code is built and CI-green on the **mock** path. That proves the *plumbing*,
not a result. This checklist is the path from "scaffold" to "a number a skeptic
believes" to "the credibility/capability/adoption goals." Work top-to-bottom;
each tier unlocks the next.

Legend: `[ ]` to do · `[~]` partially in place · `[x]` done by the implementation.

---

## Tier 0 — produce a real number (do this first)

- [x] Benchmark harness, datasets, judge, scoring, report, CLI, offline tests, CI.
- [ ] **Add API keys** to `.env` for the models you want to measure
      (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY`, etc. — see `.env.example`).
- [ ] **Run a real headline pass** with an *independent* LLM-judge (judge model
      ≠ any model under test):
      ```bash
      python tools/run_provenance_delta.py \
        --models anthropic,openai,grok,ollama:qwen2.5-7b \
        --llm-judge anthropic:claude-opus-4-8
      ```
- [ ] **Eyeball the report** (`agi-proof/benchmark-results/provenance-delta.md`).
      Sanity check: does the *alone* hallucination rate look plausible? Is
      false-positive cost low? Is coverage honest (not 100%)?
- [ ] **Spot-check ~10 judged cases by hand** — confirm the judge's
      hallucinated/abstained/affirmed labels match your reading. The lexical
      screen and even the LLM-judge will mis-grade some; you must know the rate.

## Tier 1 — make the number defensible

- [ ] **Verify Wikidata QIDs**: `python tools/fetch_wikidata_authors.py --write`,
      then re-check each resolved QID actually matches the work (search can
      mis-hit). Commit the verified snapshot.
- [ ] **Grow the dataset to ≥100 cases** (currently 25). More works, more
      traditions, more decoy authors. Keep every label externally cited. Small N
      = wide confidence intervals = easy to dismiss.
- [ ] **Report confidence intervals**, not just point rates (bootstrap over
      cases). A "9% → 0.4%" with N=12 false cases is not yet a claim.
- [ ] **Run each model ≥3 times** (temperature > 0) and report variance — single
      runs aren't reproducible evidence.
- [ ] **Add a second judge** (different model) and report inter-judge agreement.
      Disagreement is your error bar on the headline.
- [ ] **Pre-register** the protocol and thresholds in
      `agi-proof/preregistered-thresholds.md` *before* the run you'll cite.

## Tier 2 — independent replication (credibility goal)

- [ ] **Clean-clone reproduction**: have someone run it from a fresh clone with
      their own keys and confirm the delta within CI. Log it in
      `agi-proof/third-party-replication/`.
- [ ] **Recruit 1–2 collaborators** (you said this is open) to co-sign the
      replication and/or co-author the write-up.
- [ ] **Short write-up / arXiv note**: method, the non-circularity argument, the
      table, the failure cases. This is what turns a repo into a *citation*.
- [ ] **Wire the headline number into the README and thesis site** only *after*
      Tier 1 — never publish the mock numbers.

## Known gate coverage gaps (observed in real runs — scoped fixes)

The gate's regex is high-precision but missed real hallucinations during the
multi-model run. Fix these in `agent/verifiers.py:provenance_faithful` *with new
tests*, without lowering precision (verify the dispute pages still pass):

- [ ] **Quoted / punctuated titles** — `wrote "The Constitution of the Athenians"`
      isn't spanned because a quote sits between the verb and the title.
- [ ] **`attributed to X`** phrasing — the gate matches `attributed by`, not the
      far more common `attributed to`.
- [ ] **Multi-word / honorific author names** — e.g. "the prophet Daniel",
      "King David": ensure `author_markers` keys on the salient surname token.
- [ ] **Title aliases** — let a record carry alternate titles (short forms,
      Latin/English) so "Psalms" matches "the Book of Psalms".

## Tier 3 — widen the thesis (capability + adoption goals)

- [ ] **Second provenance type** beyond ancient texts — pick one: RAG
      citation-faithfulness (does an answer's cited source actually support it?),
      legal/contract attribution, or code provenance. Reuse
      `SOPHIA_DISCIPLINE_RECORDS` (already supported by the gate).
- [ ] **Ship `pip install sophia-guard`** — package the gate + guarded loop as a
      standalone primitive with a 5-line README example. Adoption needs a
      one-import surface, not a repo clone.
- [ ] **Adapter examples**: LangChain/LlamaIndex output-gate, an OpenClaw plugin
      (you have one), a raw-HTTP middleware. Make it trivial to bolt on.

## Tier 4 — public benchmark (category-ownership goal)

- [ ] **Submission format + leaderboard** so others can submit model results
      (a mini "ARC-AGI for attribution"). Hold out a private test split.
- [ ] **External hard evals**: run the gated agent on a slice of GAIA / a
      retrieval-QA set and report attribution-faithfulness there too — evidence
      outside your own dataset.

---

## Honesty guardrails (don't skip — they ARE the brand)

- [ ] Never let labels leak from the gate corpus into the test set (keep
      `provenance_bench/data/` and root `data/` separate — enforced by design,
      verify it stays true as you add cases).
- [ ] Always report **false-positive cost** and **coverage** next to the delta.
      A gate that abstains on everything is not a win.
- [ ] State N, variance, and judge method on every published number.
- [ ] Keep the mock report git-ignored; only commit numbers from real, judged,
      multi-run passes.
- [ ] Update `agi-proof/failure-ledger.md` with cases the gate misses — the
      narrow-coverage honesty is a feature, not a liability.
```bash
# quick re-verification any time
python tests/test_provenance_bench.py
python tools/run_provenance_delta.py --models mock
```
