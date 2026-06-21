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
- [x] **Report confidence intervals** — `--runs N` enables a paired bootstrap
      95% CI on the delta (`provenance_bench/aggregate.py`). Still: widen N so the
      CI is tight.
- [x] **Run each model ≥3 times** (temperature > 0) and report variance —
      `--runs 3` records per-run deltas. Still: bump runs for stability.
- [x] **Independent LLM-judge** — `--llm-judge <spec>` (judge ≠ subject), e.g.
      `--llm-judge deepseek`.
- [~] **Inter-judge agreement (CRITICAL — partially done, key open item).** A
      Claude audit panel re-judged the DeepSeek verdicts: **76% agreement**, with
      DeepSeek over-counting (10 false positives) — the validated alone-rate was
      **21.7%, not 41.3%**. Lesson: a single LLM-judge is unreliable; the headline
      needs a **≥2-judge consensus over BOTH the alone and gated arms**, reporting
      agreement. Wire a consensus judge (majority of N) into `aggregate.py` next.
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

- [x] **Quoted / punctuated titles** — gate is now quote/"the"-tolerant before a
      title (`the_q`). Verified: `wrote "The Constitution of the Athenians"` fails.
- [x] **`attributed to X`** phrasing — added, with a bounded honorific filler
      (`attributed to the prophet Daniel`). Hedged "traditionally attributed to…"
      still passes (new carve-outs).
- [x] **Multi-word / honorific author names** — `build_gate_records` reduces
      "the prophet Daniel"/"King David" to a salient marker.
- [x] **Title aliases** — records accept `altTitlesEn`; `build_gate_records`
      derives short forms ("Book of Daniel" → "Daniel", interior-"the" collapse).
- [x] **Appositive / parenthetical between author and verb** — "Enoch, the
      great-grandson of Adam, wrote …", "Lie Yukou (also known as Liezi) wrote …".
      Added a bounded, contrast-free `app` slot; independently verified **0 false
      positives on all 41 true controls** and dispute-lint still 0.
- [ ] **Accepted limitation (precision-first):** a *bare* "X is traditionally
      attributed as the author of Y" with the doubt only in the *next* sentence is
      left uncaught — narrowing the `traditionally` carve-out risked false
      positives on legitimately-hedged text, so we kept the conservative behavior.
- [ ] Still TODO: fold these alt-title/marker/appositive helpers into the seeded
      `data/*.json` corpus so the *production* gate (not just the benchmark) gets
      the wider coverage, with precision re-verified against the dispute pages.

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
