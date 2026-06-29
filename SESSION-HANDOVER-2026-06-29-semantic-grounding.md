# Session handover — Semantic Grounding & Compositional Grammar (2026-06-29)

A self-contained research program added on a feature branch: a benchmark + 3-arm ablation
asking *"can grounding word meaning in explicit OKF definitions, and composing meaning with a
small concept grammar, beat a pure distributional baseline?"* Built, run on a real model, and
honestly recorded as a **NULL** result. Nothing claimed; `canClaimAGI` stays false.

## 1. Git / repo state at handover

- **Branch:** `claude/ai-semantics-grammar-research-zmdtn0` (exclusive to this worktree).
- **HEAD:** `d56c0dc`. **Pushed** to `origin/<same branch>`. Working tree **clean**.
- **vs `origin/main`:** +7 ahead (this session), −22 behind. **Not merged to main; no PR opened**
  (none requested). Branch is the source of truth for this work.
- 7 commits this session: `1fa5c19` (research note) → `0af14ae` (Phase-0 harness) → `7929f00`
  (folds + corpus growth) → `2442e4c` (real-model wiring) → `089edc5` (run 1) → `b4c02c4`
  (arm redesign) → `d56c0dc` (run 2).

## 2. What this session did (within the no-overclaim ceiling)

- **Research note + plan:** `docs/06-Roadmap/Semantic-Grounding-And-Compositional-Grammar-Program.md`
  (thesis: distributional vs grounded/compositional semantics; maps onto the repo's existing
  neuro-symbolic machinery — `okf/`, `agent/lexical_embed.py`, `agent/datalog_engine.py`,
  concept-TBox edges).
- **Phase-0 benchmark** `eval/semantic_grounding/`: **D1** definition-faithfulness (34 closed-book
  word-sense cases from `wiki/concept` glosses + `doNotAttributeTo`) and **D2** compositional
  derivation (41 cases over 10 closed-world taxonomies; gold verdict **DERIVED** by
  `agent/datalog_engine`, not hand-labelled). Deterministic scorer + `--self-test`.
- **Decontam + pre-registration:** `tools/assert_semantic_grounding_decontam.py`,
  `agi-proof/benchmark-results/semantic_grounding/measurement_spec.json`.
- **Train/eval folds** (`build_dataset.fold_of`, ~70/30) → Phase-2 training is decontaminated by
  construction; `tools/wiki_to_sense_training.py` draws the train fold only.
- **Corpus growth:** +8 sourced, textbook-consensus science concepts in `data/science.json`
  (periodic table, telescope→`doNotAttributeTo: galileo`, universal gravitation, electron,
  smallpox vaccine, continental drift, oxygen discovery, incandescent bulb); regenerated wiki;
  `wiki_validate` + `lint_wiki_provenance` clean.
- **3-arm runner wired to real models** (`tools/run_semantic_grounding_eval.py`): DeepSeek +
  llmhub presets, real multi-seed loop, `--fold` filter, offline `--mock` self-test.
- **Two real DeepSeek runs** (`deepseek-chat`, 3 seeds, 75 cases). Run 1 was confounded
  (open-book A1 duplicated the prompt → Δ −0.071; A2 injected the engine verdict → tautology).
  **Redesigned** to closed-book A0 + genuine A1 retrieval + real CoT A2, then re-ran.

## 3. Proven vs still open

**Result (RUN 2, the clean one) — artifact `agi-proof/benchmark-results/semantic_grounding/3arm-deepseek_deepseek-chat.json`:**

| Arm | D1 sense | D1 faith | D2 | vs A0 (95% CI) |
|---|---|---|---|---|
| A0 closed-book | 0.814 | 1.0 | 0.789 | — |
| A1 +retrieved gloss | 0.824 | 1.0 | 0.789 | **+0.004 [−0.027, 0.036] NULL** |
| A2 +provenance + CoT | 0.784 | 1.0 | 0.870 | **+0.031 [−0.022, 0.084] NULL** |

- **Verdict: NO-GO / honest NULL.** All deltas' CIs include 0 and are **underpowered**
  (MDE 0.132 > pre-registered 0.10 at n=225). Faithfulness non-discriminating (1.0 — DeepSeek
  never merges a forbidden lineage on these common concepts). A2's CoT lifts D2 0.789→0.870 (a
  hint, not significant).
- **Root cause:** a strong base model is already grounded on **common** textbook concepts → no
  headroom. The signal, if any, lives in **long-tail / rare** concepts.
- **Open ledger row:** `semantic-grounding-grounding-beats-baseline-2026-06-29` in
  `agi-proof/failure-ledger.md` (carries the numbers + required next steps).
- **Known limit:** closed-book **sense** scoring is low-validity (lexical match of a free-form
  definition to provenance-focused glosses). **Faithfulness** is the robust D1 axis. An LLM-judge
  sense grader is the fix.

## 4. ▶ Next step (single most valuable)

**Make the experiment powered and discriminating** so a future run can actually GO or NO-GO:

1. **Grow long-tail D1 concepts** where a base model fails (obscure/contested attributions),
   via the wiki librarian / sourced `data/*.json` records — and **expand D2 worlds** until
   `python -c "from tools.eval_stats import mde_at_n; print(mde_at_n(N))"` ≤ 0.10
   (needs ~390+ paired items; with 3 seeds that's ~130 cases, up from 75).
2. **Add an LLM-judge sense grader** for D1 (≥2 independent families, Cohen κ ≥ 0.40) to replace
   the low-validity lexical sense axis. llmhub `gpt-4o-mini` works as one family; pick a second.
3. **Re-run** `python tools/run_semantic_grounding_eval.py --model deepseek:deepseek-chat --fold all --seeds 3 --write`
   then **gate**: `python tools/claim_gate.py --prefix semantic-grounding --spec agi-proof/benchmark-results/semantic_grounding/measurement_spec.json`.
   **Pass bar:** A1−A0 or A2−A0 Δ 95% CI excludes 0 **and** `mde_at_n` ≤ 0.10 **and** task-success
   guardrail holds. A NO-GO is a valid outcome — log it, do not tune.
4. **(Phase 2, optional, decontaminated-by-construction)** train a LoRA on the **train-fold** SFT/DPO
   from `python tools/wiki_to_sense_training.py`, via `tools/runpod_rlvr.py --task concept`
   (read `wisdom-gpu-prebaked`; never local SSH; `--dry-run` first), measure on `--fold eval`.

## 5. Read-first list

- `docs/06-Roadmap/Semantic-Grounding-And-Compositional-Grammar-Program.md` (thesis + plan)
- `eval/semantic_grounding/README.md` (tasks, run commands, honest limits)
- `agi-proof/benchmark-results/semantic_grounding/measurement_spec.json` (pre-registration + arms)
- `agi-proof/failure-ledger.md` row `semantic-grounding-grounding-beats-baseline-2026-06-29`
- `tools/run_semantic_grounding_eval.py` (arm definitions in `build_prompt`)

## 6. Don't-break list (CI gates that must stay green)

- `python -m eval.semantic_grounding.build_dataset --check` (dataset drift; D2 gold = engine-derived)
- `python tools/assert_semantic_grounding_decontam.py` (eval↔train + train/eval-fold disjointness)
- `python tests/test_semantic_grounding.py` (wired into `fast-ci.yml`)
- `make claim-check`; `python tools/wiki_sync.py check`; `python tools/lint_wiki_provenance.py`
- `python tools/validate_failure_ledger.py --check` + regen `tools/build_agi_proof_package.py`
  after any ledger edit.

## 7. Secrets note

The DeepSeek + llmhub API keys used for the real runs were provided in-session and held ONLY in
an ephemeral, untracked scratchpad env file (never committed; gitleaks/secret-scan clean). They
should be **rotated**. `api.llmhub.com.cn` is a third-party proxy (not Anthropic); on that account
only some models are provisioned (`gpt-4o-mini` works). DeepSeek's own API was the subject model.
