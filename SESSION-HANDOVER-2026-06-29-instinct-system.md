# Sophia-AGI — Session Handover (2026-06-29): Thinking-Chain Intervention & the Instinct System

> **Topic handover** for the work on branch `claude/ai-thinking-chain-intervention-sqeokn`.
> Self-contained; pairs with the master handover `SESSION-HANDOVER-2026-06-28.md` (repo thesis +
> measurement contract — read that for the no-overclaim gate). Everything here is
> `candidateOnly: true`, `canClaimAGI: false`. No promotion to `published-results.json` was done.

---

## 0. Git / repo state at handover

- Branch: **`claude/ai-thinking-chain-intervention-sqeokn`**, `HEAD == origin/<branch> == 32cbcde`.
  Working tree clean. **13 commits, all pushed.** Branch is exclusive to this worktree; every push
  was a fast-forward of my own commits (no merge/rebase, no concurrent-advisor contention).
- **Not merged to `main`** and **no PR opened** (not requested). To land: open a PR from this branch.
- All commits are additive: a new `reasoning/instinct_*` module family + one opt-in `agent/` module
  + two `tools/` runners. **No existing call sites touched**, so no behaviour change to the agent.

---

## 1. The thesis this session pursued

Operator's question: *modern models reason in a "thinking chain" — can we (a) inject that chain to
make the model change its mind, and (b) give it an instinct to change its mind early instead of
ploughing ahead and patching a wrong path?* Grounded in current literature (CoT is often post-hoc
/ unfaithful; intrinsic self-correction is marginal and can hurt; activation-steering & backtrack
tokens can flip a chain but steering is brittle — sources cited in the doc §1).

Primary deliverable (this repo's idiom): **a falsifiable, offline, seeded model + measurement
harness for each claim**, plus an honest record of what is and isn't shown. The harness is the
deliverable; no capability is claimed.

**Read first:** [`docs/06-Roadmap/Thinking-Chain-Intervention-and-Instinct.md`](docs/06-Roadmap/Thinking-Chain-Intervention-and-Instinct.md)
— the full narrative (§1 literature, §2 architecture, §3a–§3h results, §4 pre-registration, §5
risks). Then [`reasoning/README.md`](reasoning/README.md) (module table, rows #6).

---

## 2. What was built (commit → what it shows, within the ceiling)

All modules share the lane CLI (`--run` / `--self-test` / `--json`), are pure-stdlib + seeded +
**deterministic across processes** (hash-seed bugs were found and fixed twice), and each has a
test in `tests/test_<name>.py`. Results snapshots in `reasoning/results/<name>.txt`.

| # | module | what it shows (candidate) |
|---|---|---|
| `3e28397` | `reasoning/instinct_gate.py` | Policy model: early reflex **re-route beats late self-correction** above a **stable break-even d′ = 1.0** (margin-based estimator; below it a trigger-happy reflex *hurts*). Ko-bounded escalate. The ceiling is the reflex's ROC, not the policy. |
| `9b9b3c4` | `reasoning/instinct_reflex_eval.py` | Go/no-go **harness**: measures a reflex's d′/AUC vs the belief-revision oracle. Self-consistency is **borderline** (d′ 0.96 @ moderate competence); no-signal control collapses to AUC≈0.5. |
| `135d37d` | `reasoning/instinct_fusion.py` | Synthetic: a 2nd **independent** detector (okf grounding) fused with self-consistency clears the bar though neither does alone; law `d′_fused=(d_A+d_B)/√(2+2ρ)`. Later extended with `_reflex_B2` (completeness) + `fuse()` (weighted). |
| `ce6b1f5`,`2de3f35` | `tools/run_reflex_openrouter.py`, `tools/run_fusion_realmodel.py` | Real-model runners (OpenAI-compatible; **keys from env only**, fail-loud on 4xx, cost-guarded). v2 stores raw answer-sets so re-scoring is free. |
| `7d7f1cd`,`ffdabaa` | `reasoning/results/fusion_realmodel_*.json` | **Real measurements** (DeepSeek + Claude-haiku, 50 cases × 5 samples). |
| `298b73f` | `reasoning/instinct_endtoend.py` | Outcome: feeding **real operating points** into the re-route policy. With the full bus, **DeepSeek confident-wrong 0.58→0.00 (correct 0.42→0.79); Claude-haiku confident-wrong 0.98→0.00, converted to 0.92 escalation** (rescued by fail-close). |
| `1abeb73` | `reasoning/instinct_fusion.py` (+B2/fuse) | Grounding-**completeness** detector (under-abstention mirror) + quality-weighted fusion. |
| `e1cf299` | `reasoning/instinct_validation.py` | **Rigor:** LOO-CV weights + bootstrap CIs. **Key honest artifact:** B/B2 are *verifiers* (fire iff answer≠truth ⇒ near-tautological AUC); the only label-free *predictive* reflex is A (self-consistency), and it's weak (AUC 0.63, CI includes chance). |
| `8d5bbcc` | `reasoning/instinct_injection.py` | The **injection half**: editing the chain in place dominates re-route (0.66 vs 0.50, 5× cheaper) but has a **brittleness roofline** (over-steering corrupts); inject→reroute hybrid wins (0.81). Planted curves — real steering needs white-box. |
| `8895e4f` | `agent/reflex_bus.py` | **Wired (opt-in):** reflex bus + interrupt controller emitting **conscience-native verdicts** `allow/revise/escalate/abstain` (`revise` = re-route). Composes `agent.calibration` + caller-supplied okf sets. Not rewired into the guarded loop. |
| `32cbcde` | `reasoning/instinct_labelfree.py` | **Frontier:** per-element membership **instability** beats exact self-consistency (AUC 0.668 vs 0.629, CI **excludes** chance). Hard ceiling: all agreement signals are anti-predictive on a confident-wrong model (haiku ~0.02). |

---

## 3. Proven (candidate) vs still open

**Shown, within the candidate ceiling (offline, seeded, deterministic, tested):**
- Early reflex re-route > late self-correction, gated by a stable break-even d′=1.0.
- Independent-detector fusion law; a 3rd completeness detector is what real models needed.
- End-to-end: the instinct converts confident-wrong into correct-reroute **or** honest escalate —
  *as far as the detector can see the errors* (the whole payoff is detector-coverage-bounded).
- In-place injection can dominate re-route but is brittle (roofline); hybrid is best.
- A softer label-free reflex (membership instability) modestly but reliably beats exact match.

**Still open / honest limits (NOT shown):**
1. **Single seed, 1 judge-subject family per model, single run.** No CIs across seeds; the
   no-overclaim gate needs ≥3 seeds + ≥2 judge families + content decontam → not gate-eligible yet.
2. **One task** (`belief_revision_50`). `consequence_cascade` + a non-structural task untested.
3. **Claude-haiku had only 1 clean case** (base_error 0.98): its per-detector d′/AUC are not
   estimable — you cannot validate a detector on a model that fails the task ~entirely.
4. **The hard wall (hit from 3 sides): white-box access.** (a) Injection's real flip/corrupt rates,
   (b) a non-agreement predictive reflex (activation probe / logprob) to catch *confident* errors,
   (c) actual activation-steering — **all need model internals**, which the black-box hosted
   providers (and this sandbox) cannot give. This is a genuine boundary, not a gap to paper over.
5. **`agent/reflex_bus.py` is opt-in**, not adopted at any live call site.
6. **No `agi-proof/failure-ledger.md` entry was added** (it must stay in sync with
   `evidence-manifest.json` counts — a CI cross-check). Adding a formal ledger row for items 1–5 is
   a deliberate **follow-up** for whoever promotes this work.

---

## 4. ▶ Next step (single most valuable)

The science here has hit the white-box wall; the highest-value *new* result requires a GPU box
with an **open-weights model** (Spark/Mac farm or RunPod — see `spark-cluster-ops` /
`wisdom-gpu-prebaked`). Two concrete, pre-registered experiments:

**A. Non-agreement predictive reflex (closes the §3h ceiling).** On a white-box model, extract a
mid-layer **activation probe** (or logprob/perplexity confidence) as detector A′, run it through
`reasoning/instinct_reflex_eval.py` against `belief_revision_50`, and test:
*does A′ clear d′=1.0 where self-consistency (0.63–0.67) cannot, especially on confident errors?*
Pass bar: A′ AUC CI excludes chance **and** A′ separates the confident-wrong cases that agreement
signals miss (the haiku failure mode).

**B. Real activation-steering for the injection half (§3g).** Build a contrastive
"error-direction" vector; at the detected error step, add it at strength `s` and measure the real
`p_flip(s)` / `p_corrupt(s)`. Pass bar: confirm (or refute) the **brittleness roofline** —
an interior-optimal strength beyond which corruption dominates.

Both feed the *existing* harnesses (no new harness needed). Until a white-box box is available, the
cheaper alternatives are: **(C) gate rigor** — re-run DeepSeek+haiku at ≥3 seeds + add
`consequence_cascade`, report CIs, then a formal failure-ledger entry; or **(D) adopt
`agent/reflex_bus`** behind a flag at one guarded-loop call site.

---

## 5. Read-first list (for the next session)

1. `docs/06-Roadmap/Thinking-Chain-Intervention-and-Instinct.md` — full narrative + results.
2. `reasoning/README.md` (rows #6) — module map + the lane CLI.
3. `reasoning/instinct_validation.py` §3f sharp reading — *why B/B2 are verifiers not predictors*
   (the load-bearing caveat that defines the open frontier).
4. `reasoning/results/fusion_realmodel_*.json` — the only real-model measurements (with raw sets).
5. `agent/reflex_bus.py` — the opt-in component to adopt.
6. This file’s §3 (open limits) and §4 (next step).

---

## 6. Don't-break list (CI gates that must stay green)

- `python tools/lint_claims.py` — clean (no overclaims); keep candidate framing in any new prose.
- `compileall` + `pytest -q` — every new module has a test; all pass. The pytest-only test in
  `tests/test_reflex_bus.py` self-skips under direct `python` run but runs under CI pytest.
- Artifact-drift gates (`ci.yml`): unaffected (no generated artifacts changed). **If you add a
  `failure-ledger.md` row, also update `agi-proof/evidence-manifest.json`** or the count cross-check
  fails — see `ci-artifact-drift` skill.
- All `reasoning/instinct_*` modules are **deterministic across `PYTHONHASHSEED`** — preserve that
  (sort before float-summing over sets; use fixed per-stream offsets, never `hash(str)`).

---

## 7. Security note (operational)

Real-model runs used API keys the operator pasted into the chat (OpenRouter — ToS-blocked from the
sandbox; DeepSeek + an llmhub Claude proxy — worked). **Those keys are exposed in the session
transcript and should be rotated.** No key is stored in the repo: the runners read keys from env
vars only, and the committed result artifacts were secret-scanned (clean).
