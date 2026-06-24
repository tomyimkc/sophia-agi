# Honesty & Overclaim Remediation Plan (handoff)

**Audience:** another Claude/Codex session that will execute this.
**Author:** review session, against `main @ 9a86ac6` (2026-06-24).
**Goal:** remove the overclaim surface that threatens the project's credibility **without**
gutting its ambition or its genuine strengths. The failure ledger and fail-closed gate are
real assets — protect them.

> **Prime directive:** the README and all public copy must NEVER exceed what
> `agi-proof/failure-ledger.md` supports. The ledger is the single source of truth for claims.

---

## 0. Branding decision (DECIDED — apply it)

- **Drop "AGI" from the brand.** `Sophia AGI` → **`Sophia`**.
- **Brand line:** *Sophia — the Wisdom Gate.*
- **Mandatory subtitle wherever the brand appears:** *a provenance-aware, verifier-gated
  reasoning layer that abstains instead of fabricating.*
- "AGI" may appear ONLY as a research *direction* ("toward grounded AI"), never as a product
  label, repo tagline, or headline.
- Rationale: "wisdom" is a **value** claim (unfalsifiable, defensible); "AGI" is a
  **capability** claim (falsifiable, and the ledger already falsifies it). `Σοφία` already
  means wisdom, so the rename loses no meaning and sheds the liability.

**Touch points to change:** repo name/description (GitHub About), `README.md` H1 + hero,
`VISION.md`, `docs/00-Index/Home.md`, `agi-proof/README.md`, HF dataset card,
`docs/07-Growth/launch/github-repo-about.txt`, thesis-site content, badges.
Keep the `sophia-agi` repo slug if renaming the repo is too disruptive — but change the
**display** brand and all prose.

---

## 1. The critique (what is false / misleading / cannot-be-done)

### A. False or overreaching CLAIMS

1. **The name "Sophia AGI" contradicts every disclaimer.** You cannot disclaim in the body
   what you assert in the title. → fixed by §0.

2. **"AGI Missing Pillars — implemented" is the most dangerous overclaim.** Seven modules are
   named after the literal unsolved problems of AGI and described as "implemented … close
   major architecture gaps." The actual substance (code read at `9a86ac6`):

   | Module | Name implies | What it actually is |
   |---|---|---|
   | `agent/program_induction.py` | learns novel algorithms | template-matching over ~3 hand-coded fitters (affine/quadratic/string/grid) |
   | `agent/active_inference.py` | adaptive belief-update loop | a priority-sorted TODO list from report metadata |
   | `agent/planner_mcts.py` | System-2 planning under uncertainty | MCTS against a **scripted** simulator with hardcoded outcomes |
   | `agent/predictive_world_model.py` | learned world model | a `{(state,action): Counter}` lookup table over ~6 traces |
   | `agent/continual_plasticity.py` | safe online learning | a promotion scorecard that updates **no** weights |
   | `agent/layered_memory.py` | hierarchical cognitive memory | a permission-gated dict with token-overlap retrieval |
   | `agent/metacognition.py` | self-monitoring of reasoning | regex specificity scan + arithmetic; **no test file** |

   The `candidateOnly: true` flag does NOT neutralize the verb **"implemented."** These are
   *interfaces + toy reference implementations*, not the capabilities.

3. **Marketing layer broke the discipline.** Commit `9d24fbe` ("🤖 Generated with Grok")
   inserted claims the ledger does not support:
   - "the provenance gate that makes AI **safe to ship** … **trust in production without
     constant oversight**" — the validated result still leaves **23.6% hallucination after
     gating**. Not safe-without-oversight.
   - "the **only** open project shipping measured source discipline" — unfalsifiable
     superiority claim.

4. **The whole edifice rests on ONE narrow number.** "Validated proof" = Δ12.5%
   [5.6, 19.4] attribution-hallucination reduction, on **one 8B model** (dolphin-llama3),
   judged by **two LLMs**, on a **first-party benchmark + corpus**. The 0%-fabrication
   calibration result is on a **self-authored pack** (ledger admits it). Honest reading:
   *one first-party, LLM-judged benchmark on a small model.*

5. **"Self-improving honesty / flywheel" overstated.** It "closes offline" via **selection**,
   not a real weight update (ledger: *"replace selection with a live RL update (GPU)"*).
   Present-tense "writes + validates its own verifiers" implies learning that isn't happening.

6. **Anthropomorphic naming.** "Conscience," "metacognition," "moral parliament" are
   deterministic rule/regex/arithmetic layers. The names imply introspection the code does
   not perform.

7. **Vanity metrics.** "40+ MCP tools" and "spine for a solo-founder AI company / sovereign
   AI" are quantity/positioning claims with no deployment evidence.

### B. What Sophia genuinely CANNOT do (state plainly somewhere public)

- **Cannot verify anything live** — all retrieval/grounding is offline fixtures; no live
  source backend wired.
- **Cannot learn** — no weight updates; RLVR + flywheel are offline selection, not training.
- **Has never beaten a direct model on a third-party hidden eval** — every independent hidden
  run is 0/8 strict, backend-broken, or self-authored.
- **Cannot generalize** — program induction fits templates; no ARC-like novel-task result.
- **Cannot plan or act in the world** — planner runs against a mock simulator; no real
  tool-execution traces.
- **No reasoning, world-understanding, autonomy, or online adaptation** — it is a
  **fail-closed filter**, not a system that gets smarter.
- **No external replication** — benchmarks, packs, judges, corpus are overwhelmingly
  first-party.

### C. Critic's one-line verdict
Sophia is a **well-engineered, unusually honest epistemic *filter* and benchmark harness**
currently wearing an AGI costume. The engineering and the ledger are real; the "AGI" brand
and the toy "pillars" are a credibility time-bomb.

---

## 2. Remediation plan (priority-ordered, executable)

### P0 — Stop the bleeding (docs only, near-zero risk)

- [ ] **P0.1 Apply §0 rebrand** across all touch points. `Sophia AGI` → `Sophia — the Wisdom Gate`.
- [ ] **P0.2 Rewrite the two worst marketing sentences** (see §3 "say this instead").
- [ ] **P0.3 Add a CI "claims linter"** (`tools/lint_claims.py` + a CI step): fail the build if
      `README.md`, `VISION.md`, `agi-proof/README.md`, or `docs/07-Growth/**` contain any of
      `safe to ship | trust in production | the only | first AGI | breakthrough | proven AGI |
      makes AI safe` **unless** the same line cites a ledger ID (`ledger:<id>`). Provide an
      allowlist for legitimate uses. ~40–60 lines, deterministic, offline — matches repo style.
- [ ] **P0.4 Add a one-screen "What Sophia cannot do (yet)"** section to `README.md`, sourced
      verbatim from §1.B. Owning the limits makes them un-weaponizable.

**Acceptance:** `python tools/lint_claims.py` exits 0; README hero contains no capability
superlative; the "cannot do" section is present and links to the ledger.

### P1 — Reframe the pillars (don't delete; relabel honestly)

- [ ] **P1.1 Change the verb everywhere.** `docs/11-Platform/AGI-Missing-Pillars.md` and the
      report generator: "pillars **implemented**" → "fail-closed **interfaces + toy reference
      implementations** for AGI-shaped capabilities, with OPEN slots for real tasks."
- [ ] **P1.2 Add a "What this literally is" line** under each module in the doc (use the table
      in §1.A.2 verbatim). Self-stated toy-ness cannot be "exposed."
- [ ] **P1.3 De-anthropomorphize** `Conscience-Kernel.md` and module docstrings: annotate
      "conscience / metacognition / moral parliament" with "(a deterministic policy gate /
      confidence heuristic / multi-rule aggregator)."
- [ ] **P1.4 Add the missing `tests/test_metacognition.py`** (at least a shape/behaviour test;
      currently the module is untested).
- [ ] **P1.5 Ensure every pillar report keeps `candidateOnly: true`, `level3Evidence: false`**
      AND add a new field `"depth": "toy-reference"` so the artifact self-labels substance.

**Acceptance:** the word "implemented" no longer stands unqualified next to a pillar name;
each pillar carries a literal-substance line and a `depth` field; metacognition has a test.

### P2 — The substantive credibility unlock (real work, highest value)

- [ ] **P2.1 Land ONE live capability.** Wire the fact-check gate to a **live Wikidata +
      Crossref** backend (keyless), behind a flag, with an **offline fixture fallback so CI
      stays deterministic**. Run it on a **third-party-authored** claim set and report
      **fabrication rate + ECE**. This converts "designed to" → "measured against reality" and
      is already the next step in the ledger. *This is the single highest-ROI item.*
- [ ] **P2.2 Independently replicate the one number.** Execute the OPEN ledger items
      `calibration-self-authored-pack-2026-06-22` and `hidden-review-third-party-not-run`:
      commission a **third-party-authored abstain pack** + **human raters** (Prolific/Surge),
      and run the gate against a slice of an **external** hallucination benchmark (not the
      Sophia corpus). Report human-vs-scorer κ.
- [ ] **P2.3 Scope the headline precisely** until P2.2 lands: "validated: attribution-
      hallucination reduction on one 8B model, LLM-judged" — do not let it stand for "the
      system."
- [ ] **P2.4 FREEZE new pillar-naming.** No "Stage 2 pillars" until P2.1/P2.2 ship. Each new
      toy module increases the overclaim surface.

**Acceptance:** at least one result exists that uses a **live external source** OR a
**non-first-party judge/pack**, recorded in the ledger with CI-excludes-zero discipline.

### P3 — Reposition around the real moat

- [ ] **P3.1 Make honesty the brand.** Promote `agi-proof/failure-ledger.md` to a hero
      artifact (link it from the README hero). Ownable claim: *"the most rigorously falsified,
      fail-closed provenance gate in the open."*
- [ ] **P3.2 Retire capability positioning** ("spine for an AI company," "sovereign AI") until
      there is a deployment case study.

---

## 3. "Say this instead" — concrete rewrites

| Location | ❌ Current | ✅ Replace with |
|---|---|---|
| README H1 | `# Sophia AGI` | `# Sophia — the Wisdom Gate` |
| Repo tagline | "The verifiable foundation the first AGI requires." | "A provenance-aware, verifier-gated reasoning layer that abstains instead of fabricating." |
| README "What it does" | "the provenance gate that makes AI **safe to ship** … trust in production without constant oversight" | "a fail-closed provenance gate: it abstains instead of fabricating and only lets `accepted` output through. Measured effect: attribution-hallucination Δ12.5% (95% CI [5.6, 19.4]) on an 8B model — **23.6% remains; a filter, not a guarantee.**" |
| Differentiators | "the **only** open project shipping measured source discipline" | "an open project shipping *measured*, fail-closed source discipline — with a public failure ledger." |
| Pillars doc | "AGI Missing Pillars — implemented" | "AGI-shaped capability **interfaces** (toy reference implementations + OPEN slots)" |
| Flywheel | "Self-improving honesty — flywheel writes + validates its own verifiers" | "Offline self-extension demo: the loop *selects* (not yet trains) verifiers on a held-out split; live RL update is OPEN (needs GPU)." |

---

## 4. What NOT to do (guardrails)

- Do **not** weaken the fail-closed gate, BLP, or the no-overclaim aggregator gate.
- Do **not** delete the failure ledger entries — they are the asset. Only ADD honesty.
- Do **not** invent new validated numbers; only the aggregator's `_is_validated` may bless one.
- Do **not** commit secrets. (`.env`, `private/` stay gitignored. Rotate the OpenRouter,
  DeepSeek, llmhub keys used in development.)
- Do **not** rename code symbols so aggressively that you break imports/tests — prefer doc +
  docstring relabeling; reserve module renames for a dedicated, test-backed PR.
- Keep CI green: dependency-free `validate` job + full `pytest` `test` job must both pass.

---

## 5. Suggested execution order & PRs

1. **PR-1 (P0):** rebrand prose + two sentence rewrites + claims-linter + "cannot do" section.
2. **PR-2 (P1):** pillar reframing + de-anthropomorphizing + `test_metacognition.py` + `depth` field.
3. **PR-3 (P2.1):** live Wikidata/Crossref fact-check backend + offline fallback + third-party
   claim-set run recorded in the ledger.
4. **PR-4 (P2.2 / P3):** third-party pack + human-rater plan; promote ledger to hero.

Each PR: small, reviewable, CI-green, with the change reflected in BOTH the doc and the ledger.

---

## 6. Verification checklist (definition of done for the whole effort)

- [ ] No product surface says "AGI" as a label (only as research direction).
- [ ] `python tools/lint_claims.py` is wired into CI and passes.
- [ ] README has a public "What Sophia cannot do (yet)" section.
- [ ] No pillar is described as "implemented" without a literal-substance qualifier.
- [ ] `metacognition` has a test; all pillar reports carry `depth: toy-reference`.
- [ ] At least one result uses a live external source OR a non-first-party judge/pack.
- [ ] Both CI jobs green; failure ledger updated for every claim change.
