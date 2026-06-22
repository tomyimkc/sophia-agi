# MBTI Vector Agents — Spec A: Personality Measurement Gate + Level-1 Persona

**Date:** 2026-06-22
**Status:** Approved → ready for implementation plan
**Program:** "MBTI Vector Agents" — Spec **A** of 4 (A = this doc; B/C/D in §11)

## Problem

Sophia's whole identity is *refusing to assert what it cannot machine-check*.
"Personality" is the natural adversary to that discipline: it is the easiest
thing in AI to fake the **vibe** of (a system prompt that says "be extraverted")
and the hardest thing to **prove** moved. MBTI makes this sharper still — it is a
widely used framework with documented psychometric weakness (Pittenger 2005:
39–76% of takers change type on retest within weeks; SIOP advises against it for
selection).

That weakness is exactly why this belongs in *this* repo. Spec A builds the
machinery to declare a personality target, induce it with a Level-1 persona
prompt, **measure** what actually came out with a deterministic offline
instrument, and decide **faithful-vs-abstain** under the same no-overclaim gate
that governs provenance — so the system may only *claim* "this agent enacts trait
X" when a psychometric measurement **falsifiably confirms** it, and **abstains**
otherwise.

The missing artifact is a single sentence backed by a one-command run:

> A requested Big-Five trait shift is measurably present in the requested
> direction with effect size > d, replicated, halo-corrected, and capability-
> preserving — or the system abstains and says so. MBTI is only a display label
> over that measured substrate, never the thing being graded.

## The three central constraints

This design lives or dies on three boundaries. They are stated up front because
every component is shaped by them.

1. **MBTI is a one-way display veneer, never an input to judgement.**
   Big-Five (OCEAN) is the measured substrate. `mbti_to_ocean()` is allowed at
   the *request* boundary (to translate a user-facing "INTJ" into OCEAN target
   signs) and `ocean_to_mbti_letters()` at the *display* boundary. **No gate,
   verifier, effect-size, or abstention path may read the MBTI string.** A unit
   test asserts `personality_faithful` is byte-identical with the veneer on or
   off. This quarantines MBTI's invalidity from anything we report.

2. **Self-report is necessary but never sufficient (the validity threat).**
   A model can *say* "very accurate" to "I am the life of the party" without
   behaving extraverted. The headline metric therefore requires convergence of a
   deterministic self-report channel **and** an independent behavioral channel.
   Spec A ships the deterministic self-report half and **stubs-and-abstains** the
   behavioral half (Spec B produces it for real). Large self-report shift with a
   null behavioral channel resolves to **ABSTAIN ("claims, does not enact")**,
   not "faithful".

3. **Non-circularity: the scorer is arithmetic, the judge is not the subject.**
   Inventory scoring is a **pure function over a fixed published key** — zero
   model in the loop, zero circularity. Where a model *judge* is later used
   (behavioral channel, Spec B), it must be ≥2 provider families **distinct from
   the subject** (κ ≥ 0.40); judge==subject is illustrative-only.

## Scope of Spec A (decisions locked)

Per the brainstorming decisions:

- **Depth:** ship the pure-function OCEAN **scorer + golden response fixtures**
  (fully deterministic CI) **+ a `mock`-client smoke test** of the full
  induce→administer→score→verdict loop. Real hosted-model administration is
  deferred to Spec B. Spec A is **100% offline, deterministic, no-GPU**.
- **Surface:** include the **thin** MCP + portable Skill surface (2 read-only
  tools + one resource + one open-standard SKILL.md). Full server/packaging →
  Spec D.
- **Metric:** **pre-register the full PIF** (§6) in this spec as the headline
  contract; Spec A produces only the deterministic self-report half and abstains
  on the rest.
- **Benchmark scoring:** add **one new check key `mustExpressTarget`** with a
  matching `score_case` branch (honest about what the domain tests), rather than
  overloading `mustSignalNuanced`.

It is a **thin vertical slice across all four north stars** — measurement gate
(honest-steering proof), thin MCP+Skill (shippable tool), a 2-agent persona
demo (diverse council), and the Level-1 baseline that Spec B's activation
steering must *beat* (steering research).

## Components (each small, single-purpose, offline-testable)

| Module | Responsibility |
|---|---|
| `agent/personality_measure.py` (new) | **Measurement harness.** `score_items(responses, key) -> {dimensions, facets, sd, acquiescence_index}` is a **pure function** over the fixed IPIP-NEO-120 key (no model). `measure_ocean(client, *, bank, seed, k=1)` administers one item per **stateless** call (mock by default), parses one forced-choice token, reverse-keys, aggregates. |
| `agent/personality_map.py` (new) | **OCEAN substrate + MBTI veneer.** One-way `mbti_to_ocean("INTJ") -> {O:"high",C:"high",E:"low",A:"low",N:None}` validated against the 16-type allowlist; display-only `ocean_to_mbti_letters()`. Carries the verified r-value/sign table (§4). N is always `None`. |
| `agent/verifiers.py` (edit) | **`personality_faithful(spec=None) -> Verifier`** — mirrors `provenance_faithful`'s *shape* (three-way verdict enacted / contradicted / **abstained**, fail-closed; `_ok`/`_fail` helpers) but carries its own rule spec and **does not read `_PROVENANCE_FILES`**. Registered in `VERIFIERS`; optional module alias after the def (cf. `source_discipline = provenance_faithful`). |
| `tests/benchmark-personality.json` (new) | New **behavioral** benchmark domain `{version, domain:"personality", description, cases:[...]}`, each case with `id`, `question`, and the new `mustExpressTarget` check key. Mirrors `tests/benchmark-psychology.json`. |
| `agent/benchmark_checks.py` (edit) | Register `"personality"` in `DOMAIN_BENCH`; add a `personality` hint list to `infer_domain()`; add a `score_case` branch for `mustExpressTarget` following the `mustSignalNuanced` pattern. |
| `data/personality_types.json` (new) | 16-type keyed corpus (per-type OCEAN target vector + display copy). **Not** a provenance record (see ripple analysis §7) — powers the MCP resource and the Skill's canonical records. |
| `sophia_mcp/server.py` + `tools_impl.py` (edit) | Thin surface: tools `sophia_personality_target(mbti, ocean_json, prompt, model='mock', gate=True)` and `sophia_personality_faithful(text, mbti, ocean_json, model='mock')`; resource `mbti://types/{type}`. Logic in `tools_impl.py`; thin `@mcp.tool()` wrappers return `dumps(impl(...))`. All read-only/low-risk → **no `@audited`**. OCEAN vector passed as `ocean_json` (the existing `*_json` convention). |
| `skills/portable/sophia-personality-faithful/` (new) | One **open-standard** `SKILL.md` (name + description + body — works in Claude *and* Codex) + `references/facet-cues.md`, `references/frameworks.md` + `scripts/measure.py` (degrades to rules-only when the repo is absent). Parallels `sophia-source-discipline`. |
| `benchmark/personality_faithful.json` (new) | Verifier's own small corpus (`expectFaithful` booleans), mirroring `legal_holding_faithful.json` — keeps the verifier's cases separate from the scoring domain. |

## Instrument & administration protocol

**Chosen instrument: IPIP-NEO-120 (Johnson, 2014).** Public-domain *items*;
120 items = 30 facets × 4; 5-point Likert; negatively-keyed items reverse-scored
(`6 − raw`); facet = Σ 4 items, domain = Σ 6 facets. Validated on N≈320k; mean
facet reliability α > .80. Pure-function, deterministic scoring — no IRT, no
norms. *Fallback:* 50-item IPIP markers (domain-only) for a cheap smoke tier.
**Never** Mini-IPIP-20 as the authoritative gate (4 items/scale, α≈.60–.77,
fragile to one anomalous response).

**Verified corrections carried into the spec (do not regress these):**
- Cite the IPIP scoring key at **`ipip.ori.org`** directly for the
  raw-sum/public-domain/deterministic claims. The GitHub repo
  `chfhhd/ipip-neo-120-paper-pencil` is **CC-BY-SA 4.0** and its spreadsheet
  applies **normative standardization**, not raw sums — do not attribute raw-sum
  scoring to it.
- Spec A reports **within-system deltas only** (persona vs matched neutral
  baseline). **Never** human-norm percentiles ("80th-percentile Openness") — a
  human norm does not transfer to a model.

**Administration (Serapio-García 2023 + Jiang 2022 MPI + Persona Vectors 2025):**
1. One item per **stateless** request (no history) — kills order/anchoring.
2. **Neutral self-description framing** — avoid the words *test/survey/
   personality* to reduce evaluation-awareness skew (partial mitigation only;
   social-desirability bias is large and persists).
3. **Persona induced in a separate SYSTEM prompt**, kept out of the measurement
   turn (Persona-Vectors separation).
4. Decoding: prefer **log-prob argmax** over the 5 rating tokens; else constrain
   to `{1..5}`/`{A..E}`, strict-regex parse, retry out-of-set ≤3× then mark
   missing. `temperature=0, top_p=1`, fixed logged seed, pinned model snapshot.
5. Randomize item order **and** option-letter order behind the logged seed.
6. Reverse-key, mean per dimension, record SD + acquiescence index over the
   keyed-balanced subset.

**Verified determinism caveat:** hosted APIs are **not bitwise deterministic
even at temp 0** (batching/float non-associativity). Only the **pure-function
scorer** is the deterministic core; elicitation variance is treated as noise and
quantified with K replicates. (This is why Spec A's CI rests on golden fixtures +
the pure scorer, not on a live model.)

**Citation-precision flags (honesty ledger):** Serapio-García's *primary* bank
was IPIP-NEO-**300** (BFI-44 secondary) — choosing 120 is *our* call, not theirs.
Persona Vectors is steering-vector research; we borrow its induce-then-evaluate
*separation* by analogy, not as a stated self-report protocol. Option-letter
randomization is our addition.

## MBTI ↔ OCEAN mapping (verified)

Source: **McCrae & Costa (1989)**, *J. Personality* 57:17–40 (N≈468). Convention
is keyed to the **second letter** (I, N, F, P).

| MBTI axis | OCEAN factor | r (1989 point estimate) | Direction | Confidence |
|---|---|---|---|---|
| E/I | Extraversion | −0.74 (men −0.74, women −0.69) | E → high E; I → low E | highest |
| S/N | Openness | +0.72 ⁽¹⁾ | N → high O; S → low O | highest |
| T/F | Agreeableness | +0.44 | F → high A; T → low A | **lowest** (sex-confounded; 0.33–0.44) |
| J/P | Conscientiousness | ≈ −0.48 to −0.49 | J → high C; P → low C | moderate–high |
| (secondary) J/P | Openness | +0.30 | P → modestly higher O | secondary |

⁽¹⁾ **+0.72 is the 1989 point estimate, not a pooled value** — replicated/pooled
SN–Openness runs lower (~0.60–0.65). Label it as the 1989 value.

🚩 **Neuroticism gap (by design).** MBTI maps to **none** of Neuroticism — all
four dichotomies correlate near-zero (max |r|≈0.16, ~2.6% shared variance). A
4-letter code constrains only **4 of 5** OCEAN factors (E, O, A, C). **N is
undetermined by the code** → left `None`/unspecified (or supplied separately);
**never inferred from any MBTI letter.** The "-A/-T" (Assertive/Turbulent) suffix
is 16Personalities' relabeled emotional-stability axis bolted on to fill this
gap — **not** classic Myers-Briggs / McCrae-Costa.

**Worked example — INTJ → O=HIGH, C=HIGH, E=LOW, A=LOW, N=unspecified** (E and O
most confident; A least confident). General rule: E→E:high / I→E:low; N→O:high /
S→O:low; F→A:high / T→A:low; J→C:high / P→C:low; Neuroticism always unspecified.

## Data flow

```
request (MBTI code and/or OCEAN target vector, optional prompt)
  │  validate ── MBTI: 16-type allowlist; OCEAN: keys/ranges
  │             mbti_to_ocean()   [one-way; N left unspecified]
  ▼
induce Level-1 persona ── trait description in a SEPARATE system prompt
  ▼
administer inventory ── IPIP-NEO-120, one item / stateless call, neutral
  │                     framing, temp 0, logprob-argmax / constrained decode
  ▼
score OCEAN ── pure-function reverse-key + mean per dim (+facets, SD,
  │            acquiescence index)   [NO model in the loop]
  ▼
compare to target ── per-axis delta vs matched NEUTRAL baseline (same bank);
  │                  residualize target axis against off-target vector (halo)
  ▼
faithful / abstain ── personality_faithful: enacted | contradicted | abstained
  │                   (fail-closed; abstain unless the validated bar is met)
  ▼
benchmark report ── score_case → leaderboard-personality.json;
                    MCP/Skill surface returns gated, framed output
```

**Spec A produces** the deterministic self-report path (induce → administer →
score → delta → verdict) against `mock` + golden fixtures. The behavioral
cross-check and external-judge channels are **interfaces stubbed and abstained-on**
in Spec A.

## Headline metric — Persona-Induction Fidelity (PIF), pre-registered

**PIF** = count, out of **N ≥ 8** personas (spanning all five OCEAN poles), whose
**requested target axis** shifts in the requested direction with **residualized
Cohen's d > 0.5** (off-target halo removed), where the shift:

- **(a)** replicates across **K ≥ 20** seeds (randomized item + option order),
  bootstrap 95% CI excluding zero **after Holm/BH** correction across all
  axis × persona tests;
- **(b)** is corroborated in **both** a deterministic self-report family **and**
  an independent **behavioral** battery judged by **M ≥ 2** provider families
  **distinct from the subject** (inter-judge κ ≥ 0.40);
- **(c)** replicates on a **held-out** inventory + behavioral family invisible to
  the persona/optimizer (anti-gaming);
- **(d)** **preserves capability** within a pre-registered tolerance ε.

Reported as **count-out-of-N with a binomial 95% CI** + per-axis abstention rate.
**Never** a single aggregate trait score; **never** "MBTI type achieved." The
verifier mirrors `legal_holding_faithful` (three-way, fail-closed); a
`require_enactment` flag makes abstention a hard fail.

**Falsifiability:** PIF is falsified for a persona/axis if any pre-registered
threshold fails (CI includes zero; self-report vs behavior disagree; shift
vanishes held-out; off-target moves as much as target; capability drops). As a
count under stricter judges, PIF can only go **down** under more adversarial
measurement — it cannot be inflated by framing.

**Abstain conditions (each flips a cell to ABSTAIN, lowering the count):**
1. Target-axis CI includes zero after K, or fails Holm/BH.
2. Self-report shifts but behavior is null/opposite → "unfaithful self-report".
3. Shift on seen but absent on **held-out** family → suspected gaming.
4. Off-target axes move as much as target → halo / social-desirability.
5. Capability/coherence outside ε → caricature.
6. Same-family judges, judge family overlaps subject, or κ < 0.40 → illustrative.
7. Temp-0 single administration (n=1, no variance) → cannot compute CI.
8. Request framed "type achieved," or any gate input reads the MBTI veneer.
9. Any absolute trait level vs human IPIP-NEO norms → only within-system
   persona-vs-neutral **deltas** are reportable.

**Spec A's honest output:** with a `mock` client and golden fixtures, Spec A
proves the **measurement plumbing** — the loop correctly computes per-axis
deltas, applies reverse-keying/halo-residualization, and emits the three-way
verdict — while conditions 2/3/5/6 force ABSTAIN because their channels do not
exist yet. Spec A therefore reports **plumbing-grade** results (deterministic,
fixture-driven) plus an abstention ledger; the first *illustrative* numbers on
real models arrive in Spec B, and headline-grade PIF in Spec C. Spec A
pre-registers PIF; it does not claim one.

## Repo integration points

**Corpus ripple — DOES NOT apply.** `personality` is a **behavioral** benchmark
domain: no `data/*.json` provenance record, no `data/domains.json` entry, no
`dataFile`, no wiki emit, never scanned by `lint_wiki_provenance`. **Must NOT**
add a `data/domains.json` entry or an `okf/schema.py` domain (either would
wrongly trigger the wiki-emit/provenance ripple).

**Benchmark-side ripple — DOES apply (required).** Adding cases requires:
(a) a `case_map.json` entry per case → (b) a `training/examples/NNN-*.json` whose
assistant text passes each case (its text becomes the teacher reference) →
(c) the teacher reference must score **100%** for `personality`, enforced by
`test_benchmark_scorer.py`.

**Required changes** (symbols are stable; line numbers approximate):

| File | Change |
|---|---|
| `tests/benchmark-personality.json` | **NEW** — domain spec mirroring `benchmark-psychology.json`, cases use `mustExpressTarget`. |
| `agent/benchmark_checks.py` | Register `"personality"` in `DOMAIN_BENCH` (~L105); add a `personality` hint list to `infer_domain()` (~L251); add a `score_case` branch (~L164–237) for `mustExpressTarget` modeled on `mustSignalNuanced` (~L203). |
| `agent/verifiers.py` | Add `personality_faithful` factory (model on `provenance_faithful` ~L615; helpers `_ok`/`_fail` ~L42; **must not** read `_PROVENANCE_FILES`); register in `VERIFIERS` (~L811); optional alias (cf. ~L717). |
| `benchmark/reference/case_map.json` | Add top-level `"personality": { case_id: "NNN-*.json", ... }`. |
| `training/examples/NNN-personality-*.json` | **NEW**, one per case; assistant text passes the case (teacher reference). |
| `tools/run_benchmark.py` | Add `"personality"` to `DOMAINS` (~L12) **and** a seed entry (~L78–83). |
| `tools/update_leaderboards.py` | Add `"personality"` to `DOMAINS` (~L13) → generates `leaderboard-personality.json` in CI. |
| `tools/model_lab_lib.py`, `tools/rescore_model_runs.py`, `tools/build_agi_proof_package.py`, `tools/build_web_data.py`, `tools/prepare_lora_dataset.py`, `tools/run_external_models.py` | Add `"personality"` to each independent hardcoded `DOMAINS` tuple. |
| `tests/test_benchmark_scorer.py` | Add `'personality'` to the domain tuple (~L78) in the teacher-reference-100% test. |
| `tests/test_verifiers.py` | Add a `personality_faithful` unit test (in-character → True; out-of-character → False with expected reason), modeled on the `provenance_faithful` test. |

**MCP / Skill (new surface):** `sophia_mcp/server.py` (2 thin tools + 1 resource),
`sophia_mcp/tools_impl.py` (`personality_target`, `personality_faithful`,
`mbti_type_record`), `data/personality_types.json`,
`skills/portable/sophia-personality-faithful/`, optional
`skills/registry/personality-faithful.json`, `benchmark/personality_faithful.json`.

**Should-skip:** `agent/rag_sources.py` (only if a personality RAG corpus is
wanted — not now); `okf/schema.py` (skip — would imply a provenance domain);
`tools/hidden_eval_protocol.py` (seal only once the held-out family exists, Spec C).

## Risks & mitigations (all machine-gated)

| Risk | Sev | Mitigation (Spec A posture) |
|---|---|---|
| **Self-report unfaithfulness** (central validity threat) | high | Never let self-report suffice. PIF requires self-report **AND** behavioral convergence; large self-report + null behavior → ABSTAIN. Behavioral channel stubbed-and-abstained in A. |
| **Trait entanglement / halo** | high | Orthogonality null-band: target success = requested shift **AND** off-target \|d\|<0.2. Report full 5-vector delta + covariance; **residualize** target against off-target. |
| **Determinism vs sampling variance** | high | K≥20 per condition; Cohen's d + bootstrap CI. Temp-0 = reproducibility check only. CI∋0 → ABSTAIN. CI rests on the pure scorer + golden fixtures. |
| **MBTI psychometric invalidity** | med | Hard boundary: MBTI is one-way display only; veneer-invariance unit test; documented in SECURITY/spec; never "type achieved." |
| **Reward-hacking the gate** | high | Held-out anti-gaming (selfextend pattern): promote "faithful" only if the shift replicates on a held-out inventory+behavior invisible to the persona. (Spec C.) |
| **Capability/coherence retention** | med | Capability-retention guardrail (e.g. GSM8K slice + judge-coherence) neutral vs persona; require drop < ε. Headline the (shift, retention) pair. (Spec D.) |
| **Small-sample / false discovery** | high | Pre-register primary (target-axis) vs secondary (off-target); unit = per-condition mean over K; Holm/BH across all tests; aggregate count + binomial CI, not per-cell p. |
| **Circularity (judge==subject)** | high | Scorer is pure arithmetic (zero circularity). Judges (Spec B+): ≥2 families distinct from subject, κ≥0.40; same-family = illustrative. |
| **Human-norm leakage** | med | Forbid percentile claims; within-system persona-vs-neutral deltas only. |

## Reuse, don't rebuild

Built on existing pieces: the `Verifier` factory pattern + `_ok`/`_fail`
(`agent/verifiers.py`), the `DOMAIN_BENCH`/`score_case` benchmark machinery
(`agent/benchmark_checks.py`), the scorer/leaderboard tooling
(`tools/score_benchmark.py`, `run_benchmark.py`, `update_leaderboards.py`), the
`agent.model` multi-provider adapter incl. `mock`, the MCP server conventions
(`sophia_mcp/`), and the portable-skill template (`sophia-source-discipline`).
**New code is additive**: two `agent/` modules, one verifier, one benchmark
domain + cases, two data files, the thin MCP surface, and the portable skill.

## Testing

Every module offline-testable with `mock` and golden fixtures — **no API calls in
CI**:
- `test_personality_scorer` — pure `score_items` on golden responses → known
  OCEAN vector (reverse-keying, facet/domain sums, acquiescence index).
- `test_personality_map` — `mbti_to_ocean` for all 16 types; N always `None`;
  invalid code → allowlist error; **veneer-invariance**: `personality_faithful`
  identical with/without the MBTI label.
- `test_verifiers::personality_faithful` — in-character pass; out-of-character
  fail with reason; unmeasured channel → `abstained`.
- `test_benchmark_scorer` — teacher reference scores **100%** for `personality`;
  `mustExpressTarget` branch behaves.
- `mock` integration smoke — full induce→administer→score→verdict loop returns a
  delta and a three-way verdict.
- MCP tools/resource round-trip via `tools_impl` with `model='mock'`.

## Scope guardrails (YAGNI)

- **No live hosted-model administration** in Spec A (Spec B). Mock + fixtures only.
- **No behavioral battery / external judges** built here — interface only,
  returns `abstained` with `notValidated` (Spec B).
- **No held-out anti-gaming family, no hidden-eval sealing** (Spec C).
- **No capability-retention harness** (Spec D).
- **No activation steering / persona vectors / GPU** — Level-1 persona prompt is
  the only induction method in Spec A; it is the *baseline Spec B must beat*.
- **No `data/domains.json` / OKF entry** — would wrongly trigger the corpus ripple.
- **No persisted personality records** (a "persist a personality" tool would be
  `@audited(risk='medium')` + approval-gated — intentionally absent here).

## Deferred to Specs B / C / D

- **Spec B — Level-3 activation-steering engine** (PyTorch + MPS): CAA /
  persona-vector difference-of-means at ~⅔ depth, `register_forward_hook`
  steering (float32), axis composition + interference + orthogonalization,
  calibration. Plus the **behavioral half of PIF** (open-ended battery, ≥2
  judge families distinct from subject, κ≥0.40) and live hosted-model
  administration with logprob scoring. *Must beat the Spec A Level-1 baseline.*
- **Spec C — Personality-diverse council**: wire steered/persona agents into the
  existing council machinery; measure whether trait diversity **improves**
  deliberation vs a homogeneous panel (null result reported honestly). Adds the
  **held-out anti-gaming family**, trap items, hidden-eval sealing, and the
  selfextend held-out false-accept check. Full PIF at headline grade.
- **Spec D — Portable Skill + full FastMCP packaging**: complete server
  (tools/resources/prompts), MCP security gating per the repo's posture,
  cross-platform (Claude + Codex) validation. Adds the **capability-retention
  guardrail**.

## Residual uncertainty (honesty ledger)

1. SN–Openness r=+0.72 is a single 1989 point estimate, not pooled (~0.60–0.65
   in replication); J/P–C is −.48/−.49 across sources.
2. T/F–Agreeableness (r=+0.44) is the weakest, sex-confounded mapping — the
   lowest-confidence target axis.
3. Neuroticism is **not** inferable from MBTI; it must be supplied separately.
4. Cite `ipip.ori.org` directly for raw-sum/public-domain/deterministic claims
   (the `chfhhd` GitHub repo is CC-BY-SA and applies norms).
5. Hosted APIs are not bitwise-deterministic even at temp 0 — only the
   pure-function scorer is the deterministic core.
6. Two of the workflow's fact-check sub-claims (E/I r-value precision; admin
   protocol determinism) hit a transient rate limit and were not independently
   re-verified; both are corroborated by the primary McCrae-Costa / Serapio-García
   sources and flagged here for a confirming pass during implementation.

## Sources (primary)

- McCrae & Costa (1989), *Reinterpreting the MBTI from the perspective of the
  five-factor model*, J. Personality 57:17–40.
- Johnson (2014), *Measuring thirty facets of the FFM with a 120-item public-
  domain inventory: IPIP-NEO-120*, J. Research in Personality. Scoring key:
  `ipip.ori.org`.
- Serapio-García et al. (2023), arXiv:2307.00184 (administration protocol).
- Jiang et al. (2022), *MPI*, arXiv:2206.07550 (item format).
- Chen, Arditi, Sleight, Evans, Lindsey (2025), *Persona Vectors*,
  arXiv:2507.21509 (induce/evaluate separation; Spec B steering recipe).
- Pittenger (2005); SIOP guidance (MBTI psychometric-validity caveats).
