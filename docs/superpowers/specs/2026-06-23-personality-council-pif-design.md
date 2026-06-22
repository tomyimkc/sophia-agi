# MBTI Vector Agents — Spec C: Personality-Diverse Council + Held-Out Anti-Gaming/Sealed Eval + Headline-Grade PIF

**Date:** 2026-06-23
**Status:** Approved → ready for implementation plan
**Program:** "MBTI Vector Agents" — Spec **C** of 4 (stacks on Spec B; D in §12)
**Branch:** `feat/personality-council-pif` (off `feat/activation-steering-pif` / Spec B PR #66)

## Problem

Specs A and B built the measurement gate and the steering engine, and B
delivered a **decisive null**: activation steering does **not** beat a persona
prompt (illustrative SSA **0/2** on granite-2b). Spec C hardens that result to
headline grade and answers the program's last open *capability* question — does
**personality diversity** make a multi-agent council deliberate better? — while
adding the **anti-gaming machinery** the repo's ethos demands.

It delivers three related pieces, kept conceptually distinct (each CI-green
before the next) but shipped as one cohesive spec/PR:

- **C1 — Personality-diverse council:** does a trait-diverse panel out-deliberate
  a homogeneous one? (Expected: **null** — and a rigorous null is the finding.)
- **C2 — Held-out family + sealed hidden-eval + anti-gaming ABSTAIN:** a held-out
  inventory/battery the steering fit never sees, sealed and tamper-evident, with
  the selfextend "promote-or-ABSTAIN on held-out" contract mirrored onto steering.
- **C3 — Headline-grade PIF/SSA harness:** the full pre-registered N≥8 × 4-axis ×
  K≥20 grid with residualized-d + Holm/BH correction — **built CI-green; the full
  hours-long real run deferred** (it would re-confirm B's known null).

## The central constraints

1. **Two-tier execution, mirroring `tools/run_rlvr.py` / Spec B.** A
   **deterministic CI core** (pure stdlib: the held-out disjointness check, the
   sealing commitment, the new stats helpers, the synthetic-cell SSA assembly +
   ABSTAIN logic, and the council A/B on a mock client) is the **shipped,
   CI-verified contribution**. **Opt-in real runs** are cheap (C1 council A/B,
   C2 reduced slice) or **deferred** (C3 full K≥20, OPEN in the failure ledger).

2. **A null is a legitimate, publishable result.** C1 is expected to show trait
   diversity does *not* help; C3's headline is expected near-null confirming B.
   The contribution is the **rigorous, pre-registered, multiple-comparison-
   corrected** machinery and the honest number — never an inflated claim.

3. **MBTI veneer-invariance is inherited** (A+B): personas are chosen upstream
   from OCEAN signs; no gate/verifier/judge/effect-size path reads an MBTI string.

4. **Non-circularity & anti-theatre.** The C1 judge (deterministic lexical
   `score_case`) cannot collude with the seats; for any LLM-judged path the judge
   family must be disjoint from the subject/seats. The held-out family must be
   **construct-disjoint** (not just string-disjoint) from B's "seen" surface, and
   a CI **grep-gate** forbids any plaintext held-out answer from entering the repo
   or logs. If true third-party sealing is unavailable, the claim is downgraded to
   "held-out + pre-registered manifest hash" — never call an in-repo eval "sealed".

## Locked decisions (owner)

- **One Spec C, one PR**, with C1→C2→C3 as ordered task-groups (each CI-green
  before the next).
- **Build everything CI-green + reduced real demos**: C1 council A/B runs for
  real (cheap, deterministic judge); C2 runs a reduced real slice to demonstrate
  the seen→shift / held-out→ABSTAIN signature; C3 harness is CI-green + a tiny
  reduced confirmation. **The full N≥8/K≥20 PIF run is DEFERRED** (pre-committed
  command + seed manifest, OPEN in the ledger).
- **Council seats are Level-1 persona-prefixed** (offline, deterministic,
  CI-gateable). The Level-3 `SteeredClient` drop-in ships as a documented future
  hook; the steered-council arm is **deferred to D**.
- **Deterministic lexical judge** (`score_case`) for the C1 headline; LLM panels
  only behind the `_is_validated` gate.
- **Sealing = Option A**: reuse `tools/hidden_eval_commitments.py`
  (`case_digest`/`build_commitments`) for the salted per-case SHA-256 commitment;
  do **not** route through `score_pack` / add `personality` to its `DOMAINS`.

## Components

### C1 — Personality-diverse council
| Module (new) | Responsibility | Interface | Reuse |
|---|---|---|---|
| `tools/run_council_diversity.py` | Run DOMAIN_BENCH gold cases through `deliberate()` in three matched arms (single / homogeneous-persona / diverse-persona) and report `ΔQ = passrate(diverse) − passrate(homogeneous)` + paired-bootstrap CI. | `python tools/run_council_diversity.py [--model mock] [--council <id>]` | `agent/council_deliberate.py::deliberate(query, *, client, seat_clients=…)` (the **verified `seat_clients` seam**), `agent/benchmark_checks.py::score_case` (deterministic judge), `provenance_bench/aggregate.py::_ci`. |

Seats carry personality via a thin **persona-prefix client wrapper**: each
seat client wraps the base client so `generate(system, user)` →
`base.generate(persona + "\n\n" + system, user)`, where `persona` is an OCEAN
profile string (built like `measure_ocean`'s `persona` seam / `run_steering.py`'s
persona prefix). Diverse arm = N distinct OCEAN profiles; homogeneous arm = N
copies of one profile; single arm = the bare base client. **Chair, gate
(`_seat_gate` → `agent/gate.check_response`), route, max_seats, and temp-0
decoding are identical across arms** — the only manipulated variable is the OCEAN
spread.

### C2 — Held-out family + sealing + anti-gaming
| Module (new) | Responsibility | Interface | Reuse |
|---|---|---|---|
| `data/personality_items_heldout.json` | Second IPIP inventory (`ipip-spec-c-heldout`), same `{id,text,domain,keyed}` schema, `ho_*` ids, ≥4 markers/domain (≥2/pole), **zero lexical overlap** with the seen 10. | `load_bank(path=…)` | `agent/personality_measure.py` unchanged |
| `data/behavioral_battery_heldout.json` | Second battery (`ocean-behavioral-battery-heldout-v0`), **situationally-disjoint** trait-name-free prompts. | `load_battery(path=…)` | `agent/personality_behavioral.py` unchanged |
| `provenance_bench/heldout_split.py` | `held_out_disjoint()` — pure-stdlib, deterministic: asserts (a) seen∩held-out ids == []; (b) no shared content 3-gram (stopword-stripped) above threshold; (c) the fit module **never imports** the held-out path constants (greps the module). Returns `{seen_sealed, heldout_sealed, ipip_intersection, ngram_overlaps, fit_reads_heldout:false, nearest_neighbour_sim}`. | `held_out_disjoint() -> dict` | mirrors `steering_dataset._sealed`/`item_intersection` |
| sealing | Build a private personality pack under `private/hidden-evals/` (gitignored) with a secret salt + the full scoring/keying config in `scoring`; `build_commitments` emits only public per-case `{id,domain,sha256}` to `agi-proof/hidden-reviewer-packs/personality-heldout-<date>.commitments.json` (`saltStatus:"withheld until reveal"`). | `tools/hidden_eval_commitments.py::case_digest`/`build_commitments` | reused unchanged |
| anti-gaming ABSTAIN | The selfextend false-accept mirror, implemented in `agent/steering/anti_gaming.py` and unit-tested; it is wired into the live ship path when the deferred K≥20 PIF run lands (intentionally not yet called by `pif_harness.py`/`run_pif.py` because no live ship decision exists until the deferred run): fit on the SEEN split, then on the HELD-OUT family **promote-or-ABSTAIN**. | `verdict` fields below | conceptual mirror of `selfextend/{verifier_synthesis,flywheel,verified_reward,loop}.py` |

**Anti-gaming contract (mirrors selfextend verbatim in spirit):** map *verifier
synthesis → vector/coefficient fitting*; *held-out validation → held-out family*;
*promote-or-abstain → ship-steering-or-ABSTAIN*. Ship a steering vector only if
ALL invariants hold (else **ABSTAIN**, log to an abstention record, route stays
"abstain"): `target_moved_vs_baseline` on held-out targets; `generalizes_not_gamed`
= **`(fit_shift − held_shift) ≤ 0.20`** (the literal "shift-on-seen-but-absent-on-
held-out → ABSTAIN" firewall, mirroring `reward_is_hackable` gap=0.2);
`off_target_clean` = `heldoutOffTargetRate ≤ 0.10` (the fraction of held-out
off-target axes whose `|d|` exceeds the 0.20 null band — the "0% held-out
false-accept" analogue, measured ONLY on out-of-sample held-out negatives);
capability drop `< 5%` and coherence `≥ 75` preserved.
**An unvalidated steering vector is worse than none.**

### C3 — Headline-grade PIF/SSA harness
| Module (new/extend) | Responsibility | Interface | Reuse |
|---|---|---|---|
| `agent/steering/pif_harness.py` (new) | Assemble the pre-registered grid into `ssa_verdict` cells; the **pure `build_cells_from_scores()` seam** takes pre-computed per-seed score arrays → fully CI-testable with synthetic data, no model. | `build_cells_from_scores(scores, grid) -> list[cell]` | `agent/steering/stats.py` |
| `agent/steering/stats.py` (**extend in place, do NOT fork**) | Add three pure-stdlib helpers: `residualized_d` (regress target on off-target movement across seeds via OLS normal equations → the residualized `steered_d` `SSA_THRESHOLDS` already names but B never computed), `holm_bonferroni`, `benjamini_hochberg`. | three new fns | existing `cohen_d`/`bootstrap_diff_ci`/`binarize_moved`/`cohen_kappa`/`ssa_verdict`/`SSA_THRESHOLDS` |
| `tools/run_pif.py` (new) | Driver: `--dry-run`/`--model mock` = CI core (synthetic cells); `--model <hf id>` = opt-in heavy run. | `python tools/run_pif.py …` | mirrors `run_steering.py` two-tier; reuses `measure_ocean`/`score_behavioral` with `path=` held-out families |

**Pre-registered grid (fixed before any run):** Personas: **N≥8** from
`data/personality_types.json` (each of O/C/E/A high AND low in ≥3 personas).
**N (neuroticism) is null for most MBTI codes → excluded as a steered target,
retained as an off-target null axis.** Target axes: **O, C, E, A**. **K≥20** seeds
per (persona, axis, condition). Conditions: steered / level1 / neutral; the cell
contrast is `Δd = d(steered vs neutral) − d(level1 vs neutral)` — the same as the
K=2 demo, now with K≥20 paired seeds so the bootstrap CI is **real** (the demo's
degenerate `[point, point]` interval disappears). A cell counts toward the
"enacted" tally only if **both** `ssa_verdict == "enacted"` **and** it survives
**BH at q=0.05**.

## Data flow

```
                       ┌── C1: council ─────────────────────────────────────────┐
DOMAIN_BENCH gold cases ─→ deliberate() in 3 arms (single/homo/diverse persona seats)
                          → score_case (deterministic) → passrate per arm
                          → ΔQ = diverse − homo, paired-bootstrap CI → NULL-or-PASS
                       └────────────────────────────────────────────────────────┘
                       ┌── C2+C3: held-out PIF ─────────────────────────────────┐
fit CAA vector on SEEN carriers (B) ─→ FREEZE alpha
   → held_out_disjoint() gate (disjoint + sealed) ──── ABSTAIN if not disjoint
   → administer on HELD-OUT IPIP + HELD-OUT battery (measure_ocean/score_behavioral, path=)
   → per (persona×axis×K-seed): steered/level1/neutral scores
   → build_cells_from_scores(): residualized d, bootstrap Δd CI, off-target band, κ, capability
   → anti-gaming: (fit_shift − held_shift) ≤ 0.20 AND heldoutOffTargetRate ≤ ε  else ABSTAIN
   → ssa_verdict per cell + Holm/BH → headline enacted/total (expected near-null)
                       └────────────────────────────────────────────────────────┘
```

## Headline metrics + abstain conditions (falsifiable; null OK)

**C1 — Trait-Diversity Deliberation Lift (TDDL):** *"A trait-diverse council
(N seats, mean-pairwise-OCEAN-distance > 0) achieves a HIGHER held-out pass rate
than a matched homogeneous council (same N, one OCEAN profile, same chair/gate/
model/temp-0) on DOMAIN_BENCH — ΔQ > 0 with a paired-bootstrap 95% CI excluding
0."* **PASS** iff ΔQ > 0 ∧ CI excludes 0 ∧ sign test consistent; else **NULL**.
Report ΔQ vs SINGLE too (separates "council helps at all" from "diversity helps").
**Expected NULL** → honest headline: *"trait diversity does not improve council
quality on this slice; the council's value is the gate + synthesis, not OCEAN
spread."*

**C3 — PIF/SSA:** enacted tally = cells with `ssa_verdict=="enacted"` AND
surviving BH at q=0.05; headline = enacted/total. Machine-checked abstain reasons:
`steer_not_beats_baseline`, `below_floor` (residualized d), off-target halo
(≥0.20), κ<0.40, capability_drop≥5%, coherence<75, **`(fit_shift−held_shift)>0.20`**,
`heldoutOffTargetRate>0.10`. **Expected near-null** confirming B's 0/2 — *the
rigorous, pre-registered, multiple-comparison-corrected negative result IS the
contribution.*

**Cross-cutting (all real runs):** the gate is `provenance_bench.aggregate.
_is_validated` (notMock + ≥2 judge families + κ≥0.40 + ≥3 runs + 95% bootstrap CI
excludes 0). A single/mock run is illustrative, never a headline.

## Testing discipline

**CI core — `tests/test_pif_harness.py`** (plain script, no pytest, `test_steering.py`
style):
1. `held_out_disjoint()` → empty intersections, stable dual seals, `fit_reads_heldout:false`.
2. `residualized_d` / `holm_bonferroni` / `benjamini_hochberg` vs **hand-computed** values.
3. `build_cells_from_scores()` → **enacted** on a planted-strong synthetic family;
   **abstain** with the correct reason on a null family; a borderline cell that
   passes per-cell `ssa_verdict` but is **killed by BH**. (The null family is the
   realistic case — CI proves the harness correctly abstains and says why.)
4. `run_pif.main(["--dry-run"])` writes a public report and exits 0.

**Council CI — `tests/test_council_diversity.py`** (or folded into the above):
`run_council_diversity.py --model mock` is deterministic; CI asserts plumbing
runs, **judge family ∉ seat families** (fail-closed), homo-arm diversity = 0,
diverse-arm > 0, ΔQ + CI computed.

**Sealing CI:** re-running `case_digest` reproduces the committed sha256; the
commitments file has no salt; a **grep-gate** asserts no plaintext held-out answer
in the tree or logs.

**Opt-in real runs** (never asserted by exact value): `--model <hf id>` + reduced
slices are gated/illustrative; only `_is_validated` decides "validated"; seed and
record all stochastic choices. The full headline run is logged **OPEN** in the
failure ledger.

## Repo integration (file-by-file)

**New** (under the `feat/personality-council-pif` worktree):
`tools/run_council_diversity.py`, `data/personality_items_heldout.json`,
`data/behavioral_battery_heldout.json`, `provenance_bench/heldout_split.py`,
`agent/steering/pif_harness.py`, `tools/run_pif.py`,
`tests/test_pif_harness.py` (+ council tests),
`agi-proof/hidden-reviewer-packs/personality-heldout-<date>.commitments.json`,
`agi-proof/benchmark-results/council-diversity.public-report.json`, a new
`agi-proof/failure-ledger.md` entry for the OPEN heavy PIF run, and a gitignored
`private/hidden-evals/` for the unsealed held-out pack + salt.

**Extend in place (do NOT fork):** `agent/steering/stats.py` (+`residualized_d`,
`holm_bonferroni`, `benjamini_hochberg`), `.github/workflows/ci.yml` (wire the new
test files), `.gitignore` (add `private/`).

**Reused unchanged:** `agent/personality_measure.py` (`load_bank`/`score_items`/
`measure_ocean`, `path=`), `agent/personality_behavioral.py` (`load_battery`/
`score_behavioral`, `path=`), `agent/steering/stats.py` (existing fns),
`agent/steering/hooks.py` (`SteeredClient`, for the deferred steered arm),
`provenance_bench/steering_dataset.py` (`_sealed`/`item_intersection`),
`tools/hidden_eval_commitments.py` (`case_digest`/`build_commitments` — **verified
to exist**), `agent/council_deliberate.py` (`deliberate`, **`seat_clients` seam
verified**), `agent/benchmark_checks.py` (`score_case`, `DOMAIN_BENCH`),
`provenance_bench/aggregate.py` (`_ci`, `_is_validated`), `data/personality_types.json`.

## Risks & mitigations

| Risk | Sev | Mitigation |
|---|---|---|
| **C1 null mistaken for a bug** | med | Pre-register the ΔQ sign/CI rule; report ΔQ-vs-single too; frame the null as the finding (the council's value is gate+synthesis). |
| **Judge collusion / circularity** | high | C1 headline judge is deterministic lexical `score_case` (no model); any LLM-judge path asserts judge-family ∉ seat-families, fail-closed. |
| **Held-out leakage** (held-out resembles seen) | high | Construct-disjoint (hold out whole axes/clusters), no-shared-3-gram CI assert, publish nearest-neighbour seen-vs-held-out similarity, fit-never-imports-held-out grep test. |
| **Sealing theatre** (in-repo "sealed" eval) | high | Salt generated once, never committed; commit only hashes; grep-gate forbids plaintext held-out answers; downgrade claim to "held-out + manifest hash" if no external seal-holder. |
| **Re-confirming a known null at hours of cost** | med | Build CI-green + reduced; DEFER the full K≥20 run (OPEN in ledger; trigger only on a non-null reduced trend). |
| **`stats.py` extension breaks B** | med | Add-only pure-stdlib fns; the existing `test_steering.py` + the new `test_pif_harness.py` both run; no signature changes to existing fns. |

## Open decisions — resolved at recommended defaults

1. Decompose? → **one Spec C/PR, C1→C2→C3 ordered task-groups** (owner choice).
2. Council seats → **Level-1 persona-prefix** (steered arm deferred to D).
3. Heavy compute → **build all CI-green + reduced; defer full K≥20**.
4. Sealing → **Option A** (`hidden_eval_commitments` reuse, untouched).
5. C1 slice → **pre-register the slice + ΔQ sign/CI rule before running**; start
   with DOMAIN_BENCH (~36 cases), report sign+CI; larger provenance set optional.
6. C1 judge → **deterministic lexical `score_case`**.

## Explicitly deferred to Spec D

- The **full real headline PIF run** (N≥8 × K≥20 on a downloaded dense model) —
  pre-committed command + seed manifest ship here; execution stays OPEN.
- **Level-3 activation-steered diverse-council seats** as a *validated* result
  (the `SteeredClient` drop-in ships as a documented hook).
- **True third-party/external sealing** (independent seal-holder, open ceremony).
- **Model-heterogeneous × trait-diverse crossover** (real model heterogeneity —
  the only thing that has shown a win — combined with trait personas).
- **Live GRPO / verified-reward training** consuming the steering verifier (GPU).
- **Calibration/ECE tracking** of held-out shift magnitude across many real runs.
- The capability-retention guardrail as a product gate + full FastMCP packaging.

## Residual uncertainty (honesty ledger)

1. The council-reader grounding agent errored; `deliberate`'s `seat_clients` seam
   and `hidden_eval_commitments.py`'s `case_digest`/`build_commitments` were
   **manually verified to exist** — but the exact council data format / seat
   route for a *new* personality council should be confirmed during planning.
2. C1 and C3 are both expected NULL; that is the pre-registered honest outcome,
   not a failure to fix by relaxing thresholds.
3. "Sealed" is only as strong as the salt-withholding discipline; without an
   external seal-holder the claim is "held-out + manifest hash", stated plainly.

## Sources / reuse references

Spec A: `docs/superpowers/specs/2026-06-22-personality-measurement-gate-design.md`.
Spec B: `docs/superpowers/specs/2026-06-23-activation-steering-pif-design.md`.
selfextend false-accept pattern (`verifier_synthesis`/`flywheel`/`verified_reward`/
`loop`), `hidden_eval_commitments.py`, `council_deliberate.py`, `aggregate._is_validated`.
