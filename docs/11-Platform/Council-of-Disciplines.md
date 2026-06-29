# Council of disciplines — one base, many gated expert seats

> Status: **shipped registry + reference verifiers + offline demo.** The discipline taxonomy,
> chemistry/biology reference verifiers, and the council-vs-monolith harness are CI-tested. The
> per-discipline 3B adapters and the live serving are OPEN. `canClaimAGI` stays false.

## The architecture (Branch-Train-MiX + S-LoRA, not 9 separate models)

Each council seat = **one discipline served as a LoRA adapter on a shared 3B base**, routed to per
task, admitted to the shared answer only if it clears **its discipline's verifier** (the trust
boundary). This is the efficient realisation of the council `agent/swarm_router.py` already
designed (`adapter_id` carries the comment "V3 (Branch-Train-MiX)"):

- **S-LoRA** ([arXiv 2311.03285](https://arxiv.org/abs/2311.03285)) serves many adapters on one base
  (adapters in RAM, fetched per request) — "load many specialists at once" without 9× the VRAM.
- **Branch-Train-MiX** ([arXiv 2403.07816](https://arxiv.org/abs/2403.07816)) trains each domain
  expert separately (embarrassingly parallel) then mixes them into one MoE — the "graduation" path
  to a single deployable model once each adapter is gate-validated.

## The honest core: gate kind

`agent/council_registry.py` classifies every discipline by **what it can actually verify**:

| Gate kind | Meaning | Disciplines |
|---|---|---|
| `standalone` | a deterministic validator gates ANY answer | mathematics, chemistry, biology, **finance**, engineering, statistics |
| `reference` | the verifier needs a gold/test; **no reference → ABSTAIN** (fail-closed) | physics, coding |
| `composite` | a domain safety overlay THEN provenance (must clear both) | **medicine** (dose/contraindication safety + provenance) |
| `provenance` | no truth oracle — gated for source discipline / attribution (`agent.gate`); reduces fabrication, does not certify correctness | philosophy, history, religion, linguistics, psychology, sociology, political science, economics, law, business, education, environment |

A discipline with no machine verifier is **not** a verified expert — it is a provenance-gated one.
That distinction is the council's honesty. **religion and history are PROTECTED** (fixed floor,
never RL-optimised).

## Disciplines (21 + general fallback)

STEM: mathematics · physics · chemistry · biology · coding · engineering · statistics · medicine.
Humanities: philosophy · history* · religion* · linguistics. Social: psychology · sociology ·
political science · economics · finance. Applied: law · business · education · environment.
(*PROTECTED.) Each carries an `adapter_slot` (`sophia-<id>-3b`), a tool scope, a routing lexicon,
and its `verifier_ref`.

## New reference verifiers (so chemistry/biology seats are gate-able)

- `agent/chemistry_verifier.py` — element-symbol validity + **mass-balance of equations** across
  `->` (catches `H2 + O2 -> H2O`). Dependency-free; a real RDKit backend can replace it.
- `agent/biology_verifier.py` — sequence **alphabet** validity (DNA/RNA/protein), reverse-complement
  correctness, coding-length-multiple-of-3. Dependency-free; Biopython can replace it.

Both are reference-grade, candidate-only, **fail-closed** — they flag the cheap machine-checkable
errors, never claim correctness they cannot check.

Two more bring high-value seats into the gate-able set:

- `agent/finance_verifier.py` — the **accounting identity** (Assets = Liabilities + Equity) and a
  share/probability asserted > 100%. Finance is numeric, so the finance seat is genuinely
  `standalone` (must also clear provenance).
- `agent/medicine_verifier.py` — a conservative **safety overlay**: implausible dose, unknown dose
  unit, and a small hard-contraindication table. It flags gross errors and otherwise PASSES,
  deferring correctness to provenance + human. Reference-grade, **not medical advice**; medicine
  stays `composite` (overlay + provenance) because no clinical-correctness oracle exists.

## Does the council actually help? (the measured hypothesis)

`tools/eval_council_vs_monolith.py` gates labelled answers two ways — per-discipline (council) vs one
general provenance gate (monolith). On the offline fixture set the **council catches 4/4** discipline
errors vs the monolith's **2/4**; on the 20-case held-out pack (`eval/council/heldout_v1.jsonl`,
`--pack`) the **council catches 11/11 vs the monolith's 3/11**, with the per-discipline rollup showing
the gap is entirely in chemistry, biology, finance, and medicine — the seats with a machine verifier.
That delta *is* the council's reason to exist:
**it only beats a monolith where each seat is machine-verifiable.** Elsewhere (pure provenance
domains) a single strong generalist may match it — which is why this stays a pre-registered
experiment to run on real adapters, not an assumed win.

## Distillation seed packs (Stage-1 SFT nucleus per seat)

`training/council_seeds/<discipline>.jsonl` holds hand-authored teacher CoT traces — the gate-clean
nucleus each `sophia-<discipline>-3b` adapter is SFT-seeded on. `tools/gen_reasoning_distill.py` is now
**discipline-aware**: a trace with a `discipline` field is gated by THAT seat's verifier, so a
chemistry seed with an unbalanced equation is dropped by the chemistry verifier, not waved through the
general gate. `tools/build_council_seeds.py` validates every seed is gate-clean (a drop is a *seed
bug* to fix — it caught two on first run: a `math_sound` sub-expression misparse and a missing Freud
denial) and emits the combined `distill_v1.jsonl` (**68 rows across all 21 disciplines**). Real volume
is added from a teacher model through the same gate. The reference seats (physics, coding) and the
provenance seats are gated through the provenance fallback, so even their seeds are attribution- and
arithmetic-clean by construction.

## Independent v2 validation of the new verifiers

`tools/eval_discipline_verifier.py` runs a discipline's raw verifier over an independent v2 pack and
reports recall (reject the bad) + pass-rate (accept the good). On `eval/council/{finance,medicine}_
heldout_v2.jsonl` both score **recall 1.0 / pass-rate 1.0** (floor 0.9). Honest caveat: v2 is
independent of the self-checks and `heldout_v1` but authored knowing the verifier's rules — so this is
self-consistency on a fresh set, **not** blind generalisation; a truly third-party pack stays OPEN.

## Honest limits / OPEN

- Stub answers; the per-discipline **3B adapters are OPEN** (each needs gate-clean distillation data
  via `tools/gen_reasoning_distill.py`). Routing is lexical v1 (a trained router overrides it).
- chemistry/biology verifiers are reference-grade, not production oracles. medicine/finance/etc. have
  **no** machine verifier yet — they are provenance-gated only (the honest ceiling for those seats).
- The central claim — "gated council of 3B adapters beats one strong generalist on verifiable packs"
  — must be measured with CIs and ≥2 judge families, against the ledger's adapter-non-transfer risk.
  A null is a valid result.

## Sources
S-LoRA (arXiv 2311.03285) · Branch-Train-MiX (arXiv 2403.07816) · Train Separately, Merge Together
(arXiv 2604.18473).
