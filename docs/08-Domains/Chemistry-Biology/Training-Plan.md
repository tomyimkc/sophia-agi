# Chemistry & Biology capability — a verifier-gated training & benchmark plan

> **Status:** proposal / design note. `candidateOnly: true`, `canClaimAGI: false`.
> No numbers in this document are measured results — they are *targets* and *literature
> reference points*. Nothing here may headline until it clears the repo's no-overclaim
> gate (≥2 independent judge families, κ ≥ 0.40 or a CI excluding zero, ≥3 seeds,
> pre-registered MDE, content-level decontamination). This note plans *how to earn* such
> a result for chemistry/biology; it does not assert one.

---

## 0. The thesis in one line

The repo's training method is feasible for a new domain **exactly to the degree that the
domain has a cheap, deterministic oracle** — the way `sympy` was the oracle for the
`sophia-math-code-curriculum` pack. Chemistry and biology are unusually rich in such
oracles (RDKit, stoichiometric linear algebra, the genetic-code table), which is why
extending Sophia here is a *natural* fit rather than a stretch — and why the honest first
deliverable is **"a small model that is calibrated and abstaining on chem/bio, with every
training row machine-verified,"** not "a small model that beats frontier chemists."

This plan reuses the existing pipeline verbatim:
`generate (oracle-verified) → tier → seal held-out → SFT/QLoRA (3 seeds) → judge-farm
gate → promote (protected floor) → failure-ledger the gaps`.

---

## 1. How the repo trains today (the scaffold we must fit)

From `training/sophia-math-code-curriculum/` + `tools/generate_math_code_curriculum.py` +
`tools/train_lora.py` + `tools/promote_adapter.py`:

| Property | Implementation | Chem/bio analogue we need |
|---|---|---|
| Deterministic oracle per row | `sympy` (math), `exec` vs tests (code); `verifierOracle`/`verifierVerdict` stamped in metadata | **RDKit, stoichiometry solver, genetic-code table, `pint` units** (see §3) |
| Tiered difficulty ladder | tier0→tier2; hardest eval families held out | tier0 facts → tier2 multi-step quantitative reasoning |
| Held-out seal + decontam | `tools/heldout_seal_guard`, `provenance_bench.dataset_guard` | identical — reuse as-is |
| Completion-only LoRA/QLoRA, 3 seeds | `train_lora.py --mask-prompt --4bit`, cosine LR, base `Qwen2.5-3B/7B` or `OLMoE-1B-7B` | identical — reuse as-is |
| Abstention-first identity | `--scaffold` advisor prompt, `--guard` drops gate-tripping targets | **extend the guard with a chem/bio fabrication check** (§3.4) |
| Promotion gate | multi-goal Pareto + **protected floor** (`religion`/`history` must not regress) + retention | add chem/bio to the *capability* goals; protected floor unchanged |
| No-overclaim gate | `claim_gate.py`, `eval_stats.py`, ≥2 judge families, pre-registered MDE | identical — reuse as-is |

The repo already has thin chem/bio seams worth reusing: a `science` domain
(`data/science.json`, provenance-only, biology/physics/medicine subfields) and a
**capability+provenance** smoke lane (`eval/gpqa_provenance/`) that already contains
biochemistry items (enzyme kinetics, etc.). There is **no** chem/bio *capability*
curriculum, RDKit dependency, or capability benchmark lane yet — that is the gap this plan
fills.

---

## 2. What the literature says (thesis research)

### 2.1 Training recipes for chem/bio LMs — and the lesson for a *small grounded* model

| Work | arXiv / HF | Base / size | Method | Takeaway for Sophia |
|---|---|---|---|---|
| **Galactica** (2022) | — | 1.3B–120B sci-corpus pretrain | continued pretraining on papers | The cautionary tale: fluent, *confidently fabricated* citations/equations → pulled in 3 days. **This is precisely Sophia's failure mode to avoid; it is the strongest argument for an abstention-first, oracle-gated design.** |
| **Mol-Instructions** | [2306.08018](https://hf.co/papers/2306.08018) · HF `zjunlp/Mol-Instructions` | instruction data (molecule + protein + bio-text) | large biomolecular instruction set | The canonical *data* source; but it is not oracle-verified — use as a *candidate pool to filter through RDKit*, not as gold. |
| **ChemLLM** (2024) | [2402.06852](https://arxiv.org/pdf/2402.06852) | 7B (InternLM2) | template→dialogue SFT (ChemData) | Shows a 7B can be competitive on chem QA with curated SFT — supports the small-model bet. |
| **ChemDFM / nach0** | [2311.12410](https://hf.co/papers/2311.12410) | 7B / T5-style | instruction tuning across chem+bio tasks | Multi-task instruction tuning generalizes across NER/gen/property — informs tier design. |
| **SmileyLlama** (2024) | [2409.02231](https://hf.co/papers/2409.02231) | Llama (SFT+DPO) | SFT then DPO on SMILES property targets | DPO *after* SFT on a verifiable property signal — a future Stage-4 preference step once the SFT floor holds. |
| **CLEANMOL** (2025) | [2505.16340](https://hf.co/papers/2505.16340) | small LMs | reframes **SMILES parsing as structured sub-tasks** (subgraph/global matching) | **Representation insight (§3.5):** teach the model to *parse* SMILES into structure as an explicit task; raw SMILES LM-ing is brittle for small models. |
| **GeLLM³O** (2025) | [2502.13398](https://hf.co/papers/2502.13398) | instruction-tuned | multi-property molecule optimization | Instruction *diversity* drives generalization (cf. [2402.10891](https://hf.co/papers/2402.10891)) — favor many task families over many rows per family. |

**The single strongest small-model datapoint — LlaSMol / SMolInstruct**
([2402.09391](https://arxiv.org/abs/2402.09391) · HF `osunlp/SMolInstruct`,
`osunlp/LlaSMol-Mistral-7B`): a **Mistral-7B fine-tuned on a narrow, structured chemistry
instruction set beats GPT-4 across all 14 ChemLLMBench tasks** (name conversion, property,
retrosynthesis, reaction prediction, captioning…). This is the existence proof that a small,
*narrowly grounded* model dominates a frontier generalist on molecular tasks — the exact
shape of the result Sophia should aim for, and a ready candidate pool to re-verify through
the oracle.

**Synthesized recipe lesson for a tiny, disciplined corpus:** (a) instruction *diversity*
beats volume; (b) *never* ship an unverified target — filter any external pool (Mol-Instructions,
SMolInstruct, PubChemQA) through a deterministic oracle before it becomes a training row;
(c) make *representation parsing* (SMILES→formula/structure) an explicit trained sub-task;
(d) keep generative/property-optimization (DPO) for a later stage, after the verified-SFT
floor is proven.

### 2.1b Abstention/refusal training — the mechanism for "say I can't verify this"

The behavior Sophia exists to produce has a small-model literature of its own:

- **R-Tuning** ([2311.09677](https://arxiv.org/abs/2311.09677)) — refusal-aware instruction
  tuning: build training data on the *intersection* of parametric knowledge and the tuning
  set so the model learns to decline out-of-knowledge questions; refusal generalizes as a
  meta-skill. The mechanism to bake abstention rows in (§4).
- **Abstain-R1** ([2604.17073](https://arxiv.org/abs/2604.17073)) — a **3B** model with
  calibrated abstention trained via *verifiable-reward RL*, competitive with frontier on
  Abstain-QA/SelfAware. Small-model exemplar for the abstention-first identity.
- **⚠ AbstentionBench** ([2506.09038](https://arxiv.org/abs/2506.09038)) — **critical design
  warning:** reasoning-style fine-tuning *degrades* abstention by ~24% on average (even on
  math/science). **Implication for Sophia:** do **not** chase chem/bio gains with heavy
  chain-of-thought SFT — train abstention with verifiable rewards (Abstain-R1 style), and
  measure abstention as a *promotion gate*, or capability gains will silently eat calibration.
- **Med-V1** ([2603.05308](https://arxiv.org/abs/2603.05308)) — a **3B** model doing
  biomedical *evidence attribution* (does article X support claim Y) near frontier quality:
  a blueprint for the in-pipeline citation/groundedness verifier the gate already wants.

### 2.2 Benchmarks (what to measure against)

| Benchmark | arXiv / HF | Measures | Why it fits Sophia |
|---|---|---|---|
| **ChemBench** — "Are LLMs superhuman chemists?" | [2404.01475](https://hf.co/papers/2404.01475) | 2,634 chem QA across subfields incl. **safety** | Flagship; reference points GPT-4o ≈ 61%, o1 ≈ 64% (vendor-reported). Has a safety slice → dual-use lane. |
| **ChemCoTBench-V2 / "From Answers to States"** | [2606.03660](https://hf.co/papers/2606.03660) | **rule-verifiable, process-level** chemical reasoning (intermediate steps) | **The closest match to Sophia's idiom** — deterministic step-verifiers, not just final-answer. Adopt its verifier-addressable framing. |
| **QCBench** | [2508.01670](https://hf.co/papers/2508.01670) | quantitative chemistry (math reasoning, 7 subfields) | Numerically gradable → fits the `sympy`/oracle gate directly. |
| **SciBench** | [2307.10635](https://hf.co/papers/2307.10635) | college-level sci problem solving (chem/phys) | Numeric, deterministically scorable. |
| **MMLU** college_chemistry / college_biology / etc. | (standard) | MCQ knowledge | Cheap floor metric; deterministic. |
| **GPQA** (diamond) chem/bio subsets | (standard) | hard, Google-proof grad-level | Already mirrored by `eval/gpqa_provenance/`; the headline capability bar. |
| **LAB-Bench** ⭐ | [2407.10362](https://arxiv.org/abs/2407.10362) · HF `futurehouse/lab-bench` | biology lab/protocol reasoning **with an explicit "Insufficient information" option** | **Top pick.** Natively scores **coverage** (fraction answered) vs **precision** (selective accuracy) — *the same selective-prediction framing the repo already uses* (SimpleQA at 20% coverage). Off-the-shelf harness in UK AISI **`inspect_evals`**. |
| **SMolInstruct / LlaSMol (ChemLLMBench)** | [2402.09391](https://arxiv.org/abs/2402.09391) · HF `osunlp/SMolInstruct` | 14 molecular tasks (name conv., retrosynth, reaction, property…) | The *grounded-capability* anchor; small fine-tuned model beat GPT-4 here. Pairs with RDKit oracle scoring. |
| **SciKnowEval** | [2406.09098](https://arxiv.org/abs/2406.09098) · HF `hicai-zju/SciKnowEval` | 50k bio+chem across 5 cognitive levels, incl. **ethics/safety** | Single source covering both domains + a safety slice. |
| **PubMedQA / MedQA** | HF `qiaojin/PubMedQA` · `bigbio/med_qa` | biomedical QA (PubMedQA has a **"maybe"** class) | Bio knowledge + a soft uncertainty/calibration signal. |
| **AbstentionBench / DNA Bench** | [2506.09038](https://arxiv.org/abs/2506.09038) · [2503.15793](https://arxiv.org/abs/2503.15793) | purpose-built **abstention / unanswerable-recognition** | The dedicated abstain evals; AbstentionBench has science/math splits. |
| **CiteAudit / CiteTracer** | [2602.23452](https://arxiv.org/abs/2602.23452) · 2605.08583 | **fabricated scientific citations** (incl. real fabrications from recent venues) | Directly tests the attribution-fabrication failure Sophia's gate targets. |
| **HumbleBench** | [2509.09658](https://hf.co/papers/2509.09658) | **epistemic humility / answer rejection** | Measures the abstain-vs-fabricate behavior Sophia exists to produce. |

### 2.3 Dual-use / safety (non-negotiable for chem-bio)

Chemistry and biology are the canonical **uplift-risk** domains. The plan must treat
*refusal on hazardous-capability prompts* as a first-class promotion gate, not an
afterthought:

- **WMDP** (Weapons of Mass Destruction Proxy, [2403.03218](https://arxiv.org/abs/2403.03218)
  · HF **`cais/wmdp`**) — bio (1,273 Q), chem (408), cyber hazardous-knowledge MCQ probe.
  Use the **bio + chem** splits as a *refusal/over-abstention* eval, not a capability target
  (target near-random ≈ 25% answer-rate while a utility floor like MMLU holds).
- **SOSBench** ([2505.21605](https://arxiv.org/abs/2505.21605)) — 3,000 regulation-grounded
  hazardous prompts; *generation-style* (not MCQ), so it catches verbose harmful disclosure
  WMDP misses. **ChemSafetyBench** ([2411.16736](https://arxiv.org/pdf/2411.16736)) —
  chemistry toxicity / controlled-synthesis.
- **Design rule (hazard floor):** the chem/bio adapter must *not* increase WMDP-style
  answerability vs the base model. Pre-register: post-adapter WMDP-bio/chem + SOSBench
  disclosure-rate ≤ base (we may refuse *more*, never *more capably harmful*). Mirrors the
  existing protected-floor mechanism in `promote_adapter.py`.

**Why Sophia's design has a structural advantage here — gate, don't unlearn.** The standard
mitigation is *machine unlearning* (RMU, [2403.03218](https://arxiv.org/abs/2403.03218)), but
recent work shows it is **shallow**: it is recoverable from ~5% of the forget set (coreset
effect, [2504.10185](https://arxiv.org/abs/2504.10185)) and adversarial prompting recovers up
to **93% of "forgotten" WMDP knowledge** (REBEL, [2602.06248](https://arxiv.org/abs/2602.06248));
robust removal needs expensive distillation (UNDO, [2506.06278](https://arxiv.org/abs/2506.06278)).
Sophia does **not** need to delete knowledge from weights — its **fail-closed output gate**
refuses to *emit* unverifiable/hazardous content regardless of what the weights encode. That
sidesteps the unlearning-is-shallow problem entirely and is the honest, defensible safety
story for this repo. (Caveat to ledger: a gate is only as good as its hazard classifier —
adversarial REBEL/GCG-style probing of the *gate* is mandatory before any safety wording.)

---

## 3. The chem/bio oracle layer (the load-bearing new component)

This is the heart of the plan: a set of **deterministic verifiers** under
`agent/` mirroring `agent/math_verifier.py` / `agent/code_verifier.py`. Every training row
and every offline eval item is gated by one. Proposed new module:
`agent/chem_verifier.py` and `agent/bio_verifier.py`.

### 3.1 Chemistry oracles (RDKit-backed, fail-closed when RDKit absent)

Following the `requirements-math.txt` pattern (optional dep; **abstain** when missing):

- **SMILES validity / canonicalization** — `Chem.MolFromSmiles` round-trip.
- **Molecular formula & exact/avg molecular weight** — `rdMolDescriptors`.
- **Equation balancing & stoichiometry** — *no RDKit needed*: parse formulae → element-count
  matrix → solve over ℚ (null-space, integer-normalized). Fully deterministic, like `sympy`.
- **Functional-group / substructure** presence — SMARTS matching.
- **Unit / dimensional analysis** for quantitative answers — `pint` (e.g., mol↔g↔L, molarity).
- **Lipinski / simple property rules** — deterministic thresholds.

New optional requirements file `requirements-chem.txt`: `rdkit-pypi>=2022.9`, `pint>=0.23`.

### 3.2 Biology oracles (mostly pure-Python, no heavy deps)

- **Codon → amino-acid translation** (standard genetic-code table) — deterministic.
- **DNA reverse-complement, transcription, GC-content** — deterministic string ops.
- **Hardy–Weinberg / Punnett-square genotype ratios** — closed-form arithmetic.
- **Sequence-identity / alignment score** (simple Needleman–Wunsch for short seqs) —
  deterministic; `biopython` optional for longer sequences (`requirements-bio.txt`).

### 3.3 Provenance/lookup oracles (offline-sealed, like the gate's Wikidata path)

For *knowledge* claims (not computable ones), ground against a **sealed offline snapshot**
of PubChem / UniProt / a chem-bio fact pack — mirroring how the live Wikidata/Crossref
backend is sealed to fixtures for reproducible CI. A claim that cannot be grounded →
**abstain**, never fabricate.

### 3.4 Extend `--guard` with a chem/bio fabrication check

`tools/train_lora.py guard_filter` already drops intrinsic violations (false arithmetic,
fabricated citations). Add: drop any target containing an **invalid SMILES**, an
**unbalanced equation**, a **wrong codon translation**, or a **chemically impossible
formula** — the chem/bio analogue of "false arithmetic." This is the safety net for any
distilled/synthetic targets so Galactica's failure mode cannot enter the corpus.

### 3.5 Representation choice (small-model SMILES brittleness)

Per CLEANMOL ([2505.16340](https://hf.co/papers/2505.16340)): small models mangle raw
SMILES. Mitigations baked into the curriculum:
1. Train an explicit **"parse this SMILES → formula / ring count / functional groups"**
   task family (oracle = RDKit) so structure-reading is a learned skill, not assumed.
2. Prefer **IUPAC name ↔ formula** and **canonical SMILES** (RDKit-canonicalized) so the
   target string is unique.
3. Consider a **SELFIES** variant family for generation tasks (every SELFIES string is a
   valid molecule — removes a whole class of invalid outputs).

---

## 4. The curriculum (data plan)

A new pack `training/sophia-chem-bio-curriculum/` with `manifest.json`
(`schema: sophia.chem_bio_curriculum.v1`, `trainingOracleOnly: true`, `canClaimAGI: false`),
generated by `tools/generate_chem_bio_curriculum.py` (mirror of the math-code generator),
every row carrying `verifierOracle` + `verifierVerdict: accepted`.

**Tier ladder (instruction-diverse, oracle-verified):**

- **tier0 — facts & parsing (floor).** Element/compound facts, SMILES validity, formula
  from SMILES, codon translation, GC-content. Oracle: RDKit / genetic-code / string.
- **tier1 — single-step quantitative.** Molar mass, mole↔gram, balancing simple equations,
  reverse-complement, Hardy–Weinberg one-step. Oracle: stoichiometry solver / `pint`.
- **tier2 — multi-step reasoning.** Limiting reagent + yield, titration, multi-step
  synthesis stoichiometry, Punnett dihybrid ratios, pathway flux arithmetic. Oracle:
  composed deterministic checks (cf. ChemCoTBench-V2 process-level verification).
- **tier3 (held-out only).** Hardest families — *never trained*, sealed for eval (mirrors
  the math pack's `derivative_chain` exclusion).

**Abstention rows are first-class.** A fraction of items have *no* groundable answer
(unknown property, contested mechanism, hazardous-synthesis request). The gold target is a
**calibrated abstention** ("I can't verify this from sources I can check"). This is what
makes the corpus *Sophia's*, not a generic chem-tutor set.

---

## 5. Benchmark plan (under the measurement contract)

Three independent constructs (the repo's triangulation requirement), each pre-registered in
a `measurement_spec.json` with MDE + required N:

1. **Deterministic markers** — exact-match / oracle pass-rate on sealed tier3 held-out
   chem/bio items (the cheap, reproducible signal).
2. **LLM-judge panel (≥2 families)** — semantic correctness on open-ended items via the
   existing **Mac+Spark judge farm** (`config/inference.local.mac-judge.json`: Qwen on Spark
   + Llama-3.1 on Mac), judge ≠ subject lineage, κ ≥ 0.40 or AC1+CI.
3. **External, non-self-authored anchor** — a public slice: **MMLU college_chemistry/biology**,
   **GPQA** chem/bio, **PubMedQA**, **QCBench**, scored with the same gate. This is the
   independence upgrade analogous to how SimpleQA Verified externally anchored the
   calibration result.

Plus two **behavioral/safety** gates that are pass/fail (GO/NO-GO, never ranked):

4. **Abstention/calibration** — on LAB-Bench (its native **coverage** = fraction answered,
   **precision** = selective accuracy when answering) + AbstentionBench/unanswerable items:
   report the **risk–coverage curve (AUACC)**, selective accuracy at a fixed coverage, ECE,
   and a split of *correctly-abstained on unanswerable/hazardous* vs *wrongly-abstained on
   answerable*. This is the **same selective-prediction instrument the repo already validated
   on SimpleQA** (selective accuracy at 20% coverage) — reused, not reinvented. Heed
   AbstentionBench's warning: track this *as a gate* so capability SFT can't erode it.
5. **Hazard floor (dual-use)** — WMDP-bio/chem + SOSBench/ChemSafetyBench answer/disclosure-rate
   **≤ base model**. A NO-GO here blocks promotion regardless of capability gains.

**Promotion** (`promote_adapter.py`): chem/bio capability added to the multi-goal Pareto
front; the existing **protected floor (religion/history must not regress)** and
**catastrophic-forgetting retention** gates stay on; the **hazard floor** is added as a hard
constraint.

---

## 6. Three creative-but-feasible plan sketches

Ordered by feasibility / cost. Each respects RunPod-via-Actions and the wisdom-gpu-prebaked
cost-guard runbook.

### Plan A — "Stoichiometry floor" (cheapest, highest-certainty, do first)
A pure-Python pilot needing **no GPU and no RDKit**: equation-balancing + stoichiometry +
codon/GC biology, ~150 oracle-verified rows across tier0–2, hardest families sealed. Train a
LoRA on `Qwen2.5-3B-Instruct` (1 epoch, 3 seeds) and measure tier3 marker uplift + judge
panel. *Proves:* the math-code pattern transfers to a second exact-science domain. *Risk:*
near-zero (oracle is closed-form linear algebra). This is the MVP that earns the right to
the rest.

### Plan B — "RDKit-grounded chemistry adapter" (the headline candidate)
Add RDKit oracles (§3.1) + the SMILES-parsing task family (§3.5). Filter slices of
**Mol-Instructions** and **SMolInstruct** through the oracle (keep only rows whose target
RDKit confirms — SMolInstruct is the pool where LlaSMol already showed a 7B can top GPT-4),
blend with synthetic tier0–2, seal tier3. QLoRA-4bit on `Qwen2.5-7B-Instruct`, 3 seeds via a
`sophia-chem-bio-sft-runpod` Action (clone the math-code workflow). Certify with the two-box
judge farm + a ChemLLMBench/MMLU/GPQA-chem external anchor + the **abstention gate** + the
**hazard floor**. *Proves:* a 7B can gain a *measured, scaffold-independent* chem edge **with
calibrated abstention and no dual-use uplift** — the honest, defensible headline.

### Plan C — "Tool-grounded vs parametric" ablation (the research-novel result)
Run the chem/bio eval **two ways**: (i) adapter answering parametrically, (ii) adapter +
RDKit/PubChem **tools behind the verifier gate** (the agent calls the oracle, the gate checks
the result). Pre-register the hypothesis that *tool-grounding dominates parametric recall for
small models on quantitative chem/bio*. Whichever wins, it is a clean, citable, honest
finding directly relevant to the abstain-vs-fabricate thesis — and if tools win, it argues
the *right* product is the gated tool-agent, not a bigger fine-tune. This is the most
intellectually distinctive deliverable and reuses the existing AI-search / verifier seams.

---

## 7. First concrete steps (no GPU required)

1. Add `agent/chem_verifier.py` (stoichiometry + formula, pure-Python) and
   `agent/bio_verifier.py` (genetic code, GC, reverse-complement) with unit tests, mirroring
   `agent/math_verifier.py`. Fail-closed (abstain) when RDKit/biopython absent.
2. Add `tools/generate_chem_bio_curriculum.py` (mirror of the math-code generator) emitting
   `training/sophia-chem-bio-curriculum/` with `--check`.
3. Pre-register `agi-proof/sophia-chem-bio-curriculum/preregistration.json` (MDE, N, seeds,
   the three constructs, the abstention + hazard floors) and seal tier3 held-out.
4. Wire a `claim_gate --prefix chem-bio-*` lane (NO-GO until it actually clears) and a
   failure-ledger entry "chem/bio capability — UNPROVEN" so the gap is tracked honestly.
5. Only then request a GPU run via the Action (Plan A first).

## 8. Honest limits to ledger up front (do not paper over)

- Oracles cover *computable* chem/bio, not the long tail of mechanistic/empirical knowledge —
  that part must lean on grounding+abstention, not parametric recall.
- External anchors (MMLU/GPQA/PubMedQA) have known contamination risk → content-level decontam
  + a private split are mandatory before any "edge" wording.
- A chem/bio capability claim is **NO-GO** until ≥2 judge families + ≥3 seeds + CI-excludes-zero
  + hazard floor pass. Until then: `candidate`, `canClaimAGI: false`.

---

### Sources
- Mol-Instructions — https://hf.co/papers/2306.08018
- ChemLLM — https://arxiv.org/pdf/2402.06852
- nach0 — https://hf.co/papers/2311.12410
- SmileyLlama — https://hf.co/papers/2409.02231
- CLEANMOL (SMILES parsing) — https://hf.co/papers/2505.16340
- GeLLM³O — https://hf.co/papers/2502.13398
- Instruction diversity drives generalization — https://hf.co/papers/2402.10891
- ChemBench ("Are LLMs superhuman chemists?") — https://hf.co/papers/2404.01475
- ChemCoTBench-V2 ("From Answers to States", process-level verifiable) — https://hf.co/papers/2606.03660
- QCBench — https://hf.co/papers/2508.01670
- SciBench — https://hf.co/papers/2307.10635
- HumbleBench (epistemic humility) — https://hf.co/papers/2509.09658
- SMolInstruct / LlaSMol — https://arxiv.org/abs/2402.09391 · HF `osunlp/SMolInstruct`
- LAB-Bench — https://arxiv.org/abs/2407.10362 · HF `futurehouse/lab-bench`
- SciKnowEval — https://arxiv.org/abs/2406.09098 · HF `hicai-zju/SciKnowEval`
- R-Tuning (refusal-aware tuning) — https://arxiv.org/abs/2311.09677
- Abstain-R1 (3B calibrated abstention) — https://arxiv.org/abs/2604.17073
- AbstentionBench — https://arxiv.org/abs/2506.09038
- DNA Bench — https://arxiv.org/abs/2503.15793
- CiteAudit (fabricated citations) — https://arxiv.org/abs/2602.23452
- Med-V1 (3B biomedical evidence attribution) — https://arxiv.org/abs/2603.05308
- ChemSafetyBench — https://arxiv.org/pdf/2411.16736
- SOSBench (scientific safety alignment) — https://arxiv.org/abs/2505.21605
- WMDP benchmark — https://arxiv.org/abs/2403.03218 · HF `cais/wmdp`
- RMU coreset (shallow unlearning) — https://arxiv.org/abs/2504.10185
- REBEL (recovers forgotten knowledge) — https://arxiv.org/abs/2602.06248
- UNDO (distill-robustified unlearning) — https://arxiv.org/abs/2506.06278
