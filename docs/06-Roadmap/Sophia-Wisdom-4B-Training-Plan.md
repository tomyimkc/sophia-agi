# Sophia-Wisdom-4B — Training Plan (the honest, measurable, gated path)

**Status:** active execution plan · **Owner:** Sophia training lab · **Source of truth** for the
Sophia-Wisdom-4B effort. Everything here ships under the no-overclaim gate
(`tools/lint_claims.py`) and the RESULTS.md VALIDATED standard.

> Nothing here is an AGI claim. Drop "AGI" from all framing — it undermines the credible,
> provable result. Each milestone lists its kill-switch and what it does **not** prove.

---

## The target and the *only* honest win condition

**Sophia-Wisdom-4B** is a local, verifier-gated *wisdom-habit* model. It is **not** a bid for
general dominance over market models. The single win condition we will measure:

> **Sophia-Wisdom-4B + external gate beats SAME-SIZE open models** (Qwen3-4B, Phi-4-mini,
> Llama-3.2-3B, small Gemma) on provenance, contested-claim qualification, source/tradition
> separation, fake-citation detection, calibrated abstention, and moral route selection —
> **at bounded false-positive / over-abstention cost** — **WITHOUT** regressing general
> instruction-following or the protected religion/history suites.

**The differentiator, made central:** *bilingual (EN/ZH) Confucian/Daoist source discipline.*
The demo-able headline is concrete — "the small model that won't merge 儒家 and 道家, won't
misattribute the 道德經, and knows when a classical attribution is legendary vs settled" — not
abstract "wisdom."

### Winnable axes (lead with these)
False-attribution reduction · contested-claim qualification · 儒家 vs 道家 tradition separation ·
fake-citation/DOI/scripture detection · "I cannot verify" behavior · moral routing
(allow/revise/retrieve/clarify/escalate/abstain/block) · council reasoning · bilingual EN/ZH
source-aware answers · tool/MCP discipline · honest failure-ledger transparency.

### Unwinnable axes (NEVER lead with these)
General MMLU/GPQA dominance · broad coding · frontier reasoning · autonomous self-improvement ·
"true AGI" · guaranteed zero-hallucination.

---

## What we may and may NOT claim

**ALLOWED:** "Sophia-Wisdom-4B + gate beats same-size open models on provenance, contested-claim
handling, and moral routing at bounded false-positive cost"; "the weights learn source-discipline
*habits*, the external gate enforces truth."

**FORBIDDEN:** "Sophia is AGI"; "guarantees no hallucinations"; "generally smarter than
Qwen/Gemma/Phi/Llama"; "the adapter alone enforces truth"; "the model has moral consciousness."

Truth lives **outside** the model. The adapter learns habits; the external gate enforces the truth.

---

## Hard rules (violating any one is failure)

1. **MEASURE BEFORE YOU TRAIN.** Build and validate the comparison instrument before spending any
   GPU on training.
2. **The JUDGE is independent of the GATE.** The gate is *treatment only*; judges share no code
   with the gate; semantic claims need ≥2 judge families. This anti-circularity rule is enforced
   in every eval.
3. **NEVER train on ungated synthetic data.** Pipeline: teacher output → Sophia gate →
   (accepted | correctly-abstained) → train; (fabricated | unsupported | wrong-route) → reject,
   or keep as a hard negative.
4. **RETENTION IS MANDATORY.** 25–30% general-instruction replay in every training mix; run
   `tools/run_learning_shift.py` after every SFT. A model that abstains on everything or forgets
   instruction-following has **FAILED**, not won.
5. **Every run is an experiment, not a victory lap.** Update `agi-proof/failure-ledger.md`
   honestly, including negative results.
6. **Base-model choice is an EXPERIMENT, not an assumption.** Do not hard-commit to Qwen3-4B on
   faith — test it (M1).
7. **Pre-commit gates:** run `python3 tools/lint_claims.py` and
   `python3 tools/build_local_sophia_dataset.py --check` before any commit touching data or claims.

### The VALIDATED bar (from RESULTS.md — non-negotiable)
A number is **VALIDATED** only with: ≥2 independent judge families (judge ≠ subject, judge ≠ gate),
reported inter-judge agreement (Cohen's κ ≥ 0.40), ≥3 runs, and 95% CIs excluding zero. Everything
else is labelled **illustrative**. Deterministic verifiers (e.g. `legal_citation_exists`) need no
judges; semantic/moral claims need the full multi-judge bar.

### Infrastructure split
- **MLX (Apple Silicon):** fast SFT experiments.
- **CUDA (RunPod MCP):** ORPO/DPO preference + RLVR/GRPO.

---

## Behavioral target: route-first reasoning

Sophia decides a **route before answering**, emitting structured output:

```json
{
  "route": "allow | revise | retrieve | clarify | escalate | abstain | block",
  "confidence": 0.00,
  "epistemic_status": "...",
  "needed_sources": ["..."],
  "risk_flags": ["..."],
  "answer": "..."
}
```

This mirrors the existing conscience kernel / moral gate.

---

## Execution plan — four GATED milestones

Do them in order. **Do NOT advance past a milestone whose go/no-go fails** — stop and report.

### M1 — Instrument + base selection (NO training)

**Build:**
- `tools/run_same_size_market_baselines.py` — on the same cases, compares: raw Qwen3-4B /
  Phi-4-mini / Llama-3.2-3B / small-Gemma, plus Qwen3-4B+Sophia-prompt, +adapter (later),
  +adapter+gate, +adapter+gate+MCP/RAG. Metrics: `provenance_accuracy`,
  `false_attribution_rate`, `contested_fabrication_rate`, `citation_fidelity`,
  `qualification_rate_on_contested`, `tradition_merge_rate`, `moral_route_accuracy`,
  `tool_route_accuracy`, `over_abstention_rate`, `useful_correctness`,
  `protected_history_regression`, `protected_religion_regression`.
- `tools/build_wisdom_market_benchmark.py` → `data/wisdom_market_benchmark/heldout_v1.jsonl`,
  500–1000 **high-headroom** held-out cases (cases where same-size models actually fail — if raw
  accuracy is already 95%+ you cannot prove uplift). Mix: false-attribution traps,
  legendary/compiled authorship, 儒家/道家 boundary, contested religion claims,
  psychology/history myths, fake-citation traps, HK bilingual advisor traps, moral-gate cases,
  tool-use traps. Each row carries: `id, prompt, domain, language, gold_route,
  gold_claim_boundary, forbidden_assertions, acceptable_answer_features, source_refs,
  protected_suite, train_overlap_forbidden`. Wired into the contamination guard
  (`build_local_sophia_dataset.py --check`).
- Run PROMPT-ONLY Sophia (existing scaffold + gate, no training) across the candidate bases.

**GO/NO-GO:** Is there a base where **(a)** raw accuracy is low enough to leave headroom **AND**
**(b)** prompt+gate already beats raw by a CI-clean margin (≥3 runs, CI excludes 0,
over-abstention ≤0.10, no protected-suite regression)?
- **YES** → that base is the pick; the gate's value is proven *before* training.
- **NO** → the niche may not exist on these axes. **STOP and report; do not train.**

### M2 — Data pipeline (the real project, ~80% of the work)

Build `tools/build_sophia_wisdom_dataset.py` and the **teacher→gate→admission** pipeline. Produce
`training/local_sophia_v3/` (`manifest.json` + `mlx/{train,valid}.jsonl`) scaling toward
**10–20k GATE-PASSED rows**. Target mix: source-discipline 20–25% · hard provenance negatives 15% ·
council 10–15% · moral-gate routing 10% · tool/MCP 10% · HK bilingual 10% · team-agent 5–10% ·
**general-instruction retention 25–30%**. Every row carries metadata: `task_family, domain,
language, expected_route, source_ids, gate_verdict, protected_suite, license, eval_overlap=false`.
Also mine preference pairs (chosen = verify/cite/abstain/separate/retrieve/escalate;
rejected = fabricate/fake-cite/merge/guess) for later ORPO.

**GO/NO-GO:** Can you produce ≥10k rows that pass `build_local_sophia_dataset.py --check`
(decontaminated, no eval overlap) **and** the gate? If not, fix the pipeline before training.

### M3 — One SFT run, fully evaluated

Train **ONE seed** first (MLX locally or a cheap RunPod CUDA GPU). Start at the **proven seq
length (1024)** — don't jump to 2048 on faith. Then run the FULL M1 instrument plus
`tools/run_learning_shift.py` retention check and the protected suites.

**GO/NO-GO (all must hold):** adapter+gate beats raw same-size baselines on source/moral metrics ·
≥3 runs · ≥2 judge families for semantic scoring · κ≥0.40 · CI excludes 0 · over-abstention ≤0.10 ·
NO protected religion/history regression · NO material general-retention regression ·
`lint_claims.py` OK.
- **PASS** → proceed to seeds 1–2, then M4. Register the candidate in
  `training/adapters/registry.jsonl` (`candidate_only:true, validated_external:false` until
  proven); write `agi-proof/model-cards/sophia-wisdom-4b.md`.
- **FAIL** → STOP, log to failure-ledger, diagnose (data? forgetting? base?).

### M4 — Preference (ORPO) then maybe RLVR — CUDA only, after M3 passes

ORPO on the preference pairs (cite>fake-cite, abstain>fabricate, retrieve>guess, separate>merge,
escalate>unsafe-allow, useful-answer>lazy-refusal). **RLVR LAST and treated as RESEARCH, not
product** — it is the riskiest, highest-cost step; gate it behind a clearly positive ORPO result.
Use existing `tools/run_rlvr.py --reward gate` options; do NOT invent a custom weighted reward
without writing and testing the code.

---

## Current substrate (verified facts)

Trainable data today is small — ~1.3k rows total:
`training/lora/train.jsonl` (439) · `training/corpus.jsonl` (528) ·
`training/council/traces.jsonl` (125) · `training/local_sophia_v2/general_instruct.jsonl` (120) ·
`training/moral_gate_sft.jsonl` (35). Enough for a **behavioral LoRA**, NOT enough to beat market
3–4B models except on **narrow Sophia-native tasks**.

**Reuse, do not rewrite:** `tools/train_lora.py`, `train_orpo.py`, `run_rlvr.py`, `eval_ladder.py`,
`promote_adapter.py`, `run_seib.py`, `run_provenance_delta.py`, `run_council_uplift.py`,
`run_moral_public_standard_eval.py`, `run_learning_shift.py`, `mine_hard_negatives.py`,
`split_long_training_rows.py`, `build_local_sophia_dataset.py` (`--check` decontamination mode),
`lint_claims.py`.

**New tools (this effort):** `tools/run_same_size_market_baselines.py`,
`tools/build_wisdom_market_benchmark.py`, `tools/build_sophia_wisdom_dataset.py`.

---

## Decision protocol

When a real decision the plan doesn't cover arises (a base ties, a metric is borderline), **STOP
and ask the human** rather than guessing — and never let a borderline number be reported as a
headline.
