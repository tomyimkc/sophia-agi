# Scope: properly-powered third-party validation of the RLVR provenance adapter

**Goal:** turn the current *candidate-only* directional signal (adapter preferred ~52–60%
across runs, but n=94, win-rate CI spans 0.5, inter-judge κ≈0.06–0.11) into a **single
pre-registered, properly-powered result** — VALIDATED or an honest NULL. Protocol follows
`docs/methodology/llm-judge-validation.md`. **One run, no iterating until it passes.**

`canClaimAGI` stays False regardless; the ceiling of this result is a *narrow* claim
("the adapter improves provenance source-discipline"), not AGI.

---

## 1. The held-out pack (the real bottleneck — needs a human / external author)

| Property | Requirement | Why |
|---|---|---|
| **Size** | **≥ 300 cases** | Power calc below: detect a true 0.58 win-rate vs 0.5 at 80% power, two-sided α=0.05, needs **n≈304** (n≈194 if the true effect is 0.60). At n=94 we are ~30% powered — the root cause of the spanning CI. |
| **Independence** | **Third-party authored**, not by the training-data author | The #1 reviewer-dismissal gap (self-authored eval). Use the existing `agi-proof/third-party-heldout/PROTOCOL.md` + `tools/seal_third_party_heldout.py` (currently `caseCount: 0`). |
| **Composition** | false-attribution probes + true controls, **≥4 traditions/domains**, difficulty-stratified | Avoids one-domain overfit; lets per-stratum win-rates be reported. |
| **Disjointness** | entity- **and** work- **and** tradition-disjoint from training | Prevents memorization passing as capability (the `entityIntersection` check already exists). |
| **Decontamination** | exact + normalized-prompt check vs training + the 94-case pack | No leakage. |
| **Sealing** | salted hash seal before any model sees it | Anti-gaming; reviewer can recompute live. |

**Power calc (binomial, two-sided α=0.05, power 0.80):**
`n ≈ (1.96·√0.25 + 0.84·√(p₁(1−p₁)))² / (p₁−0.5)²` → p₁=0.58 → **n≈304**; p₁=0.60 → n≈194;
p₁=0.55 → n≈783. **Pre-register the target effect; ≥300 covers a ~0.58 effect.**

## 2. Generate answers (needs one GPU eval pass)

- Run **base GLM-4-9B** and the **`sophia-rlvr-v1` adapter** (weights in the committed
  `mr9sr03clgpk5g.sophia-rlvr-v1.tar.gz`, reproducible) over the sealed pack → `completion`s.
- Path: a RunPod eval via GitHub Action (this box can't SSH to RunPod). ~1 A100-hour for 300×2.
- Reshape with the existing `tools/build_rlvr_judge_answers.py` (already joins base/adapter by
  case_id and attaches faithful prompts + references).

## 3. Judge panel (pre-registered)

- **≥3 independent families, distinct vendors**, none = subject (GLM) or gate. e.g.
  `openrouter:deepseek/deepseek-chat`, `meta-llama/llama-3.3-70b-instruct`,
  `qwen/qwen-2.5-72b-instruct` (and optionally a 4th: `mistralai/mistral-large` or a Gemini).
- **Forced-choice** (`--forced-choice`), deterministic A/B swap per case (already implemented),
  length-control noted. `tools/judge_pilot_answers.py` already emits the full panel + majority.
- ~300 cases × 3 judges = **~900 calls** (OpenRouter; cents–low dollars; **no GPU**).

## 4. Pre-registered analysis (fix BEFORE running)

- **PRIMARY (the gate):** adapter win-rate vs 0.5 on the **majority-vote** label.
  PASS iff **bootstrap/Wilson 95% CI lower bound > 0.5** AND **exact binomial p < 0.05**.
- **COMPANION (report, never the gate):** κ + PABAK + bias/prevalence indices per judge pair,
  observed agreement (per `llm-judge-validation.md`; do **not** compare PABAK/AC1 to a κ bar).
- **CALIBRATION:** a **~30-case human-labeled subset**; report human↔LLM agreement (even
  human–human tops out ~80%, so this anchors judge trust).
- **ANTI-REWARD-HACKING:** consistency of the deterministic-verifier reward (proxy) vs the
  judge win-rate (gold-ish) — divergence flags reward overfit (Gao 2023).
- **A NULL is a legitimate, logged outcome.** No re-running judge variants to chase a pass.

## 5. Effort / cost / dependencies

| Item | Cost | Owner |
|---|---|---|
| Author/commission the ≥300-case third-party pack | **the real work** (external authorship) | human |
| Fresh `OPENROUTER_API_KEY` | — (rotate the leaked one) | human |
| GPU eval pass (base+adapter over 300) | ~1 A100-hr via Action; needs `RUNPOD_API_KEY` secret | me (dispatch) + human (secret) |
| Judging + analysis | ~900 OpenRouter calls, no GPU | me (turnkey now) |
| Human calibration labels (~30) | ~1–2 hrs | human |

## 6. What's already built vs. what's missing

**Built (this session):** the judge harness + honest-stats panel (`judge_pilot_answers.py`),
the RLVR→judge reshaper (`build_rlvr_judge_answers.py`), the third-party seal tooling, the
methodology + protocol docs.

**Missing (the gating dependencies):** (1) the ≥300-case independent third-party pack
[human/external]; (2) a fresh judge key; (3) a GPU eval dispatch over the new pack. Items 2–3
are hours; item 1 is the genuine bottleneck and the thing reviewers will demand.

## 7. Bottom line

The measurement machinery is ready and turnkey. **Validation is now gated on an independent,
properly-powered held-out pack — not on more code.** Until that exists, the honest status is
*candidate evidence of a directional provenance-discipline gain*, logged as OPEN in the ledger.
