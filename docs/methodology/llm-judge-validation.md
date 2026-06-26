# Validating capability gains under LLM-as-judge evaluation

**Status:** methodology note (no claim change). Motivated by the RLVR adapter κ result
(`agi-proof/benchmark-results/runpod-rlvr/mr9sr03clgpk5g.judge*.json`). It refines — and
argues for revising — the `RESULTS.md` "κ ≥ 0.40 / 2-judge-family" validated bar.

---

## 1. The problem we hit

The RLVR adapter (run `mr9sr03clgpk5g`, GLM-4-9B, 94 held-out provenance cases) was
re-scored semantically by two independent judge families (deepseek-chat, llama-3.3-70b;
≠ subject, ≠ gate). Across runs:

| Pass | deepseek win-rate | llama win-rate | inter-judge κ | observed agreement |
|---|---|---|---|---|
| tie-allowed | 0.426 | 0.606 | 0.094 | — |
| forced-choice (run A) | 0.581 | 0.596 | 0.110 | — |
| forced-choice (run B, panel) | 0.521 | 0.585 | 0.057 | 0.532 |
| **4-judge panel, majority vote** | per-judge 0.52–0.62 | (qwen 0.55, mistral 0.54) | pairwise 0.16–0.59 | **maj-vote win-rate 0.532, p=0.65** |

Three things stand out and **all** are diagnostic, not noise to be hidden:

1. **κ between a given pair is unstable** — the 2-judge κ=0.094–0.110 used the single
   *worst-agreeing* pair (deepseek↔llama). A 4-family panel shows **3/6 pairs at κ≥0.40**
   (up to 0.59), so the judges are moderately reliable once you don't rely on one pairing.
2. **But the capability signal is genuinely ~chance at this N.** Under a 4-judge **majority
   vote**, adapter win-rate = **0.532 [0.42, 0.64], binomial p=0.65 — not significant.** The
   earlier 0.58–0.60 read was inflated by one generous outlier judge (llama, 0.617).
3. **Run-to-run variance is large** (deepseek 0.581 → 0.521 between identical runs) — the
   signature of being underpowered at n=94.

Net: more/better judges fixed the *reliability* worry but confirmed the *effect* is ~0 at
n=94. The bottleneck is **power and an independent pack**, not the agreement statistic.

## 2. Why κ is the wrong *gate* here (verified literature)

Cohen's κ is **not a pure agreement measure** — its value is jointly driven by observed
agreement, between-rater **bias**, and category **prevalence** (Feinstein & Cicchetti 1990;
Byrt, Bishop & Carlin 1993). Three mechanisms apply directly:

- **First (prevalence) paradox.** When both judges skew the same direction (both prefer the
  adapter), skewed marginals inflate the chance-expected term `pe` in `κ=(p₀−pe)/(1−pe)`, so
  κ collapses *even with high raw agreement*. Zec et al. (2017) document observed agreement
  0.72–0.84 yielding κ flagged "totally unsatisfactory."
- **Second paradox (formal proof: Warrens 2010).** κ is *higher* under asymmetric marginals
  and *penalizes judges with similar marginals*. This is why forced-choice marginal-matching
  barely moved κ (0.094 → 0.110): we removed the asymmetry κ was rewarding.
- **Canonical remedy (Byrt 1993, ~1900 citations): never report κ alone.** Report it
  *alongside* observed agreement, a **Bias Index** `BI=(b−c)/N`, a **Prevalence Index**
  `PI=(a−d)/N`, and **PABAK = 2·p₀−1**.

### ⚠️ You cannot clear a κ bar by switching to Gwet's AC1
Zec et al. (2023, *MethodsX*): "Gwet's AC1 should not be seen as a substitute for Cohen's
kappa. In particular, the verbal classification of kappa values by Landis & Koch should not
be applied to Gwet's AC1." AC1 systematically produces larger numbers, so reporting AC1 and
comparing it to the **κ-derived** 0.40 threshold is statistically illegitimate. Switching
metrics *after* seeing a failing κ, to pass, is goalpost-moving. (Refuted in verification:
"AC1 is the most robust / recommended replacement" and "the paradox starts above 60%
prevalence" — neither is supported.)

> **But note our case is not *only* the paradox.** Observed agreement was just ~0.53 and
> PABAK ~0.06 — so prevalence-adjusting does **not** rescue it. At n=94 the judges genuinely
> disagree case-by-case. The paradox explains why κ is *uninformative here*; it does not turn
> a null into a win.

## 3. What inter-judge agreement is *normal* (sourced)

- **Zheng et al., MT-Bench / Chatbot Arena (NeurIPS 2023):** a strong judge (GPT-4) agrees
  with humans ~80%+, which **equals the human–human agreement level (~81%)**. The ceiling on
  these subjective judgments is ~80% *for humans too* — so demanding high *chance-corrected*
  agreement on subtle "source-discipline" calls is unrealistic.
- **Panel-of-LLM-judges / PoLL (arXiv:2404.18796):** a panel of smaller, diverse judges
  reduces single-model bias and tracks humans better than one large judge.
- Control for **position bias, verbosity/length bias, self-preference** (swap A/B order;
  length-control à la AlpacaEval-LC). Our base answers were long/repetitive — length bias is live.

## 4. The capability claim is a *win-rate*, not κ (sourced)

κ measures judge **reliability**. Whether the adapter is **better** is a **win-rate vs 0.5**:

- Report **win-rate with a Wilson + bootstrap CI** and an **exact two-sided binomial (sign)
  test vs 0.5** (Cameron Wolfe, *Applying Statistics to LLM Evaluations*; Bradley-Terry/Elo
  per Chatbot Arena for multi-system ranking).
- **Power.** Sample size scales as ~1/MDE² — halving the detectable gap needs 4× the data.
  At **n=94**, a 0.52–0.60 win-rate gives a ~95% CI of roughly **[0.42, 0.68]** with
  **binomial p = 0.12–0.76** — i.e. **not distinguishable from chance**. Detecting a true
  ~58% preference at 80% power needs **n ≈ 300–400**.

## 5. Distinguishing a real gain from reward-overfitting (sourced)

- **Gao, Schulman & Hilton, "Scaling Laws for Reward Model Overoptimization" (arXiv:2210.10760):**
  RL/BoN **reward-hacks an imperfect proxy** — proxy reward rises while *gold* reward falls.
  Guard with a **gold-vs-proxy** evaluation and KL control.
- This is the repo's own `deliberation_roofline` thesis: **a leaky verifier caps achievable
  quality.** Here *the judges are the verifier*; κ≈0.06 / observed-agreement ≈0.53 means a
  **leaky verifier that cannot certify a subtle gain** regardless of compute.

## 6. Recommended protocol (the thesis) — run ONCE, pre-registered

1. **Primary claim = win-rate vs 0.5**, validated by a pre-registered **binomial/sign test +
   bootstrap CI**, on a **properly powered (~300+), independent, third-party-authored**
   held-out set.
2. **Judges = a panel of ≥3 diverse families, majority vote** (PoLL), with **position-swap
   and length controls**.
3. **Reliability = a reported panel**: observed agreement + κ + **PABAK** + **Bias/Prevalence
   indices** (Byrt 1993). Never swap κ→AC1 to clear a κ-derived threshold.
4. **Anti-reward-hacking = gold-vs-proxy check** + held-out generalization (Gao 2023).
5. **Human-anchored calibration subset (~30 cases)** to license trusting the judges at all
   (report human↔LLM agreement; even human–human tops out ~80%).
6. **One analysis.** Fix metric, judges, N, and CI *before* running. Do not iterate judge
   protocols until one passes (that is p-hacking).

## 7. Implication for `RESULTS.md`

The current sole gate ("κ ≥ 0.40 / 2 families") is fragile on **same-direction skewed
pairwise** data and conflates *reliability* with the *capability claim*. Recommended change:
make the validated bar a **pre-registered win-rate CI excluding 0.5** (primary) **plus** a
reported reliability panel (κ + PABAK + bias/prevalence indices) as a *companion*, on a
properly powered third-party pack. Until then the RLVR adapter stays **candidate-only**:
honest directional signal, not a validated capability.

## References

- Feinstein AR, Cicchetti DV (1990). *High agreement but low kappa: I. The problems of two paradoxes.* J Clin Epidemiol 43(6):543–549.
- Byrt T, Bishop J, Carlin JB (1993). *Bias, prevalence and kappa.* J Clin Epidemiol 46(5):423–429. (PABAK)
- Warrens MJ (2010). *A formal proof of a paradox associated with Cohen's kappa.* J Classification 27:322–332.
- Wongpakaran N, et al. (2013). *Comparison of Cohen's kappa and Gwet's AC1.* BMC Med Res Methodol 13:61.
- Zec S, et al. (2017; 2023). *High agreement / low kappa* and *kappa vs AC1 chance-correction* (MethodsX 10:102212).
- Zheng L, et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS. arXiv:2306.05685.
- Verga P, et al. (2024). *Replacing Judges with Juries (PoLL).* arXiv:2404.18796.
- Gao L, Schulman J, Hilton J (2023). *Scaling Laws for Reward Model Overoptimization.* arXiv:2210.10760.
- Wolfe CR. *Applying Statistics to LLM Evaluations.* (bootstrap CIs, MDE, power.)

*Verification note: §2 (the κ paradox) passed adversarial 3-vote verification against primary
sources. §§3–6 rest on the named papers from the research fetch phase but were not individually
3-voted; verify specific figures before quoting in a headline.*
