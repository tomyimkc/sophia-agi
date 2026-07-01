# Master Training Recipe — the wisdom-gated on-device model

`canClaimAGI` **false**. This is the top-tier, enforceable structure for training a model that adopts
your **proven** integration ideas — and *only* the proven ones. The machine-checkable source of truth
is [`recipe_spec.json`](./recipe_spec.json); the gate that keeps it honest is
[`tools/lint_recipe.py`](../../tools/lint_recipe.py).

## The single most important factor
**The quality and correctness of the training SIGNAL (data + reward), verified by an external
oracle.** Not architecture, not scale, not hyperparameters — a model can only become as reliable as
the signal it learns from, and your verifiers/gates ARE that oracle. Corollary (the v5→v6 lesson):
**train the metric you will gate on**, never a proxy for it.

## The adoption rule (why this recipe can't overclaim)
An idea is folded into the model as **hard training signal** (`adopted: true`) ONLY IF it is
`validated` **AND** has a measured **ablation delta** (on/off number) **AND** names a passing
**gate**. No delta or no gate ⇒ it stays `adopted: false` (candidate/auxiliary). `lint_recipe.py`
enforces this; a candidate flipped to adopted fails CI. Today **1 of 16** ingredients is adopted
(the bench-a-06 wisdom-uplift SFT, κ=0.41, Δ+0.4534) — the rest are candidate/open and earn in by
measurement, not by hope. That selectivity *is* the frontier-grade move.

## The 6-layer structure (dependency-ordered; gate each layer before the next)
| # | Layer | Decides | Your proven / candidate blocks |
|---|---|---|---|
| 0 | **target-base** | the on-device constraint drives everything | NVFP4 low-RAM target (open — cert), OLMoE-1B-7B base |
| 1 | **data** | the signal (most important) | wisdom-uplift SFT **(adopted)**, gate-clean distill, verifier-labeled DPO, council seeds; decontam guard |
| 2 | **objective** | train the *verified* metric | v6 output-space KD+top1, abstention objective, process supervision, deterministic-verifier reward |
| 3 | **verification-in-loop** | gate = reward *and* stop-criterion | trust-boundary (validated), gate-bounded thinking, autoresearch firewall |
| 4 | **quant-serve** | deploy constraint trained-in | v6 QAT + conformal-abstention serve (ship at 0.92 by abstaining on flips) |
| 5 | **eval-promotion** | what you may claim | IEC 8-pillar `claim_gate`, public failure ledger |

The through-line: **the same gates verify at inference, reward in training, and judge in eval** — one
oracle, three roles. That coherence is what makes the ideas compound instead of collide.

## The build method (how to construct + grow the receipt)
1. **inventory + gate** — sort every idea into validated / candidate / open (audit the ledger).
2. **ablate each** — measure the on/off delta *before* folding anything in (no delta = decoration).
3. **compose 0→5** — gate each layer before the next builds on it.
4. **pre-register** each addition's expected effect (`Spark-Theory-Test-Forecast.md`).
5. **one lintable spec** — `recipe_spec.json` is the single source of truth.
6. **verifier as fitness** — iterate via `sophia_autoresearch` with the firewall intact; power-gated
   wins get promoted, the rest go to the ledger.

## Promotion workflow (how an ingredient moves candidate → adopted)
1. Run its **ablation** (adapter/feature ON vs OFF) on ≥3 seeds; record the delta.
2. Clear its **gate** (`claim_gate` GO, or the ingredient's own validated reference test) — with
   pre-registration honored.
3. Set `proofStatus: "validated"`, fill `ablationDelta`, then `adopted: true`.
4. `python tools/lint_recipe.py` must pass; add it to the model's training config.
5. If it later regresses, demote it and log the reason — the receipt tracks live truth, not history.

## Run the gate
```bash
python tools/lint_recipe.py            # OK only if every adopted ingredient is proof-gated
```
Wire it into `make claim-check` alongside `lint_claims` so the recipe is checked on every PR. The
receipt is now an **enforceable, auditable contract**: it cannot claim the model adopts an idea the
gate hasn't cleared. `canClaimAGI` stays false.
