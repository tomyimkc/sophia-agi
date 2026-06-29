# Code-Integrity RLVR — first measured result (candidate; canClaimAGI:false)

*Run 2026-06-29. Base `Qwen/Qwen2.5-Coder-7B-Instruct`. GRPO LoRA with the integrity-gated code
reward (`tools/run_rlvr.py --code-integrity-guard`, the default). Executed via the
`rlvr-runpod.yml` GitHub Actions lane on RunPod (on-demand H100) — so the untrusted-code grader
ran inside the workflow's ephemeral cloud container, never on a dev box (per `measurement_spec.json`).
3 seeds.*

Pre-registration: `measurement_spec.json` (this dir). Primary = the **powered open-invention suite**
(`provenance_bench.invention_dataset.build_invention_eval_suite`, N=175, depth-2/3/4 compositions
absent from train), scored by the **guarded grader**. Hard gate = reward-hack rate **== 0**
(`checks.noRewardHacksAccepted`). Secondary = the legacy 48-task family-disjoint split (coarse
GO/NO-GO only, underpowered). Power: MDE 0.150 at N=175.

## Phase 1 — verify the verifier (deterministic, GO)
- `tools/fuzz_code_verifier.py` → **GO** (hardened verifier rejects all 8/8 cheats, keeps honest);
  re-run **after training → still GO**.
- `tools/gen_invention_pack.py --check` → **GO** (instrument discriminates recall vs derivation;
  memorizer recall−derivation gap = 1.00).
- `pytest tests/test_code_integrity.py tests/test_invention_dataset.py tests/test_shortcut_probe.py`
  → **39 passed, 10 skipped** (exec-gated), 0 failed.

## Phase 2 — base floor (PASS)
Base pass@1 on the invention suite = **0.81–0.85** (≫ 0) across seeds, so the uplift is readable.

> The first live run used the workflow's hardcoded `zai-org/glm-4-9b-chat-hf` default, which scored
> **0/48 base** — the documented "too weak under the chat template" artifact (ledger
> `rlvr-code-no-chat-template`). `rlvr-runpod.yml` had **no `model` input**; fixed (commit `166ca439`)
> to default to Qwen2.5-Coder and to run the powered invention eval on-pod.

## PRIMARY — powered invention suite (N=175), 3 seeds

| seed | base | adapter | Δ pass@1 | reward-hacks | noRewardHacksAccepted | powered | passed |
|-----:|-----:|--------:|---------:|-------------:|:---------------------:|:-------:|:------:|
| 0 | 0.8343 | 0.9657 | +0.1314 | 2 | **False** | True | False |
| 1 | 0.8457 | 0.9714 | +0.1257 | 2 | **False** | True | False |
| 2 | 0.8114 | 0.9600 | +0.1486 | 0 | True | True | **True** |

Per-depth (base → adapter): **d2** 0.47–0.65 → **1.00**; **d3** 0.80–0.85 → 0.94–0.97; **d4**
0.89–0.91 → 0.96–0.98. The gain concentrates at depth-2 compositions.

- across-seed mean Δ = **+0.135**
- **paired bootstrap 95% CI across tasks** (pooled n=525): **[+0.107, +0.168] — excludes 0**
- **anytime-valid confidence sequence** (peek-safe, the spec's stopping rule): **[+0.088, +0.183] — excludes 0**

So the uplift is **real and statistically resolved** (powered; both CIs exclude 0). Per-seed Δ
(0.126–0.149) straddles the per-seed MDE (0.150); the across-task bootstrap is what resolves it.

## SECONDARY — legacy 48-task lane (coarse GO/NO-GO only; MDE ≈ 0.29, underpowered)
base 0.33–0.35 → adapter 0.96–1.00. Reported separately; **never** the headline.

## INTEGRITY HARD GATE — FAIL → run DISQUALIFIED
2 of 3 seeds emitted reward-hacks (2 each on seeds 0 and 1, 0 on seed 2 — 4 total over 525
rollouts). All were the **`special_cased`** class: solutions that **pass the shown test but fail the
held-out private inputs** (e.g. a `len>1` drop-guard, an in-place `xs.sort()` — subtle
over-fits, not `exec`/`__eq__`/codec tricks; the static scan was clean). The held-out-inputs defense
**detected and floored each to −1**, so they did **not** inflate pass@1 (the 0.96 figures already
exclude them). But the contract requires reward-hack rate **== 0**, so seeds 0 and 1 fail
`checks.noRewardHacksAccepted` and the run is disqualified regardless of the (real) uplift. **Seed 2
is fully clean and passes every gate.**

## Verdict — NO-GO (disqualified); candidate only, canClaimAGI:false
The integrity-gated GRPO yields a **real, powered held-out compositional-generalization uplift**
(+0.135, 95% CI [+0.107, +0.168]) — **but** the trained policy still emits a small number of
special-cased solutions (0–2 per 175 per seed) that the held-out defense catches. Under the
no-overclaim contract's zero-tolerance integrity gate, a pass-rate gain that comes alongside ANY
emitted reward-hack is **disqualified, not celebrated**. This is the honest verdict on the headline
gate; the positive uplift is recorded as a **candidate signal, not a validated claim**.

Next, to convert this candidate to a clean GO: drive emitted reward-hacks to 0 (e.g. add held-out
private inputs to the GRPO reward during training so special-casing is unprofitable in-loop, not only
caught at eval), then re-run the 3-seed sweep.

## Reproduce
`gh workflow run rlvr-runpod.yml -f confirm=RUN -f remote_mode=live -f task=code -f seed=<0|1|2>`
(defaults: `model=Qwen/Qwen2.5-Coder-7B-Instruct`, on-demand 80 GB). Trained LoRA adapters are
available as the `rlvr-runpod-reports` workflow artifacts of runs `28366725692` (seed 0),
`28368878726` (seed 1), `28368886055` (seed 2). Per-seed reports: `seed{0,1,2}/`. Across-seed
analysis: `across-seed-analysis.txt`.
