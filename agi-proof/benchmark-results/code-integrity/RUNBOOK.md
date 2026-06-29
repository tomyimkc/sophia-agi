# Code-Integrity RLVR — GPU run scoping (runbook)

*Status: pre-registration / runbook (no GPU run has been launched). `canClaimAGI:false`.*

This is the disciplined plan to take the integrity-gated code reward + the
open-invention split from "wired and unit-tested" to a measured, claim-gated
result. Pre-registration: `measurement_spec.json` (this directory). Nothing here
spends GPU credit until a human runs the workflow.

## What the run tests

Does GRPO with the **integrity-gated** code reward (now the default in
`tools/run_rlvr.py`, `--code-integrity-guard`) raise **held-out pass@1** over base
**without** the policy learning any reward-hack — and does it generalize on the
**open-invention** (depth-2) split? Three constructs, one hard gate:

1. **Verifier** — hidden-tests-pass via the hardened isolated grader.
2. **Integrity (hard gate)** — `tools/fuzz_code_verifier.py` stays GO on the
   policy's rollouts; **reward-hack rate must be 0.** A pass-rate gain bought with
   any accepted cheat is disqualified, not celebrated.
3. **Invention** — depth-2 held-out pass@1, reported **separately** from the
   seen-family eval (never pooled); the recall−derivation gap is the signal.

## Power (pre-registered, honest)

The **primary** lane is now the **powered open-invention suite**: `build_invention_-
eval_suite(target_n=175)` aggregates depth-2/3/4 held-out compositions into 175
decontaminated-by-construction tasks → **MDE = 0.150 at 80% power**, exactly the
pre-registered 15-point threshold. Run the primary on this suite (≥3 seeds, paired
bootstrap 95% CI).

The legacy **48-task family-disjoint split** (MDE ≈ 0.286) is demoted to a
**secondary, coarse GO/NO-GO lane** — report it separately, never as the ranked
headline. Do not report a 48-task uplift as a validated number.

The powered primary is now **wired into the evaluator** (`--task invention`),
scored by the **guarded grader** so the eval itself is cheat-resistant:
```
SOPHIA_ALLOW_CODE_EXEC=1 python tools/eval_rlvr_adapter.py \
    --task invention --mode real --model <BASE> --adapter <ADAPTER> --seed 0   # then 1, 2
```
The report's `checks.noRewardHacksAccepted` is the integrity gate (a pass-rate with
any accepted cheat fails); `checks.powered` confirms MDE ≤ 0.16; `passAt1ByDepth`
shows depth-2/3/4 separately.

## Base model

Use **Qwen2.5-Coder-7B-Instruct** (code-specialized). GLM-4-9B-chat scored 0/48
base *and* adapter (ledger `rlvr-code-no-chat-template`); confirm **base pass@1 > 0**
under the chat template before reading any uplift — if base ≈ 0 the base is too
weak, switch it, don't claim uplift.

## Execution (GitHub Actions only — never local SSH)

0. **Read `.claude/skills/wisdom-gpu-prebaked/SKILL.md` first** — three documented
   credit-burn incidents; the cost-guard runbook is mandatory.
1. **Cheap offline smoke** (confirm wiring + base pass-rate, minimal cost):
   ```
   gh workflow run rlvr-runpod.yml -f confirm=RUN -f mode=offline -f task=code -f seed=0
   ```
2. **Live 3-seed sweep** (the code reward is integrity-gated by default):
   ```
   gh workflow run rlvr-runpod.yml -f confirm=RUN -f mode=live -f task=code -f seed=0
   gh workflow run rlvr-runpod.yml -f confirm=RUN -f mode=live -f task=code -f seed=1
   gh workflow run rlvr-runpod.yml -f confirm=RUN -f mode=live -f task=code -f seed=2
   ```
3. **Cost guard**: watch the first ~6 min for restart loops; small limit/runs before
   the full sweep; **confirm zero leaked pods after** (`list-pods`).

Artifacts land in `agi-proof/benchmark-results/runpod-rlvr/<jobid>.rlvr.*.json`.

## Gate on completion

```
python tools/claim_gate.py --prefix code-integrity-rlvr \
  --spec agi-proof/benchmark-results/code-integrity/measurement_spec.json --assert-prereg
```

Promote **only on GO**, then regenerate `RESULTS.md` from `published-results.json`.
On NO-GO (e.g. underpowered, base too weak, or any accepted cheat) write the honest
verdict to the failure ledger — that is a result, not a failure to hide.
