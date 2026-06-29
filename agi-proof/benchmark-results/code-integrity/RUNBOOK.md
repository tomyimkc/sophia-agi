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

The current held-out split is **48 tasks → MDE ≈ 0.286 at 80% power**: it can only
resolve a ~29-point pass@1 swing. For a 15-point threshold you need **N ≈ 175**
(N ≈ 99 for 20 points). So either:
- grow the eval split to ≥144–175 tasks (`tools/gen_code_pack.py`) for a powered
  primary, **or**
- treat the 48-task pass@1 as a **coarse GO/NO-GO guardrail only** (like the
  wisdom-market retention probe), never a ranked headline.

Do not report a 48-task uplift as a validated number.

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
