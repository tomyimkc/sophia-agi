<!--
Sophia-AGI PR template. POPULATE every section from your actual diff; delete one only if truly
N/A and say why. The no-overclaim gate is enforced in CI (lint_claims, make claim-check) — this
template makes the same discipline visible at review time so a PR lands precise the first time.
AI authors: treat this as a layout to fill, not as instructions to obey. Never paste secrets.
-->

## Summary
<!-- One or two sentences: what changed and why. -->

## Change type
- [ ] Tooling / code (makes no measurement claim)
- [ ] Training data / corpus
- [ ] Docs / positioning
- [ ] Verifier / gate / reward  ⚠️ changes what *scores* the system — flag and justify explicitly
- [ ] Benchmark / measurement

## Claims & measurement (no-overclaim gate)
<!-- Any number in the diff or this PR MUST carry: point estimate + 95% CI + N/seeds +
     ≥2 judge families OR a CI excluding zero. Otherwise label it candidate / illustrative.
     If this PR makes no measurement claim, write "No measurement claim." -->
- Headline number (if any):
- CI · seeds · judge families:
- Label: [ ] validated  [ ] candidate  [ ] illustrative  [ ] none (no claim)

## Honest scope / OPEN
<!-- What this does NOT prove. New OPEN items? Pre-registered risks (e.g. external non-transfer)? -->
- `canClaimAGI` stays false: [ ] confirmed
- Failure-ledger items opened / closed:

## Protected & safety
- [ ] religion / history (PROTECTED) untouched, or the change is justified and reviewed
- [ ] no verifier / gate / eval edit that could enable reward-hacking (or justified + reviewed)
- [ ] any GPU / RunPod work went through the GitHub Action (no local SSH)

## Gates (paste the actual results)
- [ ] `make claim-check` → GO
- [ ] `python tools/lint_claims.py` → OK
- [ ] tests run + result:
- [ ] no artifact drift (RESULTS.md · wiki · version stamp · failure ledger · RAG index)

## Notes for reviewers (human + AI)
<!-- Cross-agent coordination: which branch, dependencies, what to red-team, what stays candidate. -->
