# Open-model judge family (Leiden-aligned)

**Status: proposed — not implemented.** This is a design note, not a shipped capability. It is
tracked as an open gap in [`agi-proof/leiden-compliance.json`](../../agi-proof/leiden-compliance.json)
(`open_gaps: open_model_judge_family`) and in [LEIDEN-ALIGNMENT.md](../LEIDEN-ALIGNMENT.md).

## Why

The [Leiden Declaration](https://leidendeclaration.ai/) favours non-proprietary, publicly
governed tools (value 5: autonomous direction). Today Sophia's no-overclaim gate requires
**≥2 independent judge families**, and those families are served by proprietary inference
(OpenRouter, LLMHub — see [TOOL-DISCLOSURE.md](../TOOL-DISCLOSURE.md)). That is also a documented
weakness in the failure ledger: validation that cannot be reproduced without paid third-party
APIs is harder for an independent party to replicate.

Adding an **open-weights** judge family serves the Leiden value *and* closes a real
reproducibility gap.

## Proposed shape

- Add an open-weights judge (e.g. a permissively-licensed instruction model runnable on owned
  hardware or via a self-hosted endpoint) as a first-class entry in the judge-family registry.
- Keep the existing rule unchanged: ≥2 **independent** families, judge ≠ subject, inter-judge
  agreement reported (Cohen κ ≥ 0.40 or a CI excluding zero).
- Record in each measurement spec whether the corroborating families were open or proprietary,
  so a result can advertise "validated with at least one open-weights judge".

## Acceptance criteria (before this leaves `proposed`)

1. At least one open-weights judge wired into the eval harness and the judge-family registry.
2. A headline result re-graded with an open judge as one of the ≥2 families, κ reported.
3. The receipt records judge openness; `tools/leiden_receipt.py` can surface it.
4. A failure-ledger entry is opened/closed to reflect the change in reproducibility posture.

Until all four hold, the autonomous-direction value stays **partial** in the compliance receipt.
