# Open-model judge family (Leiden-aligned)

**Status: in_progress — backend landed, not yet used to grade a headline result.** The
registry + self-hostable backend below are implemented and tested; the value is fully met only
once a headline result is corroborated on a non-proprietary path. Tracked in
[`agi-proof/leiden-compliance.json`](../../agi-proof/leiden-compliance.json)
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

## What landed

- **`agent/judge_registry.py`** — classifies any `provider:model` judge id into `open_weights`
  (model has open weights) vs `self_hostable` (inference runs without a proprietary API) vs
  `non_proprietary_path` (both). This made the real gap precise: the current headline panel is
  **all open-weights but zero self-hostable** (every family is served via OpenRouter), so it has
  **no non-proprietary path** yet.
- **`agent/open_judge.py`** — a fail-closed, self-hostable source-discipline judge that talks to
  a local OpenAI-compatible endpoint (`OPEN_JUDGE_BASE_URL` / `OPEN_JUDGE_MODEL`). Same one-word
  verdict contract as `tools/llm_judge_score.py`, so it can serve as one of the ≥2 families
  without a proprietary provider. Unconfigured ⇒ `available()` False and `score()` returns
  `None` (it never silently falls back to a proprietary judge).
- **`tools/leiden_receipt.py`** now surfaces a `judge_independence` block classifying the
  committed panel and recording that the backend exists.

The existing rule is unchanged: ≥2 **independent** families, judge ≠ subject, inter-judge
agreement reported (Cohen κ ≥ 0.40 or a CI excluding zero).

## Remaining acceptance criteria

1. ✅ A self-hostable open judge backend + an openness registry, with offline tests.
2. ⬜ A headline result re-graded with the open judge as one of the ≥2 families, κ reported.
3. ✅ The receipt records judge openness (`judge_independence` in `leiden-compliance.json`).
4. ⬜ A failure-ledger entry opened/closed to reflect the change in reproducibility posture.

Until criteria 2 and 4 hold, the autonomous-direction value stays **partial** in the compliance
receipt and this gap stays `in_progress`.
