# Decontamination & Replication Caveats — Verification Toolkit (2026-06-28)

Independent reproduction is mandatory for any expert-facing claim. This checklist records the
*honest limits* of what the `REPRODUCE.md` runbook reproduces, so a reviewer is not misled into
treating a re-run of self-authored material as an independent confirmation. It mirrors the posture
of `agi-proof/third-party-replication/README.md`: the machine-checkable items can be self-run, but
the **independence** items can only be completed by an external human. `canClaimAGI` stays **false**.

## The load-bearing caveat: the packs are self-authored

- The two evaluation packs — `agi-proof/debunk-gate/overconfident-regime-pack.json`
  (the debunk / overconfident-regime pack) and
  `agi-proof/source-verifier/source-contamination-pack.json` (the source-contamination pack) —
  are **synthetic and authored by the same project** that authored the verifiers.
- Re-running the verifiers on these packs reproduces the *numbers*, but it does **not** constitute
  an independent test: the pack designer and the verifier designer are the same party, so the
  cases may (unintentionally) sit in the verifiers' comfort zone.
- A true third-party replication MUST **swap in an independently-authored pack** — false-premise /
  contamination cases written by the reviewer, not visible to the system before evaluation — and
  re-run the same commands. Only then do catch / over-block rates carry independent weight.

## Keys and operator independence

- All 2026-06-28 live numbers were produced with **one operator's** API keys
  (`LLMHUB_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_FACTCHECK_API_KEY`, and the relay
  `OPENAI_API_KEY`). A reviewer must use **their own** keys and record their environment.
- Model endpoints drift. The committed numbers assume `llmhub:claude-sonnet-4-6` as the answer
  model and `openrouter:deepseek/deepseek-chat` as the separated judge. If those are unavailable,
  substitute comparable models and expect the percentages to move; reproduce the **shape** of the
  finding (the trade-offs), not the exact figures.

## Statistical rigor: single-run vs multi-run

- Most committed numbers are **single-run candidates** (`runs:1`). A single run has no confidence
  interval and can swing with sampling temperature and model-day variance.
- Only the **attribution-swap** result meets the rigorous protocol: 3 runs, `answer-spec` separated
  from `judge-spec`, and a bootstrap 95% CI (`10.8% [9.3, 11.6]` caught; `0.0% [0.0, 0.0]`
  over-block). To promote any other number from candidate toward validated, re-run with
  `--runs >=3` and separated specs and report the CI.
- Self-judging inflates numbers. An earlier curated over-block of **70.6%** was found to be a stale
  mis-saved retrieve report and **withdrawn** (corrected to **5.9%**); a prior **5.9%** over-block
  was an artifact of the **same model answering and judging**. Always separate answer from judge,
  and treat answer==judge runs as a weak lower bound on over-block.

## Oracle coverage is domain-bounded (not a silver bullet)

- **Google Fact Check** covered **4/21** of the debunk pack — only the viral/general myths with a
  professional ClaimReview (Great Wall, Edison, brain-10%, blue-blood). The other verified debunks
  rest on the flagged **low-independence** LLM tail, which can be wrong (it rated the Mozart-twinkle
  myth incorrectly).
- **Wikidata / Crossref authoritative oracles** covered **0/43** of the source-contamination pack's
  "a fabricated 2023 study attributes X to Y" cases — those are an *existence-check* problem
  (does the cited study exist in Crossref?), not a claim-rating problem. The provenance layer is
  correct (4/4 on a Dickens/Shakespeare/Tolstoy/Twain authorship demo) but simply does not cover
  this pack's claim type.
- This is the **no-free-lunch** law in data: catching contamination requires EITHER an oracle that
  covers the specific claim (sparse), OR fail-closed strictness (which over-blocks clean answers),
  OR model knowledge (low independence, flagged). No single verifier catches everything; the
  trustworthy posture is a labelled composition that **abstains rather than vouches**.

## Reviewer checklist (independence items — external human only)

- [ ] I used a clean clone and recorded the commit hash + my environment.
- [ ] I ran Tier A (`python3 tests/test_*.py`) keyless and it passed.
- [ ] I ran `python3 tools/verify_replication_manifest.py` and it reported PASS.
- [ ] I supplied **my own** API keys (not the operator's).
- [ ] I authored an **independent** contamination/debunk pack the system had not seen, and re-ran the bench on it.
- [ ] I ran each headline at **>=3 runs with a bootstrap CI** and **answer-spec != judge-spec**.
- [ ] I reported the **misses** beside the catches (coverage bounds, over-block, low-independence verdicts).
- [ ] I did not describe a single-run candidate as a validated result, and did not make any AGI claim.

Reviewer:

Date:

Commit:

Environment:

Signature:
