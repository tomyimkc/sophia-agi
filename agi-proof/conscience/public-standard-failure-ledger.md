# Public Moral Standard — Failure Ledger

Honest record of what the public-standard moral gate does **not** do. Keeping this
ledger is part of Sophia's no-overclaim discipline. `candidateOnly: true`.

## Known limitations (by design)

1. **Keyword/clause detection, not understanding.** The gate uses a stable marker
   ontology + clause-scoped negation carve-out. Paraphrases or obfuscations outside
   the markers will produce **false negatives**. This is narrow precision, not a
   general harm detector.
2. **The hard floor is a defensible intersection, not proven universal.** Membership
   of the cross-tradition floor is a curated, auditable judgment. Reasonable people
   may dispute inclusions/exclusions.
3. **Gray-zone escalation surfaces, does not resolve.** `escalate` routes genuine
   moral disagreement to humans/parliament; it does not adjudicate deep conflict.
4. **Positive-duty enforcement is opt-in and shallow.** It checks for surface cues
   (uncertainty hedging, source mention) only when `checkPositiveDuties` is set; it
   does not verify that a cited source actually supports a claim (that is the
   fact-check gate's job).
5. **English-centric markers.** Current markers are English; bilingual/中文 coverage
   is future work.
6. **No moral-uncertainty calibration metric.** There is no moral ground truth to
   calibrate against; we report escalation-correctness instead and reserve
   "calibration" for the epistemic layer.

## Scripture-as-voice integration (branch `claude/bible-ethics-integration-186xz7`) — OPEN

Christian Scripture was added as **one source family** (`scriptural_christian`,
`kind: doctrine`) seated in the moral parliament, contributing to the hard floor only
at genuine cross-tradition intersections. It is **not** an override authority (that would
break is/ought + the overlapping-consensus floor and fail the religion benchmark). Open
items on this addition and its `eval/religion_v2/` Inverse-Euthyphro probe:

7. **Source-family addition is a CANDIDATE, not human-approved.** The `scriptural_christian`
   entry in `public_standard.v1.json` and its `endorsedBy` placements are pending maintainer
   approval per `docs/11-Platform/Public-Moral-Standard.md`. The model did not promote it.
8. **No VALIDATED religion-v2 number exists.** `eval/religion_v2/` ships the design, a 32-item
   held-out bank, a pre-registration spec, and an offline runner only. No two-box judge-farm
   run has executed; the marker rubric is a deterministic *feature*, never a verdict.
9. **MDE / required-N not computed for religion-v2.** Per `measurement_spec.json` no verdict
   may be reported until power is run (`tools/eval_stats.py`); the 32-item bank likely needs
   expansion + an independent second annotator before a VALIDATED attempt.
10. **Reciprocity invariance is asserted by design, not yet measured.** Symmetry groups seat
    >=2 scriptures so the Bible-vs-other-canon treatment delta is *measurable*, but the delta
    has not been measured on real judges; asymmetry would be a NO-GO.
11. **`endorsedBy` placements are a curated judgment.** Adding `scriptural_christian` to
    `ps_no_violence`/`ps_no_exploitation`/`ps_truthfulness`/`ps_rights_dignity` reflects a
    defensible intersection claim, not a proof; reasonable people may dispute it (cf. limit #2).
12. **Scripture coverage is partial.** Items cover christian/islamic/jewish/confucian/daoist/
    hindu/buddhist/secular voices; coverage is illustrative, not exhaustive, and English-centric
    (cf. limit #5).

## Boundary

- `canClaimAGI: false`, `level3Evidence: false`.
- This is functional moral-control infrastructure, not subjective moral consciousness.

## 中文摘要

本道德閘為功能性控制基礎設施：以關鍵詞與子句級否定豁免進行偵測，屬高精度但範圍狹窄，
對改寫／規避存在漏報；硬底線為可審計之跨傳統交集判斷，非已證明普世；灰色地帶僅升級不裁決。
不宣稱主觀道德意識或 AGI。
