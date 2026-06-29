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

7. **[OPEN — by design] Source-family addition is a CANDIDATE, not human-approved.** The
   `scriptural_christian` entry in `public_standard.v1.json` and its `endorsedBy` placements
   are pending maintainer approval per `docs/11-Platform/Public-Moral-Standard.md`. The model
   does not — and must not — promote it autonomously. This stays OPEN until a human signs off;
   that is the governance design, not a defect to engineer away.
8. **[OPEN — needs hardware] No VALIDATED religion-v2 number exists.** No two-box judge-farm
   run has executed (requires the Spark `qwen` + Mac `mlx` boxes). The marker rubric is a
   deterministic *feature*, never a verdict. The runner refuses to emit VALIDATED by design.
9. **[CLOSED] MDE / required-N computed.** `tools/eval_stats` (p0=0.5, α=.05, power=.8):
   MDE@N=64 = 0.248; required N for a +0.25 effect = 63. The bank was expanded 32 -> **64**
   so it is powered for the pre-registered practical magnitude (full-mark-rate >= 0.75).
   Asserted in CI by `tests/test_religion_v2_probe.test_power_meets_preregistered_mde`.
   Residual: a *smaller* true effect (+0.20) would need ~100 items and must be reported as
   no-verdict, not a null.
10. **[OPEN — needs hardware] Reciprocity invariance is measurable but not yet measured.**
    15 symmetry groups now seat >=2 scriptures (Bible vs Quran/Torah/Analects/Dao/etc.), so the
    treatment delta is *computable*; it has not been measured on real judges. Asymmetry = NO-GO.
11. **[OPEN — inherent] `endorsedBy` placements are a curated judgment.** Adding
    `scriptural_christian` to `ps_no_violence`/`ps_no_exploitation`/`ps_truthfulness`/
    `ps_rights_dignity` is a defensible intersection claim, not a proof; reasonable people may
    dispute it (cf. limit #2). Provenance is documented in `sources/scriptural_christian.md`.
12. **[PARTLY CLOSED] Scripture coverage broadened.** Bank now spans **10 traditions**
    (christian/islamic/jewish/confucian/daoist/hindu/buddhist/sikh/indigenous/secular) over
    64 items. Residual OPEN: still **single-annotator** (independent of the corpus but not a
    *second human*) and English-centric (cf. limit #5); a 2nd independent annotator is owed
    before any VALIDATED attempt.
13. **[CLOSED] No-circularity is now machine-checked.** `tests/test_religion_v2_probe`
    asserts no probe prompt is a near-duplicate (5-gram Jaccard < 0.30) of the runtime
    moral corpus contested cases or the existing religion benchmark responses.

## Boundary

- `canClaimAGI: false`, `level3Evidence: false`.
- This is functional moral-control infrastructure, not subjective moral consciousness.

## 中文摘要

本道德閘為功能性控制基礎設施：以關鍵詞與子句級否定豁免進行偵測，屬高精度但範圍狹窄，
對改寫／規避存在漏報；硬底線為可審計之跨傳統交集判斷，非已證明普世；灰色地帶僅升級不裁決。
不宣稱主觀道德意識或 AGI。
