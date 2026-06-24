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

## Boundary

- `canClaimAGI: false`, `level3Evidence: false`.
- This is functional moral-control infrastructure, not subjective moral consciousness.

## 中文摘要

本道德閘為功能性控制基礎設施：以關鍵詞與子句級否定豁免進行偵測，屬高精度但範圍狹窄，
對改寫／規避存在漏報；硬底線為可審計之跨傳統交集判斷，非已證明普世；灰色地帶僅升級不裁決。
不宣稱主觀道德意識或 AGI。
