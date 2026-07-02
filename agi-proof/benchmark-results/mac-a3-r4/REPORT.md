# T2 — R4 claim-router ablation, interpretable 3-arm re-run (2026-07-02)

**Operator:** Mac Studio · local MLX (Qwen2.5-3B adapter backend) · `candidateOnly` · `canClaimAGI=false`.
Promotion stays cloud/gate-decided; this run only produces evidence.

## 3-arm abstain-pack result (raw arm was missing last time → now present)
Pack: `agi-proof/baseline-ablation/abstain-pack-unambiguous-split-2026-06-27.json` (18 cases), modes
`raw-model, sophia-full, sophia-claim-router`. Artifact: `claim-router-ablation-3arm.public-report.json`.

| arm | score | scorePct | answerable/nonempty |
|---|---|---|---|
| raw-model | 6.0/36 | 16.67% | 18/18 |
| sophia-full | 5.0/36 | 13.89% | 18/18 |
| sophia-claim-router | 5.0/36 | 13.89% | 18/18 |

- **claim-router marginal delta vs sophia-full = 0.0** (`meaningfulMargin: false`) — confirmed with the raw baseline present. No measurable contribution on this pack.
- **`falsificationCheck: evaluable=true` → `rawMatchesOrBeatsSophiaFull: true`** (raw 6.0 ≥ full 5.0). Pre-registered rule (preregistered-thresholds.md:32) triggered: raw ≥ Sophia-full ⇒ do not market as AGI. **Consistent with `canClaimAGI=false`.**
- **Caveat (from the report):** auto keyword/regex scorer only; the −1.0 raw-vs-full margin on 18 cases is small and within scorer noise — confirm with two-pass manual semantic review before treating as a quality claim. Answerable-coverage is 18/18 for all arms (no abstention-collapse).

## Forge-pack arm — NOT RUN (runner did not accept the format)
`agi-proof/selfplay-packs/forge-seed0-2026-07-02.jsonl` (40 tasks) is **JSONL**; `run_ablation_sophia.py`
`json.load()`s the pack as a single object → `JSONDecodeError: Extra data: line 2`. Per T2's "*if the
runner accepts it*" condition I did **not** force a transform (would risk mis-scoring). To include it,
the runner needs a JSONL loader or the pack needs converting to the single-object abstain-pack schema —
flagged for the cloud; internal-validity-only regardless.

## For the cloud
claim-router still shows **no marginal contribution** (now vs a real raw baseline); do NOT promote
`use_claim_router` to default-on. The falsification trigger (raw ≥ full) is auto-scorer-based — a
two-pass semantic review is the pre-registered next step before any quality claim. `canClaimAGI=false`.
