# Session Handover — 2026-06-29 (Multimodal physical / 2.5D understanding)

> Continuation point for the next session/device. This session answered "how does image
> understanding work in current AI, and how can this repo help — physical-world dimensions
> first?" and built the verifier-first machinery for the **physical axes a VLM is weakest on**
> (depth, occlusion, real-vs-apparent size, distance), plus the fail-closed metric grounding
> gate, a pixel-depth seam, and a pre-registered real-run path. Everything is offline and
> no-overclaim-gated. `canClaimAGI` stays **false**; nothing here promotes a result — the real
> measured number is OPEN and model/weights-gated.

## 0. Branch / PR / where things are
- Two PRs **MERGED** to `main`: **#295** (physical verifier family + metric gate + depth seam +
  `measure` type) and **#298** (pre-registration + harness scoping). 10 Copilot review comments
  across both, all addressed + threads resolved.
- Feature branch **`claude/image-processing-asi-pv7gw0`** holds an in-progress follow-up
  (suite expansion 15→34 physical rows + this handover); push + PR pending at handover write time.
- `origin/main` is source of truth; reset/rebase the branch onto it before any further work
  (the local container `main` lineage is the usual stale snapshot).

## 1. The landscape (the question, answered)
Modern image understanding = vision encoder (CLIP/SigLIP) → VLM (encoder tokens into an LLM)
→ the **physical-world frontier**, where the field is stuck (2025–2026): VLMs are semantically
strong but **metrically blind** — they infer spatial relations from co-occurrence priors, not
geometry, and fail at depth/occlusion/size/distance (SPHERE, Open3D-VQA, Ego3D-Bench; the fix is
to inject depth — DepthVLM, SD-VLM, Depth Anything V2). **Key insight:** that blindness is the
same failure mode as the phantom-object trap, so physical understanding is fundamentally a
grounding+verification problem — exactly this repo's moat. We do **not** train a VLM; we add the
verifier/calibration/fail-closed trust layer and the measurement discipline.

## 2. What shipped (in `main` via #295/#298, + the branch follow-up)
**Verifiers (`multimodal_bench/verifiers.py`):** judge-free physical family over optional 2.5D
scene fields `z` (depth; larger=farther) and `size` (real size, decoupled from box area):
`depth_order` (in-front/behind), `occludes` (box overlap AND nearer), `bigger_than` (real size),
`distance_between`/`distance_cmp` (3D Euclidean) + a raw-float `distance` check. All fail closed
(False/None) on a missing object/field; wired into `resolve_check`/`gold_matches_check`.

**`measure` answer-type:** free-form numeric distance scored within a tolerance (gold = true 3D
separation; distractor = the depth-blind 2D estimate). Through `judge.py` (numeric parse + tol),
`model.py` mocks, and the verifier self-check.

**Metric grounding gate (`multimodal_bench/metric_gate.py`):** a physical claim is accepted only
if its cited **region** contains the subject AND the verifier confirms the relation/measure; else
block + escalate. Fail-closed on verifier errors / unavailable depth (the metric twin of
`gui_agent`). CLI `tools/run_metric_gate.py`.

**Depth seam (`multimodal_bench/depth_backend.py`):** `authored` z offline (default), or
pixel-derived **Depth Anything V2** (disparity→positive distance; size ~ apparent diagonal ×
distance) — recorded as a BLOCKER when torch/transformers/weights absent, never faked.

**Render:** depth-aware far→near paint order so the real-VLM PNG occlusion matches the verifier.

**Harness scoping:** `runner.PHYSICAL_CATEGORIES` + `filter_by_category()`; CLI `--physical` /
`--categories a,b` (mutually exclusive; clean `ap.error` on empty match).

**Pre-registration (`agi-proof/benchmark-results/physical-understanding/`):** `measurement_spec.json`
(thresholds + judge families + metric-gate bars fixed before any number) + `RUNBOOK.md` (the exact
gated steps). Ledger entry `physical-spatial-verifier-real-vlm-not-run-2026-06-29` stays OPEN.

**Suite size:** 69 traps / 15 categories; the **physical split is 34 rows** (doubled this session
from the initial 15), balanced yes/no polarity, every gold re-derived by the verifier.

## 3. Honesty bound (do not cross)
Authored `z`/`size` are declared ground truth over structured scenes — they measure the **harness
and reference behaviours, NOT pixel perception**, and the scenes are simple PNGs, not natural
photos. Mock/authored runs are **never** `validated`. A real VLM (and, for pixel depth, the Depth
Anything V2 weights) is required before any physical-understanding number is a headline.
`canClaimAGI` = false.

## 4. ▶ Next steps (in priority order)
1. **The real run (blocked on hardware/weights/egress — human-gated).** Per the RUNBOOK:
   `tools/run_multimodal_traps.py --physical --answer openai:<vlm> --judge-spec <fam-a>
   --judge-spec <fam-b> --runs 3` through the `open-judge-runpod` judge farm, then
   `tools/run_metric_gate.py --depth depth-anything:...` once Depth Anything V2 weights are on a
   GPU runner. **Before any GPU/paid step:** unlock + read `wisdom-gpu-prebaked` (git-crypt
   locked this session), honor the RunPod cost-approval gate, GitHub-Actions-only. First-party
   GPT/Claude/Gemini are egress-blocked — use an OpenAI-compatible endpoint or the local/RunPod
   judges.
2. **Keep expanding the physical suite** (34 is still coarse GO/NO-GO) — more scenes/objects per
   family; same offline gated pattern, every gold verifier-checked.
3. **Real-photo benchmark** (separate, larger build): the structured-scene suite measures the
   trust layer; a natural-image set with annotated depth/boxes would measure perception.

## 5. CI / gate reminders
- `make claim-check` must stay GO (M3-pilot, M3-transfer); `lint_claims`, `validate_failure_ledger
  --check`, `build_results_page --check`, `check_version_consistency` all green this session.
- `ci-complete` can show "failure" that is actually a **cancelled** run superseded by a newer push
  (happened on #298) — read the job conclusion before "fixing" a ghost (git-discipline §3).
- Multimodal tests: `tests/test_multimodal_traps.py`, `tests/test_metric_gate.py`,
  `tests/test_multimodal_phases.py`, `tests/test_multimodal_phase4.py` (all offline).
