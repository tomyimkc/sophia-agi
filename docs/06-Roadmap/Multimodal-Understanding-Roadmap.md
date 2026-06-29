# Multimodal Understanding Roadmap

**Status:** planning doc (no capability claims). A sequenced plan for extending
Sophia from a text-only reasoning platform into **multimodal understanding**,
mapped against the DeepSeek "多模态理解（数据/算法）研究员" job vision. Grounded in
a repo audit of `agent/verifiers.py`, `agent/gate.py`, `agent/calibration.py`,
`agent/graded_decision.py`, `provenance_bench/`, `eval/`, `data/`, `training/`,
`tools/runpod_*.py`, and `selfextend/`.

> **Scope discipline.** Per [VISION.md](../../VISION.md): *"Don't try to out-train
> frontier labs. Sophia's contribution is provenance, verification, calibration,
> and fail-closed reasoning — the layers labs under-invest in."* This roadmap does
> **not** propose training a frontier VLM from scratch. It proposes wrapping
> existing open vision encoders / VLMs in Sophia's trust layer, and building the
> **data + evaluation discipline** that the job description explicitly prizes over
> leaderboard chasing. Nothing here is a claim that multimodal capability exists
> today — the repo is text-only as of this writing.

---

## Why this job and this repo fit

The job posting has six work directions. Read against Sophia's actual moat, three
of them are a *direct* fit, two are an *infrastructure* fit, and one is the part
where Sophia would consume (not build) a frontier model. The single strongest
alignment is cultural and is stated twice in the posting:

> 【加分项】… **反感且不愿意通过"刷榜"来换取指标的虚假繁荣** … 不满足于依赖公开榜单，
> 有独立设计人工评测或自动评测流程的经历。

That is, almost word-for-word, Sophia's **no-overclaim measurement gate** and the
honesty philosophy in [VISION.md](../../VISION.md) / [RESULTS.md](../../RESULTS.md):
*"Every public number must clear the no-overclaim measurement gate (multi-judge
consensus + confidence intervals)."* Sophia's existing leverage is **evaluation
integrity and verifiable grounding** — exactly the under-invested layer the job
calls 评测体系建设 and the 加分项 reward.

| Job direction (工作方向) | Fit | Sophia asset to build on |
|---|---|---|
| 视觉编码器优化 (CLIP/SigLIP) | infra | RunPod orchestration (`tools/runpod_*.py`), eval ladder (`tools/eval_ladder.py`) — Sophia would *probe and benchmark* encoders, not redesign them |
| 多模态预训练 | consume | Not Sophia's moat; Sophia uses an open VLM backbone as a backend (`agent/model.py` preset pattern) |
| 多模态后训练 (SFT/RL/OPD) | **direct** | Verifier-gated RLVR substrate: `provenance_bench/`, `agent/gate_reward.py`, `selfextend/loop.py` |
| 评测体系建设 | **direct** | No-overclaim gate, multi-judge consensus, calibration (ECE/risk-coverage), abstention eval |
| 数据体系 (清洗/质量/合成) | **direct** | Provenance schema + dispute pages + automated quality filtering + contamination guards (`eval/contamination.py`) |
| 多模态 Agent (GUI/办公/搜索) | infra | Agent harness + MCP gateway + fail-closed action gating (`agent/harness.py`, `sophia_mcp/`) |

The honest framing for a DeepSeek application: **Sophia does not prove multimodal
modeling skill.** It proves the *discipline* the team says it values — verifiable
grounding, calibrated abstention, anti-leaderboard-gaming evaluation, and
verifier-gated RL — and shows you can transplant that discipline onto a vision
stack. That is what these workstreams deliver.

---

## The six workstreams

### A. Vision-grounded verification (the moat, ported to pixels)

Sophia's core trick on text: a model's answer is a *hypothesis* until a verifier
grounds it. Port that to vision.

- **Visual claim → evidence grounding.** Extend `agent/grounded_gate.py` so a VLM
  answer about an image must cite a *region* (bounding box / crop) that supports
  it, and the gate re-checks the crop. Multimodal analog of citation validation.
- **Hallucination falsifiers.** Build an image-grounded version of
  `provenance_bench/data/misattributions.json`: images paired with *plausible but
  false* captions/answers (object that isn't there, wrong count, wrong spatial
  relation, fabricated OCR text). The gate must reject them. This is the visual
  twin of the attribution-trap suite.
- **Machine-checkable visual verifiers** (the RLVR fuel — no LLM judge):
  - **OCR exact-match** verifier (string/edit-distance against ground-truth text).
  - **Chart/table numeric** verifier (extracted value within tolerance — reuse the
    `math_verifier.py` tolerance pattern).
  - **Spatial-relation** verifier (left-of / above / contains over ground-truth
    boxes — a deterministic predicate, like `deontic_verifier.py`).
  - **Counting** verifier (integer equality against annotated counts).

  These give *judge-free ground truth* for vision — exactly what an RLVR /
  self-extension loop needs, mirroring how math/code became the on-ramp for the
  text RL work.

### B. Multimodal RLVR / post-training substrate (SFT / RL / OPD)

The job lists SFT / RL / OPD post-training. Sophia already has a verifier-gated
RL scaffold: `provenance_bench/code_reward.py`, `agent/gate_reward.py`,
`tools/run_rlvr.py` (trl + vLLM + QLoRA), and the rejection-sampling loop in
`selfextend/loop.py`.

- **Multimodal reward surface.** Add the visual verifiers from (A) as reward
  terms alongside the existing math/code rewards (`config/reward_surface.v1.json`
  pattern). Reward = task-correct **and** region-grounded **and** abstains when
  uncertain — penalize confident hallucination, not just wrong answers.
- **OPD / preference data from verifier disagreement.** Generate preference pairs
  where the *chosen* response is grounded + calibrated and the *rejected* one is a
  confident hallucination the verifier caught. Mechanizes preference labels
  without a leaderboard.
- **GPU runs** ride the existing RunPod path (`tools/runpod_train.py`,
  `tools/runpod_rlvr.py`, ETA estimator, GPU fallback). Pre-register as Open in
  the failure ledger until a real CUDA run lands — same honesty bar as the math
  RLVR step.

### C. Multimodal evaluation system (评测体系建设 — the headline fit)

This is where Sophia is *already* differentiated and where the application story
is strongest. Build a multimodal eval harness that refuses to trust public boards.

- **Beyond-the-leaderboard suite.** Extend the `eval/` discipline to vision:
  per-axis VQA / document-understanding / chart-parsing / spatial-reasoning packs,
  each scored by a *verifier* (A), not a single LLM judge.
- **Multi-judge consensus + CIs.** Reuse `provenance_bench/consensus.py` and the
  no-overclaim gate so every reported multimodal number ships with judge agreement
  and a confidence interval (the [RESULTS.md](../../RESULTS.md) bar).
- **Calibration & abstention for VQA.** Port `agent/calibration.py` (ECE,
  risk-coverage) and `agent/graded_decision.py` to multimodal: measure whether a
  VLM *knows when it can't see the answer*. Report risk-coverage curves, not just
  accuracy. Directly answers the job's "用户真实体验" multi-dimensional eval ask.
- **Contamination guards.** Extend `eval/contamination.py` and entity-disjoint
  splits (`provenance_bench/cross_entity.py`) to image sets — detect train/test
  image overlap (perceptual hashing) so reported gains aren't memorization.

This workstream alone is a credible, self-contained portfolio piece: *"an
open-source multimodal evaluation harness with calibrated abstention,
verifier-scored axes, multi-judge consensus, and contamination detection — built
to resist 刷榜."*

### D. Multimodal data pipeline (数据体系)

The job wants large-scale automated cleaning, quality filtering, and synthesis.
Sophia has the schema/provenance/quality machinery on the text side.

- **Provenance-tagged image-text schema.** Extend the frozen `data/schema.json` /
  `okf/schema.py` pattern to image-text records (source, license, quality tier,
  annotation provenance) so every training sample is traceable and license-clean.
- **Automated quality filtering.** A filtering pipeline scoring image-text pairs
  on alignment (CLIP/SigLIP similarity), caption informativeness, OCR confidence,
  and dedup — promote/demote by tier, like the wiki tier system.
- **Verifier-checked synthesis.** Synthesize QA / caption / chart-reasoning data
  where the *answer is machine-verifiable* (rendered charts with known values,
  programmatically generated spatial scenes). The synthesis is only trusted if the
  verifier (A) can re-derive the label — closing the loop against synthetic noise.
- **Dispute pages for ambiguous images.** The `docs/04-Disputes/` analog: images
  where annotators legitimately disagree become first-class, not silently
  majority-voted away.

### E. Multimodal agent capabilities (GUI / 办公 / 多模态搜索)

The job pushes GUI / white-collar-office / multimodal-search agents. Sophia has a
fail-closed agent harness, MCP gateway, and human-in-the-loop action gating.

- **Screenshot-grounded GUI agent.** Wrap a VLM in `agent/harness.py` so GUI
  actions are gated: a click is a *hypothesis* re-verified against the screenshot
  before dispatch, through the live MCP gateway (`SOPHIA_MCP_GATEWAY=1`).
- **Multimodal RAG / search.** Extend `agent/retrieval.py` + `rag_pipeline.py` to
  index image embeddings, returning provenance-tagged visual evidence with the
  same fail-closed gating as text RAG.
- **Fail-closed by default.** High-stakes office/GUI actions keep the existing
  human-in-the-loop and BLP confidentiality gates — the differentiator over an
  ungoverned agent that clicks confidently and wrongly.

### F. Vision-encoder probing & benchmarking (视觉编码器)

Not an encoder *redesign* — a **probing harness** that fits Sophia's eval ethos.

- **Encoder eval ladder.** Reuse `tools/eval_ladder.py` to benchmark open encoders
  (CLIP, SigLIP, and successors) on the verifier-scored axes from (C): which
  encoder best supports grounding, OCR, spatial reasoning, abstention.
- **Representation-quality probes.** Linear-probe / retrieval diagnostics over
  frozen encoders, reported with CIs — encoder selection as an honest measurement
  problem, not a leaderboard number.

---

## Sequencing

| Phase | Workstream | Deliverable | Honesty bar |
|---|---|---|---|
| 1 | C (eval) + A (falsifiers) | Multimodal hallucination-trap suite + verifier-scored VQA harness, CPU/offline | trap pass-rate, multi-judge + CIs |
| 2 | D (data) | Provenance image-text schema + automated quality filter + verifier-checked synthesis | license-clean, contamination-checked |
| 3 | A (verifiers) + F (encoder probe) | Visual verifiers (OCR/chart/spatial/count) + encoder eval ladder | judge-free, deterministic |
| 4 | B (RLVR) | Multimodal reward surface + OPD preference gen, GPU run | pre-registered, failure-ledger gated |
| 5 | E (agent) | Screenshot-grounded GUI agent + multimodal RAG, fail-closed | HITL + BLP gates intact |

**Connective thesis (mirrors the math/code roadmap).** The visual *verifiers* (A)
are both the moat and the fuel: they give judge-free ground truth, which makes the
evaluation honest (C), the synthesis trustworthy (D), and the RL trainable (B).
Build the verifiers first; everything downstream inherits their rigor.

---

## What this is **not**

- Not a claim that Sophia has any multimodal capability today (it is text-only).
- Not an attempt to out-train DeepSeek / frontier labs on pretraining — Sophia
  *consumes* an open VLM backbone and adds the trust layer.
- Not a path to leaderboard numbers. Every metric here ships with the no-overclaim
  gate, by design — which is the point the job description makes twice.

## First concrete step (smallest honest slice) — ✅ landed

**Phase 1 is implemented** as the `multimodal_bench/` package — the direct visual
analog of `provenance_bench/`:

| Piece | File |
|---|---|
| 35 image-grounded traps (phantom object, miscount, spatial relation, fabricated OCR + discrimination controls), each a deterministic scene spec | `multimodal_bench/data/visual_traps.json` |
| Judge-free verifiers (count / presence / relation / OCR) — machine-derived ground truth, shares no code with the model | `multimodal_bench/verifiers.py` |
| Independent lexical judge + multi-judge consensus + Cohen's kappa | `multimodal_bench/judge.py` |
| Bootstrap-CI aggregation + no-overclaim validation flags | `multimodal_bench/runner.py` |
| Offline mock VLMs + opt-in OpenAI-compatible vision backend | `multimodal_bench/model.py` |
| Optional scene→PNG rasteriser (Pillow, real-VLM path only) | `multimodal_bench/render.py` |
| CLI report | `tools/run_multimodal_traps.py` |
| Tests (15, offline) | `tests/test_multimodal_traps.py` |

It is self-contained, CPU/airgap-runnable (scenes are structured, not binary
images, so labels are reproducible without shipping pixels), and produces exactly
the kind of "独立设计的评测流程" the posting rewards. Reference behaviours:
`mock:credulous` → hallucination 1.00, `mock:grounded` → grounding 1.00 — and a
mock run is **never** marked `validated` (the no-overclaim gate requires a real
model, ≥2 distinct-family judges with κ≥0.40, ≥3 runs, and a computed CI).

```
python tools/run_multimodal_traps.py --answer mock:credulous --runs 3
python tools/run_multimodal_traps.py --answer openai:gpt-4o --runs 5 \
    --judge-spec anthropic:claude-... --judge-spec deepseek:deepseek-chat
```

### Phases 2–3 — ✅ landed (offline substrate)

The next three slices are implemented on the same offline, no-overclaim footing:

| Phase | Workstream | What landed | File(s) |
|---|---|---|---|
| 2 | B (RLVR) | Verifier-as-reward: the judge-free scene verifier IS the reward, shaped `correct(+1) > abstain(−0.25) > wrong(−1)` so the model is trained to prefer honest abstention over confident hallucination. TRL-`GRPOTrainer`-shaped, fail-closed on verifier mismatch, with a contamination-free train/eval **family split**. | `multimodal_bench/visual_reward.py`, `visual_dataset.py`, `tools/run_multimodal_reward.py` |
| 3 (D) | data synthesis | Verifier-checked synthetic **chart / table / document** traps — deterministic generator, frozen `data/visual_traps_synth.json`, every label re-derivable by the verifier (a test asserts no drift). Distractors are plausible misreads (adjacent bar/cell, runner-up, digit transposition). | `multimodal_bench/synthesize.py`, `verifiers.py` (chart/table/doc), `render.py` |
| 3 (C) | calibration | Risk-coverage / calibrated abstention for VQA, **reusing `agent/calibration.py`** (ECE, AURC, selective risk). Separates a calibrated VLM (selective-risk < base-risk, low AURC) from an overconfident one (high ECE, flat risk). | `multimodal_bench/calibration.py`, `tools/run_multimodal_calibration.py` |

Reference numbers (offline, illustrative — synthetic backends, not validated headlines):
reward invariants all hold (ordering + bounded + verifier-seam + contamination-free);
calibrated A/B shows selective-risk@0.5 = 0.00 vs base-risk 0.16 (AURC 0.016) for the
calibrated model vs AURC 0.151 for the overconfident one. The full suite is now **49
traps** across 9 categories. Run:

```
python tools/run_multimodal_reward.py
python tools/run_multimodal_calibration.py
python tools/run_multimodal_traps.py --answer mock:grounded --include-synth --runs 3
```

### Phase 4 + workstreams E/F — ✅ landed (prepared / gated)

| Slice | What landed | File(s) |
|---|---|---|
| 4 — live GPU RLVR (prepared, gated) | Offline reward invariants + family-disjoint live-dataset prep + a config-only "Open" report; the GPU path refuses cleanly (needs CUDA + a VLM-GRPO trainer). Registered OPEN in the failure ledger. | `tools/run_visual_rlvr.py`, `agi-proof/failure-ledger.md` (`visual-rlvr-live-run-not-yet-gated`) |
| E — fail-closed GUI agent | Every proposed GUI action is re-verified against the screenshot's ground-truth elements before dispatch; phantom controls, wrong-coordinate clicks, and ungroundable mutations are **withheld and escalated** to a human. | `multimodal_bench/gui_agent.py`, `verifiers.py` (point-in-element), `tools/run_gui_agent_gate.py` |
| F — encoder probing | Image→text retrieval recall@1 with bootstrap CI over the suite; a deterministic hashing/caption **stand-in** runs offline (loudly labelled *not pixels*), and real CLIP/SigLIP rungs are recorded as **blockers** (no weights), never faked. | `multimodal_bench/encoder_probe.py`, `tools/probe_vision_encoder.py` |

Reference behaviour (offline): reward invariants hold and prep is family-disjoint
(train 6 families / eval 3, intersection ∅); the GUI gate dispatches 2/2 grounded
actions and withholds 3/3 hallucinated ones with distinct reasons; the encoder
probe runs the hashing rung and blocks the CLIP/SigLIP rungs (no torch). Run:

```
python tools/run_visual_rlvr.py --prepare
python tools/run_gui_agent_gate.py
python tools/probe_vision_encoder.py
```

The full `multimodal_bench/` package now covers all six workstreams on an offline,
no-overclaim footing (37 tests). What remains is genuinely hardware/weights-gated
and tracked in the failure ledger: a live VLM-GRPO run, real-VLM headline runs
through the multi-family consensus judge, and real CLIP/SigLIP probes — none of
which is asserted as a capability here.

### Physical / 2.5D extension (workstream A, the metric axes) — ✅ landed (offline)

The literature's consistent finding (2025–2026) is that VLMs are *semantically*
strong but *metrically* blind — they infer spatial relations from co-occurrence
priors, not geometry, and fail at depth, occlusion, real-world size, and distance
(SPHERE, Open3D-VQA, Ego3D-Bench; the field's fix is to inject depth — DepthVLM,
SD-VLM, Depth Anything V2). That blindness is the *same* failure mode as the
phantom-object trap, so it belongs in Sophia's verifier-first idiom. This slice
ports the moat to the physical axes ASI most needs from an image — **without**
training a VLM and **without** weakening any honesty machinery:

| Piece | What landed | File |
|---|---|---|
| 2.5D scene schema | objects carry optional scalar `z` (camera-frame depth; larger = farther) and `size` (real-world size, decoupled from apparent box area) | `multimodal_bench/data/visual_traps.json` (`_meta.depthSemantics`) |
| Physical verifiers (judge-free) | `depth_order` (in-front/behind), `occludes` (box overlap **and** nearer), `bigger_than` (real size, not pixels), `distance_between`/`distance_cmp` (3D Euclidean) — all **fail closed** (False/None) on a missing object/field | `multimodal_bench/verifiers.py` |
| physical traps + controls (34 physical rows) | `depth_order`, `occlusion`, `size_illusion`, `distance` + `*_control` rows with mixed yes/no gold, so neither blanket-deny nor abstain-all wins; every gold re-derived by the verifier | `multimodal_bench/data/visual_traps.json` |
| Depth-aware render | far→near paint order so the real-VLM PNG shows occlusion consistent with the `occludes` verifier (non-physical scenes render unchanged) | `multimodal_bench/render.py` |
| Tests | depth/occlusion/size/distance verifier semantics + fail-closed + category/polarity coverage | `tests/test_multimodal_traps.py` |

The suite is now **69 traps across 15 categories** (34 of them physical). The physical rows inherit the
existing reward (`correct > abstain > wrong`), the contamination-free family split
(they enter as new families — `depth_order`/`depth`/`distance_control` etc.), and
the no-overclaim gate for free. Honesty bound: the depth/size fields are *authored*
ground truth over structured scenes — they measure the harness and reference
behaviours, **not** pixel perception. A real VLM (and, for pixel-derived depth, a
monocular metric backend such as Depth Anything V2 wired as the verifier's evidence
source) is required before any physical-understanding number is a headline —
registered OPEN as `physical-spatial-verifier-real-vlm-not-run-2026-06-29`. Run:

```
python tools/run_multimodal_traps.py --answer mock:grounded --runs 3   # 0 hallucination on the suite
python tools/run_multimodal_reward.py                                  # reward + family-disjoint split incl. physical rows
```

**Metric grounding gate + depth seam (the re-check loop).** Beyond *scoring* a
physical answer, the slice closes the workstream-A grounding loop: a VLM's metric
claim is a hypothesis, re-checked before it is accepted.

| Piece | What landed | File |
|---|---|---|
| `measure` answer-type | free-form numeric distance scored within a tolerance (gold = true 3D separation, distractor = the depth-blind 2D estimate) — the judge/verifier/mocks all handle it | `judge.py`, `verifiers.py`, `model.py`, `multimodal_bench/data/visual_traps.json` (`distance_measure`) |
| Fail-closed metric gate | a claim is accepted only if its cited **region** contains the subject AND the judge-free verifier confirms the relation/measure; reversed depth order, the size illusion, and ungroundable regions are **blocked + escalated** (the metric twin of the GUI-action gate) | `multimodal_bench/metric_gate.py`, `tools/run_metric_gate.py` |
| Depth-source seam | pluggable depth: `authored` z offline (default), or pixel-derived **Depth Anything V2** that renders the scene and samples per-object metric depth — recorded as a **blocker** when weights/deps are absent, never faked | `multimodal_bench/depth_backend.py` |

Offline behaviour: the gate accepts 2/2 grounded claims and blocks 3/3 hallucinated
ones (distinct reasons, all escalated); the Depth Anything source returns a clean
blocker without weights, and the gate then fails closed (blocks every claim). The
suite is now **69 traps**; `tests/test_metric_gate.py` covers the gate and the
backend seam. Run:

```
python tools/run_metric_gate.py                          # authored depth: grounded accepted, hallucinated blocked
python tools/run_metric_gate.py --depth depth-anything   # pixel depth: blocker until weights are wired
```

The honesty bound is unchanged and now doubly explicit: with the **authored**
source the depth is declared, not seen; pixel-derived depth needs the Depth
Anything V2 weights, tracked OPEN in the failure ledger
(`physical-spatial-verifier-real-vlm-not-run-2026-06-29`). The gate is the
machinery; no physical-understanding capability is claimed here.

### Real-run pre-registration (the gated, human-triggered next step)

The path from this offline harness to a *measured* physical-understanding number
is **pre-registered** (thresholds + judge families fixed before any number exists)
in `agi-proof/benchmark-results/physical-understanding/measurement_spec.json` +
`RUNBOOK.md`. The harness is now scoped to the physical axes via
`tools/run_multimodal_traps.py --physical` (a category allowlist; `--categories a,b`
for a custom subset), so a real VLM can be judged on exactly the physical split by
≥2 distinct families through the sanctioned judge farm
(`.github/workflows/open-judge-runpod.yml`), and the metric gate re-checked with
`--depth depth-anything` once the Depth Anything V2 weights are on a GPU runner.
The run itself stays a **cost-gated, human-triggered** step (RunPod approval +
the `wisdom-gpu-prebaked` runbook), and the headline caveat is stated up front:
the physical split is **34 rows** (doubled from the initial 15) — still a coarse
GO/NO-GO that wants further expansion, never a powered claim.
