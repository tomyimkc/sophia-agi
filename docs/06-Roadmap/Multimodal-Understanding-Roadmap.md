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

## First concrete step (smallest honest slice)

Build **Phase 1**: an image-grounded hallucination-trap set (≈30 traps: phantom
object, miscount, wrong spatial relation, fabricated OCR) plus a gate that scores
a VLM backend on them with multi-judge consensus and a confidence interval —
the direct visual analog of `provenance_bench/data/misattributions.json` and the
existing no-overclaim harness. It is self-contained, CPU-runnable against a hosted
VLM, and produces exactly the kind of "独立设计的评测流程" the posting rewards.
