# Huawei Noah PLM repo — takeaways for Sophia's training methodology

Source repo: <https://github.com/huawei-noah/Pretrained-Language-Model> (Huawei Noah's Ark Lab).
Status of everything in this doc: **candidate / proposal** — nothing here is adopted until it clears
the Master-Training-Recipe adoption rule (validated + measured ablation delta + a named passing gate).
`canClaimAGI` stays false.

## 1. Why this repo is the right one to study

The Huawei PLM repo is not a "we trained the biggest model" repo. It is a **compression-and-efficiency
lab**: a lab that could not win on raw scale and instead published a coherent stack of techniques —
distillation (TinyBERT), adaptive architecture (DynaBERT), ultra-low-bit QAT (TernaryBERT/BinaryBERT),
NAS under latency budgets (AutoTinyBERT), memory-efficient optimization (CAME), byte-level tokenization
(BBPE) — each shipped with a paper, benchmarks, and ablations.

That is structurally the same bet as Sophia's charter ("don't out-train frontier labs — innovate at the
trust layer") and the same publishing discipline as the adoption rule in
`docs/06-Roadmap/Master-Training-Recipe.md`. The meta-takeaway: **a small lab compounds by shipping one
measured technique at a time, each with its own ablation row — not by chasing scale.** The repo below is
mined technique-by-technique against the recipe's 6 layers.

## 2. Technique → Sophia mapping (recipe-layer indexed)

| Huawei technique | What it showed | Recipe layer | Where it lands in this repo |
|---|---|---|---|
| **TernaryBERT** (EMNLP 2020) — distillation-aware ultra-low-bit QAT | 2-bit weights recover FP-level GLUE **only when QAT is driven by layer-wise distillation from the FP teacher** (MSE on embeddings + hidden states + attention scores, not just output logits) | 2 (objective) + 4 (quant-serve) | The NVFP4 QAT program (`training/qat.py`, v6/v7 recipes) is currently **output-space** KD (KL + top1-margin). Intermediate-layer distillation is the literature's answer to exactly the failing gate class (see §3.1) |
| **DynaBERT** (NeurIPS 2020) — adaptive width/depth with importance-ordered rewiring | Rank units by **loss-sensitivity importance**, rewire so the most important come first, then any width prefix is a servable model | 4 (quant-serve) | `--keep-top-experts` / `keep_layers` protection levers (`tools/expert_protection.py`). Holding by projection type didn't move top1 (v7 doc, Lever D); DynaBERT says the selection metric should be **measured importance on the gated metric**, not a structural category (§3.2) |
| **TinyBERT** (Findings-EMNLP 2020) — two-stage transformer distillation + data augmentation | General distillation on broad corpus, then task-specific distillation **on an augmented task set**; attention + hidden-state alignment; 7.5× smaller / 9.4× faster | 1 (data) + 2 (objective) | The W1 substrate step "distill regex gates → learned verifiers" (`AGI-Substrate-Plan.md`) and `tools/distill_*.py` (§3.3) |
| **AutoTinyBERT** (ACL 2021) — model zoo per latency budget via one-shot NAS | Treat the deploy constraint as a **search budget**, not a post-hoc filter | 0 (target-base) + 4 | The low-RAM cert (`mem_ratio`, `LowRamGate`) already defines the budget; the v7 levers (rank alloc × keep_layers × precision mix) form a small pre-registerable grid instead of hand-ordered single levers (§3.4) |
| **CAME** (ACL 2023, Noah-adjacent) — confidence-guided memory-efficient optimizer | Adafactor-class memory with Adam-class convergence via a confidence correction on the factored second moment | 0/2 | `MegaTrain-Memory-Centric-Training.md` (params in host RAM, GPU transient) — optimizer state is the other memory wall on the 128 GB Spark (§3.5) |
| **BBPE** — byte-level BPE vocabulary builder | Byte-level vocab removes OOV across scripts; vocab is a **design surface**, not a default | 0/1 | `From-Scratch-LLM-Brainstorm.md` ("provenance-native from token zero") and the bilingual corpus (§3.6) |
| **NEZHA** — functional (non-learned) relative position encoding + whole-word masking + LAMB | Cheap, deterministic pretraining tricks measured one at a time | pretraining lab | `pretraining/` toy-lab study candidates (method-and-taste scope) |
| **PanGu-α / PanGu-Bot** | Build the dialog product **on top of** the pretrained base with small curated data, not by re-pretraining | 1 | Confirms the existing adapter-first stance; PanGu-Bot is the precedent for "small high-quality pack on a strong base" |

## 3. Candidate ideas (cheapest-first, pre-registration style)

Each idea below is written as a lever with the gate it targets. Per the recipe: a lever that does not
move its gated metric by the pre-registered threshold is dropped, not hopefully stacked.

### 3.1 TernaryBERT-style intermediate-layer distillation in QAT — targets `protected_max_kl`

The live frontier (`nvfp4-gptq-experts-cert-preregistered-2026-07-02`): GPTQ clears the top1 primary
gate (0.9805) but full cert is NO-GO on `protected_max_kl` 0.155 > 0.1. The v6/v7 QAT objective is
output-space only. TernaryBERT's central result is that ultra-low-bit fidelity is recovered by
**matching internal representations of the FP teacher**, layer by layer, during QAT — output-logit KD
alone under-constrains where the quantization error accumulates.

Candidate lever for a v8 recipe (or a GPTQ+QAT hybrid): add to `training/qat.py` an optional
`--distill-hidden` term — MSE on hidden states (and for OLMoE, **KL on router logits**) against the FP
teacher, computed **on the protected-slice prompts specifically**. This is the repo's own thesis
("train the metric you gate on") applied one level deeper: the failing gate is a KL on a protected
slice, so put that slice's internal agreement into the loss. Gate: pre-registered v7 gate unchanged
(`top1 ≥ 0.97 ∧ mean_kl ≤ 0.05 ∧ protected_max_kl ≤ 0.1`, n=256).

### 3.2 DynaBERT-style measured-importance expert protection — targets the same cert

`keep_top_experts` selected what to hold in higher precision by structural category, and it didn't move
top1. DynaBERT's rewiring lesson: rank units by **measured loss-sensitivity on the target metric** and
protect the top of that ranking. Concrete: for each expert (and each depth), measure per-unit
contribution to protected-slice KL (ablate-to-quant one unit at a time on a small calibration set, or
use a first-order Taylor sensitivity), then pass the top-k of *that ranking* to
`tools/expert_protection.py`. Composes with v7 Lever D (depth-based holding) — depth may simply be
what the measured ranking recovers, but then it's measured, not guessed.

### 3.3 Two-stage distillation for learned verifiers (W1), upgraded to on-policy

For "distill regex gates → learned verifiers": TinyBERT's recipe transfers directly — stage 1 general
distillation (broad verifier-labeled corpus), stage 2 task-specific distillation **on an augmented set**
(TinyBERT's ablations show the augmentation is load-bearing; the repo's trajectory packs and council
seeds are the augmentation source). The 2023–2026 literature then upgrades the loop: **on-policy /
generalized knowledge distillation (GKD)** — the student generates, the teacher (here: the
deterministic gate/oracle) scores the student's own outputs — which reduces compounding error versus
static imitation packs and is cheaper per step than GRPO (no group rollouts; the "teacher" is the
existing verifier, which is free). This is a natural third arm next to the SFT and RLVR arms, gated
exactly like RLVR: pass@1 / VSC load-bearing, never meanReward, ≥3 seeds.

### 3.4 The low-RAM cert as a search budget (AutoTinyBERT framing)

The v7 levers A–D are hand-ordered. AutoTinyBERT's framing: fix the budget (`mem_ratio` bar), define
the small discrete space (β × margin × rank-alloc × keep_layers × precision-mix), pre-register the
grid, and let the cert pick. On this repo's compute that is not a NAS supernet — it is a ~10–20-cell
pre-registered grid on the n=256 cert, which also kills the "hopeful stacking" failure mode by
construction: every cell is its own ledger row.

### 3.5 CAME optimizer in the memory-centric path

For MegaTrain / any Spark-side full-parameter work: Adam's second moment doubles optimizer memory;
Adafactor factors it but costs convergence; CAME (ACL 2023) adds a confidence correction that, per the
paper, keeps Adam-class convergence at Adafactor-class memory (validated on BERT/GPT-2/T5 pretraining,
incl. large-batch). Cheap to trial in the `pretraining/` toy lab first (the lab already studies
optimizers against a known loss floor), then in any LoRA run where optimizer state binds.

### 3.6 Byte-level, provenance-aware vocabulary for the from-scratch path

If/when From-Scratch-LLM-Brainstorm's "gate-shaped from token zero" model is attempted: BBPE's
byte-level vocab is the OOV-free base for a bilingual corpus, and the vocab step is where
provenance-nativeness can be made structural — e.g. reserved token space for provenance/attribution
markers so citation structure is first-class in the sequence, not a post-hoc annotation. Design-only
until P0 capability-delta gating clears (per `Distributed-Training-FSDP-Path.md`).

## 4. What NOT to copy

- **Scale-track (PanGu-α 200B, MindSpore/Ascend auto-parallel):** out of charter and out of budget;
  the only import is the staged-growth mindset, already covered by the recipe's layer gating.
- **BERT-era task framing:** GLUE-style encoder distillation targets don't transfer literally;
  the transferable content is the *loss structure* (which internal signals to match), not the tasks.
- **Their evaluation standard:** the repo reports single-run benchmark numbers without CIs or seeds —
  Sophia's IEC (multi-seed, CIs, decontam, pre-registration) is strictly stronger. Keep ours.

## 5. Methodology sketch for "training your own model later"

Dependency-ordered, mapped to the Master-Training-Recipe layers; each stage gates the next:

1. **Stay adapter-first (layer 0–1).** PanGu-Bot precedent: strong open base + small verified pack.
   From-scratch remains research-only until a measured capability delta demands it.
2. **Data = verifier-labeled, decontam-guarded packs (layer 1).** Unchanged; TinyBERT adds: budget for
   *augmentation* of the task-specific pack, it is load-bearing in distillation.
3. **Objective = distill the gate (layer 2).** SFT (M3 rank16, adopted) → add the on-policy
   verifier-distillation arm (§3.3) as the cheap middle ground between SFT and GRPO.
4. **Verification-in-loop (layer 3).** Unchanged — this is the moat; nothing in the Huawei repo has it.
5. **Quant-serve trained-in (layer 4).** v7 levers + §3.1 intermediate-layer distillation +
   §3.2 measured-importance protection + §3.4 grid framing, all under the existing cert.
6. **Claim only what the gate passes (layer 5).** Unchanged; see §4 third bullet.

## Sources

- Huawei Noah PLM repo: <https://github.com/huawei-noah/Pretrained-Language-Model>
- TinyBERT: <https://arxiv.org/abs/1909.10351> · <https://aclanthology.org/2020.findings-emnlp.372/>
- TernaryBERT: <https://arxiv.org/abs/2009.12812> · <https://aclanthology.org/2020.emnlp-main.37.pdf>
- KD-for-QAT analysis (transformer encoders): <https://arxiv.org/abs/2211.11014>
- CAME optimizer: <https://aclanthology.org/2023.acl-long.243/> · <https://github.com/yangluo7/CAME>
- On-policy distillation / GKD: <https://arxiv.org/abs/2306.13649> ·
  <https://thinkingmachines.ai/blog/on-policy-distillation/> · survey <https://arxiv.org/abs/2604.00626>
