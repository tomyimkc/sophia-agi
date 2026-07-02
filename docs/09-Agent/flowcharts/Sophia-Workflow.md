# Sophia — Combined Workflow Walk-through

> **Single-document tour of the whole system**, stitched from the per-subsystem charts in this folder. Built from the working clone (branch `feat/oscillatory-crosspollination` @ `4f1059a0`, 388 uncommitted local mods — may differ from any pushed commit; `origin/main` was `2cfa3c63` at build time, not verified against). Every node names a real file. For thesis use: each section's chart also exists as a standalone `.md`, and as `.png`/`.svg` under `png/` and `svg/`.

## How to read the system

Sophia is organised as **one per-case inference spine** wrapped by **two slower loops**. A case flows left-to-right through the spine (intake → grounded context → answer → epistemic gate → result). Around it sit the **proof harnesses** that measure the spine, and the **self-improvement loop** that turns measured failures into gated updates to memory, skills, and verifiers — and, only via an explicit training run, into model weights. The design invariant that makes the whole thing measurable: every spine step is a *suppressible* stage in one shared `run_case()`, which is why the ablation runner can price each component independently.

---

## 0 · Master chart — Sophia — Master Workflow Flowchart

One master chart that ties every Sophia subsystem together, plus links to the per-subsystem charts that expand each block. Built from the actual code in the working clone (the `run_case()` pipeline in `tools/run_hidden_eval_sophia.py`, the `agent/` module wiring, and `docs/09-Agent/Sophia-Architecture.md`) — not from a hand-drawn map. Every node names a real file.

```mermaid
flowchart TB
    CASE([Case / query / task]) --> INTAKE

    subgraph INFER["Per-case inference pipeline · run_case()"]
        direction TB
        INTAKE["**Intake & routing**<br/>agent/intake · claim_router"] --> CTX
        CTX["**Grounded context**<br/>retrieval · web_evidence · memory"] --> ANSWER
        ANSWER["**Answer**<br/>council + model backend<br/>agent/model.py"] --> GATE
        GATE{"**Epistemic gate**<br/>agent/gate.py"} -->|fail · repair on| REPAIR["Bounded repair<br/>agent/correction_loop.py"]
        REPAIR --> ANSWER
        GATE -->|pass| RESULT["Per-case result<br/>+ rubric review"]
    end

    RESULT --> HARNESS
    RESULT --> CALIB

    subgraph HARNESS["Evidence & proof harnesses"]
        direction LR
        HID["Hidden eval"] --> PROOF
        ABL["Baseline / ablation"] --> PROOF
        SHIFT["Learning-under-shift"] --> PROOF
        LH["Long-horizon"] --> PROOF
        PROOF["AGI-candidate proof package<br/>agi-proof/ + evidence-manifest.json"] --> LADDER["Pre-registered claim ladder"]
    end

    subgraph LEARN["Self-improvement loop (no weights unless trained)"]
        direction TB
        CALIB["**Calibration & abstention**<br/>calibration · abstention_scoring"] --> SIGNAL
        SIGNAL["**Training signals**<br/>gate_reward · multiaxis_reward · rlvr"] --> EVOLVE
        EVOLVE["**Self-evolution / RSI**<br/>self_evolving_agent · governed_rsi"] --> UPDGATE
        UPDGATE{"**Update gate**<br/>continual_plasticity.evaluate_update<br/>protected-regression + retention"} -->|reject| DROP["Discard candidate"]
        UPDGATE -->|promote| ASSETS["Updated assets:<br/>memory · skills · verifiers"]
    end

    ASSETS -.->|feeds next case| CTX
    ASSETS -.-> TRAIN["Optional weight training<br/>training/ · mlx_lm lora<br/>(gated, ASI-precursor)"]
    PROOF -.->|failures become tasks| SIGNAL

    classDef spine fill:#1f5fa8,stroke:#12365f,color:#fff
    classDef gate fill:#b8860b,stroke:#6b4e00,color:#fff
    classDef loop fill:#2e7d32,stroke:#184a1b,color:#fff
    class INTAKE,CTX,ANSWER,RESULT spine
    class GATE,UPDGATE gate
    class CALIB,SIGNAL,EVOLVE,ASSETS loop
```

---

## 1 · 1 · Intake & Routing

*The master spine's first block — **Intake & routing** — expands here.*

The front gate: turns a raw case into a typed, contract-checked request and decides which downstream paths (retrieval, council, claim-verification) it needs. Ablation flag `use_intake`. Suppressing it sends the raw query straight to context-gathering.

```mermaid
flowchart TD
    Q([Raw case / query / task]) --> INTAKE["Request triage + intake contract<br/>agent/intake · sophia_contract/intake.py<br/>schema/intake-contract-1.0.0.json"]
    INTAKE --> VALID{Contract valid?}
    VALID -->|no| REJECT["Reject / clarify<br/>fail-closed"]
    VALID -->|yes| TYPE["Classify request type<br/>coding · figure · planning · learning · QA"]
    TYPE --> CROUTE{"Route atomic claims?<br/>agent/claim_router.py<br/>flag: use_claim_router"}
    CROUTE -->|yes| CLAIMS["Split into atomic claims<br/>→ claim-type verifiers"]
    CROUTE -->|no| PASS["Whole-response path"]
    TYPE --> SWARM{"Multi-agent task?<br/>agent/swarm_router.py"}
    SWARM -->|yes| DISPATCH["Route to sub-agents<br/>agent/subagent.py · team_agents.py"]
    SWARM -->|no| SOLO["Single-agent path"]
    CLAIMS --> OUT([To grounded context →])
    PASS --> OUT
    DISPATCH --> OUT
    SOLO --> OUT
```

**Thesis note.** The intake contract is what makes every downstream step *suppressible and
measurable* — it stamps the request shape so the ablation runner can toggle each pipeline stage
independently. That toggle-ability is the basis of the whole baseline/ablation evidence story.

---

## 2 · 2 · Grounded Context (RAG + Evidence + Memory)

*The **Grounded context** block: what the model is given before it answers.*

Assembles the grounded context the model answers from: retrieved passages, local/web evidence, and prior memory — each with a provenance/trust tag. Ablation flags `use_kb` (retrieval), `use_evidence` (evidence), `use_memory` (memory).

```mermaid
flowchart TD
    IN([Typed request]) --> RET["RAG retrieval<br/>agent/retrieval.py · rag_pipeline.py<br/>flag: use_kb"]
    IN --> EVI["Local + web evidence<br/>agent/web_evidence.py · live_sources.py<br/>flag: use_evidence"]
    IN --> MEM[("Append-only memory<br/>agent/memory.py · memory/*.jsonl<br/>flag: use_memory")]

    RET --> EMB["Embed & score<br/>agent/rag_embed.py · embedding_backends.py"]
    EMB --> RANK["Trust-rank sources<br/>agent/source_ranking.py<br/>OKF/belief 0.95 → web 0.86 → generic 0.55"]
    EVI --> RTG["Realtime grounding gate<br/>agent/realtime_grounding.py<br/>→ conformal_gate · fact_check_gate"]
    RANK --> PACK
    RTG --> PACK
    MEM --> PACK
    PACK["Compose grounded context<br/>+ context-pack cards<br/>schema/context-pack-card-1.0.0.json<br/>flag: use_context_packing"] --> OUT([To council / answer →])

    RANK -.->|low-trust or no source| ABSTAIN["Flag for abstention<br/>(feeds calibration)"]
```

**Thesis note.** Two traps worth stating in a methods chapter: (1) `rag_local_embed.py` is *also*
hash-based (`local-hash-v1`), so it is not a semantic upgrade over the lexical embedder — confirm the
live backend via `agent.vector_store.embedding_backend_id()`. (2) Source trust rank is deterministic
(`agent/source_ranking.py`), which is what lets provenance become a *weight* on downstream loss (see
the untapped-training W3 direction), not just a display tag.

---

## 3 · 3 · Council & Answer Generation

*The **Answer** block: council deliberation then the frozen model backend.*

Produces the candidate answer. For coding/figure/planning cases a multi-seat **council** deliberates before the model call; otherwise the composed prompt goes straight to the frozen model backend. Ablation flag `use_council`.

```mermaid
flowchart TD
    CTX([Grounded context]) --> ROUTE{"Council-eligible?<br/>coding · figure · planning · learning"}
    ROUTE -->|yes| COUNCIL["Council deliberation<br/>agent/coding_council.py<br/>council_deliberate.py · council_personas.py"]
    ROUTE -->|no| COMPOSE
    COUNCIL --> SEATS["Multi-seat panel<br/>agent/sector_council.py · council_registry.py"]
    SEATS --> AGG["Aggregate seats<br/>agent/council_format.py"]
    AGG --> COMPOSE["Compose prompt<br/>MODE_PROMPTS vs RAW_SYSTEM_PROMPT<br/>agent/prompts.py · flag: raw_system"]
    COMPOSE --> TOOLS["Operational tools<br/>run_operational_tools · flag: use_tools"]
    TOOLS --> LLM["Model backend<br/>agent/model.py · default_client().generate()<br/>grok · deepseek · anthropic · mlx"]
    LLM --> GUARD{"Real backend?<br/>cfg.kind != 'mock'"}
    GUARD -->|mock, no key| FAILCLOSED["Fail-closed<br/>no fabricated score"]
    GUARD -->|live| ANS([Candidate answer → gate])
```

**Thesis note.** Two facts a reviewer will check: the real adapter is
`agent.model.default_client(spec).generate(system, user) -> ModelResult` (not a `complete(prompt)`
call), and `agent.model._auto_provider()` returns `"mock"` with no API key — mock `.generate()`
fabricates text at `ok=True`. Any measured claim must assert `cfg.kind != "mock"`. Council catch-rate
(1.0 vs 0.27 monolith) in the repo's reports is on *stub* seats — real trained discipline adapters are
a named open gap.

---

## 4 · 4 · Epistemic Gate & Verification

*The **Epistemic gate** — the master's first gold gate, and the largest subsystem (49 modules).*

The heart of the repo: after generation, the answer must pass an **epistemic gate** before it becomes a result. The gate composes many verifiers (claim-type, citation, code, formal/Lean, conformal) and can send a failing answer back for bounded repair. Ablation flag `use_gate`. This is the largest subsystem — 49 `agent/` modules.

```mermaid
flowchart TD
    ANS([Candidate answer]) --> GATE["Epistemic gate<br/>agent/gate.py"]
    GATE --> VER["Verifier suite<br/>agent/verifiers.py"]
    GATE --> CR["Claim-type routing<br/>agent/claim_router.py"]
    GATE --> SC["Sector council check<br/>agent/sector_council.py"]

    CR --> CV["Citation existence<br/>agent/citation_existence_verifier.py"]
    CR --> AV["Attribution swap<br/>agent/attribution_swap_verifier.py"]
    CR --> CODEV["Code verifier<br/>agent/code_verifier.py · execution_verifiers.py"]
    CR --> FV["Formal / Lean / SMT<br/>agent/formal_verifier.py · smt_verifier.py"]
    CR --> CONF["Conformal gate<br/>agent/conformal_gate.py"]

    VER --> VERDICT{"Gate verdict<br/>accepted · rejected · abstain"}
    CV --> VERDICT
    AV --> VERDICT
    CODEV --> VERDICT
    FV --> VERDICT
    CONF --> VERDICT

    VERDICT -->|rejected · repair on| FIX["Bounded repair<br/>agent/correction_loop.py<br/>flag: allow_repair"]
    FIX -.->|re-generate| ANS
    VERDICT -->|abstain| ABS["Abstain<br/>→ calibration / abstention"]
    VERDICT -->|accepted| RV["Rubric review<br/>agent/rubric_review.py"]
    RV --> OUT([Per-case result])

    VERDICT -.->|verdict as signal| REWARD["gate_reward · multiaxis_reward<br/>→ training signals"]
```

**Thesis note.** The gate verdict is three-way (`accepted` / `rejected` / `abstain`), and the same
verdict is the seed for the training-signal path (`gate_reward.reward()`). That dual use — verdict as
inference-time gate **and** as a reward — is the bridge from "measured epistemics" to "learned
epistemics" (the untapped-training W1/W2 directions). `gate_reward` deliberately drops the question,
so as a raw reward it cannot tell abstain-on-answerable from abstain-on-trap — a known reward-hacking
surface worth naming.

---

## 5 · 5 · Calibration & Abstention

*The result feeds **Calibration & abstention**, entry to the self-improvement loop.*

Turns gate verdicts and confidence signals into a calibrated answer/abstain decision, and measures whether the system knows what it doesn't know. This is where the repo's validated headline result lives (self-consistency selective prediction). Feeds the self-improvement loop with the honesty signal.

```mermaid
flowchart TD
    IN([Answer + gate verdict + samples]) --> SC["Self-consistency<br/>agent/calibration.py:self_consistency()<br/>→ (answer, confidence)"]
    SC --> CAL["Calibration report<br/>agent/calibration.py<br/>ECE · selective_risk · base_risk"]
    IN --> ABST["Abstention scoring<br/>agent/abstention_scoring.py<br/>λ ∈ (0,0.5,1,2,3,5)"]
    ABST --> ACT{"Action<br/>answer · hedge · abstain"}
    CAL --> RISK["Selective risk / AURC<br/>agent/selective_risk.py<br/>coverage_fabrication_at · paired_aurc_delta_ci"]
    ACT --> CONF["Conformal policy<br/>agent/conformal_gate.py"]
    RISK --> DECIDE
    CONF --> DECIDE{"Confident enough<br/>at target coverage?"}
    DECIDE -->|yes| ANSWER([Emit answer])
    DECIDE -->|no| ABSTAIN([Abstain / defer])

    ABST -.->|fabrication propensity| FP["agent/fabrication_propensity.py<br/>temptation.py"]
    CAL -.->|proper-scoring target| TRAIN["→ calibration training signal<br/>(untapped W2)"]
```

**Thesis note.** This subsystem is measurement-only today: `calibration.py`,
`abstention_scoring.py`, and `selective_risk.py` contain **no differentiable loss** — they score, they
don't train. `abstention_scoring.py` cites Kalai et al. (*Why Language Models Hallucinate*) on the
binary-scoring incentive to guess (repo's own citation, not independently verified here). The largest
untapped lever in the whole repo is turning these proper-scoring metrics into an actual training
objective (W2) — the measurement→learning gap.

---

## 6 · 6 · Self-Evolution / RSI + Update Gate

***Self-evolution / RSI** — the master's green loop and its second gold gate (`evaluate_update`).*

The slow loop: turn measured failures into candidate improvements (knowledge, skills, verifiers), then **gate every promotion** through a protected-regression + retention check so the system improves without catastrophic forgetting — and without touching weights unless an explicit training run is invoked. 37 `agent/` modules, incl. the whole `ssil_*` safety family.

```mermaid
flowchart TD
    FAIL([Measured failures / new tasks]) --> EVOLVE["Self-evolving agent<br/>agent/self_evolving_agent.py<br/>evolve → no-hack → promote → retain → commit"]
    EVOLVE --> RSI["Governed RSI<br/>agent/governed_rsi.py<br/>→ verifier_synthesis · poison_resistant_ingestion"]
    RSI --> PROP["Propose candidate<br/>knowledge · skill · verifier"]

    PROP --> KIND{Candidate kind}
    KIND -->|verifier| VSYN["Verifier synthesis<br/>agent/verifier_synthesis.py · verifier_proposer.py"]
    KIND -->|skill code| SKILL["Skill library<br/>agent/skill_library.py<br/>anti-forgetting gate"]
    KIND -->|knowledge| FLY["Fact-check flywheel<br/>agent/fact_check_flywheel.py"]

    VSYN --> CAND
    SKILL --> CAND
    FLY --> CAND
    CAND["UpdateCandidate<br/>agent/continual_plasticity.py<br/>metrics + verifier_artifacts"] --> GATE

    GATE{"evaluate_update()<br/>min_target_delta ≥ 0.03<br/>max_protected_regression ≤ 0.01<br/>require_artifacts ≥ 2 · retention"}
    GATE -->|reject| DROP["Discard<br/>(regression or insufficient evidence)"]
    GATE -->|promote| RETAIN["Retention check<br/>agent/continual_retention.py · cls_consolidation.py"]
    RETAIN --> COMMIT["Commit updated asset<br/>memory · skill · verifier"]
    COMMIT -.->|next case| REUSE([Back into pipeline])
    COMMIT -.->|distill corpus| WEIGHTS["Optional weight training<br/>training/ · mlx_lm lora<br/>(ASI-precursor, re-audit calibration)"]

    subgraph SAFETY["RSI safety envelope · ssil_*"]
        SG["ssil_guardian · ssil_compute_governor<br/>ssil_eval_awareness · ssil_capability_ceiling"]
    end
    SAFETY -.->|bounds| RSI
```

**Thesis note.** `evaluate_update()` is the repo's central safety primitive — a promotion is accepted
only if it clears a target-improvement floor **and** does not regress any protected metric beyond
tolerance, backed by ≥2 verifier artifacts. Today this loop improves *assets* (memory/skills/
verifiers), never weights. The dashed `WEIGHTS` edge is the one move that crosses into actual model
training (the SkillOpt-style skill→weight distillation, and W-series live runs) — and it is precisely
where the abstention the gate enforced can be un-learned, so post-distillation calibration re-audit is
mandatory.

---

## 7 · 7 · Evidence & Proof Harnesses

*The **proof harnesses** that measure the whole spine and gate every claim.*

The measurement layer. Every harness drives the *same* `run_case()` pipeline with different toggles/datasets, writes a `*.public-report.json`, and feeds the AGI-candidate proof package and its pre-registered claim ladder. This is what keeps claims honest: nothing is a result until a harness produces a decontaminated, gated report.

```mermaid
flowchart TD
    PIPE["run_case() shared pipeline<br/>retrieval · evidence · council · gate · memory · tools · repair"] --> H
    PIPE --> A
    PIPE --> L
    PIPE --> LC

    H["Hidden eval<br/>tools/run_hidden_eval_sophia.py"] --> REPORTS
    A["Baseline / ablation<br/>tools/run_ablation_sophia.py<br/>toggles each flag independently"] --> REPORTS
    L["Learning-under-shift<br/>tools/run_learning_shift.py"] --> REPORTS
    LC["Long-context<br/>tools/run_long_context_sophia.py"] --> REPORTS
    LH["Long-horizon autonomy<br/>tools/run_long_horizon.py"] --> REPORTS
    REP["Third-party replication<br/>tools/run_replication_check.py"] --> REPORTS

    REPORTS[("agi-proof/benchmark-results/<br/>*.public-report.json (~130)")] --> DECON
    DECON["Decontamination guard<br/>shingle / Jaccard · make_independent_hidden_pack"] --> MANIFEST
    MANIFEST["evidence-manifest.json<br/>openCount · resolved · backendFailureCount"] --> LADDER
    LADDER["Pre-registered claim ladder<br/>agi-proof/preregistered-thresholds.md"] --> CLAIM{"Level 3 evidence?<br/>independent · gated · CI excludes 0"}
    CLAIM -->|no| OPEN["Row stays Open<br/>candidateOnly:true"]
    CLAIM -->|yes| RESOLVED["Row Resolved<br/>claim promoted"]
```

**Thesis note.** The design invariant worth foregrounding: every subsystem in charts 1–6 is a
*suppressible step* in one shared `run_case()`, which is exactly why the ablation runner can measure
each component's marginal contribution. Reports carry `candidateOnly` / `level3Evidence` /
`canClaimAGI` flags; a claim is promoted on the ladder only when an independent, decontaminated, gated
harness clears a pre-registered threshold with a CI excluding zero. This is the repo's answer to "how
do you prove an AGI-candidate claim without fooling yourself."

---

## 8 · 8 · Weight-Training Path (SFT / DPO / RLVR → MLX LoRA)

*The dashed **weight-training** crossover — the one path that changes model weights.*

The one path that changes **model weights** — the dashed `WEIGHTS` node in the master chart. Everything in charts 1–7 improves behavior without touching weights; this subsystem is where measured epistemic signals (gate verdicts, calibration, provenance) are distilled into training corpora and folded into a frozen base model via MLX LoRA. It is the ASI-precursor loop, and the point where inference-time safety must be re-audited post-training.

```mermaid
flowchart TD
    subgraph SIGNALS["Epistemic signals (from charts 4–6)"]
        GV["Gate verdicts<br/>gate_reward · multiaxis_reward"]
        CAL["Calibration targets<br/>train_calibration_objective.py"]
        PRM["Process reward<br/>distill_process_reward_model.py"]
        RLVR["Verified-trace RLVR<br/>tools/run_rlvr.py"]
    end

    GV --> CORPORA
    CAL --> CORPORA
    PRM --> CORPORA
    RLVR --> CORPORA

    subgraph CORPORA["Training corpora · training/*.jsonl"]
        SFT["SFT traces<br/>wiki_provenance_sft · sft_source_discipline<br/>sft_moral_gate · sft_council_traces · tool_use/sft_traces"]
        DPO["DPO pairs<br/>dpo_wiki_provenance · dpo_hard_negatives<br/>tool_use/dpo_pairs"]
        CUR["Curricula<br/>sophia-math-code-curriculum/sft_{math,code,all}<br/>self_evolve/distill · prosoche/attention_sft"]
    end

    SFT --> HOLDOUT
    DPO --> HOLDOUT
    CUR --> HOLDOUT
    HOLDOUT["Decontam + held-out split<br/>training/local_sophia_v2/holdout.jsonl<br/>shingle/Jaccard guard"] --> TRAIN

    TRAIN["MLX LoRA fine-tune<br/>python3 -m mlx_lm lora --train<br/>--model Qwen/Qwen2.5-3B-Instruct<br/>--iters 500 --batch-size 4 --mask-prompt"] --> ADAPT
    GRPO["GRPO path<br/>training/swarm_router/train_grpo.py"] -.->|router policy| TRAIN
    QAT["Quantization / sharding<br/>training/qat.py · sharding.py"] -.-> ADAPT

    ADAPT[("MLX adapters<br/>sophia-v2 → v3 → v4-religion-repair → v5<br/>training/mlx_adapters/")] --> EVALGATE
    EVALGATE{"eval-ladder gate<br/>eval_ladder_baseline vs _adapter<br/>+ POST-TRAIN calibration re-audit"}
    EVALGATE -->|regressed / calibration lost| REJECT["Reject adapter<br/>weights can un-learn abstention"]
    EVALGATE -->|clears ladder| PROMOTE["Promote adapter<br/>→ agent/model.py backend"]
    PROMOTE -.->|frozen backend| BACKINFER([Back into inference pipeline])
```

**Thesis note.** Three points a training-chapter reviewer will want stated: (1) the base is a **frozen
Qwen2.5-3B-Instruct** adapted only by LoRA — sophia does not pre-train. (2) The corpora are the
epistemic loop's *output* (provenance discipline, hard-negative abstention, moral-gate, council
traces) turned into supervision — this is the concrete measurement→learning bridge the W-series
proposes to complete. (3) The **post-training calibration re-audit** (the `EVALGATE` node) is not
optional: distilling gated *behaviors* into weights removes the inference-time gate that made them
safe, so an adapter must be re-checked for abstention/calibration regression before promotion, under
the same `candidateOnly` / `canClaimAGI:false` discipline as every other harness.

---

## Where the leverage is (for a thesis 'future work' section)

The recurring pattern across charts 4–8 is a **measurement→learning gap**: Sophia measures epistemics thoroughly (gate verdicts, calibration, provenance, fixed-point residuals) but rarely turns those measurements into a *training signal*. The highest-leverage moves — each a documented seam in the current code — are (1) the epistemic gate verdict as a process reward, (2) the calibration proper-scoring metric as an actual loss, (3) provenance rank as a per-example loss weight, and (4) the post-training calibration re-audit that keeps the weight-training path from un-learning the abstention the gate enforced. All four are additive to the pipeline and preserve the repo's honesty discipline (`candidateOnly` / `canClaimAGI:false` until a gated harness says otherwise).
