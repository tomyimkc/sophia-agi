# 6 · Self-Evolution / RSI + Update Gate

**Role in the master flow.** The slow loop: turn measured failures into candidate improvements
(knowledge, skills, verifiers), then **gate every promotion** through a protected-regression +
retention check so the system improves without catastrophic forgetting — and without touching weights
unless an explicit training run is invoked. 37 `agent/` modules, incl. the whole `ssil_*` safety
family.

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

**Modules:** `self_evolving_agent.py`, `governed_rsi.py`, `continual_plasticity.py`,
`continual_retention.py`, `cls_consolidation.py`, `fact_check_flywheel.py`, `verifier_synthesis.py`,
`verifier_proposer.py`, `skill_library.py`, `habit_strength.py`, `plasticity_probe.py`, and the
`ssil_*` family (`ssil_guardian`, `ssil_compute_governor`, `ssil_eval_awareness`,
`ssil_capability_ceiling`, `ssil_moral_parliament`, …).

**Thesis note.** `evaluate_update()` is the repo's central safety primitive — a promotion is accepted
only if it clears a target-improvement floor **and** does not regress any protected metric beyond
tolerance, backed by ≥2 verifier artifacts. Today this loop improves *assets* (memory/skills/
verifiers), never weights. The dashed `WEIGHTS` edge is the one move that crosses into actual model
training (the SkillOpt-style skill→weight distillation, and W-series live runs) — and it is precisely
where the abstention the gate enforced can be un-learned, so post-distillation calibration re-audit is
mandatory.