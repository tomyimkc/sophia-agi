# Sophia Architecture — One Flow

This diagram ties the Sophia subsystems — retrieval (RAG), council, epistemic
gate, append-only memory, operational tools (a fixed tool-dispatch step, not a
general executor), hidden evaluation, baseline/ablation, and the AGI-candidate
proof package — into a single flow. It satisfies the Architecture Claim item in
[`agi-proof/TODO.md`](../../agi-proof/TODO.md).

Every component below is a discrete, suppressible step in the shared per-case
pipeline `run_case()` in
[`tools/run_hidden_eval_sophia.py`](../../tools/run_hidden_eval_sophia.py), which
is why the baseline/ablation runner can toggle each one independently.

## Per-case reasoning pipeline

```mermaid
flowchart TD
    Q[Hidden / benchmark case] --> R[RAG retrieval<br/>agent/retrieval.py]
    Q --> E[Local + web evidence<br/>agent/web_evidence.py]
    Q --> C{Coding / figure council?<br/>agent/coding_council.py}
    R --> P[Compose prompt<br/>MODE_PROMPTS or RAW_SYSTEM_PROMPT]
    E --> P
    C -->|coding/tool/planning/learning| P
    M[(Append-only memory<br/>agent/memory/*.jsonl)] --> P
    T[Operational tools<br/>run_operational_tools] --> P
    P --> LLM[Model backend<br/>grok / deepseek / anthropic]
    LLM --> G[Epistemic gate<br/>agent/gate.py]
    G --> S[Score: alias/regex + tool/memory checks<br/>hidden_eval_protocol.score_pack]
    S --> RP{Pass + gate ok?}
    RP -->|no, repair enabled| FIX[Bounded repair<br/>agent/correction_loop.py]
    FIX --> LLM
    RP -->|yes| RV[Rubric review<br/>agent/rubric_review.py]
    RV --> OUT[Per-case result]
```

## Evidence pipeline (Level 2 → Level 3)

```mermaid
flowchart LR
    subgraph Run[run_case shared pipeline]
        RC[retrieval · evidence · council · gate · memory · tools · repair]
    end
    RC --> H[Hidden eval<br/>run_hidden_eval_sophia.py]
    RC --> A[Baseline / ablation<br/>run_ablation_sophia.py]
    RC --> L[Learning-under-shift<br/>run_learning_shift.py]
    LH[Long-horizon harness<br/>run_long_horizon.py] --> PROOF
    H --> PROOF[AGI-candidate proof package<br/>agi-proof/ + evidence-manifest.json]
    A --> PROOF
    L --> PROOF
    REP[Third-party replication<br/>run_replication_check.py] --> PROOF
    PROOF --> CLAIM[Pre-registered claim ladder<br/>preregistered-thresholds.md]
```

## Component → file map

| Component | Implementation | Ablation flag |
|---|---|---|
| RAG retrieval | `agent/retrieval.py` | `use_kb` |
| Local/web evidence | `agent/web_evidence.py` | `use_evidence` |
| Coding/figure council | `agent/coding_council.py` | `use_council` |
| Epistemic gate | `agent/gate.py` | `use_gate` |
| Append-only memory | `agent/memory.py`, `run_learning_probe` | `use_memory` |
| Operational tools | `run_operational_tools` | `use_tools` |
| Bounded repair | `agent/correction_loop.py` | `allow_repair` |
| Prompt discipline | `agent/prompts.py` (`MODE_PROMPTS`) vs `RAW_SYSTEM_PROMPT` | `raw_system` |

## Proof harnesses

| Harness | Tool | Produces |
|---|---|---|
| Hidden eval | `tools/run_hidden_eval_sophia.py` | `agi-proof/benchmark-results/*.public-report.json` |
| Baseline/ablation | `tools/run_ablation_sophia.py` | `agi-proof/baseline-ablation/ablation-deltas-*.public-report.json` |
| Learning-under-shift | `tools/run_learning_shift.py` | `agi-proof/learning-under-shift/shift-result-*.public-report.json` |
| Long-horizon autonomy | `tools/run_long_horizon.py` | `agi-proof/long-horizon-runs/*.public-report.json` |
| Third-party replication | `tools/run_replication_check.py` | `agi-proof/third-party-replication/replication-check-*.json` |
