# 8 · Weight-Training Path (SFT / DPO / RLVR → MLX LoRA)

**Role in the master flow.** The one path that changes **model weights** — the dashed `WEIGHTS` node
in the master chart. Everything in charts 1–7 improves behavior without touching weights; this
subsystem is where measured epistemic signals (gate verdicts, calibration, provenance) are distilled
into training corpora and folded into a frozen base model via MLX LoRA. It is the ASI-precursor loop,
and the point where inference-time safety must be re-audited post-training.

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

**Files:** corpora under `training/` (`wiki_provenance_sft.jsonl`, `moral_gate_sft.jsonl`,
`local_sophia_v2/*.jsonl`, `sophia-math-code-curriculum/*.jsonl`, `tool_use/*.jsonl`,
`self_evolve/distill.jsonl`, `prosoche/attention_sft.jsonl`); code `training/qat.py`,
`training/sharding.py`, `training/swarm_router/train_grpo.py`; adapters `training/mlx_adapters/sophia-v2…v5`;
eval gates `training/local_sophia_v2/eval_ladder_*.json`; reward tools
`tools/run_rlvr.py`, `tools/distill_process_reward_model.py`, `tools/train_calibration_objective.py`.

**Thesis note.** Three points a training-chapter reviewer will want stated: (1) the base is a **frozen
Qwen2.5-3B-Instruct** adapted only by LoRA — sophia does not pre-train. (2) The corpora are the
epistemic loop's *output* (provenance discipline, hard-negative abstention, moral-gate, council
traces) turned into supervision — this is the concrete measurement→learning bridge the W-series
proposes to complete. (3) The **post-training calibration re-audit** (the `EVALGATE` node) is not
optional: distilling gated *behaviors* into weights removes the inference-time gate that made them
safe, so an adapter must be re-checked for abstention/calibration regression before promotion, under
the same `candidateOnly` / `canClaimAGI:false` discipline as every other harness.