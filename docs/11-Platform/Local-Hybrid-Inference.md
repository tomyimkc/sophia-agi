# Local + Hybrid Inference (Stage E)

> Status: design + config scaffold. This documents the inference topology Sophia
> targets on the maintainer's actual hardware — a **MacBook Pro M4 Max / 48 GB**.
> It is **not** the M3 Ultra / 96 GB assumed by some external write-ups; sizing
> here is for 48 GB unified memory.
>
> Claim boundary: this is *infrastructure* for an AGI-candidate framework. It does
> not change the verification gate or any AGI claim. The gate
> (`tools/run_agi_verification_gate.py`) remains the source of truth.

## Why hybrid

Sophia's guarantees are enforced in deterministic code (the gateway interceptor,
the CaMeL-style constrained interpreter, the formal verifier). The *model* tier is
swappable. The goal is to keep the **majority of turns local** (cost + privacy)
and escalate only hard, low-confidence reasoning to an API model.

## Three tiers

| Tier | Engine | Role | Constrained decoding |
|---|---|---|---|
| **Orchestrator** | MLX | Planning, routing, cheap tool-calls — the CaMeL *privileged planner* | No (MLX lacks native grammar) |
| **Tool-calls / quarantine** | llama.cpp / KoboldCpp | Guaranteed well-formed JSON tool-calls + CaMeL *quarantined extractor* (no tool access) | Yes — GBNF / JSON-schema |
| **Escalation** | API (Anthropic) | Hard reasoning flagged low-confidence | n/a |

On 48 GB the realistic local sweet spot is a **Qwen3-class 30B-A3B MoE at 4-bit**
(~3B active params/token, leaves context headroom) or a **7–14B dense** model.
A 70B dense model is out of practical range at 48 GB.

## How it maps to existing modules

- **Privileged/quarantined split** is already implemented:
  `agent/dataflow/interpreter.py` (P-LLM plans over symbolic vars, Q-LLM extracts
  as data-only, deterministic interpreter propagates taint).
- **Guaranteed JSON tool-calls** use the Stage-D bridge:
  `agent/structured_output.py:schema_to_gbnf` emits a GBNF grammar from
  `schemas/gateway_call.schema.json` that a llama.cpp server loads at decode time.
  Even before constrained decoding, `validate_tool_call` rejects malformed calls.
- **Escalation trigger** reads calibrated confidence from `agent/calibration.py`;
  escalate only above the uncertainty threshold.
- **Semantic check after syntax**: a schema-valid call still passes through the
  gateway's semantic verifier — syntactic validity is necessary, not sufficient.

## Config

Copy `config/inference.local.example.json` to `config/inference.local.json` and
edit model paths. Key targets:

```json
{ "local_turn_ratio_min": 0.8, "unsafe_accepted_outputs": 0, "tool_call_json_validity_min": 0.99 }
```

## Engine notes (Apple Silicon)

- **MLX / MLX-LM** — fastest raw throughput + native fine-tuning (QLoRA on a
  4-bit base). No grammar enforcement, no batching. Use for orchestration + FT.
- **llama.cpp / KoboldCpp** — GBNF grammars + JSON-schema constrained decoding,
  IQ low-bit quants. Use for tool-calls and the quarantined extractor.
- **LM Studio** — GUI / OpenAI-compatible endpoint for local dev + eval.
- **vLLM / SGLang** — NVIDIA path, not Apple Silicon; relevant only if a Linux/GPU
  node is added later (SGLang + XGrammar is the structured-output reference there).

## Fine-tuning path (MLX-LM, optional)

1. Curate 2,000–5,000 high-quality tool-use / reasoning traces (quality > quantity).
2. `mlx_lm.convert --hf-path <model> -q --q-bits 4` to quantize the base.
3. `mlx_lm.lora --train --data ./data --iters 500-1000 --batch-size 4 --mask-prompt`
   (auto-QLoRA on a 4-bit base; train on responses only).
4. Evaluate on a held-out tool-use set; watch for overfitting.
5. `mlx_lm.fuse --de-quantize` then export GGUF for llama.cpp + grammars.
6. Keep adapters separate per role (router / verifier / KG-writer) and hot-swap.

## Honest limits

- Numbers (model names, tok/s) move monthly — re-benchmark on the actual M4 Max.
- CaMeL-style defense reduces but does not eliminate prompt-injection risk; keep
  defense-in-depth (interceptor + interpreter + hooks), not a single layer.
- This file is a topology + config scaffold; the live router wiring is future work
  and gated by the same verification pipeline.
