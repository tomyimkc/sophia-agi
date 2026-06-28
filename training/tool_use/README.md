# Tool-use / MCP curriculum

This directory is the durable scaffold for Sophia's tool-use and MCP training data. It is
candidate-only curriculum material: it can teach habits around when to call tools, how to
handle tool errors, and when to abstain, but it does not prove capability or justify an AGI
claim. `canClaimAGI` stays false.

中文摘要：這裏只保存工具使用與 MCP 行為訓練的候選資料；外部 verifier/MCP gate 仍然是正確性的依據，不能當作 AGI 證明。

## Present pack

`dpo_pairs.jsonl` is a small DPO preference pack. It contrasts grounded answers against
negative behaviors that showed up in tool-use loops:

- `over_call`: calling a tool when the answer is already source-disciplined and direct.
- `ignored_error`: failing to recover from or surface a tool/gate error.
- `spurious_extra`: adding unsupported extra text after a correct answer.
- `schema_invalid`: returning malformed structured output.
- `mis_ground`: preferring an unsupported attribution or author claim.
- `wrong_tool`: choosing an irrelevant tool path.

The file is intentionally small and reviewable. It is ingested as an optional DPO source by
`tools/build_local_sophia_dataset.py` and validated by `tools/validate_tool_use_curriculum.py`.

## MCP trace rows

`mcp_trace_schema.json` defines the row shape for future first-class MCP traces. The schema
requires a chat row, a structured `toolTrace`, and metadata that records whether the trace was
human-reviewed and kept disjoint from held-out evals. Raw private logs, credentials, benchmark
answers, and unreviewed self-generated traces are excluded from this curriculum.

## Feedback return path

Tool-use feedback follows the same default-deny loop as source-discipline feedback:

1. Runtime failures or reviewer notes become candidates, not training rows.
2. A human reviewer approves only verified, license-clean examples.
3. Approved SFT rows land in `training/feedback/sft_from_feedback.jsonl`; optional DPO rows can
   land here when they match the manifest and validator.
4. `tools/build_local_sophia_dataset.py` decontaminates against eval, holdout, and sealed
   tool-use benchmark prompts before writing local training packs.
5. Promotion still requires external eval gains at acceptable false-positive and over-call cost.

Validate without training or GPU:

```bash
python3 tools/validate_tool_use_curriculum.py
python3 tools/build_local_sophia_dataset.py --check
```
