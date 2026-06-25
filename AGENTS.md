# Sophia AGI — AI Assistant Guide

Open corpus for **provenance-aware philosophy** and AGI-shaped epistemic reasoning.

## Read first

1. `README.md`
2. `docs/00-Index/Home.md`
3. `data/attributions.json`
4. `docs/04-Disputes/` for contested cases
5. Training/RSI/continual work: `docs/06-Roadmap/Training-RSI-Continual-Convergence.md`,
   `docs/11-Platform/Local-Sophia-Training.md`, `agi-proof/failure-ledger.md`

## Rules

- Never attribute a text to a figure listed in `doNotAttributeTo`.
- Flag `authorConfidence: legendary` or `compiled` as uncertain — do not state as settled fact.
- Keep 儒家 (Confucian) and 道家 (Daoist) lineages distinct unless evidence supports a link.
- Training outputs: bilingual EN + 中文 summary; JSONL-compatible structure.

## Guardrails (enforced — do not bypass)

- **No overclaiming.** `python tools/lint_claims.py` must pass before every commit. Never claim
  AGI, validated uplift, or that an adapter is promoted unless the gate says so.
- **Promotion is decided by `tools/promote_adapter.py`** (the W2 `agent/continual_plasticity.py`
  gate), not by judgment. `religion`/`history` are protected suites — a regression forces reject.
- **Contamination guard must stay CLEAN.** Never put eval/holdout prompts in training;
  `tools/build_local_sophia_dataset.py` decontaminates and fails closed.
- **Record runs (including failures) in `agi-proof/failure-ledger.md`** with the numbers and
  what is not yet proven.
- **RunPod GPU jobs: GitHub Actions only** (local SSH egress to mapped pod ports is unreliable).
  Use `.github/workflows/runpod-sophia-7b-sft.yml` or `train-runpod` / `speedup-runpod` / `rlvr-runpod`.

## Agents

| Agent | Entry | Role |
|-------|-------|------|
| **Sophia Advisor** | `python tools/sophia_agent.py advisor "..."` | Project & epistemic decisions |
| **Sophia Repo** | `python tools/sophia_agent.py repo "..."` | Repo ops + approved tools |
| **Sophia Life** | `python tools/sophia_agent.py life "..."` | General decisions + guardrails |
| **Online RAG** | `python tools/sophia_rag.py "..."` | Curated index + Gemini/Vertex + gate |
| `grok-cli-teacher` | `.grok/agents/grok-cli-teacher.md` | Training pair generation |

## Skill + MCP

| Layer | Path |
|-------|------|
| **Grok skill** | `.grok/skills/sophia-agi/SKILL.md` — `/sophia-agi` |
| **Portable skill** | `skills/portable/sophia-source-discipline/` — `/sophia-source-discipline` (any project) |
| **MCP server** | `sophia_mcp/server.py` — validate, gate, benchmark, lookup, disputes, export |
| **Install** | `python tools/install_skills.py --all` — [Skills-Install.md](docs/09-Agent/Skills-Install.md) |

Docs: `docs/09-Agent/Sophia-Agent.md`, `docs/09-Agent/Online-RAG.md`, `docs/09-Agent/LoRA-Experiment.md`

## Validation

```bash
python tools/validate_attribution.py
python tools/export_training_jsonl.py
```