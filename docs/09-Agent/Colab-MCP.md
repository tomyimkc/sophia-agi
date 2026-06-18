# Colab MCP (agent-driven training)

Bridge a local Cursor/Grok agent to a **Colab tab in your browser** so it can create/run cells, install packages, and drive [Sophia-LoRA-Colab.ipynb](../../notebooks/Sophia-LoRA-Colab.ipynb) without manual copy-paste.

Upstream: [googlecolab/colab-mcp](https://github.com/googlecolab/colab-mcp)

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| `uv` / `uvx` | `pip install uv` — adds `uv.exe` and `uvx.exe` (often under `%LOCALAPPDATA%\Programs\Python\Python312\Scripts\`) |
| MCP client | Must support `notifications/tools/list_changed` (Cursor, Grok CLI, Claude Code, Windsurf) |
| Open Colab tab | Agent cannot start Colab; you open the notebook URL first |

## Install uv (Windows)

```powershell
pip install uv
uv --version
uvx --version
```

If `uvx` is not on PATH in Cursor, use the full path in `mcp.json` (see repo `.cursor/mcp.json`).

Optional: add Python Scripts to user PATH so `uvx` resolves everywhere:

`%LOCALAPPDATA%\Programs\Python\Python312\Scripts`

## Cursor wiring

Copy or merge from [`.cursor/mcp.json.example`](../../.cursor/mcp.json.example):

```json
"colab-mcp": {
  "command": "uvx",
  "args": ["git+https://github.com/googlecolab/colab-mcp"],
  "timeout": 30000
}
```

Reload MCP in Cursor after saving (`Settings → MCP → Refresh`).

First `uvx` run downloads ~100 packages; allow up to 60s on slow networks.

## sophia-v2 workflow (agent + Colab)

1. Open notebook (GPU runtime):

   https://colab.research.google.com/github/tomyimkc/sophia-agi/blob/main/notebooks/Sophia-LoRA-Colab.ipynb

2. **Runtime → Change runtime type → T4 GPU** → **Restart session**
3. Reload MCP so `colab-mcp` tools appear
4. Ask the agent to run the **sophia-v2** section: HF pull `tomyimkc/sophia-agi-lora-v1` → train `sophia-v2` → download zip
5. Eval: [Sophia-LoRA-Eval-Colab.ipynb](../../notebooks/Sophia-LoRA-Eval-Colab.ipynb) with v2 adapter path

**Base model:** `Qwen/Qwen2.5-3B-Instruct` only (must match v1 for `--resume-adapter`).

See [LoRA-Experiment.md](LoRA-Experiment.md) for dataset and local fallback.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `uvx` not found in MCP | Use full path to `uvx.exe` in `mcp.json` |
| No Colab tools after reload | Confirm Colab tab is open and logged in |
| MCP timeout on first start | Increase `timeout` to `60000`; first install is slow |
| Training wrong base | Stop; v2 must resume Qwen2.5-3B, not 7B |

## Related

- [MCP-Server.md](MCP-Server.md) — local `sophia-agi` validate/gate/benchmark tools
- [LoRA-Experiment.md](LoRA-Experiment.md) — v1/v2 training and eval