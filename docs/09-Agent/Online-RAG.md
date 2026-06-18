# Online RAG — curated corpus + Gemini / Vertex

Trap-aware retrieval over the **Sophia curated corpus only** (no open-web grounding), then generation via **Gemini API** or **Vertex AI**, with the same epistemic gate as the agent CLI.

## Architecture

```text
Question → curated retrieve (rag/index) → Gemini or Claude → epistemic gate
                ↑
    data/, docs/, disputes/, reference/, teacher examples
    (benchmark holdouts excluded — same rules as LoRA training)
```

| Path | When to use |
|------|-------------|
| **Gemini API** | Local dev, Colab, quick MVP — `GOOGLE_API_KEY` |
| **Vertex AI** | Cloud Run production — `GOOGLE_GENAI_USE_VERTEXAI=true` + GCP project |

## Quick start (local)

```powershell
pip install -r requirements-rag.txt

# Build keyword index (no API key)
python tools/build_rag_index.py

# Optional: vector embeddings (requires Google key or Vertex)
python tools/build_rag_index.py --embed

# Ask with gate
python tools/sophia_rag.py "Did Confucius write the Dao De Jing?"
python tools/sophia_rag.py "Did Confucius write the Dao De Jing?" --json
```

Set in `.env` (see `.env.example`):

```env
GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash
SOPHIA_RAG_BACKEND=auto
```

`SOPHIA_RAG_BACKEND`:

| Value | Behavior |
|-------|----------|
| `auto` | Gemini if key/Vertex available; else Claude fallback |
| `gemini` | Force Gemini API |
| `vertex` | Force Vertex (`GOOGLE_GENAI_USE_VERTEXAI=true`) |
| `claude` | Claude generation; curated retrieve still used |
| `keyword` | Keyword retrieve + Claude |

## Index contents

Built by `tools/build_rag_index.py` from `agent/rag_sources.py`:

- `data/*.json` — attributions, traditions, domain records
- `docs/04-Disputes/*.md`, `docs/08-Domains/*.md`
- `benchmark/reference/responses-*.json` — teacher reference answers
- `training/examples/*.json` — **excluding** benchmark holdouts (`benchmarkCase`, trap IDs, benchmark questions)

Output: `rag/index/chunks.jsonl` (committed). Optional `rag/index/embeddings.npz` (gitignored; rebuild with `--embed`).

## Agent integration

`sophia_agent.py` and `serve_web.py` call `agent.retrieval.retrieve()`, which **automatically prefers** `rag/index` when present. Legacy inline corpus scan remains the fallback.

## HTTP API (Cloud Run)

```powershell
pip install -r requirements-rag.txt
uvicorn services.rag_api.main:app --host 127.0.0.1 --port 8080
```

```http
POST /ask
{"question": "Did Confucius write the Dao De Jing?", "mode": "advisor", "top_k": 8}
```

Deploy:

```powershell
.\tools\deploy_rag_api.ps1 -ProjectId YOUR_GCP_PROJECT -Region us-central1
```

Vertex on Cloud Run: set `GOOGLE_GENAI_USE_VERTEXAI=true`, attach a service account with Vertex AI User, and omit `GOOGLE_API_KEY` (uses ADC).

## Tests

```powershell
python tests/test_rag_index.py
```

## Related

- [Sophia-Agent.md](Sophia-Agent.md) — advisor / repo / life CLI
- [LoRA-Experiment.md](LoRA-Experiment.md) — local fine-tuned model path
- [MCP-Server.md](MCP-Server.md) — validate / gate / benchmark tools