# Online RAG — curated corpus, web evidence, Gemini / Vertex

Trap-aware retrieval starts with the **Sophia curated corpus**, then can
optionally add web evidence from provider APIs. Generation can use **Gemini
API** or **Vertex AI**, with the same epistemic gate as the agent CLI.

## Architecture

```text
Question → curated retrieve (rag/index) → optional web evidence → Gemini/Claude → gate → rubric review
                ↑                         ↑
    data/, docs/, disputes/, reference/    Brave / Tavily / SerpAPI
    teacher examples                       (off by default for hidden evals)
    (benchmark holdouts excluded)
```

| Path | When to use |
|------|-------------|
| **Gemini API** | Local dev, Colab, quick MVP — `GOOGLE_API_KEY` |
| **Vertex AI** | Cloud Run production — `GOOGLE_GENAI_USE_VERTEXAI=true` + GCP project |
| **Web evidence** | Fresh sources, official docs, academic papers, thesis/source checks |

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
SOPHIA_WEB_SEARCH_PROVIDER=off # off | auto | brave | tavily | serpapi
BRAVE_SEARCH_API_KEY=
TAVILY_API_KEY=
SERPAPI_API_KEY=
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

Web evidence is a separate, opt-in layer:

```powershell
# Local RAG evidence only; no external prompt sharing
python tools/sophia_agent.py web_evidence "ARC-AGI skill acquisition efficiency"

# Online evidence through the configured provider
python tools/sophia_agent.py web_evidence "ARC-AGI skill acquisition efficiency" --web-evidence --web-provider auto

# Hidden eval runner, reviewer-approved only
python tools/run_hidden_eval_sophia.py private/hidden-evals/pack.json `
  --responses-out private/hidden-evals/responses.json `
  --private-report-out private/hidden-evals/private-report.json `
  --public-report-out agi-proof/benchmark-results/hidden-pack.public-report.json `
  --web-evidence --web-provider brave
```

Do not enable `--web-evidence` on hidden packs unless the reviewer accepts that
sanitized or full prompt text may be sent to the configured search provider.

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
python tests/test_web_evidence.py
python tests/test_rubric_review.py
```

## Is Online RAG Good For AGI Development?

Yes, but only as a governed evidence system. Online RAG helps Sophia handle
fresh facts, papers, APIs, docs, and unfamiliar domains without pretending that
the base model already knows everything. It improves AGI-candidate evidence for
tool use, source seeking, and learning under novelty.

It is not AGI proof by itself. For proof-quality evaluation, online RAG needs:

- source ranking that prefers primary, official, academic, or curated-local
  sources;
- exact source labels and URLs in the answer;
- logs of queries, providers, timestamps, and failures;
- anti-contamination rules for hidden tests;
- append-only memory writes, so new evidence does not overwrite old records;
- baseline and ablation comparisons with web search on/off.

## Related

- [Sophia-Agent.md](Sophia-Agent.md) — advisor / repo / life CLI
- [LoRA-Experiment.md](LoRA-Experiment.md) — local fine-tuned model path
- [MCP-Server.md](MCP-Server.md) — validate / gate / benchmark tools
