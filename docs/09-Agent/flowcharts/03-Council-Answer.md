# 3 · Council & Answer Generation

**Role in the master flow.** Produces the candidate answer. For coding/figure/planning cases a
multi-seat **council** deliberates before the model call; otherwise the composed prompt goes straight
to the frozen model backend. Ablation flag `use_council`.

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

**Modules:** `agent/coding_council.py`, `council_deliberate.py`, `council_personas.py`,
`council_registry.py`, `sector_council.py`, `council_format.py`, `prompts.py`, `model.py`,
`subagent.py`, `team_agents.py`.

**Thesis note.** Two facts a reviewer will check: the real adapter is
`agent.model.default_client(spec).generate(system, user) -> ModelResult` (not a `complete(prompt)`
call), and `agent.model._auto_provider()` returns `"mock"` with no API key — mock `.generate()`
fabricates text at `ok=True`. Any measured claim must assert `cfg.kind != "mock"`. Council catch-rate
(1.0 vs 0.27 monolith) in the repo's reports is on *stub* seats — real trained discipline adapters are
a named open gap.