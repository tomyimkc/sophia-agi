# 1 · Intake & Routing

**Role in the master flow.** The front gate: turns a raw case into a typed, contract-checked request
and decides which downstream paths (retrieval, council, claim-verification) it needs. Ablation flag
`use_intake`. Suppressing it sends the raw query straight to context-gathering.

```mermaid
flowchart TD
    Q([Raw case / query / task]) --> INTAKE["Request triage + intake contract<br/>agent/intake · sophia_contract/intake.py<br/>schema/intake-contract-1.0.0.json"]
    INTAKE --> VALID{Contract valid?}
    VALID -->|no| REJECT["Reject / clarify<br/>fail-closed"]
    VALID -->|yes| TYPE["Classify request type<br/>coding · figure · planning · learning · QA"]
    TYPE --> CROUTE{"Route atomic claims?<br/>agent/claim_router.py<br/>flag: use_claim_router"}
    CROUTE -->|yes| CLAIMS["Split into atomic claims<br/>→ claim-type verifiers"]
    CROUTE -->|no| PASS["Whole-response path"]
    TYPE --> SWARM{"Multi-agent task?<br/>agent/swarm_router.py"}
    SWARM -->|yes| DISPATCH["Route to sub-agents<br/>agent/subagent.py · team_agents.py"]
    SWARM -->|no| SOLO["Single-agent path"]
    CLAIMS --> OUT([To grounded context →])
    PASS --> OUT
    DISPATCH --> OUT
    SOLO --> OUT
```

**Modules:** `agent/intake/`, `agent/claim_router.py`, `agent/swarm_router.py`, `agent/subagent.py`,
`agent/team_agents.py`. **Contract:** `schema/intake-contract-1.0.0.json`.

**Thesis note.** The intake contract is what makes every downstream step *suppressible and
measurable* — it stamps the request shape so the ablation runner can toggle each pipeline stage
independently. That toggle-ability is the basis of the whole baseline/ablation evidence story.