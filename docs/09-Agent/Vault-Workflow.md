---
title: Vault → Contract Gate Workflow
tags: [workflow, obsidian, contract, diagram]
generated_by: tools/vault_workflow_diagram.py
---

# Vault → Contract Gate Workflow

> **Generated file.** Do not hand-edit — run `python tools/vault_workflow_diagram.py`.
> The routing is derived from the code (`route_after_verify` + `service._decide`),
> so the diagram cannot drift from what the gate actually does.

**View it on your iPhone two ways:**

1. **GitHub app / mobile Safari** — open this file in the repo; GitHub renders the
   Mermaid blocks below into diagrams.
2. **Obsidian mobile** — copy this note into your vault (or run
   `python tools/vault_workflow_diagram.py --out /path/to/your/Vault/`); Obsidian
   renders Mermaid natively, so you get the interactive diagram in the app.

## 1 · Note lifecycle — record → verify → route → publish

Every Obsidian note is a *claim*. `VaultGate.gate_note()` records it, runs the
fail-closed `_decide` rule ladder, routes the verdict, and stamps the result back
into the note's frontmatter. **A note ships only when `gate_status == accepted`.**

```mermaid
flowchart TD
    N["📝 Obsidian note<br/>(frontmatter: role, blp_level, sources, parents)"]
    N --> G["VaultGate.gate_note()"]
    G --> IK["idempotency_key =<br/>vault:relpath:body-fingerprint"]
    IK --> REC["contract.record_claim()"]
    REC -->|error| REJ
    REC --> VER["contract.verify_claim() → _decide()"]

    %% _decide rule ladder (first match wins, fail-closed)
    VER --> D1{"1· BLP no-read-up?"}
    D1 -->|violation| H_BLP["held · blp_violation"]
    D1 -->|ok| D2{"2· verify budget left?"}
    D2 -->|exhausted| H_BUD["held · over_budget"]
    D2 -->|ok| D3{"3· superseded?"}
    D3 -->|yes| SUP["superseded"]
    D3 -->|no| D4{"4· prior human ruling?"}
    D4 -->|yes| PREF["verdict = human ruling<br/>(feedback loop short-circuit)"]
    D4 -->|no| D5{"5· sources present & valid?"}
    D5 -->|none| H_NOS["held · no_source"]
    D5 -->|"refuted / invalid"| R_REF["rejected"]
    D5 -->|all stale| H_STALE["held · stale_source"]
    D5 -->|ok| D6{"6-7· low-risk &<br/>confidence ≥ 0.75 & cited?"}
    D6 -->|"yes · blp ∈ UNCLASSIFIED"| ACC["accepted"]
    D6 -->|no| H_HUM["held · needs_human"]

    %% verdict -> route (route_after_verify: the single authority)
    PREF --> RT{"route_after_verify"}
    ACC --> RT
    H_BLP --> RT
    H_BUD --> RT
    H_NOS --> RT
    H_STALE --> RT
    H_HUM --> RT
    SUP --> RT
    R_REF --> RT
    RT -->|accepted| PUB["✅ publishable"]
    RT -->|held| HUMR["🧑 human review"]
    RT -->|"superseded / rejected"| REJ["🚫 rejected / not published"]

    %% stamp verdict back into the note's frontmatter
    PUB --> STAMP["stamp frontmatter:<br/>provenance_id, gate_status, confidence, reasons"]
    HUMR --> STAMP
    REJ --> STAMP
    STAMP --> PUBGATE["publish_if_accepted()<br/>— only gate_status == accepted ships"]

    %% approve-by-exception feedback loop
    HUMR -.founder approves.-> HV["record_human_verdict(accepted)"]
    HV -.re-gate.-> VER

    classDef ok fill:#1b5e20,stroke:#0b3d0b,color:#fff;
    classDef hold fill:#8d6e00,stroke:#5c4700,color:#fff;
    classDef bad fill:#7f1d1d,stroke:#4c0f0f,color:#fff;
    class ACC,PUB,PUBGATE ok;
    class H_BLP,H_BUD,H_NOS,H_STALE,H_HUM,HUMR,PREF hold;
    class R_REF,SUP,REJ bad;
```

### Verdict → route (the single authority: `route_after_verify`)

| Verdict | `_decide` branch | Goes to |
|---------|------------------|---------|
| `accepted` | low-risk, high-confidence, cited | **publishable** |
| `held` | blp_violation | **human review** |
| `held` | over_budget | **human review** |
| `held` | no_source | **human review** |
| `held` | stale_source | **human review** |
| `held` | needs_human | **human review** |
| `superseded` | successor exists | **rejected / not published** |
| `rejected` | source refuted / invalid | **rejected / not published** |

Only `accepted` is publishable. `held` waits for a founder ruling
(approve-by-exception); once approved, the human verdict is recorded and the note
is re-gated — the preference feedback loop then short-circuits it to `accepted` on
every future pass. `superseded` / `rejected` never publish.

## 2 · Copywriting pipeline over the vault

`CopywritingPipeline` (`sophia_contract/pipelines/copywriting.py`) is the same gate
applied to a bespoke-voice drafting loop: a brief note becomes a gated draft in
`06_Review/`, and only an accepted (or founder-approved) draft is published.

```mermaid
flowchart LR
    B["📥 brief note"] --> DR["draft()<br/>Role · Data · Requirements · Format"]
    DR --> RV["06_Review/&lt;brief&gt;.draft.md"]
    RV --> GT["VaultGate.gate_note()"]
    GT -->|accepted| P["publish()"]
    GT -->|"held: needs_human"| A["founder approve()"]
    A -->|"re-gate → accepted"| P
    classDef ok fill:#1b5e20,stroke:#0b3d0b,color:#fff;
    classDef hold fill:#8d6e00,stroke:#5c4700,color:#fff;
    class P ok;
    class A hold;
```

## Keeping this note honest

`tests/test_vault_workflow_diagram.py` regenerates this file and fails CI if the
committed copy drifts from the generator — the same discipline the repo uses for
its other generated artifacts.
