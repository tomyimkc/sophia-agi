#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Render the Obsidian-vault → contract-gate workflow as a Mermaid Markdown note.

The "Obsidian connection system" is the ``VaultGate`` bridge
(``sophia_contract/vault.py``) plus the copywriting pipeline
(``sophia_contract/pipelines/copywriting.py``): an Obsidian/Markdown note flows
``record_claim -> verify_claim (_decide rules) -> route -> stamp frontmatter ->
publish``, with an approve-by-exception human loop.

This tool turns that flow into a **Mermaid** diagram so it can be *seen* on a
phone. Mermaid renders natively in two places you already use:

  * GitHub markdown (view the committed note in the GitHub app / mobile Safari), and
  * Obsidian mobile (drop the note into your vault, open it in the Obsidian app).

The routing (which verdict publishes vs. escalates vs. rejects) is not hand-typed
— it is derived from ``route_after_verify`` in ``langgraph_nodes`` and the verdict
vocabulary of ``service._decide``, so the diagram cannot drift away from the code.

    python tools/vault_workflow_diagram.py                 # -> docs/09-Agent/Vault-Workflow.md
    python tools/vault_workflow_diagram.py --stdout        # print, do not write
    python tools/vault_workflow_diagram.py --out ~/Vault/  # drop into an Obsidian vault
    python tools/vault_workflow_diagram.py --check         # drift check (CI): fail if stale

A companion test (``tests/test_vault_workflow_diagram.py``) runs ``--check`` so the
committed note stays in lockstep with the source of truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sophia_contract.langgraph_nodes import route_after_verify  # noqa: E402
from sophia_contract.service import AUTO_ACCEPT_CONFIDENCE, LOW_RISK_LEVELS  # noqa: E402

DEFAULT_OUT = ROOT / "docs" / "09-Agent" / "Vault-Workflow.md"

# Every verdict service._decide can emit, with the _decide branch (held_reason) that
# produces it. route_after_verify() is the single authority for where each verdict goes.
VERDICTS: list[tuple[str, str]] = [
    ("accepted", "low-risk, high-confidence, cited"),
    ("held", "blp_violation"),
    ("held", "over_budget"),
    ("held", "no_source"),
    ("held", "stale_source"),
    ("held", "needs_human"),
    ("superseded", "successor exists"),
    ("rejected", "source refuted / invalid"),
]

# Human-readable label for each terminal route that route_after_verify can return.
ROUTE_LABEL = {
    "publish": "publishable",
    "review": "human review",
    "reject": "rejected / not published",
}


def _routes() -> dict[str, str]:
    """verdict -> terminal route, computed from the code (never hand-typed)."""
    return {verdict: route_after_verify({"verdict": verdict}) for verdict, _ in VERDICTS}


def _decide_flow() -> str:
    """Mermaid flowchart of gate_note: note -> record -> _decide branches -> route."""
    routes = _routes()
    lines = [
        "flowchart TD",
        '    N["📝 Obsidian note<br/>(frontmatter: role, blp_level, sources, parents)"]',
        '    N --> G["VaultGate.gate_note()"]',
        '    G --> IK["idempotency_key =<br/>vault:relpath:body-fingerprint"]',
        '    IK --> REC["contract.record_claim()"]',
        '    REC -->|error| REJ',
        '    REC --> VER["contract.verify_claim() → _decide()"]',
        "",
        "    %% _decide rule ladder (first match wins, fail-closed)",
        '    VER --> D1{"1· BLP no-read-up?"}',
        '    D1 -->|violation| H_BLP["held · blp_violation"]',
        '    D1 -->|ok| D2{"2· verify budget left?"}',
        '    D2 -->|exhausted| H_BUD["held · over_budget"]',
        '    D2 -->|ok| D3{"3· superseded?"}',
        '    D3 -->|yes| SUP["superseded"]',
        '    D3 -->|no| D4{"4· prior human ruling?"}',
        '    D4 -->|yes| PREF["verdict = human ruling<br/>(feedback loop short-circuit)"]',
        '    D4 -->|no| D5{"5· sources present & valid?"}',
        '    D5 -->|none| H_NOS["held · no_source"]',
        '    D5 -->|"refuted / invalid"| R_REF["rejected"]',
        '    D5 -->|all stale| H_STALE["held · stale_source"]',
        f'    D5 -->|ok| D6{{"6-7· low-risk &<br/>confidence ≥ {AUTO_ACCEPT_CONFIDENCE} & cited?"}}',
        f'    D6 -->|"yes · blp ∈ {"/".join(LOW_RISK_LEVELS)}"| ACC["accepted"]',
        '    D6 -->|no| H_HUM["held · needs_human"]',
        "",
        "    %% verdict -> route (route_after_verify: the single authority)",
        '    PREF --> RT{"route_after_verify"}',
    ]
    # Wire every verdict node into the route decision, then to its terminal, from code.
    verdict_node = {
        "accepted": "ACC", "superseded": "SUP", "rejected": "R_REF",
    }
    held_nodes = ["H_BLP", "H_BUD", "H_NOS", "H_STALE", "H_HUM"]
    for node in [verdict_node["accepted"], *held_nodes,
                 verdict_node["superseded"], verdict_node["rejected"]]:
        lines.append(f"    {node} --> RT")
    lines += [
        f'    RT -->|accepted| PUB["✅ {ROUTE_LABEL[routes["accepted"]]}"]',
        f'    RT -->|held| HUMR["🧑 {ROUTE_LABEL[routes["held"]]}"]',
        f'    RT -->|"superseded / rejected"| REJ["🚫 {ROUTE_LABEL[routes["rejected"]]}"]',
        "",
        "    %% stamp verdict back into the note's frontmatter",
        '    PUB --> STAMP["stamp frontmatter:<br/>provenance_id, gate_status, confidence, reasons"]',
        "    HUMR --> STAMP",
        "    REJ --> STAMP",
        '    STAMP --> PUBGATE["publish_if_accepted()<br/>— only gate_status == accepted ships"]',
        "",
        "    %% approve-by-exception feedback loop",
        '    HUMR -.founder approves.-> HV["record_human_verdict(accepted)"]',
        "    HV -.re-gate.-> VER",
        "",
        "    classDef ok fill:#1b5e20,stroke:#0b3d0b,color:#fff;",
        "    classDef hold fill:#8d6e00,stroke:#5c4700,color:#fff;",
        "    classDef bad fill:#7f1d1d,stroke:#4c0f0f,color:#fff;",
        "    class ACC,PUB,PUBGATE ok;",
        "    class H_BLP,H_BUD,H_NOS,H_STALE,H_HUM,HUMR,PREF hold;",
        "    class R_REF,SUP,REJ bad;",
    ]
    return "\n".join(lines)


def _pipeline_flow() -> str:
    """Mermaid flowchart of the copywriting pipeline lifecycle over the vault."""
    return "\n".join([
        "flowchart LR",
        '    B["📥 brief note"] --> DR["draft()<br/>Role · Data · Requirements · Format"]',
        '    DR --> RV["06_Review/&lt;brief&gt;.draft.md"]',
        '    RV --> GT["VaultGate.gate_note()"]',
        '    GT -->|accepted| P["publish()"]',
        '    GT -->|"held: needs_human"| A["founder approve()"]',
        '    A -->|"re-gate → accepted"| P',
        "    classDef ok fill:#1b5e20,stroke:#0b3d0b,color:#fff;",
        "    classDef hold fill:#8d6e00,stroke:#5c4700,color:#fff;",
        "    class P ok;",
        "    class A hold;",
    ])


def render() -> str:
    """Build the full Mermaid Markdown note (deterministic; no timestamps)."""
    routes = _routes()
    route_rows = "\n".join(
        f"| `{verdict}` | {reason} | **{ROUTE_LABEL[routes[verdict]]}** |"
        for verdict, reason in VERDICTS
    )
    return f"""---
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
{_decide_flow()}
```

### Verdict → route (the single authority: `route_after_verify`)

| Verdict | `_decide` branch | Goes to |
|---------|------------------|---------|
{route_rows}

Only `accepted` is publishable. `held` waits for a founder ruling
(approve-by-exception); once approved, the human verdict is recorded and the note
is re-gated — the preference feedback loop then short-circuits it to `accepted` on
every future pass. `superseded` / `rejected` never publish.

## 2 · Copywriting pipeline over the vault

`CopywritingPipeline` (`sophia_contract/pipelines/copywriting.py`) is the same gate
applied to a bespoke-voice drafting loop: a brief note becomes a gated draft in
`06_Review/`, and only an accepted (or founder-approved) draft is published.

```mermaid
{_pipeline_flow()}
```

## Keeping this note honest

`tests/test_vault_workflow_diagram.py` regenerates this file and fails CI if the
committed copy drifts from the generator — the same discipline the repo uses for
its other generated artifacts.
"""


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, default=None,
                    help="write to this path (a directory → <dir>/Vault-Workflow.md); "
                         "default is docs/09-Agent/Vault-Workflow.md")
    ap.add_argument("--stdout", action="store_true", help="print to stdout, do not write")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed note differs from freshly generated")
    args = ap.parse_args(argv)

    content = render()

    if args.stdout:
        sys.stdout.write(content)
        return 0

    if args.check:
        current = DEFAULT_OUT.read_text(encoding="utf-8") if DEFAULT_OUT.exists() else ""
        if current != content:
            sys.stderr.write(
                f"DRIFT: {DEFAULT_OUT.relative_to(ROOT)} is stale. "
                "Run: python tools/vault_workflow_diagram.py\n")
            return 1
        print(f"ok: {DEFAULT_OUT.relative_to(ROOT)} is in sync")
        return 0

    out = args.out or DEFAULT_OUT
    if out.exists() and out.is_dir() or (args.out and str(args.out).endswith("/")):
        out = out / "Vault-Workflow.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
