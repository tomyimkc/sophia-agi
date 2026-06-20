# Discipline Layer (small-model source discipline)

*Added in v0.7.3.* A layer that lets **any local/small model** inherit Sophia's
"never merge lineages" rule at run time, plus the data to train the rule in. It
builds on the [Source-Discipline Gate](./Source-Discipline-Gate.md) (`provenance_faithful`)
and reuses the existing `doNotAttributeTo` corpus. All runtime paths are offline
and CPU-only; only the DPO *training* step needs a GPU.

## Phase 0 — user-supplied records

`agent.verifiers._load_provenance_records` also merges JSON records pointed to by
the `SOPHIA_DISCIPLINE_RECORDS` env var (a directory of `*.json`, a glob, a single
file, or several joined by the OS path separator). Each record needs a
`doNotAttributeTo` list; malformed/skipped records emit a warning. This lets a user
enforce their **own** attribution rules (legal, corporate, code provenance) through
the same machine-checked gate, beyond the seeded domains.

```bash
export SOPHIA_DISCIPLINE_RECORDS=~/my-provenance-rules/
# now provenance_faithful / check_claim also enforce your records
```

## Phase 1 — guarded completion loop

`agent/guarded.py`:

- `guarded_complete(query, on_fail=…)` — retrieve + format context → generate →
  judge with `provenance_faithful`. On a violation it branches by `on_fail`
  (default `$SOPHIA_ON_FAIL` then `repair`):
  - **repair** — one bounded re-generation that must clear the gate; if it still
    fails, a **cited abstention** (which itself passes the gate).
  - **abstain** — go straight to the cited abstention.
  - **hedge** — keep the answer but prepend a visible "unverified attribution" banner.
  - **passthrough** — return the unguarded answer (explicit opt-out).
- `check_claim(text)` — the mode-free verifier surface (`{passed, reasons,
  violations}`), unlike the moded `gate_check`. Exposed as the **`sophia_check_claim`**
  MCP tool; the CLI equivalent is `tools/source_discipline_cli.py`.

## Phase 2 — best-of-N, belief graph, confidence injection

- `agent/best_of.py` — `best_of(query, n=…)` samples N candidates and ranks by the
  gate (passing > violating, then fewer violations, then an optional `score_fn`),
  with early-exit on the first gate-passing sample.
- `okf.belief(graph, entity)` — per-entity belief record exposing
  `effectiveConfidenceRank` (min over the `derivesFrom` chain) and a
  `confidenceLaundered` flag, so a confident claim resting on weak provenance is
  caught structurally. Exposed as the **`sophia_belief`** MCP tool.
- `harness._memory_recall` — recalled pages are annotated with that **effective**
  (laundering-aware) confidence and a "confidence capped by weak provenance"
  warning, instead of their face-value `authorConfidence`.

## Phase 3 — hard-negative DPO miner

`tools/mine_hard_negatives.py` turns every `doNotAttributeTo` edge into contrastive
DPO pairs in four shapes — **direct**, **sibling** (forbidden author who really
authored a sibling work), **alias**, and **laundering** (the merge laundered
through passive/possessive/"a work by" grammar). Every candidate is
**self-validated** through `provenance_faithful`: the `rejected` must trip the gate
and the `chosen` must pass, so the dataset is honest by construction.

```bash
python tools/mine_hard_negatives.py        # training/hard_negatives_dpo.jsonl
```

CPU-only data generation; the DPO training that consumes the pairs needs a GPU. The
generated `.jsonl` is regenerable output and is not committed.

## Phase 4 — sophia-guard CLI

`tools/sophia_guard.py` runs any model from the unified adapter (ollama, llama.cpp,
grok, openclaw, …) behind the guarded loop:

```bash
echo "Who wrote the Dao De Jing?" | python tools/sophia_guard.py
python tools/sophia_guard.py --query "..." --provider ollama:llama3.2 --on-fail abstain --json
```

Exit `0` when an answer is surfaced (clean / repaired / abstained / hedged), `1` on
a model error or a violating passthrough.

## Honest limits

Reuses the ~31-record `doNotAttributeTo` corpus (high precision, **not** a general
hallucination catcher). The repair/abstain paths need a working model; the gate
itself is the same local-regex check. Adds nothing to, and claims nothing about, the
AGI-candidate proof package.
