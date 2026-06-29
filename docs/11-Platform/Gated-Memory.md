# Gated Memory — verifier-gated persistent agent memory

`agent/gated_memory.py` implements **durable agent memory where a write persists only if it
clears the machine gate**. A hallucination is never written into the readable store.

`canClaimAGI` stays **false**. This is a filter on persistence, not a claim about truth or
general intelligence.

## Inspiration and the difference

The "ECC" agent-memory pattern persists *everything* an agent emits into a SQLite database via
lifecycle hooks, so a sibling session can read it back later. That makes memory durable — and
durable-ly wrong: a single hallucination, once written, is recalled forever.

Sophia keeps the durable-SQLite half and **fuses it with the existing verifier**
(`agent.gate.check_response`). The trust boundary Sophia already enforces on every answer is
made durable across sessions and processes:

| | ECC pattern | Gated Memory (Sophia) |
|---|---|---|
| What persists | every emitted message | only gate-cleared writes |
| Store on hallucination | yes (recalled forever) | held in a separate quarantine table |
| What a sibling reads | all of it | only the `accepted` table |

## API

```python
from agent.gated_memory import GatedMemory

mem = GatedMemory("agent_memory.db")           # verifier=None -> real agent.gate
mem.remember(text, *, question=None, source=None)
#   -> {"stored": True,  "verdict": "accepted"}                 (cleared the gate)
#   -> {"stored": False, "verdict": "held", "reasons": [...]}   (gate flagged it)

mem.recall(query=None, limit=100)  # ONLY accepted rows; optional LIKE %query% on text
mem.quarantined()                  # audit-only view of held rows + their reasons
mem.audit()                        # {"accepted": n, "held": m}
```

- `GatedMemory(db_path=":memory:", *, verifier=None, mode="advisor")`.
- Two SQLite tables are created if absent: `accepted(id, ts, source, text, question)` and
  `quarantine(id, ts, source, text, question, reasons)`.
- Default `verifier=None` uses the real gate: a write is clean iff
  `check_response(text, mode=mode, question=question or text, route_claims=True)["violations"]`
  is empty.
- A custom `verifier(text, question) -> (clean: bool, reasons: list[str])` may be injected; tests
  and the offline invariants use a deterministic stub so no model is required.

### The trust boundary, made durable

`remember` is the boundary. Clean text is `INSERT`ed into `accepted` (the only table `recall`
reads). Flagged text is `INSERT`ed into `quarantine` with its reasons — kept for audit, **never**
returned by `recall`. Because the data lives in SQLite, a brand-new `GatedMemory` opened on the
same db file (a sibling session, a later process) sees exactly the previously-accepted rows.

## Falsifiable invariants

`offline_invariants()` (and `tests/test_gated_memory.py`) prove, deterministically with an
injected stub verifier:

- a clean claim is stored and recalled;
- a flagged claim is quarantined, **not** recalled, and its reasons are retained;
- a brand-new `GatedMemory` instance opened on the **same temp db file** sees the
  previously-accepted claim (cross-session persistence);
- `recall` never returns a held row;
- `audit` totals reconcile (`accepted` + `held`).

One additional check exercises the **real gate** (no stub): remembering
*"Confucius wrote the Dao De Jing."* is **held**, while the corrected
*"No, Confucius did not write the Dao De Jing; it is a Daoist text attributed to Laozi. This is a
common Confucian misconception."* is **accepted** — matching `agent/gate.py`.

Run them:

```
python -m agent.gated_memory      # prints PASS/FAIL per check, exits 0/1
python tests/test_gated_memory.py # prints "PASS N tests"
```

## Honest limits

- **The gate is a filter, not a truth oracle.** It catches the machine-checkable failures it
  knows about (false equalities, forbidden attributions, unverifiable citations, ...). A false
  statement with **no detectable violation can still be stored**. This *bounds* persisted
  hallucination to "passed the gate"; it does not eliminate it.
- The store inherits the verifier's coverage and blind spots. Strengthening the gate strengthens
  the memory; gaps in the gate are gaps in the memory.
- `quarantine` is an audit surface only. It must never be re-injected as readable context — doing
  so would defeat the whole point.
- Timestamps for the live store use `time.time()`; the invariants do not assert on timestamps and
  inject a stub verifier so they stay deterministic and offline.
