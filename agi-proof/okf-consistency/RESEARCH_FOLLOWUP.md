# Research follow-up: capturing the real belief-dynamics signals

The belief-dynamics/forgetting layer (`okf/decay_okf`, `frontier_demotion`,
`forgetting_audit`) is **wired into the consolidation audit ledger** as of this change,
but it remains **`level3Evidence: false`**. The reason is concrete, not procedural:

> The OKF frontmatter records only provenance. It records **none** of the signals the
> dynamics layer needs to do its real work. Today every belief is projected with
> **unrecorded placeholders** (timestamps = `GENESIS_EPOCH`, `surprise = 0`,
> `reinforcement_count = 0`) — see `okf/belief_state_projection.py`. Those placeholders
> cannot, by construction, move a belief toward suppression; they can only leave it
> as-is. So the layer's *time-decay* and *surprise-gating* are effectively no-ops over
> the real corpus, and the only thing the wiring earns honestly today is a **real
> tamper-evident audit trail of every consolidation selection**.

Earning `level3Evidence: true` means capturing the real signals. This file scopes that
work so it is named, not buried. It is a research roadmap, not a sprint plan.

## The three signals, by cost

### 1. Temporal — `written_at` / `last_reinforced_at`
**Cost: low (~1 day). Engineering.** Add two frontmatter fields, stamp them in
`tools/wiki_sync.py` (the generator) on first write, add an update hook for page edits,
and backfill the 96 existing pages with a fixed "genesis" epoch honestly marked as
"arrival time unknown." No new instrumentation — pure plumbing. This unblocks
*meaningful* time-decay.

### 2. Usage — `reinforcement_count`
**Cost: moderate (~3–5 days). Instrumentation on the live path.** A belief is
"reinforced" when it is grounded in a real retrieval/answer and survived the gate. No
counter exists today. Recording it honestly requires hooking the grounding path
(`agent.continual_retention.belief_state` / the gate-cleared set) to increment a
per-belief counter on each grounded use, with a persistence story (append-log folded
into frontmatter). Must not regress the existing offline tests.

### 3. Predictive — `surprise`
**Cost: high (weeks). Open research.** "Surprise" = how unexpected a belief was, given
current memory — a predictive probability under the model: the actual engram-
consolidation signal. Nothing measures this today. Doing it honestly means defining the
predictive model (a retrieval/likelihood over the existing graph), computing it at first
observation, and auditing it. **This is the only signal whose capture would make the
layer genuinely evidence-grade**, and it is research, not engineering.

## Net

`written_at` is cheap plumbing; `reinforcement_count` is a moderate instrumentation
task on the grounding path; `surprise` is the real research follow-up. Until `surprise`
exists, the layer's `level3Evidence` stays `false` — wiring the audit ledger (this
change) is the honest ceiling without it.
