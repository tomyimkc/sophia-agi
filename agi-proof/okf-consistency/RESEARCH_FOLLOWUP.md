# Research follow-up: capturing the real belief-dynamics signals

> **STATUS UPDATE — signal #3 (`surprise`) is DELIVERED.** `okf/surprise_signal.py` now
> measures surprise as a **leave-one-out retrieval-likelihood over the OKF graph** (the
> approach this file scoped below), and `tools/eval_okf_surprise.py` validates it on three
> deterministic, offline panels (falsifiable separation; a real, auditable corpus run that
> the placeholder could not produce; and a proof the anti-forgetting gate stays the floor).
> That report carries `level3Evidence: true` **scoped to the surprise signal only**, with
> `canClaimAGI: false`. The honest caveat is in §3 and the report: this is the
> *leave-one-out* substitute (`P(belief | rest of memory)`), **not** the temporal
> "surprise at first observation" — that still needs `written_at` (signal #1). Signals #1
> and #2 remain open, so time-decay and usage-reinforcement are still no-ops and the
> **broader** forgetting layer stays `level3Evidence: false`.

The belief-dynamics/forgetting layer (`okf/decay_okf`, `frontier_demotion`,
`forgetting_audit`) is **wired into the consolidation audit ledger** as of #163, and its
**surprise** signal is now measured (above). The reason the broader layer remained
`level3Evidence: false` was concrete, not procedural:

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

### 3. Predictive — `surprise`  — **DELIVERED (leave-one-out variant)**
**Originally scoped: high (weeks), open research.** "Surprise" = how unexpected a belief
was, given current memory — a predictive probability under the model: the actual engram-
consolidation signal. The honest predictive model named here was *"a retrieval/likelihood
over the existing graph,"* and that is exactly what shipped: `okf/surprise_signal.py`
computes the per-token NLL of a belief's content under a smoothed interpolated unigram
model built from the **rest** of the corpus, focused on the belief's provenance
neighbourhood. No GPU, no neural model, deterministic. (Approach B — a small model's NLL —
was rejected: the repo's logprob path is MLX/Apple-Silicon-only, so it cannot run
reproducibly here, and a neural NLL over 96 pages invites the same overclaim the
`predictive_world_model → tabular_transition_model` rename corrected.)

**What is NOT yet delivered (the honest gap):** true *surprise at first observation* needs
an arrival-time ordering of the corpus (`written_at`, signal #1). Lacking it, we compute
the **leave-one-out** surprise `P(belief | the rest of memory)` — a weaker but genuinely
measured substitute. When `written_at` lands, the same machinery restricts to the
arrival-time prefix to recover the temporal signal. Evidence: `tools/eval_okf_surprise.py`
→ `agi-proof/okf-consistency/surprise-signal.public-report.json`.

## Net

`written_at` is cheap plumbing; `reinforcement_count` is a moderate instrumentation task
on the grounding path; `surprise` **is now measured** via a leave-one-out
retrieval-likelihood (the scoped, honest variant). The **surprise signal** is
`level3Evidence: true` (scoped); the **broader** forgetting layer stays `false` until
`written_at` and `reinforcement_count` land, because time-decay and usage-reinforcement
remain no-ops without them. Next wiring step: adopt `project_corpus_measured` in
`tools/run_cls_consolidation.py` so the live consolidation manifest projects measured
surprise (kept out of this change to preserve that run's landed honesty tests).
