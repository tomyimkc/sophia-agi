# Habit-formation experiment lane

Pre-registration lane for the *Atomic Habits for Sophia* design note
(`docs/06-Roadmap/Atomic-Habits-for-Sophia.md`).

- **`measurement_spec.json`** — pre-registration for the flagship experiment
  **Habit-Strength Transfer (HST)**: does *difficulty-graded, reward-positive
  abstention* (H2) yield higher **identity-consistency on novel entities** (H5)
  than flat abstention reward? It also registers the 7 remaining roadmap
  hypotheses (H1, H3, H4, H6, H7, H8, H9) with their falsifiable claim, the
  metric each needs, and `status: not-yet-powered`.

**Status: pre-registration only — no result artifact exists yet.** The spec is
committed *before* any run so `tools/claim_gate.py --prefix HST --assert-prereg`
can prove the criteria predate the data. `candidateOnly: true`,
`canClaimAGI: false`. `primaryN` / `primaryMDE` are deliberately left
*to-be-computed by `tools/eval_stats.py`* — inventing a power number here would
itself violate the measurement contract (power-before-you-run).

Nothing in this lane is wired into CI yet (CI gates only the `M3-pilot` /
`M3-transfer` prefixes); it becomes a gated claim only once a result is produced
and clears `claim_gate`.
