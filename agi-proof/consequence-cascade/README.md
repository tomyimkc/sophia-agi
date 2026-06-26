# Consequence-cascade evidence (candidate)

A deterministic, offline report that **tunes** the ConsequenceGate's `flipSeverityEscalate`
threshold — the fraction of the OKF belief graph a retraction must orphan before the 8th
conscience path forces `escalate` (stronger process) rather than `allow`. Candidate-only
(`candidateOnly: true`, `level3Evidence: false`) — a reproducible evidence artifact about
*how the gate's threshold separates severe from routine cascades*, not a capability or AGI
claim, and not validation against real belief graphs.

| Report | What it measures | Reproduce |
| --- | --- | --- |
| `../benchmark-results/consequence-cascade.public-report.json` | Per-case verdict accuracy + flip-severity-band match over 38 synthetic OKF retraction graphs, plus a threshold sweep that picks the max-accuracy `flipSeverityEscalate`. | `python tools/run_consequence_cascade_benchmark.py` |

## What the report covers

The pack (`eval/consequence_cascade/consequence_cascade_40_v1.jsonl`) is 38 synthetic OKF
graphs with a retraction target each, labeled by **graph structure** (never a model
judgement, so the gate is never grading itself):

- **Severe cascades** (13): retracting a high-fanout root or mid-tree node orphans a large
  fraction of the graph (severity band 0.3–1.0) → expected `escalate`.
- **Routine retractions** (15): retracting an isolated node or a single leaf orphans a tiny
  fraction (severity band 0.0–0.15), and multi-source nodes survive → expected `allow`.
- **Unbounded targets** (5): a ghost retraction target that does not resolve → expected
  `abstain` (fail-closed: consequence cannot be bounded).
- **Boundary cases** (5): severities deliberately placed in the 0.14–0.30 discriminator zone
  (e.g. retracting one leaf of a 5-node star → 1/5 = 0.20), where the threshold's exact value
  matters. These are the cases that make the sweep discriminate.

The **threshold sweep** re-classifies every case at each candidate in
`[0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40, 0.50]` and picks the value with
maximum verdict-accuracy, ties broken toward the value with the greatest margin to the
nearest case severity (a wide clean gap, not a knife-edge).

## Result

`recommendedFlipSeverityEscalate = 0.15`, at 100% verdict accuracy. This is the value that
already shipped as a hand-pick placeholder; the sweep gives it a **data-derived
justification**: 0.15 sits in the clean gap between the worst `allow` severity (0.143, a
7-node-tree leaf) and the worst `escalate` severity (0.167, a 6-node-star leaf). Thresholds
below 0.12 leak routine 0.143 cases into escalate; thresholds above 0.18 leak the 0.167
boundary escalate into allow.

## Honest bound

Deterministic, offline, pure-stdlib over **synthetic** graphs. The ground-truth label is the
structural fraction a retraction orphans — there is no model in the loop and no
self-grading. The recommended threshold is data-derived from synthetic graph structure; it is
**not validated against real belief graphs** (which may have different severity distributions),
and it earns `level3Evidence: true` only after a real run routes retraction decisions through
the live ConsequenceGate and the escalate/allow split is shown to match operational intent.
