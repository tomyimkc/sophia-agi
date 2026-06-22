# Capability-Retention Guardrail + Program Packaging (Spec D)

Spec D closes the MBTI-Vector-Agents program: it builds the capability-retention
guardrail that produces Spec B's SSA `capability_drop`/`coherence` inputs, and
exposes Specs A–D through the MCP server + portable skill.

## D1 — Capability-retention guardrail
`agent/steering/capability.py` scores a bundled arithmetic slice
(`data/capability_arithmetic.json`) deterministically: answer-correctness vs a
gold number, `arithmetic_sound` soundness, and a deterministic coherence proxy
(0–100) that catches the degeneracy high-alpha steering produces.
`capability_cell(base, steered)` assembles the SSA cell —
`capability_drop = max(0, (base_acc − steer_acc)/base_acc)`,
`retains = capability_drop < 0.05 and coherence ≥ 75` — the exact predicate
`ssa_verdict` applies. `tools/run_capability.py --dry-run` runs the deterministic
core (`CAPABILITY RETENTION VERIFIED ✓`); `--model granite` runs a reduced real
steered-vs-unsteered slice.

**Expected result:** a capability *drop* under steering. That is the honest,
on-brand finding — steering strong enough to move a trait degrades reasoning,
which *explains* Spec B's SSA null. A drop is reported as a drop.

## D2 — Full MCP / skill packaging
Read-only tools (`sophia_ocean_measure`, `sophia_capability_retention`,
`sophia_council_diversity`, `sophia_pif_dryrun`) + a `sophia://program/status`
resource expose A–D, reusing the existing `@audited` gating posture (read-only ⇒
ungated). The tool logic lives in `sophia_mcp/tools_impl.py` (importable without
FastMCP), so CI validates the surface with no `mcp` package. The portable
`SKILL.md` covers the whole program and works in Claude and Codex.

## Two-tier discipline (inherited)
Pure-stdlib deterministic CI core + opt-in reduced real run. SSA=0/N, ΔQ≤0, and
a measured capability drop are all legitimate honest results.

## Deferred frontier (OPEN in the failure ledger)
Full N≥8/K≥20 PIF headline run; a real capability cell wired into a live SSA
headline; an LLM-judge coherence channel; validated Level-3 steered council
seats; true external sealing; model×trait crossover; live GRPO; calibration.
