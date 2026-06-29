# Prompt authoring as a verifier-gated skill

> Status: **shipped machinery + design.** The PR template (Layer 1), the `prompt-author` skill
> (Layer 2), and the `prompt_quality_verifier` metric (Layer 3) are shipped and CI-tested. The
> live forge-promoted prompt skill (Layer-C through `tools/sophia_skill_forge.py`) is OPEN.
> Makes no capability or AGI claim — it scores prose for checkability.

## Thesis

A structured scaffold measurably improves an LLM's output by pre-committing it to a shape —
meta-prompting lifts accuracy ~62% → ~79.6% (Suzgun & Kalai 2024), and a "logical certificate"
format pushes code-review accuracy to ~93% (Meta semi-formal reasoning). The next step — distilling
a scaffold into a reusable, optimised prompt skill — is DSPy's "prompts as programs": define a
signature + a *metric*, then compile the prompt against data, so prompt quality becomes a function
of a metric rather than intuition.

Sophia's contribution is the missing **metric**. A DSPy-style optimiser or a self-improving prompt
loop has the same failure mode as any unguarded optimiser — it overfits / reward-hacks the prompt
metric — *unless the metric is machine-checked*. Sophia already gates facts with `claim → verify →
accept/abstain`; this applies the identical discipline to prompts. The result is **verifier-gated
prompt distillation**: only prompts that clear a machine bar on held-out tasks get promoted. That is
the honest, falsifiable version of "a system that writes the best prompt possible" — not "self-
improving prompt intelligence" (which would trip `lint_claims`).

## Three layers

| Layer | Artifact | Role | State |
|---|---|---|---|
| 1 — static scaffold | `.github/pull_request_template.md` | forces CI/seeds/label + gates at review time | shipped |
| 2 — distilled skill | `.claude/skills/prompt-author/SKILL.md` | auto-triggering prompt/PR/handoff discipline | shipped |
| 3 — the metric | `agent/prompt_quality_verifier.py` | machine score: success-criterion · scope · abstention · no-overclaim · grounding | shipped, 6 invariants + 7 tests |
| C — gated skill | via `tools/sophia_skill_forge.py` (`synthesize_gate`) | a generator whose prompts must clear Layer 3 on held-out tasks before promotion | **OPEN** |

## The metric (`agent/prompt_quality_verifier.py`)

`score_prompt(text)` returns `{passed, score, dimensions, reasons, overclaims}`. `passed` requires
the four load-bearing dimensions (success_criterion, bounded_scope, abstention_path, no_overclaim);
grounding is scored and reported but not fail-closed, so a short well-formed prompt is not punished
for brevity. The **no_overclaim** check imports `FORBIDDEN` and `ALLOW_MARKER` directly from
`tools/lint_claims.py`, so the prompt verifier and the claims linter can never drift apart — one
source of truth for what counts as an overclaim, whether in marketing prose or in a prompt.

`prompt_quality_ok(text) -> bool` is the predicate; `prompt_quality()` is a verifier-style callable
(`v(text, record, ctx) -> {passed, reasons, detail}`) matching the `agent/*verifier*` convention, so
it drops straight into the gate and the forge.

## Layer-C: the forge-gated prompt skill (OPEN design)

`tools/sophia_skill_forge.py` already does `task → verifier → skill → eval suite → registration`,
with LLM-proposed predicates AST-sandboxed and required to clear held-out validation before
promotion (`gateway.skill_flywheel.synthesize_gate`). To make a self-improving prompt skill:

1. **Held-out prompt-task set** — a pack of `(situation → ideal prompt traits)` cases, decontaminated.
2. **Gate predicate = `prompt_quality_ok`** (this module), optionally tightened per task family.
3. **Generator** proposes candidate prompts; only those clearing the predicate on the *held-out*
   split are promoted — exactly the `gen_verifier_dpo` / `sophia_autoresearch` pattern: the verifier
   is the labeller and the firewall, never a soft LLM judge.
4. **Honest trail** — rejected candidates log to the failure ledger; a promoted prompt-skill stays
   CANDIDATE until it clears the project's κ ≥ 0.40 / CI gate on a third-party prompt-task pack.

This composes with `sophia_autoresearch.py`: a prompt skill is just another editable surface the
gated loop can optimise — under the same firewall (it may edit the generator, never the verifier).

## Honest limits

- A passing prompt is **well-formed, not correct** — the metric scores checkability, not truth.
- The dimension detectors are lexical/regex; they can be gamed by keyword-stuffing. The Layer-C
  held-out eval is what defends against that, the same way the held-out split defends the verifier
  forge. Until that runs, treat Layer 3 as a *lint*, not a *proof*.
- Recursive prompt-improvement looks like RSI; it is safe **only** because the gate is machine-
  checked. Claim "verifier-gated prompt distillation with falsifiable metrics," never "self-
  improving prompt intelligence."

## Sources

- Meta-Prompting (Suzgun & Kalai 2024, arXiv 2401.12954)
- Meta semi-formal reasoning / structured prompting (2025)
- DSPy / "treat prompts as code" (arXiv 2507.03620)
- GenAI scaffolding reduces hallucination (arXiv 2508.05929)
- This repo: `docs/09-Agent/Skill-RSI-Thesis-and-Review.md`, `tools/sophia_skill_forge.py`
