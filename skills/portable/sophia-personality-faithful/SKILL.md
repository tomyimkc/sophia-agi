---
name: sophia-personality-faithful
description: |
  Score personality (Big Five / OCEAN) claims for faithfulness in any project:
  does a trait claim match the validated construct, or is it pop-psych overclaim,
  cross-framework merge (MBTI / Enneagram / Type A / astrology presented as Big
  Five), or a debunked myth? Use when rating openness / conscientiousness /
  extraversion / agreeableness / neuroticism claims, judging whether a description
  SUPPORTS the OCEAN facet it cites, or when the user runs /personality-faithful.
  Big Five is the substrate; MBTI is a display veneer only. Works without the
  sophia-agi repo; prefer sophia-agi MCP tools when that server is connected.
metadata:
  short-description: "Portable Big Five / OCEAN faithfulness scoring"
---

# Sophia personality faithfulness (portable)

**Wisdom before intelligence.** Construct validity before vibes. Never merge frameworks.

## When to invoke

- "Is this an accurate description of high openness / low conscientiousness?"
- Big Five / OCEAN / five-factor trait and facet questions.
- Faithful vs misstated personality claims (does the text support the facet cited?).
- Pop typing vs validated trait science (MBTI types, Type A, astrology, left/right brain).

## Hard rules

1. Big Five (OCEAN) is the measured substrate. MBTI is a display veneer — never a Big Five trait.
2. Neuroticism has no MBTI correlate; never infer it from a type code.
3. Refuse framework merges (MBTI = Big Five, astrology predicts a trait, Type A is an OCEAN dimension).
4. Label myths: "myth", "misconception", "pop psychology", 迷思.
5. Report within-system comparisons, never human-norm percentiles.
6. A VALIDATED faithfulness number needs the no-overclaim gate (>=2 independent judges, kappa>=0.40, >=3 runs), not one judge.
7. End teaching answers with a concise 中文 summary.

## Common traps (deny these)

See `references/trap-patterns.md`.

## If sophia-agi MCP is connected

Prefer `sophia_personality_faithful` (deterministic merge/abstain verdict) and
`sophia_personality_target` over guessing.

## The MBTI-Vector-Agents program (Specs A–D)

When the `sophia-agi` MCP server is connected, these read-only tools expose the
whole program. Big Five (OCEAN) is the measured substrate; MBTI is a one-way
display veneer no tool reads for a decision.

- `sophia_ocean_measure(answers)` — score a `{item_id: 1..5}` IPIP map into OCEAN.
- `sophia_personality_faithful(text, mbti, ocean)` — is a trait claim faithfully enacted, contradicted, or abstained?
- `sophia_capability_retention()` — does steering degrade reasoning? Returns `capability_drop` + `coherence` + `retains` on a deterministic arithmetic slice.
- `sophia_council_diversity()` — does a trait-diverse council deliberate better? (The honest finding: ΔQ did not replicate.)
- `sophia_pif_dryrun()` — the personality-injection / steering-superiority harness invariants.
- Resource `sophia://program/status` — what shipped and the honest nulls.

**No-overclaim posture:** the program ships its machinery and its honest results.
Two headline findings are NULL — activation steering did not beat a persona
prompt (SSA 0/2), and personality diversity did not reliably improve a council
(ΔQ did not replicate). A capability *drop* under steering is expected and
reported as such. SSA=0/N, ΔQ≤0, and a measured capability drop are legitimate.
