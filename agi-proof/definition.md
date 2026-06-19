# Operational AGI Definition For Sophia

There is no single universal AGI definition. This project uses a conservative
operational definition drawn from recurring themes in major AGI discussions.

## Common Definition Pattern

An AGI-capable system should demonstrate:

- broad competence across domains;
- transfer to unfamiliar tasks;
- efficient skill acquisition;
- tool use and planning;
- long-horizon autonomy;
- improvement from feedback without corrupting prior knowledge;
- performance above strong baselines;
- reproducibility by independent reviewers.

## Source Families

| Source family | Emphasis | How Sophia tests it |
|---|---|---|
| OpenAI Charter | Highly autonomous systems that outperform humans at economically valuable work | Long-horizon tasks, repo tasks, and baseline comparisons |
| Google DeepMind Levels of AGI | Generality, performance, and autonomy | Proof ladder with explicit levels |
| Legg and Hutter | Goal achievement across a wide range of environments | Cross-domain hidden tasks and transfer tests |
| Chollet / ARC | Skill-acquisition efficiency on novel tasks | ARC-style and distribution-shift protocols |

## Sophia-Specific Definition

For this repository, Sophia reaches a credible AGI-candidate threshold only if it
can repeatedly solve novel, hidden, cross-domain tasks better than raw model
baselines, while preserving source discipline, using tools safely, learning from
new evidence append-only, and producing logs that independent reviewers can
reproduce.

This definition intentionally excludes consciousness, sentience, or human-like
inner experience. The proof target is capability and reproducibility.
