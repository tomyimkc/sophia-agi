# Psychology domain (active)

## Source discipline (philosophy methodology)

Same provenance rules as philosophy attributions — applied to concepts instead of texts:

- Who **coined** this term? (`attributedAuthor`, `doNotAttributeTo`)
- Which **subfield** owns it? (`cognitive`, `clinical`, `pop_myth`)
- What **pop myths** must be denied?

## Data center

- **Records:** `data/psychology_concepts.json`
- **Hub training:** `training/examples/018-psychology-source-discipline-hub.json`
- **Benchmark:** `tests/benchmark-psychology.json` (4 cases)

## Example traps

| Trap | Record | Deny / tag |
|------|--------|------------|
| Freud → cognitive dissonance | `cognitive_dissonance` | Deny `sigmund_freud`; affirm Festinger |
| Left-brain personality | `left_brain_right_brain_pop` | `pop_myth` |
| Chemical imbalance slogan | `chemical_imbalance_slogan` | Nuance / biopsychosocial |
| Universal Stockholm | `stockholm_syndrome` | `clinical` + deny universal; tag `pop_myth` misuse |

## Status

**Active** — scope locked in [Expansion-Questionnaire.md](Expansion-Questionnaire.md). See [Source-Discipline-Methodology.md](Source-Discipline-Methodology.md).