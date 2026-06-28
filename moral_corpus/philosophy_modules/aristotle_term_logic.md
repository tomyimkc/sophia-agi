# Module: Aristotelian term logic (assertoric syllogistic)

- id: `aristotle_term_logic`
- tradition: `aristotelian`
- maxVerdict: `accepted` (finite/decidable → machine-checkable)
- checker: `agent/philosophy_modules.py::aristotelian_syllogism_valid`
- eval: `moral_corpus/philosophy_modules/aristotle_term_logic.v1.jsonl`

## Source

Aristotle, *Prior Analytics* I.1–7. Categorical (assertoric) syllogistic over the
four figures and the four statement forms A/E/I/O. The traditional logic admits 24
valid moods (19 "principal" + 5 subaltern/weakened that assume existential import).
A modern proof-assistant encoding exists (arXiv:1904.01422), confirming the system
is finite and decidable.

## Scope

This checker decides validity of a categorical syllogism presented in the
**structured** form `(figure ∈ {1,2,3,4}, mood ∈ {A,E,I,O}^3)`. It is sound and
total over that form. It deliberately does **not** parse natural language into a
figure/mood — that extraction is the brittle step (Logic-LM-style self-refinement
+ type checking belongs there), so eval items carry the structured form alongside
their prose, and a mis-extraction is a separate, visible failure.

## Statement forms

| form | reading |
| --- | --- |
| A | All S are P |
| E | No S are P |
| I | Some S are P |
| O | Some S are not P |

## Figures (middle term M position)

| figure | major | minor | conclusion |
| --- | --- | --- | --- |
| 1 | M–P | S–M | S–P |
| 2 | P–M | S–M | S–P |
| 3 | M–P | M–S | S–P |
| 4 | P–M | M–S | S–P |

## What "accepted" means here

`accepted` means the syllogistic *form* is valid — it says nothing about whether
the premises are true. Soundness of the argument still depends on the (sourced,
retractable) truth of its premises. See the claim-boundary doc.
