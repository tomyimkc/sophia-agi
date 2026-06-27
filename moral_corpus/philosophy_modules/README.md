# Philosophy modules (scoped, sourced, tested, retractable)

Philosophy enters Sophia **only** as human-authored, sound checkers — never as a
verifier "synthesized" from ingested text (the synthesis engines fit
substring/numeric stumps and provably cannot represent predication/essence/genus;
their correct behaviour on such a task is to abstain).

Each module is:

- **scoped** — it states the tradition and the respect-of-comparison it covers;
- **source-grounded** — every claim cites a source (no axiom without provenance);
- **tested** — it ships a `*.jsonl` eval set graded by its checker;
- **retractable** — its edges live in the graph and can be `retract()`-ed;
- **gradient-capped** — per-module `maxVerdict` (see `agent/philosophy_modules.py`).

## The formalizability gradient (why Aristotle first)

| module | tradition | maxVerdict | why |
| --- | --- | --- | --- |
| `aristotle_term_logic` | aristotelian | `accepted` | finite/decidable assertoric syllogistic → machine-checkable |
| `kant_universal_law` | kantian | `candidate` | encodable in dyadic deontic logic, but maxim formulation ≠ deduction |
| `virtue_care_contractualist` | virtue_ethics | `quarantine` | defeasible; route through argumentation, not crisp subClassOf |
| `wittgenstein_family_resemblance` | wittgensteinian | `polythetic` | no necessary-and-sufficient core (PI §65–71) |
| `nagarjuna_catuskoti` | madhyamaka | `abstain` | needs paraconsistent logic; a dialetheia would explode a classical ABox |

Only `aristotle_term_logic` ships a machine checker today. The others are
registered with their cap so a later human-authored checker can be added without
ever letting a downstream phase promote a claim past its tradition's ceiling.
