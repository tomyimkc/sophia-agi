# Ontology Claim Boundary (the honesty contract for the concept TBox)

*Added with the ontology-native concept-discipline work.* This is the contract every
later phase (the `ontology_edge` schema, the `ontology_edge_faithful` verifier, the
Datalog concept-edge program, the graph TBox checks, the SSIL ontology seat, the
philosophy modules, and the concept-edge RLVR reward) points back to. It exists so
that adding a concept-level TBox does **not** quietly turn Sophia's provenance ABox
into a truth oracle.

`canClaimAGI` stays **false**. Nothing in this work changes that.

## What we are building

A small, versioned, provenance-tagged **concept TBox** (`subClassOf`, `disjointWith`,
and *scoped* cross-tradition analogy edges) plus the machine-checked discipline that
governs it. The deliverable is **epistemic restraint + concept-consistency relative
to a closed world** — drawing distinctions, detecting category errors, abstaining when
a question is ill-posed, and demoting unscoped cross-tradition identity to scoped
analogy — measured on a purpose-built eval. It is **not** "an AGI philosopher" and it
is **not** a new source of truth.

## The five rules every later phase must honour

1. **The TBox is not a truth oracle.** A derivable `subClassOf` / `disjointWith`
   verdict is a statement about the *closed world of axioms we wrote down*, each of
   which is itself an untrusted, sourced claim. The engine is sound over the facts it
   is given; it says nothing about whether those facts are true. (See the
   "grounding gate, not a truth gate" finding, commit `da6570eb` / #202.)

2. **New concept edges are untrusted until independently grounded.** An edge enters
   the graph as a *candidate*. The ontology gate can **veto** a bad edge (policy /
   structural violation), but it **cannot admit** a cross-tradition identity claim as
   true, because no independent ground-truth channel exists in-repo. The only honest
   verdict for an unverifiable cross-tradition claim is `quarantine` / `abstain`.

3. **Synthetic philosophy is never evidence.** Counterfactual / synthetic concept
   material (e.g. "what would Aristotle conclude if he had read the Dao De Jing")
   is structurally barred from promotion. It may improve *reasoning* via RL, never
   *content* via SFT, and "non-promotable" is enforced as an information-flow
   invariant at the promotion choke-point, not merely a label.

4. **Cross-tradition identity defaults to abstain.** `sameAs` / `equivalentClass` /
   `exactMatch` / bare `subClassOf` across two distinct traditions is effectively
   forbidden (always-abstain). A cross-tradition relationship is admissible **only**
   as a *sourced, scoped analogy* (`scopedAnalogy` / `closeMatch`) that states the
   respect-of-comparison and a contrast — never bare identity. This follows the SKOS
   `closeMatch` (non-transitive) / `owl:sameAs` (transitive, Leibniz-substitution)
   distinction and the comparative-philosophy norm against "descriptive chauvinism".

5. **No AGI claim.** Adding structural consistency + trained abstention is not
   "conceptual reasoning unlocked" and not an "AGI unlock". `tools/lint_claims.py`
   must stay exit-0 and `canClaimAGI` stays `false` in every registry.

## Why Datalog, not OWL

The reasoner extends the existing closed-world `agent/datalog_engine.py`
(stratified negation-as-failure, least-fixed-point, finite Herbrand universe). We do
**not** add OWL / `owlready2` / `rdflib` / `pyshacl`. OWL's open-world assumption
("unknown ≠ false") directly fights an abstain-on-absence gate; the real
differentiator is **CWA vs OWA**, not "OWL is heavy". Datalog handles transitive
`subClassOf` + disjointness violations, which is exactly enough for subsumption +
disjointness + scoped analogy. If existentials are ever needed, the answer is
Datalog±, still not OWL.

## What you may and may not say

| You MAY claim | You may NOT claim |
| --- | --- |
| "abstains when its closed world is silent" | "knows when it does not know" |
| "detects declared structural contradictions in the TBox" | "detects all conceptual errors" |
| "demotes unscoped cross-tradition identity to scoped analogy or abstains" | "correctly aligns concepts across traditions" |
| "sound over the axioms it is given" | "the axioms are true" |
| "concept-consistency relative to a closed world" | "conceptual reasoning" / "AGI" |

Every axiom added to the TBox needs a source. A TBox without provenance is a second
source of unfalsifiable claims; keep it small, versioned, and provenance-tagged.
