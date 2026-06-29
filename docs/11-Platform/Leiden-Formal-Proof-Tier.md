# Formal-proof verifier tier (Leiden-aligned)

**Status: proposed — not implemented.** This is a design note, not a shipped capability. It is
tracked as an open gap in [`agi-proof/leiden-compliance.json`](../../agi-proof/leiden-compliance.json)
(`open_gaps: formal_proof_verifier_tier`) and in [LEIDEN-ALIGNMENT.md](../LEIDEN-ALIGNMENT.md).

## Why

The [Leiden Declaration](https://leidendeclaration.ai/) puts **proof and certainty** first
(value 1): a mathematical result is trusted because its argument is transparent and
independently verifiable. Sophia's gate today verifies *attributions and provenance*; it does
not machine-check *formal proofs*. For any formal or mathematical claim, the highest available
form of certainty is a proof checked by a proof assistant (e.g. Lean, Coq, Isabelle).

## Proposed shape

Add a verifier tier alongside the existing ones:

- A claim tagged as *formal* must carry a machine-checked proof artifact (e.g. a Lean file the
  toolchain compiles) **or** be labelled `informal-unverified` and gated accordingly.
- Extend `tools/claim_gate.py` with a `formal_proof_verified` pillar: present + checks ⇒ may
  back a formal claim; absent ⇒ the claim is capped at `informal-unverified` and may not be
  published as certain.
- Keep the fail-closed posture: no proof artifact ⇒ abstain from the certainty claim, never
  fabricate one.

## Risks / honesty notes

- Proof-assistant toolchains are heavy and environment-specific; the verifier must degrade to a
  clear `unverified` label when the toolchain is unavailable, rather than silently passing.
- This must not be advertised as general theorem-proving capability. It is a *gate*: it checks
  proofs that are supplied; it does not claim to discover them.

## Acceptance criteria (before this leaves `proposed`)

1. A formal-claim schema + the `formal_proof_verified` pillar wired into `tools/claim_gate.py`.
2. A worked example: one formal claim with a checked artifact passes; the same claim without it
   is correctly capped at `informal-unverified`.
3. Deterministic offline tests for both the pass and the fail/abstain path.
4. A failure-ledger entry opened to track real coverage (which claim classes are in scope).

Until all four hold, the proof-and-certainty value notes this as open work in the receipt.
