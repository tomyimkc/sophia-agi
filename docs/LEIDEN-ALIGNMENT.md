# Sophia × the Leiden Declaration on AI and Mathematics

_How this repository aligns with the [Leiden Declaration on Artificial Intelligence and
Mathematics](https://leidendeclaration.ai/) (2026)._

The Leiden Declaration is a set of recommendations asking researchers to keep integrity
at the centre of AI-assisted work. Sophia's design goal is to turn that kind of discipline
from prose people promise into machinery a CI gate checks. This document is the crosswalk:
for each Leiden value and recommendation, the concrete Sophia mechanism that serves it, and
— honestly — the gaps that remain.

Two artifacts back this page and are drift-checked in CI:

- [`agi-proof/leiden-compliance.json`](../agi-proof/leiden-compliance.json) — machine-readable
  receipt, regenerate with `python tools/leiden_receipt.py`.
- [`docs/TOOL-DISCLOSURE.md`](TOOL-DISCLOSURE.md) — generated tool & resource disclosure,
  regenerate with `python tools/build_tool_disclosure.py`.

> **Scope.** This is an alignment statement, not a capability claim. `canClaimAGI: false`
> everywhere. Where a value is only partly served, it is marked **partial** with the open
> work named — consistent with the Declaration's own call to state when something is not
> yet done.

---

## The five core values

### 1. Proof & certainty — **operationalized**

> _Leiden: results should carry the highest justified degree of certainty._

- Fail-closed verifier gate: `claim → verify against sources → accept · abstain · block`.
- The Instrumented Evaluation Contract (`agi-proof/measurement-thesis.md`): confidence
  intervals required, pre-registered minimum detectable effect, anytime-valid confidence
  sequences for iterated metrics, before any number backs a claim.
- Enforced by `tools/claim_gate.py` / `tools/eval_stats.py`; verifier tests in
  `tests/test_verifiers.py`, `tests/test_proof_search.py`.

**Open work:** a formal-proof verifier tier (machine-checked proofs for formal/mathematical
claims) is proposed but not implemented — see
[`docs/11-Platform/Leiden-Formal-Proof-Tier.md`](11-Platform/Leiden-Formal-Proof-Tier.md).

### 2. Attribution & responsibility — **operationalized**

> _Leiden: results are attributable to specific humans who take responsibility for their
> correctness; credit is not given to automated systems._

- `canClaimAGI: false` carried through the adapter registry and architecture-bets registries.
- A generated **Tool & Computational Resource Disclosure** ([`TOOL-DISCLOSURE.md`](TOOL-DISCLOSURE.md))
  — the Declaration's most concrete individual-researcher ask, here auto-generated and
  drift-checked rather than hand-maintained.
- A **no-AI-authorship** rule in `tools/lint_claims.py`: public copy may not state that an AI
  system authored, invented, or discovered a result.
- The gate **abstains** when provenance cannot be established, rather than fabricating one.

### 3. Transparency & independent verifiability — **operationalized**

> _Leiden: arguments are transparent and subject to independent verification._

- A public **failure ledger** (`agi-proof/failure-ledger.md`) of what is **not** proven, with
  claim-impact and required next step; structurally validated by
  `tools/validate_failure_ledger.py`.
- Results are published as aggregates (rates + confidence intervals) from a single source of
  truth; hand-edits are blocked in CI.
- A clean-clone replication path (`agi-proof/REPLICATION.md`).

### 4. Shared standards of evaluation — **partial**

> _Leiden: work is evaluated by collectively established criteria._

- The Instrumented Evaluation Contract is published for reuse, and the **PROTECTED** per-domain
  standards (religion, history) in `data/traditions.json` model how a field can publish explicit
  shared criteria.

**Open work:** several benchmarks are self-authored. Driving the external-replication path and
adding more non-self-authored evaluation reduces this gap — see
[`agi-proof/REPLICATION.md`](../agi-proof/REPLICATION.md).

### 5. Autonomous direction & non-proprietary tooling — **partial**

> _Leiden: humans shape research directions; the field favours non-proprietary, publicly
> governed tools._

- Local-first, Apache-2.0, runs on owned hardware; metered compute is the exception, not the
  default (`VISION.md`, `SECURITY.md`).

**Open work:** validation still leans on proprietary judge inference. An open-weights judge
family would let the ≥2-judge-family rule run without proprietary services — see
[`docs/11-Platform/Leiden-Open-Judge-Family.md`](11-Platform/Leiden-Open-Judge-Family.md).

---

## Recommendation-by-recommendation crosswalk

| Leiden recommendation | Sophia mechanism | Status |
|---|---|---|
| Disclose tool & computational resource use | generated [`TOOL-DISCLOSURE.md`](TOOL-DISCLOSURE.md) from registry + `.mcp.json` + declared config | operationalized |
| Responsibility stays with human authors | `canClaimAGI:false`; no-AI-authorship lint | operationalized |
| Do not credit automated systems as authors | no-AI-authorship lint in `tools/lint_claims.py` | operationalized |
| State when attribution is impossible | gate abstains; provenance accuracy is measured | operationalized |
| Adhere to FAIR / UNESCO open science | FAIR self-assessment in `pretraining/data_passport/passport.py` | partial |
| Provide human descriptions of automated arguments | measurement-thesis + method notes | operationalized |
| Publish in scrutinised venues / peer review | external-replication path; aggregates-only publishing | partial |
| Prefer non-proprietary tools | local-first, Apache-2.0; open-judge family proposed | partial |
| Keep abreast of capabilities; informed recommendations | failure ledger + architecture-bets registries | operationalized |
| Assess consequences; withdraw from harmful use | `SECURITY.md` acceptable-use policy; conscience kernel | operationalized |

---

## Open gaps (tracked)

These are declared, not hidden. Each is mirrored in `agi-proof/leiden-compliance.json` under
`open_gaps`:

1. **Open-model judge family** (`proposed`) — remove the proprietary dependence in validation.
2. **Formal-proof verifier tier** (`proposed`) — machine-checked proofs for formal claims.
3. **External replication** (`open`) — reduce reliance on self-authored benchmarks.

---

_Maintained by hand for the prose; the receipts it links are generated. If the prose here ever
exceeds `agi-proof/failure-ledger.md`, the claims linter (`python tools/lint_claims.py`) fails
the build — this file is in its scan list._
