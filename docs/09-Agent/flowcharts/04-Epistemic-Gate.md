# 4 · Epistemic Gate & Verification

**Role in the master flow.** The heart of the repo: after generation, the answer must pass an
**epistemic gate** before it becomes a result. The gate composes many verifiers (claim-type,
citation, code, formal/Lean, conformal) and can send a failing answer back for bounded repair.
Ablation flag `use_gate`. This is the largest subsystem — 49 `agent/` modules.

```mermaid
flowchart TD
    ANS([Candidate answer]) --> GATE["Epistemic gate<br/>agent/gate.py"]
    GATE --> VER["Verifier suite<br/>agent/verifiers.py"]
    GATE --> CR["Claim-type routing<br/>agent/claim_router.py"]
    GATE --> SC["Sector council check<br/>agent/sector_council.py"]

    CR --> CV["Citation existence<br/>agent/citation_existence_verifier.py"]
    CR --> AV["Attribution swap<br/>agent/attribution_swap_verifier.py"]
    CR --> CODEV["Code verifier<br/>agent/code_verifier.py · execution_verifiers.py"]
    CR --> FV["Formal / Lean / SMT<br/>agent/formal_verifier.py · smt_verifier.py"]
    CR --> CONF["Conformal gate<br/>agent/conformal_gate.py"]

    VER --> VERDICT{"Gate verdict<br/>accepted · rejected · abstain"}
    CV --> VERDICT
    AV --> VERDICT
    CODEV --> VERDICT
    FV --> VERDICT
    CONF --> VERDICT

    VERDICT -->|rejected · repair on| FIX["Bounded repair<br/>agent/correction_loop.py<br/>flag: allow_repair"]
    FIX -.->|re-generate| ANS
    VERDICT -->|abstain| ABS["Abstain<br/>→ calibration / abstention"]
    VERDICT -->|accepted| RV["Rubric review<br/>agent/rubric_review.py"]
    RV --> OUT([Per-case result])

    VERDICT -.->|verdict as signal| REWARD["gate_reward · multiaxis_reward<br/>→ training signals"]
```

**Modules (subset of 49):** `gate.py`, `gate_reward.py`, `gate_feedback.py`, `verifiers.py`,
`claim_router.py`, `conformal_gate.py`, `constitutional_gate.py`, `consequence_gate.py`,
`honeypot_gate.py`, `fact_check_gate.py`, `citation_existence_verifier.py`,
`attribution_swap_verifier.py`, `code_verifier.py`, `execution_verifiers.py`, `formal_verifier.py`,
`smt_verifier.py`, `deontic_verifier.py`, `layered_verifier.py`, `rubric_review.py`,
`correction_loop.py`.

**Thesis note.** The gate verdict is three-way (`accepted` / `rejected` / `abstain`), and the same
verdict is the seed for the training-signal path (`gate_reward.reward()`). That dual use — verdict as
inference-time gate **and** as a reward — is the bridge from "measured epistemics" to "learned
epistemics" (the untapped-training W1/W2 directions). `gate_reward` deliberately drops the question,
so as a raw reward it cannot tell abstain-on-answerable from abstain-on-trap — a known reward-hacking
surface worth naming.