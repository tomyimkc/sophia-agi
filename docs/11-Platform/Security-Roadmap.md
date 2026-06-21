# Security roadmap — provenance-aware, verifiable, fail-closed local reasoning

Deterministic, externally-checkable security for the guarded reasoning loop. The
guiding principle: **never trust the LLM** — assume it is compromised and prove
the *code* contains it. Ranked by leverage (impact ÷ effort).

| # | Change | Impact | Effort | Status |
|---|--------|:------:|:------:|--------|
| 1 | Injection / containment red-team (assume-compromised-model) | High | S | **shipped (M1)** — `eval/security/` |
| 2 | Out-of-prompt data-flow firewall (CaMeL: capabilities + taint) + HITL on the action path | High | L | planned (M-next) |
| 3 | Biba integrity axis + bounded, logged declassification (fights label creep) | High | M | planned |
| 4 | Corroboration-aware confidence (Dempster–Shafer / log-odds) | Med | S–M | planned |
| 5 | External fact-checking (NLI cross-encoder) as a `claim_supported` verifier | Med-High | M | planned |
| 6 | Tool least-privilege + dual-LLM (privileged planner / quarantined extractor) | High | M | planned |
| 7 | LoRA leakage guard + contamination-controlled eval splits | Med | S–M | planned |

Trade-offs are real and stated: CaMeL-style enforcement (#2) costs ~7 utility
points on AgentDojo; Dempster–Shafer misbehaves under high conflict (cap it);
`no_secret_leak` is a verbatim tripwire, not a guarantee.

## M1 — shipped: injection / containment red-team

`eval/security/` measures whether the deterministic gate contains a compromised
model. Run: `python tools/run_security_redteam.py`. See
[eval/security/README.md](../../eval/security/README.md).

**Contained at 0% ASR (gating, CI-enforced):** forbidden attribution, false
arithmetic, topic-mismatch citation. **Secret exfiltration: 100% baseline → 0%**
with the new deterministic `no_secret_leak` tripwire (`agent/verifiers.py`) and the
`confidentiality` policy (`agent/policies.py`).

**Vulnerabilities the harness found (reported, drive later milestones):**
1. *Citation subject-match* — lexical overlap passes a wrong predicate when the
   subject matches the source → **M-#5** (NLI fact-checking).
2. *Negation-evasion in `provenance_faithful`* — a carve-out trigger word ("it is
   a myth, but in truth …", "contrary to the claim that he did not …") in the
   **same sentence** as a forbidden attribution makes the sentence-scoped carve-out
   skip it → **M-#3** (gate hardening). The failing red-team case is committed and
   will drive the fix.

## Next milestones (not "AGI")

- **M2 — "Prove it, then move it":** harden the carve-out (close the negation-
  evasion the red-team found), then the CaMeL-style data-flow firewall (#2) with
  HITL on shell/git/file writes. DoD: red-team ASR <2% incl. the negation probes;
  utility within ~10 pts; no tainted→write/egress path reachable without an
  approved capability.
- **M3 — "Honest labels":** Biba integrity axis + bounded/logged declassification
  (#3) and corroboration-aware confidence (#4), with a label-creep dashboard and an
  ECE calibration report.
- **M4 — "Faithful & contamination-clean":** NLI `claim_supported` verifier (#5) +
  a false-attribution P/R benchmark on contamination-controlled splits + the LoRA
  leakage guard (#7).
