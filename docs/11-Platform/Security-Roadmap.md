# Security roadmap — provenance-aware, verifiable, fail-closed local reasoning

Deterministic, externally-checkable security for the guarded reasoning loop. The
guiding principle: **never trust the LLM** — assume it is compromised and prove
the *code* contains it. Ranked by leverage (impact ÷ effort).

| # | Change | Impact | Effort | Status |
|---|--------|:------:|:------:|--------|
| 1 | Injection / containment red-team (assume-compromised-model) | High | S | **shipped (M1)** — `eval/security/` |
| 2 | Out-of-prompt data-flow firewall (CaMeL: capabilities + taint) + HITL on the action path | High | L | **M2 v1 (engine + live airgap) + M2.2 v1 (interpreter w/ sound taint propagation + control-flow integrity) shipped** — `agent/dataflow/`; real planner LLM = M2.3 |
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

**Vulnerabilities the harness found:**
1. *Negation-evasion in `provenance_faithful`* — **FIXED**. A carve-out trigger word
   ("it is a myth, but in truth …", "contrary to the claim that he did not …") in the
   **same sentence** as a forbidden attribution made the sentence-scoped carve-out
   skip it. Fix: the carve-out is now **clause-scoped** (split on contrastive
   connectors + a leading subordinate clause, never on commas, so appositive
   matching is preserved). 4 variants now gate at 0% ASR; 0 false positives on 68
   corpus pages; locked by `test_verifiers.py`.
2. *Citation subject-match* — lexical overlap passes a wrong predicate when the
   subject matches the source → still open, **M-#5** (NLI fact-checking).

## Next milestones (not "AGI")

- **M2 v1 — shipped: enforcement engine + live airgap kill-switch.** `agent/dataflow/`
  is a deterministic capability + taint firewall: tools classified READ/WRITE/EGRESS
  (default-deny for unknown); the policy blocks a tainted value reaching a write/egress
  sink (or routes to a fail-closed human approver); `taint_of` recurses into
  containers. **Live and enforced now:** the **airgap** profile fail-closes egress at
  every model/network chokepoint — the model adapter (`agent/model.py`), `web_search`,
  the Google GenAI client, and the MCP egress tools — and the firewall is wired at the
  live `wiki_upsert` WRITE sink. The red-team scores the engine + airgap (lethal-trifecta
  **ASR 0%**, reads not over-blocked). `tests/test_dataflow.py`, `eval/security/`.
- **Honest gap (found by adversarial review, tracked as M2.2):** the engine is not yet
  fed by automatic taint *propagation* on the live autonomous path — untrusted retrieved
  content isn't auto-`Labeled` into tool args, and taint does not survive arbitrary
  Python (f-strings launder). So today the firewall guarantees the **airgap egress
  kill-switch** and blocks **explicitly-labelled** tainted writes; it does not yet
  auto-contain a laundered value on the live path.
- **M2.2 v1 — shipped: constrained interpreter (dual-LLM execution model).**
  `agent/dataflow/interpreter.py` removes the model from the data path: a trusted
  PLAN (Const/Retrieve/Extract/Concat/Call over symbolic vars) is the only control
  flow; variables hold `Labeled` values and **every step propagates taint via
  `combine`**, so a value derived from untrusted input stays untrusted however it is
  transformed (the laundering hole, closed soundly — the interpreter, not the model,
  does the transform). Three falsifiable properties, scored by the red-team and
  `tests/test_interpreter.py`: (1) taint propagates through every step; (2)
  **control-flow integrity** — an injection in retrieved content cannot cause a tool
  call not in the plan; (3) a tainted value into a write/egress sink is blocked. The
  planner/extractor are injectable (mocked in CI) — security lives in the
  deterministic interpreter.
- **Honest scope (M2.3+):** a real privileged-planner LLM that generates plans for
  open-ended tasks (the instruction set here is small — the planner's *expressiveness*
  is the limit, not the safety); a quarantined-extractor model; per-tool airgap
  classification for `agent/tools.run_tool`; an AgentDojo-style end-to-end suite
  (DoD: ASR <2%, utility within ~10 pts) once a planner is wired.
- **M3 — "Honest labels":** Biba integrity axis + bounded/logged declassification
  (#3) and corroboration-aware confidence (#4), with a label-creep dashboard and an
  ECE calibration report.
- **M4 — "Faithful & contamination-clean":** NLI `claim_supported` verifier (#5) +
  a false-attribution P/R benchmark on contamination-controlled splits + the LoRA
  leakage guard (#7).
