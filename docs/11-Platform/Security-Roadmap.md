# Security roadmap — provenance-aware, verifiable, fail-closed local reasoning

Deterministic, externally-checkable security for the guarded reasoning loop. The
guiding principle: **never trust the LLM** — assume it is compromised and prove
the *code* contains it. Ranked by leverage (impact ÷ effort).

| # | Change | Impact | Effort | Status |
|---|--------|:------:|:------:|--------|
| 1 | Injection / containment red-team (assume-compromised-model) | High | S | **shipped (M1)** — `eval/security/` |
| 2 | Out-of-prompt data-flow firewall (CaMeL: capabilities + taint) + HITL on the action path | High | L | **M2 v1 (engine + live airgap) + M2.2 v1 (interpreter w/ sound taint propagation + control-flow integrity) shipped** — `agent/dataflow/`; real planner LLM = M2.3 |
| 3 | Biba integrity axis + bounded, logged declassification (fights label creep) | High | M | planned |
| 4 | Corroboration-aware confidence (Dempster–Shafer / log-odds) | Med | S–M | **shipped (#4)** — `agent/corroboration.py`; complements `okf` min-over-chain |
| 5 | External fact-checking (NLI cross-encoder) as a `claim_supported` verifier | Med-High | M | **shipped (M-#5)** — `agent/verifiers.claim_supported` + `nli` policy; closes the citation subject-match probe (model opt-in) |
| 6 | Tool least-privilege + dual-LLM (privileged planner / quarantined extractor) | High | M | planned |
| 7 | LoRA leakage guard + contamination-controlled eval splits | Med | S–M | **shipped (#7)** — `agent/training_safety.py`, `eval/contamination.py`; guard wired into the export |

Trade-offs are real and stated: CaMeL-style enforcement (#2) costs ~7 utility
points on AgentDojo; Dempster–Shafer misbehaves under high conflict (cap it);
`no_secret_leak` is a verbatim tripwire, not a guarantee.

## #4 — shipped: corroboration-aware confidence (`agent/corroboration.py`)

The OKF graph's min-over-chain (`okf/graph.py`) correctly stops confidence
*laundering* through a weak ancestor, but it ignores *corroboration*. This adds the
missing axis: a Bayesian **log-odds pool** that raises belief when **independent**
sources agree and lowers it on dissent, after collapsing dependent sources
(same `independence_group`) so duplicates can't inflate. (Log-odds over raw
Dempster–Shafer to avoid Zadeh's high-conflict paradox.)

- `python tools/run_corroboration.py` — the **gated** invariants are structural and
  robust (across 40 seeds): confidence is monotone in independent agreeing sources,
  *rewards independent agreement unlike the baselines* (3 indep @0.7 → 0.93 vs mean
  0.7 / min 0.7), idempotent under duplicates, lowered by dissent. Inputs are
  validated (NaN/inf/out-of-range rejected); unknown method raises.
- **Honest scope:** decision accuracy *ties* a mean-of-opinions baseline; the
  selective-risk delta is favourable but **noisy at this N**, so it (and ECE) are
  reported, not gated — a single source is trivially calibrated. `min` is shown only
  for contrast (a laundering guard, not a classifier). Independence groups are a
  caller-supplied input the combiner cannot verify.

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

## #7 — shipped: LoRA leakage guard + contamination control

Anything in a fine-tuned model's weights is extractable, so the rule is: never
train on confidential data, and never let eval leak into train.

- **Leakage guard** (`agent/training_safety.py`): a deterministic pre-export filter
  drops any example that is metadata-flagged (`classification` ∈ confidential/
  secret/restricted, or `doNotTrain`), matches a PII pattern (email/SSN/card/phone/
  secret-kv), or contains a known secret value — **wired into
  `tools/export_training_jsonl.py`** so it guards the real corpus. 0 false positives
  on the 518-example public corpus; a planted confidential example is dropped. Plus
  a **canary harness** (`make_canary` / `canary_extraction_rate`) for a direct
  post-train regurgitation test the maintainer runs against the trained LoRA.
- **Contamination control** (`eval/contamination.py`): word-n-gram **shingle**
  containment catches *near-duplicate / paraphrased* train↔eval overlap that the
  existing by-ID holdout (`prepare_lora_dataset.py`) misses; a planted near-dup is
  flagged, disjoint text is clean.
- **Honest scope:** the canary harness *measures* extraction but cannot run a real
  LoRA in CI (the maintainer runs it post-train); membership-inference is not
  implemented; PII regexes are conservative (high precision, not exhaustive recall).

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
- **M2.3 v1 — shipped: planner + fail-closed plan-validator + end-to-end suite.**
  `agent/dataflow/planner.py`: a `template_planner` (deterministic, offline) and a
  `model_planner` (real P-LLM via the adapter, mockable) turn a TRUSTED request into
  a plan — never reading untrusted data. `parse_plan` is the **trust boundary**: it
  admits only known ops + manifest tools, forces `retrieve` to a READ tool, and
  **fails closed** on anything malformed, so even an adversarial planner can lose
  utility but not safety. End-to-end red-team invariant `e2e_planner_contains_injection`:
  a planner-driven run over attacker-poisoned retrieved content fires no out-of-plan
  tool. `tests/test_planner.py`.
- **M2.4 v1 — shipped: quarantined extractor + AgentDojo-style end-to-end suite.**
  `agent/dataflow/extractor.py` formalises the Q-LLM — a pure-`generate`, no-tools
  reader of untrusted content whose output the interpreter labels untrusted (even a
  fully-subverted extractor produces only data). `eval/security/agentdojo.py` +
  `tools/run_agentdojo.py` run the full planner→interpreter pipeline on benign
  requests with injected, poisoned retrieved content and report **ASR + utility**:
  first run **ASR 0% / utility 100%** across the suite (a tainted-write task is
  safely refused) — attacks are contained by construction, offline (real
  planner/extractor opt-in).
- **Honest scope (M2.5+):** the template planner + suite cover a handful of task
  shapes; a real P-LLM needs prompt engineering for broad tasks; the AgentDojo
  *official* dataset (not a hand-built analogue) for a citable cross-system number;
  per-tool airgap classification for `agent/tools.run_tool`.
- **M3 — "Honest labels":** Biba integrity axis + bounded/logged declassification
  (#3) and corroboration-aware confidence (#4), with a label-creep dashboard and an
  ECE calibration report.
- **M4 — "Faithful & contamination-clean":** NLI `claim_supported` verifier (#5) +
  a false-attribution P/R benchmark on contamination-controlled splits + the LoRA
  leakage guard (#7).
