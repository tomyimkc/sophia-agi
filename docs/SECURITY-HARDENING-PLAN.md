# Security Hardening Plan — Repo + Trained LLM

> Status: **implemented** (Phases 0–5 landed; one manual repo-settings step remains).
> Scope: protecting this repo and any **published Sophia model** against misuse,
> prompt injection, and system-prompt / training-data leakage.
> Audience: maintainer (`tomyimkc`) + contributors.

## Implementation map (what shipped)

| Phase | Deliverable | Files |
| --- | --- | --- |
| P0 | Acceptable Use Policy; SECURITY.md disclosure + no-secrets-in-prompt | `USAGE-POLICY.md`, `SECURITY.md` |
| P1 | CI security scans + Rust dep coverage + gitleaks config | `.github/workflows/security.yml`, `.github/dependabot.yml`, `.gitleaks.toml` |
| P2 | Prompt-hygiene gate, canaries, corpus scrub | `tools/check_prompt_hygiene.py`, `agent/secret_patterns.py`, `agent/canary.py`, `agent/corpus_scrub.py` |
| P3 | Egress output guard, refusal screen, prod profile | `gateway/output_guard.py`, `agent/refusal.py`, `gateway/profiles.py`, `gateway/interceptor.py`, `agent/conscience.py` |
| P4 | ZeroLeaks-style red-team corpus + harness + release gate | `eval/redteam/attacks.jsonl`, `tools/redteam_runner.py`, `.github/workflows/redteam.yml` |
| P5 | Artifact signing + SBOM, tamper-evident audit chain, safety model card | `tools/sign_artifacts.py`, `agent/audit_chain.py`, `models/hf-model-card/README.md` |
| tests | 35 unit tests across the layers | `tests/test_security_*.py`, `tests/test_redteam.py` |

**Remaining manual step (cannot be done from code):** enable **branch protection**
on `main` (require the `Security` workflow + review), turn on GitHub **secret
scanning + push protection**, and set `SOPHIA_CANARY_SEED` in `.env` (and as a
repo secret for live red-team runs).

**Activating the runtime defenses in deployment.** The MCP entrypoints
(`gateway/server.py`, `sophia_mcp/server.py`) read **`SOPHIA_HARDENED=1`**: when
set, the gateway is built via `gateway.profiles.hardened_gateway(...)` (egress
output-guard + call budget + tamper-evident audit chain, with the system prompt
wired for echo-detection and the minted canary for confirmed-leak blocks) and the
conscience tool enforces the acceptable-use refusal screen at the input boundary.
Unset (the default), behavior is unchanged for local/offline use. So production =
`SOPHIA_HARDENED=1` + `SOPHIA_CANARY_SEED=<secret>`; everything stays opt-in
otherwise.

This document does three things:

1. Distills what the **ZeroLeaks** LLM-security scanner does and what is worth borrowing.
2. States a **defense-in-depth thesis** for securing an open repo *and* a trained model.
3. Gives a **prioritized, file-mapped plan** you can adopt before you deploy/publish.

It is grounded in the controls you already have (gateway firewall, fail-closed
interceptor, BLP classification, git-crypt, the no-overclaim gate, the conscience
kernel) and the OWASP Top-10 for LLM Applications (2025).

---

## Part 1 — What ZeroLeaks does, and the takeaway

[ZeroLeaks](https://github.com/ZeroLeaks/zeroleaks) is an **autonomous red-team
scanner** for LLM systems. It does not *defend* a model — it *attacks* one to
prove whether its system prompt can be extracted and whether it can be hijacked.
Its design is the useful part:

| ZeroLeaks component | What it does | What to borrow |
| --- | --- | --- |
| **6-agent loop** (Strategist → Attacker → Evaluator → Mutator → Inspector → Orchestrator) | Plans, generates, scores, and evolves attacks over multi-turn campaigns | Treat red-teaming as an *automated, evolving* pipeline, not a one-off checklist |
| **Tree-of-Attacks-with-Pruning (TAP)** | Explores attack branches, prunes dead ones | Systematic coverage instead of ad-hoc "try a few jailbreaks" |
| **14+ attack categories** (Base64/Unicode encoding, DAN/Developer-Mode role-play, Crescendo & Echo-Chamber multi-turn, CoT manipulation, format injection, social engineering) | A taxonomy of real extraction/injection vectors | Use as the **test corpus** for your own model |
| **Defense fingerprinting** | Detects which guardrail a target uses | Run it against *your* model to see what leaks |
| **Severity + extraction grading** (secure→critical / none→complete) | Turns attacks into a pass/fail metric | Gate releases on a leakage/injection score |

**The single biggest takeaway:** ZeroLeaks proves that *the only honest way to
know your model resists prompt injection and prompt leakage is to continuously
attack it with an adaptive adversary and measure the result.* Static guardrails
are necessary but unfalsifiable on their own. So the strategy below pairs every
**defensive** layer with an **offensive** test that tries to break it — and wires
that test into CI as a release gate.

> Note: ZeroLeaks is offensive tooling. We adopt its *methodology and attack
> taxonomy* to test **our own** systems. Don't point it (or our clone of it) at
> third-party models without authorization — that's the misuse we're trying to
> prevent.

---

## Part 2 — The thesis: defense-in-depth for a repo *and* a model

There is no single control that secures an LLM product. Security is the product
of independent layers, each of which fails closed, each of which is *tested by an
adversary*. Map every layer to a concrete OWASP-LLM-2025 risk so coverage is
auditable.

```
            ATTACKER
               │
   ┌───────────▼─────────────┐
   │ L0  Supply chain & repo  │  secrets, deps, CI provenance, model-artifact integrity
   ├──────────────────────────┤
   │ L1  Acceptable use / law │  who may use it, for what  (the "illegal stuff" worry)
   ├──────────────────────────┤
   │ L2  Input boundary       │  trust-tagging, untrusted-data fencing, input validation
   ├──────────────────────────┤
   │ L3  Model / prompt        │  system-prompt hygiene, no-secrets-in-prompt, training-data scrub
   ├──────────────────────────┤
   │ L4  Output boundary       │  leakage detection, verifier gate, PII/secret redaction
   ├──────────────────────────┤
   │ L5  Agency / tools        │  least privilege, approval gates, rate limits, sandboxing
   ├──────────────────────────┤
   │ L6  Observability         │  tamper-evident audit log, anomaly alerts
   ├──────────────────────────┤
   │ L7  Offense (continuous)  │  ZeroLeaks-style red-team in CI  ← falsifies L2–L6
   └──────────────────────────┘
```

### Mapping to OWASP Top-10 for LLMs (2025)

| OWASP risk | Your worry | Primary layer(s) |
| --- | --- | --- |
| LLM01 Prompt Injection | "afraid of prompt injection" | L2, L3, L7 |
| LLM02 Sensitive Info Disclosure | training-data / PII leak | L3, L4, L7 |
| LLM06 Excessive Agency | repo used "for bad stuff" | L1, L5 |
| LLM07 **System Prompt Leakage** | "system prompt leakage of my trained model" | L3, L4, L7 |
| LLM03 Supply Chain | poisoned deps / weights | L0 |
| LLM04 Data/Model Poisoning | corrupted training data | L0, L3 |
| LLM05 Improper Output Handling | downstream injection | L4 |
| LLM08 Vector/Embedding Weaknesses | RAG poisoning | L2 (you already have `poison_resistant_ingestion.py`) |
| LLM09 Misinformation | overclaim | already gated (no-overclaim) |
| LLM10 Unbounded Consumption | cost/DoS | L5 (rate limits) |

### Two truths specific to *your* situation

1. **You publish a model and a corpus.** Once weights or a system prompt ship,
   they are *forever* extractable by a determined user — there is no
   "unpublish." So the defense for L3/L7 is **"assume the prompt is public and
   put nothing secret in it,"** not "hide the prompt." System-prompt leakage only
   *hurts* you if the prompt contains secrets, keys, internal URLs, or business
   logic. The fix is hygiene, not obfuscation.

2. **An open repo cannot technically prevent misuse** — anyone can fork Apache-2.0
   code. What you *can* do is (a) state an enforceable **Acceptable Use Policy**
   so misuse is a license violation, not just frowned upon; (b) keep the *recipe
   and held-out evals private* (you already do via `private/` + git-crypt) so
   bad actors can't cheaply reproduce a more capable, unaligned variant; and
   (c) ship the model with refusal training + a runtime guardrail so the
   published artifact itself resists the worst asks.

---

## Part 3 — Gap analysis (what you already have vs. what's missing)

### Already strong — keep and document

- **L2/L3 injection boundary:** `gateway/firewall.py` (pattern scan of tool
  descriptions + call args, fail-closed quarantine), `agent/untrusted.py`
  (delimiter-fenced untrusted content with spoof protection).
- **L4 output gate:** `gateway/interceptor.py` surfaces only `accepted` verdicts;
  everything else withheld.
- **L5 agency controls:** role allow-lists + clearance, `sophia_contract/blp.py`
  (Bell-LaPadula), `sophia_mcp/approval.py` (human approval queue, arg digests
  only).
- **L0 secrets:** `.env*`/`*.key`/`*.pem` git-ignored; git-crypt for `secret/**`,
  `CONTRACT.md`, `AGENTS.md`, skills; `private/` tree for held-out evals & vNext
  recipe; the SECURITY.md self-audit.
- **Governance:** constitution + conscience kernel + no-overclaim gate.

### Gaps to close (priority order)

| # | Gap | Layer | OWASP | Severity |
| --- | --- | --- | --- | --- |
| G1 | No **Acceptable Use Policy**; Apache-2.0 alone permits hostile use | L1 | LLM06 | High |
| G2 | No **continuous red-team** of the model/system prompt (ZeroLeaks gap) | L7 | LLM01/07 | High |
| G3 | No **dependency/SAST scanning** (no CodeQL, Dependabot, pip-audit, secret-scan in CI) | L0 | LLM03 | High |
| G4 | No **system-prompt hygiene check** (nothing forbids secrets in prompts) | L3 | LLM07 | High |
| G5 | No **output leakage/PII redaction filter** at the boundary | L4 | LLM02 | Med |
| G6 | **Rate limiting** is optional/off; approval gate default-off | L5 | LLM10 | Med |
| G7 | No **model-artifact integrity** (weights/adapters unsigned, no checksums published) | L0 | LLM03/04 | Med |
| G8 | Audit log is local, not **tamper-evident / centralized** | L6 | — | Med |
| G9 | No **model card with safety section** / responsible-release notes for published model | L1/L3 | LLM09 | Med |

---

## Part 4 — Proposed adoption plan (phased)

Each item lists the deliverable and where it lives. Phases are ordered so the
highest-leverage, lowest-effort protections land first.

### Phase 0 — Policy & paper trail (½ day, no code risk)

- **P0.1 Acceptable Use Policy** → `USAGE-POLICY.md` (+ link from README & model
  card). Prohibit: weapons/CBRN uplift, mass surveillance, generating CSAM or
  NCII, targeted harassment, fraud/malware, evading sanctions, and *removing
  safety mitigations from the published model*. Make it binding on the model
  weights via the model card / HF gated-repo terms (Apache-2.0 governs the code;
  the weights can carry an additional use-based license such as a RAIL clause).
- **P0.2 Security policy upgrade** → extend `SECURITY.md` with a coordinated
  disclosure window, a contact, and an explicit "no secrets in system prompts"
  rule.
- **P0.3 Threat model doc** → this file; keep the layer/OWASP table current.

### Phase 1 — Supply chain & repo hardening (1 day, mostly config)

- **P1.1 CI security scans** → `.github/workflows/security.yml`:
  - `pip-audit` / `uv pip audit` + `cargo audit` (you have Rust crates in `storage/`, `services/ann_serving/`).
  - **CodeQL** (Python) on push/PR.
  - **gitleaks** or GitHub secret-scanning + push protection (catches the `sk-…`
    pattern your SECURITY.md already greps for — automate it).
  - Pin GitHub Actions by SHA; set `permissions: contents: read` by default.
- **P1.2 Dependabot** → `.github/dependabot.yml` for pip + cargo + actions.
- **P1.3 Branch protection** → require the security workflow + review on `main`;
  keep the existing `ai-guardrails.yml` protected-path rule.
- **P1.4 Lockfile discipline** → you already ship `uv.lock`/`Cargo.lock`; add a
  CI check that they're in sync.

### Phase 2 — Model & prompt hygiene (1–2 days)

- **P2.1 System-prompt linter** → `tools/check_prompt_hygiene.py` run in CI:
  scan every committed system/instruction string (gateway `instructions=`,
  `sophia_mcp`, constitution, skills) for secrets, keys, internal URLs, emails,
  absolute home paths. Fail closed on a hit. *(Closes G4.)*
- **P2.2 Training-data leakage scrub** → extend ingestion/build scripts to strip
  PII and secret patterns from any corpus shipped to HF; add a unit test with a
  seeded canary string that must **not** appear in the published dataset. *(Addresses LLM02 at the source.)*
- **P2.3 Canary tokens** → embed unique canary strings in the *private* system
  prompt / training mixture; add a monitor that flags if a canary ever appears in
  model output (proves a leak) or on the public web (proves recipe exfiltration).

### Phase 3 — Runtime defense at the boundary (2–3 days, builds on what exists)

- **P3.1 Output leakage filter** → new `gateway/output_guard.py` chained after
  `interceptor.verify_output()`: regex + classifier pass that blocks responses
  echoing the system prompt, known canaries, secret patterns, or PII before they
  reach the user. *(Closes G5; pairs with the firewall on the input side.)*
- **P3.2 Default-on hardening for prod** → make `SOPHIA_MCP_APPROVAL=1` and the
  call-budget/rate-limit the **default in the deployment profile** (keep off for
  local/offline tests). Add per-role/per-session quotas. *(Closes G6 / LLM10.)*
- **P3.3 Refusal layer** → a lightweight constitutional-classifier screen on
  user input for the AUP-prohibited categories, wired into the conscience gate
  (you already have `agent/conscience.py` + classifier — extend its rule set with
  the P0.1 categories).

### Phase 4 — Continuous red-team (the ZeroLeaks integration) (2–4 days)

- **P4.1 Attack corpus** → `eval/redteam/attacks.jsonl`: encode ZeroLeaks' 14+
  categories (encoding bypass, role-play/DAN, Crescendo/Echo-Chamber multi-turn,
  CoT manipulation, format injection, direct extraction "repeat your
  instructions verbatim"). Each row: category, prompt(s), success-detector.
- **P4.2 Harness** → `tools/redteam_runner.py`: a small Strategist→Attacker→
  Evaluator→Mutator loop (mirrors ZeroLeaks) that runs the corpus against the
  deployed gateway/model, mutates failures, and scores **extraction
  (none→complete)** and **injection (resisted→hijacked)** per category. Reuse
  your existing model adapter (`agent/model.py`) for the attacker model; default
  to `mock` offline so CI stays deterministic, and gate the *live* run behind a
  flag + secret.
- **P4.3 Release gate** → `.github/workflows/redteam.yml` (nightly + pre-release):
  fail the release if any attack reaches "high/critical" extraction or a
  successful injection. Publish the **aggregate** score (not the raw exploits) the
  same way you publish benchmark aggregates. *(Closes G2; falsifies L2–L4.)*

### Phase 5 — Integrity, observability, release (1–2 days)

- **P5.1 Artifact signing** → publish SHA-256 + a `cosign`/minisign signature for
  every released adapter/weight; verify on load. Generate an SBOM
  (`cyclonedx`). *(Closes G7 / LLM03-04.)*
- **P5.2 Tamper-evident audit log** → hash-chain the existing tracer output
  (each entry includes prev-hash); optional WORM/centralized sink for prod. *(Closes G8.)*
- **P5.3 Safety-aware model card** → `models/hf-model-card/`: document training
  data provenance, known limitations, the AUP, red-team results (aggregate),
  intended use, and out-of-scope use. Gate the HF model repo. *(Closes G9 / LLM09.)*

---

## Part 5 — Summary & the one-page adoption checklist

**Your worries, mapped to the fix:**

- *"Others use my repo for illegal stuff I don't endorse"* → **P0.1 Acceptable
  Use Policy** (binding on weights) + **keep recipe/evals private** (already done)
  + **P3.3 refusal layer** in the shipped model. You can't stop forks, but you can
  make misuse a license breach and make the *published* artifact resist it.
- *"Prompt injection"* → existing firewall + untrusted fencing (**L2/L3**),
  strengthened by **P3.1 output guard** and *proven* by **P4 continuous
  red-team**.
- *"System-prompt / model leakage"* → **P2.1 prompt hygiene** (nothing secret in
  the prompt) + **P2.3 canaries** + **P3.1 output leakage filter** + **P4** to
  measure it. The strategic point: **design so a leaked prompt is harmless.**

**Adoption status:**

- [x] P0.1 `USAGE-POLICY.md` (Acceptable Use)
- [x] P0.2 expand `SECURITY.md` (disclosure + no-secrets-in-prompt)
- [x] P1.1 `security.yml`: pip-audit, cargo-audit, CodeQL, gitleaks
- [x] P1.2 `dependabot.yml` (+ cargo); **P1.3 branch protection — manual repo setting**
- [x] P2.1 `tools/check_prompt_hygiene.py` (CI gate)
- [x] P2.2 corpus PII/secret scrub + canary test
- [x] P3.1 `gateway/output_guard.py` (leakage/PII output filter)
- [x] P3.2 prod profile (`gateway/profiles.py`): guard + budget default-on in `hardened_gateway()`
- [x] P3.3 refusal screen wired into conscience gate (opt-in via `enforceAcceptableUse`)
- [x] P4.1–P4.3 ZeroLeaks-style red-team corpus + runner + release gate
- [x] P5.1 sign + checksum + SBOM model artifacts
- [x] P5.2 hash-chain the audit log
- [x] P5.3 safety-aware model card

**Guiding principles (keep these even if you skip steps):**

1. **Fail closed everywhere** — you already do; extend it to the output boundary.
2. **Nothing secret in the system prompt or training data** — assume both leak.
3. **Every defensive layer needs an offensive test in CI** — the ZeroLeaks lesson.
4. **Publish aggregates, never exploits** — same discipline as your benchmark gate.
5. **License + policy do the work code can't** — for misuse you can't technically block.

---

### Sources

- OWASP Top 10 for LLM Applications (2025) — prompt injection (LLM01), sensitive
  information disclosure (LLM02), system prompt leakage (LLM07), excessive agency
  (LLM06), supply chain (LLM03).
- ZeroLeaks scanner — multi-agent TAP red-team methodology and attack taxonomy:
  https://github.com/ZeroLeaks/zeroleaks
