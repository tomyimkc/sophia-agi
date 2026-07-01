---
name: sophia-security-audit
description: >
  Run BEFORE a release / tag / publish, or when the user says "security audit", "audit the repo",
  "before release", "secure the repo", "pre-release check", or "/sophia-security-audit". Bundles the
  repo's three local security gates into one deterministic, offline pre-flight: the no-overclaim
  claims linter, the ReDoS/robustness verifier regression, and a heuristic secret-shape scan over the
  source tree. Use it to catch an overclaim, a re-introduced catastrophic-backtracking regex, or a
  leaked-key-shaped string LOCALLY before CI does. It COMPLEMENTS — does not replace — the
  authoritative CI scans (pip-audit, CodeQL, gitleaks). A clean audit is a filter, NOT a guarantee
  of security.
metadata:
  short-description: "One local pre-release security pre-flight: no-overclaim + ReDoS-robustness + heuristic secret scan (complements CI, not a guarantee)"
---

# sophia-security-audit — local pre-release security pre-flight (this repo)

One command to run the repo's security gates together before you cut a release. Inspired by an
"AgentShield" security-auditing agent: bundle the cheap, deterministic checks so a regression is
caught locally instead of in CI (or, worse, after publish).

```bash
python tools/security_audit.py --check    # JSON result, exit 0 iff ok
python tools/security_audit.py            # PASS/FAIL offline invariants
```

## When to run

- Before a release, tag, or publish; before pushing copy that touches public-facing prose.
- After changing a discipline verifier's regexes (re-confirm ReDoS-safety).
- When the user says "security audit", "before release", "secure the repo", or `/sophia-security-audit`.

## The three checks

1. **no_overclaim** — runs `tools/lint_claims.py` (`main([])`). Public copy must not exceed the
   failure ledger (no AGI/safety overclaims, no AI-as-author attribution). Pass iff exit 0.
2. **redos_robustness** — runs `tests/test_verifier_robustness.py` (`main()`). The discipline
   verifiers run over untrusted model output, so their regexes must stay bounded (no catastrophic
   backtracking). Pass iff returns 0.
3. **secret_scan** — a heuristic regex scan over a bounded set of tracked non-binary files
   (`agent/`, `tools/`, `training/`, `eval/`, `docs/`, `.claude/`; files > 1 MB and binaries skipped)
   for high-entropy / secret-shaped patterns (AWS access-key ids, PRIVATE KEY headers, `sk-`/`ghp_`/
   `hf_`/`xai-` provider tokens). Honors the `.gitleaks.toml` allowlist conventions (docs, tests,
   `.example`, placeholder values). Pass iff no match.

`run_audit()` returns `{"checks": {...}, "ok": bool}`. `ok` is True iff no check **failed** — a check
that could not RUN (ImportError, missing module) is reported as `skipped` with a reason and does NOT
flip `ok` to False. We fail-closed only on a real positive finding, never on inability to run.

## Honest boundary

This is a **local pre-flight**, not the authoritative scan. The real coverage lives in
`.github/workflows/security.yml`: **pip-audit** (dependency CVEs), **CodeQL** (static analysis), and
**gitleaks** (full secret-scan ruleset + history). The local secret scan is a cheap heuristic subset
of gitleaks — it can miss secrets gitleaks catches.

A clean audit means "no positive finding from these three cheap checks." It is **not** a guarantee of
security and makes no such claim. Do not describe the repo as "guaranteed secure" or claim this tool
"makes the AI safe" — `tools/lint_claims.py` (check 1) would reject that copy anyway.

## Do not

- Do not treat a clean local audit as a substitute for the CI security workflow.
- Do not weaken a check to make the audit pass — a real finding is the signal, fix the cause.
- Do not write a real credential anywhere to "test" the secret scanner — pass synthetic strings to
  `scan_text(text)` instead (see `tests/test_security_audit.py`).
