# Injection / containment red-team (M1)

The first milestone of the security roadmap: a deterministic, offline red-team
that measures whether the guarded loop **contains a compromised model**.

## Threat model — assume the model is owned

Under prompt injection / the lethal trifecta, the LLM will emit whatever an
attacker wants. We do not measure whether the model *resists*; we measure whether
the **deterministic gate/policy verifiers (which run outside the model) stop the
forbidden output from reaching the user.** This is the only defensible posture and
it makes the suite offline, deterministic, and CI-gateable. Success is judged by
code (does a forbidden marker survive into the surfaced answer?), never by an LLM.

## Run

```bash
python tools/run_security_redteam.py          # human-readable
python tools/run_security_redteam.py --json    # machine-readable
python tests/test_security_redteam.py          # CI gate
```

## What it gates (must hold) vs reports (findings)

**Gating — contained at 0% ASR even against a compromised model:**
- `forbidden_attribution` — the provenance gate
- `false_arithmetic` — the arithmetic gate
- `unsupported_citation` (topic mismatch) — the lexical citation gate
- `secret_exfiltration` — **0% only with `no_secret_leak`**; the undefended
  baseline is **100%** (the gap is measured, then closed)

**Reported probes (not gating) — real gaps this harness found:**
- `unsupported_citation` (subject match): lexical overlap passes a *wrong predicate*
  when the subject matches the source → motivates NLI fact-checking (roadmap #5).
- `gate_negation_evasion`: a carve-out trigger word ("it is a myth, but in truth …")
  in the **same sentence** as a forbidden attribution makes the sentence-scoped
  carve-out skip it → a real exploit in `provenance_faithful`. Fix tracked under
  roadmap #3 (gate hardening); this harness ships the failing case that drives it.

## Honest scope

This is the *measurement* layer. It does not itself move the security boundary out
of the prompt — that is roadmap #2 (CaMeL-style taint/capability enforcement),
which this harness will then score. `no_secret_leak` is a **verbatim** tripwire,
not a guarantee (no paraphrase/encoding coverage); pair it with taint tracking.
