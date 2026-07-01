<!-- SPDX-License-Identifier: Apache-2.0 -->
# sophia-lex — a deterministic single-DFA scanner for the measurement gate

> Status: **shipped (optional accelerator)**, parity-verified, opt-in. The
> pure-Python gate tools remain the reference oracle and the CI default.
> `canClaimAGI` unaffected — this is instrument engineering, not a capability claim.

## Motivation

This repo's flagship thesis (the Instrumented Evaluation Contract) is that in a
small-corpus pipeline *the measurement instrument*, not the model, is the
dominant source of wrong conclusions — so evaluation is engineered as a
first-class, fail-closed instrument. Two of those instruments are text scanners:

- `tools/lint_claims.py` — the **no-overclaim gate**: ~18 forbidden-phrase
  regexes over public-facing prose.
- `tools/assert_decontam.py` — **train/eval decontamination**: exact overlap +
  a content-shingle (word k-shingle Jaccard) near-duplicate scan.

Both are pure-Python regex / `str.split`. Adopting a `logos`-generated DFA
(see the upstream project, [maciejhirsz/logos](https://github.com/maciejhirsz/logos))
strengthens the *instrument* in three ways the repo already values:

1. **Determinism** — a single DFA, no backtracking, guaranteed linear time. A
   regex list can backtrack and its `\b` semantics are implicit; a token DFA
   makes word boundaries exact by construction.
2. **Auditability** — the forbidden-phrase and claim-grammar tables are one
   diffable declaration of exactly what is recognized.
3. **An on-thesis coverage fix** — `assert_decontam.py` caps the eval surface at
   `--max-eval-shingle 4000` "for perf". That silent coverage bound is precisely
   the failure mode the measurement thesis condemns ("a silent cap reads as
   covered-everything when it didn't"). The accelerated path is fast enough to
   drop the cap and scan the **full** eval surface.

## What was built

A standalone Rust crate, `tools/sophia-lex/` (its own workspace — deliberately
**not** a member of the storage virtual workspace, so the gates stay
pure-Python by default). No pyo3: the Python tools shell out to a CLI binary and
parse line-oriented JSON, so there is **no compiled dependency in the gate**.

| Component | Role |
|---|---|
| `overclaim` | `logos` DFA tokenizes each (lowercased) line into Unicode word tokens; forbidden phrases are matched over the token stream with explicit separator rules (whitespace / hyphen / apostrophe) that reproduce the regexes' literal-space + `\b` semantics. Emits `{file,line,col,why}` with the SAME `why` strings as `lint_claims.py`. |
| `normalize` + `shingle` | Byte-exact mirror of `dataset_guard.normalize` and the `_shingles` / `_jaccard` near-dup math. |
| `scl` | **Sophia Claim Language** — a tiny deterministic surface syntax for provenance assertions, lexed by `logos` + a hand parser, compiling to a canonical claim triple (see below). |

### Wiring (opt-in, fail-safe)

- `tools/_lex_accel.py` — a thin bridge that locates/optionally builds the binary
  and exposes `overclaim_scan(...)` and `decontam_near(...)`. Any failure raises
  `LexUnavailable`; callers fall back to Python.
- `tools/lint_claims.py --accel` — swaps only the prose scan for the Rust scanner
  (registry/recipe/architecture checks always run in Python). Default off.
- `tools/assert_decontam.py --accel` — uses the Rust near-dup scan with **full**
  eval coverage (no `--max-eval-shingle` cap). Default off.

### Parity gate

`tools/test_lex_parity.py` asserts the Rust scanner and the Python oracle produce
**identical** `(line, why)` verdicts on (a) a shared fixture suite of 30
positive/negative vectors (`tools/sophia-lex/fixtures/overclaim_vectors.jsonl`)
and (b) every file `lint_claims.py` actually scans, plus a decontam-parity check
on a synthetic corpus. It **skips** unless the binary is already built or
`SOPHIA_LEX_BUILD=1` — so the default `pytest -q` gate stays green with no Rust
toolchain. Verified locally: **0 divergence across 30 vectors + 29 corpus files**,
decontam match.

## Sophia Claim Language (SCL)

A deterministic, non-LLM path from a claim to a belief-graph triple:

```text
attribute("Analects" school:"Confucian" confidence:0.6)
  not_to("Confucius")
  source:"wikidata:Q17592"
```

compiles to

```json
{"subject":"Analects","school":"Confucian","confidence":0.6,
 "not_to":["Confucius"],"source":"wikidata:Q17592"}
```

This is net-new (duplicates nothing): a fail-closed *interface* with a tested
reference implementation, in the repo's idiom — a deterministic ground-truth
anchor alongside the existing extractor, not a replacement for it. Confidence is
range-checked `[0,1]`; malformed input fails closed with a JSON error.

## Honest scope / limits

- **Optional, not load-bearing.** Acceleration is opt-in and fails closed to the
  Python oracle. Nothing here can make a gate PASS that Python would FAIL.
- **Agreement, not proof of identity.** The overclaim scanner agrees with the
  oracle on the fixtures and committed corpus; it is not proven byte-identical on
  every adversarial input (the `first .{0,12} agi` fuzzy gap is reproduced as a
  byte-distance window). Python stays the source of truth.
- **The decontam cap is not yet removed in CI.** The full-coverage scan is
  available behind `--accel`; promoting it to the CI default is a follow-up that
  requires the Rust toolchain in the relevant CI lane.

## Follow-ups (not done here)

1. Promote `assert_decontam --accel` to the CI default once a CI lane carries the
   Rust toolchain; then the `--max-eval-shingle` cap can be retired and that
   instrument limitation documented as closed.
2. Wire SCL into `okf/` as a deterministic claim-ingest path beside the extractor.
3. Extend the overclaim fixture suite as new forbidden phrases are added to
   `lint_claims.FORBIDDEN` (keep the two in lockstep via the parity test).
