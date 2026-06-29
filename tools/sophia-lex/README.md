<!-- SPDX-License-Identifier: Apache-2.0 -->
# sophia-lex

A deterministic, single-DFA text scanner for the Sophia **measurement gate**,
built on [`logos`](https://github.com/maciejhirsz/logos) (compile-time lexer
generation). It is an **optional accelerator** for the pure-Python gate tools —
the Python tools stay the reference oracle and the CI default. Acceleration is
opt-in and parity-tested against Python: `tools/test_lex_parity.py` asserts the
same set of `(line, why)` verdicts — not byte-identical output, columns, or
ordering.

## Why a lexer here

The repo's thesis is that *the measurement instrument*, not the model, is the
dominant source of wrong conclusions. Two of those instruments are themselves
text scanners (`tools/lint_claims.py`, `tools/assert_decontam.py`) built on
Python regex / `str.split`. A `logos` DFA gives three properties the repo
values more than raw speed:

1. **Deterministic by construction** — one DFA, no backtracking, linear time.
2. **The spec is the artifact** — the token tables are an auditable, diffable
   declaration of exactly what is recognized (vs. ~18 scattered regexes).
3. **Compile-time, offline, dependency-free at the gate** — no pyo3; the Python
   tools shell out to a CLI and parse line-oriented JSON.

It also unlocks an on-thesis instrument fix: the Python decontam scan caps the
eval surface at `--max-eval-shingle 4000` "for perf" — a silent coverage bound
of exactly the kind the measurement thesis condemns. The accelerated path is
fast enough to drop the cap and scan the **full** eval surface.

## Components

| Module | Mirrors / provides | Parity |
|---|---|---|
| `overclaim` | `lint_claims.py` FORBIDDEN phrase scan | agrees on the fixture suite + committed corpus |
| `normalize` + `shingle` | `assert_decontam.py` normalize + k-shingle Jaccard | byte-exact algorithm; full eval coverage (no cap) |
| `scl` | **Sophia Claim Language** — deterministic claim → triple (net-new) | n/a (new capability) |

## Build & use

```bash
cd tools/sophia-lex && cargo build --release
cargo test                      # 18 unit tests

# Python opt-in acceleration (auto-falls-back to Python if the binary is absent):
python tools/lint_claims.py --accel
python tools/assert_decontam.py --accel      # full eval coverage, no --max-eval-shingle cap

# Parity gate (skips unless the binary is built or SOPHIA_LEX_BUILD=1):
SOPHIA_LEX_BUILD=1 pytest tools/test_lex_parity.py
```

## Honest scope

- The `overclaim` scanner **agrees** with the Python oracle on the shared
  fixtures and the committed corpus; it is **not** proven byte-identical on every
  adversarial input (the fuzzy `first .{0,12} agi` regex is reproduced as a
  byte-distance window). Python remains the source of truth.
- Nothing here can make a gate PASS that Python would FAIL: acceleration is
  opt-in, fails closed to the Python path, and is parity-tested.
- SCL is a fail-closed *interface* with a tested reference implementation, not a
  replacement for the LLM/regex claim extractor — a deterministic anchor beside it.

See `docs/11-Platform/Sophia-Lex-Deterministic-Scanner.md` for the design note.
