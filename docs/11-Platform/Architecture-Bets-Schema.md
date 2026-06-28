# Architecture-Bets registries: two schemas, two files

## Why this document exists

`agi-proof/architecture-bets.json` was, for a period, demanded in **two mutually
incompatible JSON schemas** by two different test suites, and a tool reads it on top:

| Consumer | Schema it expects |
| --- | --- |
| `tests/test_architecture_bets.py` (governance, from commit `ab736a0`) | **module-wiring** registry: each bet has `id`, `module`, `status` (`scaffold\|wired\|measured\|retired`), `live_caller`, `ablation_flag`, `closing_experiment`, `ledger_id` |
| `tests/test_long_context_runner.py` (long-context candidate runner, also on `claude/sophia-agi-architecture-review-ucvzyl`) | **measurement-target** registry: exact set-equality on 7 bets (`verifier-gated-long-context`, `council-small-models`, …) with `honest_status`, `blocked_on`, `implementation_files` |
| `tools/lint_claims.py` | reads the file and asserts `canClaimAGI === false` and that no bet is `status: "wired"` without a `live_caller` |

One file cannot satisfy both without fabricating a hybrid that guts the governance
tracking. (See `HANDOVER.md` §4, which deferred the `ucvzyl` branch for exactly this
reason.) At one point the long-context schema had simply overwritten the file and the
governance test had been deleted — a destructive partial resolution.

## The decision

The two registries are **split into two non-colliding files**:

- **`agi-proof/architecture-bets.json`** stays the canonical **module-wiring governance
  registry** (module/`live_caller`/`status`/`ablation_flag`/…). This is the schema
  `tools/lint_claims.py` is written against (it gates `status: "wired"` ⇒ non-null
  `live_caller`) and the schema the W0 governance test (`tests/test_architecture_bets.py`)
  enforces. Governance, not measurement, owns the canonical filename.
- **`agi-proof/long-context-bets.json`** holds the 7-bet **long-context
  measurement-target registry** (`honest_status`/`blocked_on`/`implementation_files`).
  Its invariants are enforced by the new `tests/test_long_context_bets.py`.

Both test suites now pass against their own file; neither file pretends to be two schemas.
`tools/lint_claims.py` applies its `canClaimAGI === false` check to both files.

This is honest, minimal, and reversible: the long-context data was copied verbatim from
the prior `architecture-bets.json` (byte-identical to the `ucvzyl` branch's copy), and the
module-wiring `architecture-bets.json` + its test were restored verbatim from `ab736a0`.

## What the `ucvzyl` branch needs to land cleanly

The branch `claude/sophia-agi-architecture-review-ucvzyl` carries
`tests/test_long_context_runner.py`, whose
`test_architecture_bets_root_map_has_required_fields` reads
`agi-proof/architecture-bets.json` and expects the 7-bet schema. After this split that
function would read the module-wiring file and fail. The one-line retarget:

```python
# tests/test_long_context_runner.py, in test_architecture_bets_root_map_has_required_fields
- bets = json.loads((ROOT / "agi-proof" / "architecture-bets.json").read_text(encoding="utf-8"))
+ bets = json.loads((ROOT / "agi-proof" / "long-context-bets.json").read_text(encoding="utf-8"))
```

Equivalently, that whole assertion can be deleted from `test_long_context_runner.py` since
`tests/test_long_context_bets.py` now owns those invariants.

Note: this same retarget is already needed on `main`/the current branch, because
`tests/test_long_context_runner.py` is present there too. Until that one line is changed,
`test_long_context_runner.py::test_architecture_bets_root_map_has_required_fields` is the
single test that fails against the canonical module-wiring file — by design, it is the
hand-off marker for whoever lands the retarget. Every other test
(`tests/test_architecture_bets.py`, `tests/test_long_context_bets.py`,
`tests/test_architecture_scaffolding.py`) and `tools/lint_claims.py` pass.
