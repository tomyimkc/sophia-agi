# Handover: ConsequenceGate follow-on development

**For:** the next AI session continuing the ConsequenceGate / OKF belief-dynamics work.
**From:** the session that completed the two PR-#162 deferred items.
**Date:** 2026-06-27. **Repo:** `tomyimkc/sophia-agi`. **All work on `main`.**

---

## TL;DR — where things stand

The ConsequenceGate (the **8th conscience path**, PR #162) had two deferred items. **Both are now done and merged:**

| PR | What | Status |
|---|---|---|
| **#162** | ConsequenceGate itself — `simulate_cascade`, the ko-detector, wired into `ConscienceKernel`. | MERGED |
| **#176** | Consequence-cascade benchmark + **data-derived** `flipSeverityEscalate` (was a 0.15 hand-pick placeholder). | MERGED |
| **#182** | Ko-guarded **iterative revise loop** (`run_revise_loop`) + wired the orphaned `koMaxRounds` config key — the ko-detector's first real consumer. | MERGED |

Everything below is `candidateOnly: true`, `level3Evidence: false`. The layer is **structurally complete and tested** but earns level3 only after a real run routes decisions through it.

---

## The layer as it now exists (all on `main`)

```
agent/consequence_gate.py          simulate_cascade(graph, move) -> ConsequenceReport
                                   _load_consequence_config()  [reads flipSeverityEscalate + koMaxRounds, fail-safe]
                                   module globals: flip_severity_escalate, ko_max_rounds, KO_MAX_ROUNDS (imported SSOT)

reasoning/consequence/
  ko_detector.py                   detect_ko(revision_states, *, max_rounds) -> KOAlert
                                   is_ko(...) -> bool   [tracks MOST-RECENT occurrence, not first]
  revise_loop.py                   run_revise_loop(graph, *, retraction_schedule, ko_max_rounds, escalate_threshold, by) -> ReviseLoopState
  __init__.py                      exports KO_MAX_ROUNDS, KOAlert, detect_ko, is_ko, ReviseLoopState, run_revise_loop

config/consequence.json            flipSeverityEscalate (0.15, DATA-DERIVED), koMaxRounds (4)

tools/run_consequence_cascade_benchmark.py   JSONL-driven runner + threshold sweep
eval/consequence_cascade/consequence_cascade_40_v1.jsonl   38 synthetic cases (severe/routine/unbounded/boundary)

agent/conscience.py                8th path: consequence routing runs AFTER hard blocks, BEFORE benign_boundary allow

tests/                             test_consequence_gate.py (16), test_ko_detector.py (10), test_revise_loop.py (7),
                                   test_consequence_cascade_benchmark.py (9). House pattern: sys.path bootstrap +
                                   def main() + inspect runner; pytest also works.
```

**Key contracts to honor if you extend this:**
- `simulate_cascade` reuses `okf.revise` **verbatim** — it invents no facts; the abstain set names only existing nodes.
- A ko → **`escalate`, NEVER `abstain`** (load-bearing: a ko needs a human/new source, not a silent drop).
- `revise()` is **non-destructive**; the loop re-calls it on the **original** graph with the round's cumulative targets — never threads a reduced graph forward.
- `flipSeverity = |abstainSet| / |graph.nodes|`; `verdict = escalate if severity >= threshold else allow`.
- Import layering: `reasoning.consequence.revise_loop` resolves its agent.* defaults **lazily** (inside `run_revise_loop`) to avoid a circular import. Do NOT add an agent.* import at any `reasoning.consequence.*` module top.

---

## Honest gaps — the real next steps (in rough priority order)

These are NOT regressions. They are the deliberate `level3Evidence: false` boundaries that, if closed, would graduate the layer from candidate to evidenced.

### 1. Wire the loop into a LIVE caller (highest value, currently zero live consumers)
`run_revise_loop` exists and is tested, but **nothing calls it at runtime.** The natural homes:
- `agent/belief_revision_policy.py` → `resolve_conflicts` is currently **single-pass** (decides each declared contradiction once, no iteration). A real contradiction-resolution flow that revises → observes the abstain set → reasserts in response would be the canonical ko surface. **Caution:** adding iteration changes what `resolve_conflicts` returns and could break its existing consumers — extract the decision logic into a helper the loop imports, don't embed iteration directly.
- The MCP `sophia_revise` tool (`sophia_mcp/server.py`) is single-shot; a multi-round variant is a possible consumer.

**Do this last** — it's the step that could earn `level3Evidence: true` (a real run routing decisions through the loop, with evidence that escalate-on-ko is the right operator response).

### 2. Validate `flipSeverityEscalate` against REAL belief graphs
The #176 threshold (0.15) is data-derived from **synthetic** graphs. The pack's own README states it "earns `level3Evidence: true` only after a real run ... against real belief graphs (which may have different severity distributions)." If/when Sophia has a real OKF corpus, re-run the sweep (`tools/run_consequence_cascade_benchmark.py`) against real-graph cases and check the optimum holds. The runner already supports `--in <custom.jsonl>` and `--threshold`.

### 3. Author a SEIB-Go-Flip benchmark pack (separate from #176's cascade pack)
#176's pack tunes the *severity threshold*. A distinct pack could tune/validate the **ko window** (`koMaxRounds`) — e.g. synthetic multi-round revise schedules with known-ko vs known-benign sequences, swept over window sizes. The runner template (`run_consequence_cascade_benchmark.py`) generalizes.

### 4. The "God's Touch" detector (from the original brainstorm, `sess_c0691230`)
The brainstorm flagged that nobody has wired `|claims_to_abstain| / |graph|` as a *flip-severity score whose low-probability-high-magnitude tail is the "God's Touch" move*. The severity is computed; the *distributional* detector (a cascade in the tail) is not. This is a Medium-variant extension — would live in `reasoning/consequence/`.

---

## Repo working conventions (learned the hard way)

- **`main` moves fast** — many concurrent advisors. ALWAYS `git fetch origin main` and check `behind`/`ahead` immediately before any push or merge. A branch that was clean 10 minutes ago is stale.
- **Branch protection on `main`:** required checks `fast` + `ci-complete`; `required_review_thread_resolution: true` (the **silent killer** — green checks + BLOCKED usually means unresolved Copilot threads); `required_approving_review_count: 0`; `required_linear_history: true`. Use `gh pr merge --rebase`.
- **Copilot auto-reviews every PR** and its threads are *frequently real correctness bugs* — read every one, fix the valid ones in code, reply, resolve. Do not rubber-stamp. (This session: 4 valid threads on #176, 3 valid on #182, all fixed.)
- **No-overclaim discipline:** every artifact carries `candidateOnly`/`level3Evidence`/`validated`/`claimBoundary`. `tools/lint_claims.py` must be clean. Never assert `level3Evidence: true` without a real run.
- **Worktree discipline:** the shared main worktree (`/Users/tom/Documents/GitHub/sophia-agi`) is contended by multiple live sessions — its HEAD changes under you. Do consequence work in the **`cg-worktree`** (`/Users/tom/Documents/GitHub/cg-worktree`); `git add <explicit files>` never `-A`.
- **Merge-preflight skill:** `.agents/skills/multi-agent-merge-preflight/` — run before any PR. (Its `merge_blockers.py` script path moved; query threads directly via `gh api graphql` if needed.)
- **Test house pattern:** `sys.path` bootstrap + `def main() -> int` with `inspect`-based `test_*` discovery + `if __name__ == "__main__": raise SystemExit(main())`. Both pytest and standalone `python tests/test_x.py` must work.
- **uv** for the env: `uv sync --extra dev` (the `dev` extra has pytest/numpy/sympy); `uv run pytest ...`, `uv run python tools/...`.

---

## Reproduce / verify the current state

```bash
cd /Users/tom/Documents/GitHub/cg-worktree
git fetch origin && git checkout main && git reset --hard origin/main
uv sync --extra dev
uv run pytest tests/test_consequence_gate.py tests/test_ko_detector.py tests/test_revise_loop.py \
              tests/test_consequence_cascade_benchmark.py tests/test_conscience.py -q   # ~46 pass
uv run python tools/run_consequence_cascade_benchmark.py    # ok:true, recommended=0.15, deterministic
uv run python tools/lint_claims.py                          # OK
```

---

## Pointers to the originating context

- **Original brainstorm:** session `sess_c0691230` ("Sophia ConsequenceGate Implementation Brainstorm", `completed`). Its `searchable_text` (in `~/.zcode/v2/tasks-index.sqlite`) contains the full 5-section design (Minimal/Medium/Maximal variants); #162/#176/#182 implemented Variant A.
- **PR #162** body: the layer's founding design + the 3 Copilot correctness fixes (ko most-recent tracking, config fail-safe, conscience routing order).
- **PR #176** body: the threshold-sweep methodology + why 0.15 wins (clean gap between worst-allow 0.143 and worst-escalate 0.167).
- **PR #182** body: the loop's verified-correct revise pattern + the lazy-import layering fix.
