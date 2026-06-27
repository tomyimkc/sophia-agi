# HANDOVER — verifiable-sophia arc (for the next AI session)

> **Purpose:** hand this entire document to the next AI session (Claude / Grok /
> ZCode / other) as the first thing it reads. It is self-contained: it states the
> project context, the strict style rules the user (Tom) requires, the exact
> current state of the verifiable-sophia workstream (the arc that converted
> Sophia's one validated claim into a bounded, machine-checkable substrate), the
> one open blocker, and the prioritized action plan. The next AI should not
> re-derive any of this — read, ground against the repo, then act.
>
> **Author of this handover:** ZCode (GLM-5.2), 2026-06-27.
> **User:** Tom (tomyimkc). Address him as Tom. Sole author of the repo.
> **Repo:** https://github.com/tomyimkc/sophia-agi · **Branch:** work on a fresh
> branch off `origin/main` (see "workflow rules" below).

---

## 0. Non-negotiable style rules (follow in EVERY response)

- **Address the user as Tom.** Be critical and honest about AGI claims — no hype,
  no sugar-coating gaps. The thesis is *wisdom-before-intelligence*: provenance
  discipline, 0% fabrication on traps, fail-closed abstention, epistemic integrity.
- **Base every recommendation on the latest deployed repo state.** Verify against
  `origin/main` before asserting anything — multiple concurrent AI sessions edit
  this repo, so working-tree state is **unreliable**; always `git fetch origin main`
  and diff against `origin/main`, never the local working tree. (This bit the prior
  session repeatedly: concurrent sessions force-pushed branches, reverted files,
  and killed background processes mid-run. Work in an isolated `git worktree`.)
- **No-overclaim discipline is the core constraint.** `canClaimAGI` stays `false`
  for all verifiable-sophia work. Every result artifact carries `candidateOnly: true`
  and `canClaimAGI: false`. Every new mechanism gets a failure-ledger entry with
  boundary conditions BEFORE building.
- **Read `agi-proof/failure-ledger.md` FIRST** — especially the entries this arc
  produced (listed in §3). They contain the measured boundaries, not aspirations.
- **Workflow rules:** do ALL work in a dedicated `git worktree` off `origin/main`,
  keep the main checkout clean, never `--force` push (use `--force-with-lease`),
  rebase onto current `origin/main` before merging, and verify with the
  `pr-merge-verification` skill before any merge to main.

---

## 1. The arc, in one paragraph

Sophia's one validated claim (+12.5pt hallucination Δ on dolphin-llama3:8b) is a
**decaying asset** — the repo's own failure-ledger entry
`calibration-advantage-is-model-dependent-2026-06-25` proves the advantage → 0 on
strong base models. The Verifiable-Sophia strategic plan (commit cb887e5, on
`feat/governed-sparse-quant`) pivoted to the non-decaying asset: machine-checkable
verification. This arc executed that pivot. The result (all on `origin/main`,
`canClaimAGI: false` throughout): the fail-closed abstention rule is now a
derivable Datalog theorem (byte-identical to the production gate, runtime-viable),
the model-side delta is precisely bounded (~9pt on weak models, confirmed Δ=0 on a
strong base), and the repo gained its first third-party-labeled pack — but no
external REVIEWER has run the reproducer yet, so external validation status is
unchanged. The headline validated claim (+12.5%) is unchanged.

---

## 2. What landed on `origin/main` (this arc's commits)

Verify with: `git fetch origin main && git log --oneline origin/main | grep -iE "datalog|claimreview|provenance|decay|consolidate|llmhub|judgefree|reproducer|verifiable-sophia|moves|checkpoint"`

| Commit | What |
|---|---|
| `514b5fb1` | merge: Datalog substrate + judge-free +9.0% + 2-family +9.4% + reproducer + llmhub wiring |
| `512487df` | strong-base decay test (qwen3:30b-a3b, Δ=0.0% confirmed) |
| `d2af2a5f` | ClaimReview pack (303 claims) + retriever + eval (NULL on famous claims) |
| `37b459cd` | qwen3 confirmation (0/60 raw endorsement on famous claims) |
| `2777bcb7` | consolidation: retire superseded retriever; Phase summary |
| `898b831b` | obscure-pack re-harvest (hypothesis REFUTED, near-zero headroom) |

A concurrent session also landed **v0.10.0** (`19379c93`: SimpleQA cross-model
validation + C1–C5 candidate mechanisms — `prover_verifier`, `abstention_scoring`,
`conformal_policy`, `activation_probes`, `graded_decision`; all `validated: false`,
`candidateOnly: true`). That work is complementary, not overlapping: this arc
bounded the EXISTING claim; C1–C5 explore NEW mechanisms.

---

## 3. The five experiments + their results (read the ledger entries for detail)

All on `agi-proof/failure-ledger.md`. **Read these before acting** — they hold the
measured boundaries, and several were honest negatives the repo values.

| # | Experiment | Result | Ledger entry |
|---|---|---|---|
| 1 | Datalog port of `provenance_faithful` | **957/957 byte-identical** to the Python gate; 0.5ms runtime backend; opt-in `backend="datalog"` on `check_claim` | `datalog-provenance-faithful-port-preregistered-2026-06-27` |
| 2 | Judge-free reproduction | **+9.0%, CI [+4.9, +13.9]**, excludes 0 — NOT an LLM-judge artifact | `provenance-delta-survives-judge-free-2026-06-27` |
| 3 | 2-family multi-judge (gpt-4o + claude) | **+9.4%, CI [+4.2, +15.6]**, κ=0.81 (2 runs; run 3 lost to churn) | `provenance-delta-multijudge-2family-2026-06-27` |
| 4 | Strong-base decay (qwen3:30b-a3b) | **Δ=0.000, CI [0,0]** — decays to zero on strong base, as predicted | `provenance-delta-decays-to-zero-on-strong-base-2026-06-27` |
| 5a | ClaimReview famous pack | **NULL**: raw 3.3%→grounded 3.3% (models know famous claims) | `claimreview-third-party-axis-null-on-famous-claims-2026-06-27` |
| 5b | ClaimReview obscure pack | **Hypothesis REFUTED**: raw 1.7% (lower, not higher); near-zero headroom | `claimreview-obscure-pack-hypothesis-refuted-2026-06-27` |

**Phase summary** ties these together: `verifiable-sophia-phase-summary-2026-06-27`.

---

## 4. Key code/assets on main (where to find them)

- `agent/datalog_engine.py` — minimal stdlib Datalog (stratified negation, ~250 LOC).
- `agent/datalog_provenance.py` — the abstention rule as one Horn clause:
  `violation :- asserted_in_clause(C,W,A), forbidden(W,A), not carveout(C).`
- `agent/guarded.py::check_claim(..., backend="datalog")` — opt-in logic backend (default `"regex"`).
- `agent/verifiers.py::provenance_faithful(..., return_specs=True)` — read-only hook the port uses.
- `tools/run_datalog_provenance_audit.py` — the 957-case audit harness.
- `tools/run_datalog_reproducer.py` — **THE turnkey third-party reproducer** (hash-pinned,
  trusts no committed artifact, self-verified PASS/tamper-FAIL). This is the thing a reviewer runs.
- `tools/run_unified_uplift.py` — the provenance uplift harness (now has per-run checkpointing;
  `--judges` omitted = deterministic lexical judge = judge-free).
- `agent/model.py` + `provenance_bench/aggregate.py` — `llmhub` aggregator preset +
  `_LLMHUB_FAMILY` map (so gpt-4o + claude behind one key count as 2 honest families).
- `tools/build_claimreview_pack.py --set {famous,obscure}` + `provenance_bench/data/claimreview_pack{,_obscure}.json`.
- `tools/run_claimreview_eval.py --pack` — raw-vs-grounded endorsement eval.
- Note: `GoogleFactCheckBackend` in `agent/live_sources.py` (v0.10.0) is the canonical Fact
  Check integration now; the standalone `agent/claimreview_retriever.py` was retired.

---

## 5. THE binding constraint (and why this arc is "done" but not "won")

**Everything internally falsifiable has been falsified.** The repo now has ~7
candidate mechanisms + 1 validated claim, all `canClaimAGI: false`. The bottleneck
is NOT code — it's that no external REVIEWER has run the reproducer. The strategic
plan was explicit: *"one real third-party pack/run is worth >10 more self-runs."*

**The single highest-leverage next action needs Tom (a human):** solicit one
external reviewer to run, on a clean host:
```
python tools/run_datalog_reproducer.py
```
It's one command, hash-pins the data files, recomputes the 957-case audit live,
prints PASS/FAIL. That converts candidate-grade → externally-validated. No AI
session can do this alone.

---

## 6. Prioritized action plan for the next AI session

1. **(Needs Tom — flag it, don't attempt it)** Solicit the external reviewer run
   above. This is the only thing that moves external validation status.
2. **(If Tom wants more internal work — diminishing returns, recorded honestly)**
   The ClaimReview axis's open thread: the true/false-QA frame has near-zero
   headroom because models rarely endorse explicit misinformation when asked
   directly. A frame surfacing *implicit/confident-but-wrong* endorsement (open-
   ended generation, not true/false prompts) is the plausible substrate. Recorded
   in `claimreview-obscure-pack-hypothesis-refuted-2026-06-27` as the real next step.
3. **(The 2-family run is missing its 3rd run.)** The +9.4% multi-judge result
   clears 4 of 5 validation flags; only `atLeast3Runs` fails (run 3 killed by
   concurrent churn). A clean 3rd run on a quiet host flips it. Re-run:
   `tools/run_unified_uplift.py --model ollama:dolphin-llama3:8b --judges
   "llmhub:gpt-4o,llmhub:claude-sonnet-4-6" --runs 3 --limit 48 --levers +gate`.
   (Needs the `LLMHUB_API_KEY`; see §7.)
4. **(Consolidation)** v0.10.0's C1–C5 candidates are all `candidateOnly: true`.
   Reconciling/deduping across my arc + C1–C5 + the other concurrent work is
   lower-glamour but high-value housekeeping.

---

## 7. Secrets / credentials note (security)

Two API keys were provided this session and are referenced in artifacts/evals:
- `LLMHUB_API_KEY` — the llmhub.com.cn proxy key (for multi-judge runs). Stored
  only in `/tmp/.sjk`, never committed. **Rotate it** — it's now in the chat
  transcript and the proxy 301-redirects HTTP→HTTPS (key in plaintext to the redirect).
- `GOOGLE_FACTCHECK_API_KEY` / `GFC_API_KEY` — the Google Fact Check Tools key.
  Stored only in `/tmp/.gfc`, never committed. **Rotate it too** (same reason).

Neither key is in the repo. The tooling reads them from env/files at runtime.

---

## 8. Hazards the prior session hit (so the next one avoids them)

- **Concurrent-session churn is severe in this repo.** Force-pushes dropped my
  commits twice; checkouts reverted my files ~5×; background processes were killed
  mid-run 3×. Mitigation: isolated `git worktree` per task; commit early/often;
  per-run checkpointing (now in `run_unified_uplift.py`); never trust the shared
  working tree. If a long run dies, the `.partial.json` checkpoint preserves
  completed runs — aggregate it honestly rather than re-running from zero.
- **The `pr-merge-verification` skill is mandatory before any merge to main.**
  Run it; work from an isolated worktree; verify `canClaimAGI: false`,
  `lint_claims`, `validate_failure_ledger`, and the relevant tests post-merge.
- **Don't fabricate a "run" of something that abstains.** If a toolchain is
  missing (Lean, a model), record it as deferred/gated — never silently skip or
  claim it ran. The Lean soundness experiment is correctly deferred (toolchain
  gated) — do NOT attempt the multi-GB install locally.
