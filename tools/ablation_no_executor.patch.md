# PATCH (REVISED after adversarial review): the "no-executor" ablation is confounded — do NOT add a naive `use_executor` flag

> **Review verdict: REDESIGN, not a one-line flag.** The original plan (add a single
> `use_executor` bool + a `sophia-no-executor` mode) is **withdrawn**. Ground-truthing
> against the real `tools/run_hidden_eval_sophia.py` (defects D2/D6) shows it would produce
> an uninterpretable delta. This document now records why, and the two defensible options.

## Why a single `use_executor` gate is wrong (review D2)

There is **no single "executor" dispatch site** to gate. "Execution" is spread across at
least four flag-gated stages in `run_case`:
- `run_operational_tools(case)` — the subprocess tool loop, dispatched at ~:1276 under `use_tools`;
- the learning-probe model calls under `use_memory` (~:1279);
- the primary answer generation (~:1313);
- the bounded repair attempt under `allow_repair` (~:1366).

A single `use_executor` flag would leave the other three live. **Worse (the confound):**
bypassing the tool stage empties `tool_log` / `memory_diff`, and `score_case` reads those
directly to emit **forced `operationalFailures`** (`tools/hidden_eval_protocol.py` ~:212–227).
So a full-vs-no-executor delta would conflate two different things — "the model fabricates
more without tools" and "the harness auto-fails the case because a required log is missing" —
and there is no dedicated fabrication scorer to isolate the first from the second. Not
apples-to-apples.

## Field-count correction (review D6)

The real `Ablation` dataclass (`tools/run_hidden_eval_sophia.py` ~:1069–1081) has **two more
fields** than the original patch assumed: `use_context_packing: bool` and
`context_packing_policy: str`. Any patch that reconstructs the dataclass positionally, or
asserts a field count, breaks. **Construct `Ablation(...)` with keyword args only.**

## DECISION (2026-07-01, maintainer session): **Option 1 — config-only. Option 2 rejected.**

Ground-truthed against the current tree (`ABLATION_MODES` verified live). Correction to an
earlier draft: **there is NO `sophia-no-tools` mode in `ABLATION_MODES`** (keys are raw-model,
raw-model-plus-tools, sophia-full, sophia-no-intake, sophia-no-kb, sophia-no-gate,
sophia-no-memory, sophia-no-council, sophia-claim-router). But the tool/executor stage is
already isolable **two ways**, both keyword-safe (D6):

1. **Zero code — an existing pair already isolates `use_tools`.** Verified live:
   `raw-model` and `raw-model-plus-tools` differ in **exactly one field, `use_tools`**
   (F vs T; every other field identical). So `raw-model-plus-tools − raw-model` IS the
   clean operational-tool/executor delta in the raw arm — no code change at all.
2. **One line — isolate it in the full-Sophia arm.** Add
   `"sophia-no-tools": Ablation(label="sophia-no-tools", use_tools=False)` to `ABLATION_MODES`
   (keyword args only; verified it builds and preserves `context_packing_policy`). Then
   `sophia-full − sophia-no-tools` differ in exactly one field (`use_tools`).

**Both must exclude cases with `requiresToolLog`/`requiresMemoryDiff` from the delta** — those
cases auto-fail when `tool_log`/`memory_diff` are empty (`hidden_eval_protocol.py` ~:212–227),
which would masquerade as a fabrication signal (D2 confound).

**Why not Option 2:** a new `use_executor` bool + 4 dispatch-site edits on a *frozen* dataclass
re-invents what `use_tools` already does; there is no distinct "executor" site to gate. More
code, more risk, no cleaner signal than case-excluded `use_tools`. Rejected.

**Live-run status:** not yet run — the only backend available this session is the agentic grok
CLI (~66 s/case), too slow/costly for a multi-mode × multi-seed ablation pack. The ablation
CELL stays **Open**; closing it needs a cheap batch backend (DeepSeek/OpenRouter) per the gate
below. This is a decision + zero/one-line implementation path, not a measured delta.

---

### (superseded draft) Option 1 — "reuse the existing `sophia-no-tools` mode"
`use_tools=False` already gates `run_operational_tools`. If the question is "does the
operational-tool/executor stage reduce fabrication," a `use_tools=False` mode answers it. Run
`--modes sophia-full,sophia-no-tools` (after adding the one-line mode above) and report the
delta on the subset of cases that do **not** set `requiresToolLog`/`requiresMemoryDiff`.

**Option 2 — a real, multi-site "no-executor" mode (more work, only if Option 1 is insufficient).**
Add `use_executor: bool = True` (keyword-only, defaulting True so every existing mode is
unchanged), gate **all** execution sites listed above on it, AND exclude cases that carry
`requiresToolLog`/`requiresMemoryDiff` from the delta so the missing-log auto-fail cannot
masquerade as a fabrication signal. This is the only way to get an apples-to-apples
"executor off" cell, and it must be reviewed against current line numbers because it touches
the frozen dataclass and four dispatch sites.

## Original (WITHDRAWN) single-flag diff — kept for the record, do not apply

---

## 1. `tools/run_hidden_eval_sophia.py` — add the flag to the `Ablation` dataclass

```diff
 @dataclass(frozen=True)
 class Ablation:
     label: str = "sophia-full"
     raw_system: bool = False
     use_kb: bool = True
     use_evidence: bool = True
     use_council: bool = True
     use_gate: bool = True
     use_memory: bool = True
     use_tools: bool = True
     allow_repair: bool = True
+    use_executor: bool = True  # code/tool executor stage; off => planner output is
+                               # NOT executed/verified by running it
```

(Keep it defaulting `True` so `SOPHIA_FULL` and every existing mode are unchanged.)

## 2. Add the mode to `ABLATION_MODES`

```diff
     "sophia-no-council": Ablation(label="sophia-no-council", use_council=False),
+    "sophia-no-executor": Ablation(label="sophia-no-executor", use_executor=False),
```

## 3. Gate the executor stage in `run_case`

Find the executor dispatch in `run_case` (where the planned action / generated code is
actually executed — grep for the executor call, e.g. `execute(`, `run_executor`,
`code_verifier`, or the tool-execution block). Wrap it:

```diff
-    exec_result = _run_executor(plan, ...)
+    if ablation.use_executor:
+        exec_result = _run_executor(plan, ...)
+    else:
+        # fail-closed: with the executor off, an answer that WOULD require execution
+        # to verify is emitted un-executed and flagged, never silently "passed".
+        exec_result = ExecResult(ran=False, verified=False,
+                                 note="executor ablated (sophia-no-executor)")
```

The invariant: with `use_executor=False`, any claim whose correctness depends on running
code/tools is emitted **unverified-and-flagged**, so the fabrication scorer counts it
honestly rather than crediting an unexecuted guess.

## 4. Add the mode to the `--modes all` set

Confirm `sophia-no-executor` is included in the canonical modes iterated by
`tools/run_ablation_sophia.py` (it reads `ABLATION_MODES`, so adding it above is
sufficient unless there is an explicit allow-list — grep `ABLATION_MODES` there).

## Acceptance gate (ledger: `no-executor-ablation-cell`)

`python3 tools/run_ablation_sophia.py <pack.json> --backend <live> --modes sophia-full,sophia-no-executor`
produces a fabrication delta between the two through the **same deterministic scorer**,
with a CI, committed under `agi-proof/baseline-ablation/`. Closes the one missing cell.
The unit test below asserts the flag exists, the mode builds, and the executor is bypassed
when off — WITHOUT a live backend.
