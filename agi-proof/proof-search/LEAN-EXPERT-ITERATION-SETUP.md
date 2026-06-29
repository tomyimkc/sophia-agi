# Lean Verifier-as-Reward Expert Iteration — Provisioning Runbook

**Scaffold bet:** `verifier_synthesis_over_proof_kernel`
**Cluster E reframe:** the earlier Lean-4 proof-search "blocked / negative" was an
**infra blocker**, not a capability failure. Lean / elan / LeanDojo were never installed
on the CI host, so only the scripted stub applier ran and the real path fail-closed
`lean_unavailable` — exactly as designed. The reframe: a machine-checked Lean proof is the
ideal **non-gameable** verifier, so we run **verifier-as-reward expert iteration**
(propose → keep only Lean-verified AND novel proofs → train the proposer on them → repeat),
harnessed by the already-validated SSIL verifier-gated loop (`selfextend.proof_verifier`).

This document is the precise, reproducible runbook to stand up the real compute, point the
backend at a real Lean repo, and run the loop FOR REAL. **What is gated on real compute is
called out honestly throughout.**

---

## 0. What runs WITHOUT this runbook (CI-safe, already green)

```bash
python3 tests/test_lean_expert_iteration.py        # deterministic, no Lean/model/network
python3 tools/run_lean_expert_iteration.py --stub --rounds 2
```

The `--stub` path exercises the loop STRUCTURE against a **scripted applier** that stands in
for the kernel. It is **NOT** a real verification and **does not close the bet**. The real
path (no `--stub`) abstains `lean_unavailable` on a host with no Lean toolchain and writes
that status — it never fabricates a proof. Everything below is what converts that honest
negative into a real result.

---

## 1. Stand up a GPU host (RunPod MCP)

This repo has the **RunPod MCP wired** (see the `runpod-mcp` skill and the
`mcp__runpod__*` tools: `list-gpu-types`, `create-pod`, `get-pod`, `start-pod`,
`stop-pod`, `delete-pod`, `list-data-centers`). The LLM tactic proposer (`default_proposer`)
needs a GPU; the Lean kernel itself is CPU-bound but tracing Mathlib is memory-heavy.

Recommended pod:
- **GPU:** 1× A100 80GB or H100 (for the proposer model; pick via `list-gpu-types`).
- **Disk:** >= 150 GB container + volume — a traced Mathlib cache is tens of GB.
- **Image:** a CUDA + Python 3.11 base (e.g. `runpod/pytorch`), Ubuntu 22.04.

Sketch (exact tool args resolved at call time from `list-gpu-types` / `list-data-centers`):

```
mcp__runpod__list-gpu-types
mcp__runpod__create-pod   { name: "lean-expert-iter", gpuTypeId: <chosen>, ... }
mcp__runpod__get-pod      { podId: <id> }     # wait for RUNNING, grab ssh
```

`stop-pod` / `delete-pod` when done — Lean + Mathlib tracing is expensive; don't leave it up.

---

## 2. Install elan + Lean 4 + a pinned Mathlib + LeanDojo

On the pod (these are the heavyweight, compute-gated steps):

```bash
# 2a. elan (the Lean toolchain manager) — installs `lean`, `lake`, `elan` on PATH.
curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
  | sh -s -- -y
source "$HOME/.elan/env"
lean --version          # must print a Lean 4 version (e.g. Lean 4.31.0)

# 2b. A PINNED Mathlib. Pin the commit so verification is reproducible across seeds/runs.
#     (Pin to the SAME Lean 4.31 the closed-smoke.jsonl notes were verified under.)
git clone https://github.com/leanprover-community/mathlib4.git
cd mathlib4
git checkout <PINNED_MATHLIB_COMMIT>     # record this commit in the public report
lake exe cache get                       # download the prebuilt Mathlib oleans (avoids a multi-hour build)
lake build                               # finish/verify the build

# 2c. LeanDojo (the programmatic Lean interaction layer; the repo's opt-in extra).
pip install -r requirements-theorem.txt  # the repo's pinned lean-dojo + deps
python -c "import lean_dojo; print(lean_dojo.__version__)"
```

> **lean-dojo 4.x note (already handled in code):** `agent/lean_backend.py` documents that
> 4.x removed the stateless `run_code` API and that `trace()` deadlocks on this machine
> class. The **trace-free L0 bypass** (`verify_lean_source` → the `lean` CLI) and the
> stateful `LeanProofSession` (`agent/proof_search.py`) are the two working paths. For the
> expert-iteration loop, the kernel gate is `agent.lean_verifier.check_proof`, which uses
> the `lean` CLI directly — so **step 2a alone (elan + `lean` on PATH)** is enough for the
> verifier; LeanDojo (2c) is only needed for the stateful tactic-search path.

---

## 3. Point the backend at the real Lean repo

`agent.lean_verifier.lean_available()` probes the `lean_dojo` import; the kernel check in
`agent.lean_verifier.check_proof` shells out to the `lean` CLI on PATH. So:

1. Ensure `source "$HOME/.elan/env"` is in the shell that runs the driver (so `lean` is on
   PATH and `lean_cli_available()` is True).
2. For **project / Mathlib lemmas**, run the driver from inside the built lake project (or
   set the working directory so `lake env lean` resolves Mathlib). The free-standing
   prelude-only snippets in `BUNDLED` need only the toolchain; Mathlib lemmas need the
   built project on the import path.
3. If you use the stateful search path, `agent/proof_search.py::LeanProofSession` already
   takes `repo_url` (defaults to mathlib4) and pins `"master"` — **change that to your
   `<PINNED_MATHLIB_COMMIT>`** so the traced repo matches step 2b.

No code edit is required for the kernel-gated expert-iteration loop itself — it reads the
toolchain from PATH. The only pin to set is the Mathlib commit (in `LeanProofSession` if you
use the search path, and recorded in the report).

---

## 4. Reproduce a REAL single-theorem verification first

Before the loop, confirm the kernel actually accepts a real proof (not `--stub`):

```bash
# (a) Real single-proof search — must NOT abstain lean_unavailable now.
python tools/run_proof_search.py --theorem add_comm        # real path (no --stub)
#   Expected with a live toolchain: verdict "proved", leanVerdict "accepted",
#   novelty probe recorded. If it still says lean_unavailable, the toolchain/PATH is wrong.
```

This is the gate: until `run_proof_search.py --theorem add_comm` (no `--stub`) yields a
**real** Lean `accepted`, the expert-iteration loop will (correctly) fail-close.

---

## 5. Run the expert iteration FOR REAL

```bash
# (b) Verifier-as-reward expert iteration, real kernel, multiple rounds.
python tools/run_lean_expert_iteration.py --rounds 3 --theorem add_zero
#   Real path: each round proposes (LLM proposer), the Lean kernel verifies, only
#   verified+novel proofs are kept into the corpus, anti-gaming runs each round.
```

To actually CLOSE the bet, run on a **held-out theorem the proposer was not trained on**,
across **multiple seeds**, and record every run. Vary the seed via the proposer/model config
(the driver threads one growing corpus per run; reproduce the whole run per seed).

---

## 6. SUCCESS CRITERION (explicit, falsifiable)

> **>= 1 Lean-verified proof with `novelty=true` on a HELD-OUT theorem, REPRODUCED across
> seeds**, with the anti-gaming check showing no kernel-gaming (drop 0.0).

Meeting this:
- closes the scaffold bet **`verifier_synthesis_over_proof_kernel`**, and
- converts the Cluster E negative into a **Level-3 datapoint** — but only under the repo's
  **no-overclaim gate** (>= 3 runs, CI excluding 0, independent review).

A `--stub` run is **NOT** sufficient: it verifies via a scripted applier, not the real
kernel. The report's `verdict: "novel_proof_kept"` under `mode: "stub"` is explicitly
flagged `candidateOnly` and "does NOT close the bet".

---

## 7. What is honestly gated on real compute (not done here)

- **The Lean toolchain + Mathlib build + LeanDojo** (§2) — not installed on the CI host;
  this is the entire infra blocker the Cluster E reframe identified. Without it the real
  path abstains `lean_unavailable` (correct, fail-closed).
- **A real LLM tactic proposer on a GPU** (§1) — `default_proposer()` falls back to the
  deterministic stub when no model is resolvable, which cannot prove nontrivial theorems
  (by design).
- **Held-out theorem curation + multi-seed reproduction** (§5–6) — the open-problems split
  (`formal_proofs/eval/open-problems.jsonl`) stays EVAL-ONLY and must never enter the
  train/kept corpus; a non-abstention there is breakthrough-or-bug, flagged not promoted.
- **The no-overclaim gate** — `canClaimAGI` stays **false**; `candidateOnly`/`validated:
  false` until the gate passes. This runbook provisions the experiment; it does not, by
  itself, assert the result.

---

## Appendix — files in this loop

| File | Role |
| --- | --- |
| `tools/run_lean_expert_iteration.py` | the expert-iteration driver (this runbook's target) |
| `tests/test_lean_expert_iteration.py` | deterministic, fail-closed tests (no Lean/model/network) |
| `selfextend/proof_verifier.py` | `kernel_verifier`, `kernel_reward_is_hackable`, `close_loop_on_proofs` |
| `agent/lean_verifier.py` | `check_proof` (the kernel gate), `lean_available` |
| `agent/lean_backend.py` | `verify_lean_source` (L0 CLI bypass), `novelty_check` |
| `agent/proof_search.py` | `LeanProofSession` (stateful LeanDojo path), `search_proof` |
| `agent/tactic_proposer.py` | `stub_proposer` (CI) / `default_proposer` (live LLM) |
| `formal_proofs/eval/` | smoke split (train-eligible) + open-problems (EVAL-ONLY) |
