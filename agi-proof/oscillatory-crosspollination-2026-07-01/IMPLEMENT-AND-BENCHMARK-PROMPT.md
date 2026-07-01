# Handover: understand, implement, and benchmark the oscillatory epistemics tools (O1–O5)

You are an autonomous engineering agent working in a local clone of the `tomyimkc/sophia-agi`
repo. A prior research session designed five directions and shipped them as **fail-closed,
unit-tested instruments** on branch `feat/oscillatory-crosspollination` (HEAD authored by
tomyimkc). Your job has three phases, in order: **(1) understand the research, (2) take each
instrument live, (3) benchmark it against a real dataset and report PASS/FAIL vs its acceptance
gate.** Read the whole brief before you touch anything.

---

## Phase 0 — Understand the research (do this first, change nothing)

Read these, in order:
1. `agi-proof/oscillatory-crosspollination-2026-07-01/Sophia-Oscillatory-CrossPollination.md`
   — the prospectus: the thesis, the literature survey, and the strongest objection to each
   direction.
2. `agi-proof/oscillatory-crosspollination-2026-07-01/README.md` — what each tool binds to and
   proves offline.
3. `agi-proof/oscillatory-crosspollination-2026-07-01/failure-ledger-additions.md` — the five
   Open rows and the exact acceptance gate for each.
4. The six tools in `tools/` and their tests in `tests/`.

**The one-sentence thesis.** sophia is an unusually complete epistemic *measurement* engine
(rules, verifiers, calibration, provenance) but it has no readout for one axis: **how coherently
an answer settles.** `unconv-ai/Un-0` generates images by reading the settled state of a coupled
Kuramoto oscillator population; these five tools borrow that convergence/coherence signal and use
it for the opposite purpose — *verification and abstention*, not generation.

| Dir | Tool | Idea |
|---|---|---|
| **O1** | `consensus_gate_oscillator.py` | Kuramoto order-parameter *r* over sampled answers as a self-consistency confidence gate. |
| **O2** (flagship) | `energy_verifier_head.py` | A learned scalar *energy* over (answer, evidence); min-energy = Best-of-N self-verification; high energy = abstain. |
| **O3** | `fixedpoint_stability_gate.py` | Treat verification as a DEQ fixed point: does claim+evidence settle? Non-convergence → abstain. |
| **O4** | `adaptive_compute_convergence.py` | Adaptive test-time compute: keep sampling until *r* stabilizes; spend more on hard inputs. |
| **O5** (horizon) | `oscillator_substrate_sim.py` | **Simulation only** — energy verification mapped onto oscillator-Ising relaxation. Do NOT take past simulation. |

**The honesty contract you inherit (non-negotiable).**
- These are instruments. **A run is not a result.** No metric may be claimed without a real
  backend and a real, decontaminated, labelled dataset. Keep `candidateOnly:true` until a gate is met.
- **Negative results are results.** If a tool does not beat its baseline, record that plainly —
  do not tune fixtures until it passes.
- Every artifact carries `candidateOnly / level3Evidence / canClaimAGI` flags. Never flip
  `canClaimAGI`.

Establish your baseline before editing anything:
```
PYTHONPATH=. python -m pytest \
  tests/test_oscillator_core.py tests/test_consensus_gate_oscillator.py \
  tests/test_energy_verifier_head.py tests/test_fixedpoint_stability_gate.py \
  tests/test_adaptive_compute_convergence.py tests/test_oscillator_substrate_sim.py -q
# expect: 30 passed. Full package suite (WS-A–D + W1–W5 + O1–O5): 88 passed, 0 skipped.
```

---

## Phase 1 — Repo reality you MUST respect (verified against the tree)

1. **Model adapter.** The real interface is
   `agent.model.default_client(spec).generate(system, user) -> ModelResult` (and the
   module-level `complete(system, user, *, max_tokens=2400, spec=None) -> str`). There is **no
   `Model(spec).complete(prompt)`** — a prior session had to fix exactly that drift.
2. **Mock guard.** `agent.model._auto_provider()` returns `"mock"` when no API key is present,
   and mock `.generate()` returns *fabricated* text with `ok=True`. Any live run MUST
   `assert cfg.kind != "mock"` (or check `ModelResult.provider`) or you will benchmark a
   hallucinated backend and score noise.
3. **The `hash_embed` seam is lexical, not semantic.** O1/O3/O4 embed text with
   `tools/oscillator_core.hash_embed` (blake2b token hashing) — it captures lexical overlap only.
   The repo's real embedders live in `agent/rag_embed.py`
   (`embed_texts(texts, *, batch_size=16)`, `embed_query(text)`). **Caution:**
   `agent/rag_local_embed.py` is ALSO a hash-feature embedder (backend id `local-hash-v1`), so
   swapping to it is *not* a semantic upgrade — you need `rag_embed` with a real embedding backend
   configured. Use `agent.vector_store.embedding_backend_id()` to confirm which backend is live
   (`local-hash-v1` = still lexical; a real model id = semantic). Every tool surfaces
   `hashEmbedSeam:true`; the coherence/residual geometry only becomes meaningful once a real
   embedder is in.
4. **Do not `git add -A`.** The working tree has ~380 pre-existing modified files that are the
   maintainer's. Stage only files you author, by explicit path.

---

## Phase 2 — Implement + benchmark, per tool

For each tool: (a) swap the seam / wire the live path, (b) assemble a real labelled dataset,
(c) run the named benchmark harness, (d) report the gate metric with its CI and PASS/FAIL.
**Reuse the existing `tools/eval_*.py` harness family — do not invent a new benchmark format.**

### O1 — consensus gate  *(do first: training-free, cheapest, directly falsifiable)*
- **Wire:** replace `hash_embed` with `agent.rag_embed.embed_texts` (confirm backend is not
  `local-hash-v1`).
- **Dataset:** the same human-authored SimpleQA slice behind the repo's validated self-consistency
  result. Each row = k sampled answers + a correctness label. Reuse
  `tools/analyze_simpleqa_calibration.py` / `tools/eval_graded_confidence.py` to source and score.
- **Run:** `tools/consensus_gate_oscillator.py --records <sampled.jsonl>`.
- **Gate (`o1-consensus-gate-not-validated-vs-self-consistency`):** `verdict ==
  consensus_beats_baseline` with the paired-AURC 95% CI lower bound > 0, reproduced across ≥2
  seeds. If it ties or loses, that is a publishable negative result — report it.

### O2 — energy verifier head  *(flagship; needs a GPU/MLX backend)*
- **Wire:** implement the shared seam `agent.activation_probes.build_hidden_state_featurizer`
  (also unlocks W1/W5), so the energy head is a learned scalar over hidden states rather than the
  linear probe over `featurize_text`. Train the compatibility energy on real verifier-labelled
  (answer, evidence) pairs from `agent.verified_trace_rlvr` outputs. The repo's real LoRA
  invocation, for reference:
  ```
  python3 -m mlx_lm lora --train --model Qwen/Qwen2.5-3B-Instruct \
    --data training/local_sophia_v2/mlx --iters 500 --batch-size 4 --mask-prompt \
    --adapter-path training/mlx_adapters/sophia-vNEXT \
    --steps-per-report 50 --steps-per-eval 250 --save-every 250 --max-seq-length 1024
  ```
- **Benchmark:** `tools/eval_faithfulness.py` for verification quality; calibration of energy vs
  correctness via `tools/calibration_check.py` / `calibrate_graded_thresholds.py`.
- **Gate (`o2-energy-verifier-linear-stub-not-hidden-state`):** calibrated energy (AUROC/ECE,
  paired-bootstrap CI excluding 0) AND held-out-**domain** `goodhartGap ≤ 0.15`. Then wire
  min-energy Best-of-N as a reward path in `tools/run_rlvr.py`.

### O3 — fixed-point stability
- **Wire:** host `iterate_fixedpoint` inside `agent/realtime_grounding.py:ingest_one` as a
  pre-ingestion gate; use the real embedder.
- **Benchmark:** the C1 labelled fact pack via `agent/realtime_benchmark.py`; grounding quality via
  `tools/eval_grounded_search.py`.
- **Gate (`o3-fixedpoint-stability-not-validated-on-c1`):** residual/non-convergence gate rejects
  unsupported live claims with F1 beating the current admission arm at matched coverage.

### O4 — adaptive compute
- **Wire:** wrap the convergence stopping rule around the live self-consistency sampler in
  `agent/long_horizon.py`, bounded by its cooperative `deadline_monotonic`.
- **Benchmark:** `tools/eval_graded_confidence.py` on a real query set, adaptive-k vs fixed-k.
- **Gate (`o4-adaptive-compute-not-wired-into-long-horizon`):** adaptive-k matches fixed-k
  selective accuracy (paired CI contains 0) while cutting mean samples ≥ 25%.

### O5 — oscillator substrate  *(leave as simulation)*
- Do **not** take past simulation. Any hardware/energy claim needs a real oscillator-Ising mapping
  and a measured energy comparison — out of scope for a software repo. Confirm it stays
  `simulationOnly:true`.

---

## Phase 3 — Report back

Produce a short report. For each of O1–O4:
- dataset used + decontamination method (how you ensured no train/eval overlap);
- backend + model id, with proof it was not mock (`ModelResult.provider`);
- the gate metric with its confidence interval;
- **PASS / FAIL** vs the acceptance gate, and the exact command to reproduce;
- if FAIL: the honest negative result, not a fixture tuned until it passes.
For O5: confirm simulation-only.
Keep every new artifact `candidateOnly:true` unless a gate is met, and update the matching
failure-ledger row (Open → Resolved) only when it is.

**Related work you can build on:** the training-loss line (W1 process-reward model, W2 calibration
objective, W5 probe-as-loss) shares the same `build_hidden_state_featurizer` seam — see
`agi-proof/untapped-training-2026-07-01/CONTINUATION-PROMPT.md`. Implementing that seam once
unlocks O2, W1, and W5 together; it is the single highest-leverage task in this whole set.