# DSpark / DeepSpec Applicability to Sophia

> **Question.** Does the DeepSeek **DSpark** speculative-decoding theory (paper
> arXiv:2606.19348; **DeepSpec** is the open training/eval codebase) apply to this
> repo, given what the active Claude sessions are building?
>
> **One-line answer.** It applies *strongly and synergistically* to the low-RAM
> **serving** stack you are currently developing, has **no effect** on the core
> provenance/abstention thesis, and rhymes with the gate architecture only as an
> **analogy**. This doc states which, why, and a feasible, no-overclaim plan to
> integrate and benchmark it.

This is an **applicability analysis + plan**, not a capability claim. Nothing here
licenses a speedup number; the speedup is a thing to *measure* against the gate in
§4, exactly as `serving/lowram_eval.py` measures the low-RAM claim.

---

## 1. What DSpark actually is

Speculative decoding for **serving latency**, with the defining property that it is
**distribution-preserving** — it changes *how fast* tokens are produced, never
*which* tokens:

- **Draft → verify → accept/reject.** A small **draft model** proposes several
  tokens; the large **target model** verifies them in one batched forward pass;
  accepted tokens are kept, the first rejection resamples from the corrected
  distribution. Output distribution is provably unchanged.
- **Semi-autoregressive drafting.** Mostly-parallel proposal with light sequential
  coupling → longer accepted runs than purely-parallel drafting.
- **Load-aware confidence scheduling.** Verify more speculative tokens when GPUs are
  idle; get stricter under high concurrency. Trades acceptance for tail-latency
  under load.
- **Reported result.** 60–85% faster per-user generation over the **MTP-1** baseline
  on DeepSeek-V4.
- **Prerequisites.** A *real* GPU inference engine, a *trained* draft model, a
  KV-cache, and spare verification compute. DeepSpec ships data-prep, draft-model
  implementations (DSpark / DFlash / Eagle3), training code, and eval scripts.

Sources: paper arXiv:2606.19348 · DeepSpec repo (`deepseek-ai/DeepSpec`) ·
[MarkTechPost summary](https://www.marktechpost.com/2026/06/27/deepseek-releases-dspark-a-speculative-decoding-framework-that-accelerates-deepseek-v4-per-user-generation-60-85-over-mtp-1/).

---

## 2. Three layers of this repo — does it apply?

| Layer | Applies? | Why |
|---|---|---|
| Low-RAM serving (`serving/`, PRs #219/#221, `moe/`) | **Yes — synergistic** | Spec-decoding amortizes layer-streaming's *per-token* disk cost; load-aware scheduler fits your tiered-memory budgeting. |
| Core provenance / abstention thesis (the Wisdom Gate) | **No** | Spec-decoding preserves the output distribution → **zero** effect on hallucination / abstention metrics. |
| Gate *architecture framing* (`claim → verify → accept·abstain·block`) | **Analogy only** | Same propose-then-verify *shape*; different object (token identity vs. epistemic grounding). |

### 2.1 Serving layer — the real, non-obvious synergy

Your headline low-RAM lever, `serving/layer_stream.py`, **re-streams every layer
from disk *per token*** (it says so at `layer_stream.py:11`: "every token re-streams
every layer"). That makes the target-model forward pass **I/O-bound**.

Speculative decoding amortizes **one** target forward pass over **k** accepted
tokens. So when the forward pass is the bottleneck — which layer-streaming
deliberately makes it — you re-stream the layer stack **once to verify k tokens
instead of k times**. The two levers **multiply** rather than add:

```
naive stream:        stream(N layers) × T tokens
+ spec decoding:     stream(N layers) × ceil(T / k_accepted)
```

Concretely the integration points already exist in-repo:

- `serving/layer_stream.py` / `serving/expert_offload.py` — the streamed/tiered
  forward pass that the verify step rides on; their byte-accounting is exactly the
  signal DSpark's **load-aware scheduler** consumes (resident budget → how many
  speculative tokens to verify this step).
- `serving/kv_cache.py` / `serving/kv_quant.py` — speculative decoding needs the
  KV-cache to roll back rejected tokens; your paged/quantized KV is the substrate.
- `moe/router.py` + `serving/expert_offload.py` — for an MoE target, the draft can
  also *predict the expert route*, pre-warming `expert_offload` promotion (a
  repo-specific extension beyond vanilla DSpark).
- `docs/11-Platform/Inference.md` already lists `vLLM --speculative-model … 
  --num-speculative-tokens 5` and `ARCHITECTURE.md` already cites MTP — so this is
  an *extension of a documented lever*, not a new direction.

**Honest blocker (do not skip).** The `serving/` modules are explicitly **policy
reference implementations that do not move real GPU tensors** (`layer_stream.py:30-35`
— payloads are opaque byte sizes). There is no real inference engine *in-repo* to
bolt a draft model onto. So a *measured* speedup is a **deployment-time** result
(vLLM/SGLang or DeepSpec directly) and is gated on the **same real-GPU blocker** as
the OPEN RLVR / live-weight items in `agi-proof/failure-ledger.md`. In-repo we can
build and CI-test the **policy + the measurement gate**; the GPU number comes from a
RunPod run (§4.3).

### 2.2 Core thesis — explicitly does **not** apply

The validated Sophia result is attribution-hallucination reduction
(36.1% → 23.6%) and abstain-vs-fabricate behavior. Because speculative decoding is
distribution-preserving, running the gated model under DSpark leaves **every one of
those numbers bit-identical**. Do **not** expect — or claim — any accuracy or
hallucination effect from it. Its only deliverable is **latency/throughput/cost** at
**unchanged quality**. (This is a feature for the plan: distribution-preservation is
*itself* a falsifiable invariant, §4.1.)

### 2.3 Architecture framing — analogy, not theory transfer

DSpark: `draft → target verifies → accept/reject`. Sophia:
`claim → verify against sources → accept · abstain · block`. Both are
**propose-then-verify** with a cheap generator and a stricter verifier, and DSpark's
load-aware confidence scheduling even rhymes with the selective-prediction/coverage
gate. Borrowing the *framing* ("a verifier-gated decoder, at the semantic level") is
fair. Claiming "DSpark validates the gate" would be overclaiming — DSpark verifies
*token identity* for speed; Sophia verifies *epistemic grounding* for trust. Keep
them separate in any public wording.

---

## 3. Implementation plan (feasible, staged, no-overclaim)

The plan mirrors the repo's established discipline: a **pure-Python policy reference
module** with CI-gated falsifiable invariants (like `layer_stream.py` /
`expert_offload.py`), a **measurement gate** that fails closed (like
`lowram_eval.py`), and a **gated RunPod** lane for the one number that needs a real
GPU (like the #192 GRPO workflow / #221 launcher). Each phase is independently
useful and independently gated.

### Phase 0 — Decide the integration surface (no code)

Two non-exclusive paths; recommend **both**, in order:

1. **Deployment path (fastest real number).** Use DSpark via the engine the adapter
   already speaks to. `agent/model.py` is OpenAI-compatible, so a vLLM server with
   `--speculative-model <draft> --num-speculative-tokens k` (or a DeepSpec-trained
   Eagle3/DSpark draft) is a **drop-in** — *no Sophia code changes*. This yields the
   §4.2 benchmark immediately on any rented GPU.
2. **In-repo policy + gate path (CI-testable, owns the contract).** Add the
   reference modules below so the *scheduling policy* and the *correctness gate* live
   in-repo, CI-tested on synthetic data, model-agnostic — matching how every other
   `serving/` lever is structured.

### Phase 1 — `serving/spec_decode.py` (policy reference, pure-Python, CI-gated)

The **mechanism**, not a CUDA kernel. Opaque payloads (token ids + a distribution
the caller produced), dependency-light, deterministic.

- `class SpecDecodeScheduler` — given current resident-byte budget / concurrency
  (read from `expert_offload`/`layer_stream` accounting), choose `num_spec_tokens`
  this step → encodes DSpark **load-aware confidence scheduling**.
- `def verify(target_probs, draft_tokens, draft_probs) -> AcceptResult` — the
  standard speculative-decoding accept/reject + corrected resample. Returns
  `accepted_len`, `resampled_token`, and a flag that the realized distribution equals
  the target's (the lossless property).
- `def semi_ar_block(...)` — model the semi-autoregressive block length vs. a
  purely-parallel baseline (longer accepted runs).
- `def offline_invariants()` — CI entrypoint, same pattern as the other modules.

**Falsifiable offline invariants (CI-gated):**
1. **Lossless.** Over random target distributions, the accept/resample procedure
   reproduces the target's sampling distribution within MC error (this is the whole
   point — speed without distribution drift).
2. **Acceptance ∈ (0,1]**, and realized speedup tracks `accepted_per_pass`
   (acceptance ↑ ⇒ passes ↓).
3. **Load-aware bound.** Under high simulated load the scheduler verifies *fewer*
   speculative tokens (stricter) and never exceeds the resident budget — composing
   with `expert_offload`/`layer_stream` byte accounting.
4. **Semi-AR ≥ parallel.** Semi-autoregressive accepted-run length ≥ the
   purely-parallel baseline on the same draft agreement profile.
5. **Fail-closed.** A draft that *shifts the argmax distribution* is surfaced as a
   correctness regression by the §3.2 gate, never silently accepted.

### Phase 2 — `serving/spec_decode_eval.py` (the no-overclaim gate)

The **measurement**, mirroring `serving/lowram_eval.py` one-to-one. Because DSpark is
distribution-preserving, the correctness gate is an **equivalence** gate:

- Input: target greedy/sampled distributions vs. spec-decoded output distributions
  over the same positions, **plus** timing (`target_forward_passes`,
  `wall_clock_s`).
- Gate (`SpecDecodeGate`): **mean KL(target ‖ spec) ≈ 0** and **top-1 agreement =
  100%** on a held-out batch (lossless contract) — reuse `_row_kl` from
  `lowram_eval.py`. **Report `speedup` / `acceptance_rate` alongside the equivalence
  verdict, never one without the other** (same rule `LowRamReport` enforces for
  `mem_ratio`).
- Fail-closed on: any KL above ε, any argmax flip, missing timing, shape mismatch.

This makes "DSpark gave us X% speedup" a *gated* claim: it only headlines if
equivalence holds. A draft that quietly changed outputs fails the gate even if it was
faster.

### Phase 3 — Draft-model production (DeepSpec)

Use DeepSpec's training stack rather than reinventing it:

- Reuse your **distill→serve loop** (`Inference.md` §"Distill → serve loop",
  `tools/distill_export.py` / `tools/collect_traces.py`): the same verified teacher
  traces that train the LoRA student are training data for an Eagle3/DSpark **draft**
  head of that student.
- Target = the low-RAM Sophia-V1 student (PR #221); draft = a tiny head trained with
  DeepSpec. Wire the resulting draft into the vLLM `--speculative-model` slot.

### Phase 4 — Compose with the low-RAM frontier (the repo-specific win)

Run the draft+verify loop *on top of* `layer_stream` + `expert_offload` so the
amortization in §2.1 is realized: one streamed pass verifies k tokens. For the MoE
target, prototype **draft-predicted expert routing** to pre-warm `expert_offload`
promotion. This is the novel contribution beyond vanilla DSpark and the reason the
theory is more — not less — valuable here.

---

## 4. Benchmark test plan

Three tiers, weakest-but-cheapest first. Tiers 1–2 run in CI / on any box; Tier 3 is
the gated real-GPU number.

### 4.1 Tier 1 — Offline equivalence + policy (CI, no GPU, no keys)

- `python -m serving.spec_decode` and `python -m serving.spec_decode_eval` run the
  `offline_invariants()` (§3.1, §3.2) on synthetic distributions. Wire into the same
  CI lane as the other `serving/` modules. **Gate:** all invariants pass; equivalence
  holds at ε≈0. Proves the *policy and the contract* are correct independent of any
  model.

### 4.2 Tier 2 — Real engine, small model, single box (no rented GPU needed)

- Stand up two vLLM configs behind the adapter: **baseline** (no draft) vs.
  **+spec** (`--speculative-model <draft> --num-speculative-tokens k`), same student,
  same quant.
- Drive both with the existing harness — `tools/eval_agent.py --provider vllm` — which
  already aggregates `cost_usd` / `latency_sec` per `ModelResult` (`Inference.md`
  §"Cost/latency tracking").
- **Metrics:** tokens/s, p50/p95 latency, acceptance rate, **price-per-passed-task**;
  and the decisive correctness check — **run `tools/run_ablation_sophia.py` under
  both and confirm the Sophia gate verdicts are byte-identical** (distribution
  preservation ⇒ identical accept/abstain/block decisions). If they diverge, the
  draft integration is wrong, full stop.
- Sweep `num_speculative_tokens ∈ {2,4,6,8}` and draft size to find the
  acceptance/overhead knee. **Gate:** speedup > 1.0 at equivalence on this hardware,
  honestly reported with the config.

### 4.3 Tier 3 — Gated RunPod, low-RAM frontier (the headline number)

- Reuse the **gated launcher** pattern from PR #221 / the #192 GRPO RunPod workflow
  (`mcp__runpod__*` tools are available). One run = Sophia-V1 student, layer-streamed
  + expert-offloaded (§4 compose), baseline vs. +spec-decoding.
- Emit a report into `agi-proof/benchmark-results/` (same convention as the existing
  `seib-*` / `runpod-rlvr` artifacts) carrying **both** the speedup **and** the
  `spec_decode_eval` equivalence verdict.
- Apply the project's standard headline discipline: **≥3 runs**, CIs, and the
  equivalence gate must pass, or the number stays labelled *illustrative*. Until a
  third-party / multi-run pass exists, treat it like every other single-run serving
  result — mechanism shown, capability not yet certified.

---

## 5. Bottom line

- **Apply it to serving — and expect a *multiplicative* win with layer-streaming**,
  not a token saving. That is the strongest, most repo-specific reason DSpark is
  worth integrating here.
- **Do not connect it to the hallucination/abstention thesis** — it is provably
  neutral there; saying otherwise would fail your own no-overclaim gate.
- **Build the policy + equivalence gate in-repo (Phases 1–2, CI-testable now); get
  the speedup number from a gated GPU run (Tier 3).** The equivalence gate is what
  turns "faster" into a *measured, lossless* claim rather than an asserted one.

### Open blockers (for `agi-proof/failure-ledger.md` if pursued)

- No real in-repo inference engine — speedup requires a deployment-time run (shared
  with the existing GPU blocker).
- Draft model must be trained (DeepSpec) and *verified lossless* before any speedup
  headlines.
- Real-GPU benchmark needs ≥3 runs + CIs + passing equivalence gate to leave
  *illustrative* status.
