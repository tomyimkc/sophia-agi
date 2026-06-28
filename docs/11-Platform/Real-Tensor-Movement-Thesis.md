# Moving Real Tensors — a thesis for closing the policy→tensor gap, and one idea worth testing

**Status:** research + design thesis (no capability claim; `canClaimAGI` stays `false`).
Companion to [Governed-Scaling.md](Governed-Scaling.md), [DGX-Spark-Maximization.md](DGX-Spark-Maximization.md),
and [DSpark-Applicability.md](DSpark-Applicability.md).

> **The question.** Given the repo's current state, what is the honest thesis for
> "moving real tensors" (versus the opaque-byte policy stubs), and what is *one
> ultra-creative, repo-native idea worth testing*?
>
> **The short version.** (1) You already move real tensors — in the *kernel* layer.
> The gap is the *serving-policy* layer. (2) On the DGX Spark's **unified memory**,
> the entire tiering model the policy layer assumes is the wrong cost model — there
> is no PCIe bus to cross, so "move a tensor" means *bytes read by the GEMM*, not
> *bytes copied over a bus*. (3) That reframing unlocks an idea the offloading SOTA
> structurally cannot use: **Governed Speculative Sparsity** — use a cheap draft pass
> to *not read* most of the weights, and make the skip **provably lossless** with the
> same accept/reject that speculative decoding already runs.

---

## 1. Where the repo actually stands (diagnosis)

| Layer | File(s) | Moves real tensors? |
|---|---|---|
| **Kernel** | `kernels/src/nvfp4_gemm.py` | **Yes.** Triton fused dequant-GEMM streams packed 4-bit weights on CUDA, FP32 accumulate, validated `rel_err < 5e-2` vs the NumPy oracle, self-rooflined vs the Spark profile. |
| **Quant scheme** | `moe/quant.py`, `moe/adapt.py` | Real arrays; produces the bytes the kernel streams. Bits allocated by measured output-KL sensitivity. |
| **Serving policy** | `serving/expert_offload.py`, `serving/layer_stream.py`, `serving/kv_cache.py` | **No.** Payload is an *opaque byte count*; tracks tier transitions, LRU, prefetch hits, byte budgets — the *decision logic*, not the data. |
| **Measurement gate** | `serving/lowram_eval.py` | Real arrays (KL + top-1 vs FP16); the no-overclaim contract. |

So "move real tensors" is **not** a from-scratch problem. It is: **bind the policy
layer's `register/route_select/step` decisions to real `torch` tensors** (mmap/
safetensors load → pinned host buffer → device), preserving each module's existing
`offline_invariants()` as the correctness contract. The kernel is the proof that the
hard part (a correct, rooflined GPU compute path) is already done.

### The naive bridge (and why it's a trap on the Spark)

The obvious move: make `_promote_to_gpu` do `tensor.to('cuda', non_blocking=True)`
from a pinned buffer, double-buffered on a side CUDA stream (AirLLM v2.5's
overlap-load-with-compute). That is correct **on a discrete GPU** — and it is what
[AirLLM](https://github.com/lyogavin/airllm), [MoE-SpeQ](https://arxiv.org/abs/2511.14102),
and [SP-MoE](https://arxiv.org/abs/2510.10302) optimize: hide the **PCIe/host-DRAM
transfer latency** behind compute.

**On the GB10 Spark it is largely a no-op — and that is the whole insight.**

---

## 2. The unified-memory reframing (the load-bearing correction)

`DGX-Spark-Maximization.md` already states the hardware truth: **128 GB unified
LPDDR5x at 273 GB/s**, shared by Grace CPU and Blackwell GPU. Consequences the
current tiered policy does *not* yet encode:

- **There is no PCIe bus.** Experts/layers in "CPU DRAM" are in the *same physical
  pool* the GPU reads. Promoting CPU→GPU is not a copy across a bus; with managed/
  unified allocations it is at most a page-migration hint (`cudaMemPrefetchAsync`),
  often nothing. **The `GPU`/`CPU` tiers in `expert_offload.py` collapse into one
  "resident" tier; only `DISK` is a genuine other tier.**
- **The bottleneck is not transfer latency — it is bandwidth.** The dominant cost
  per decode token is *bytes the GEMM reads from the 128 GB pool*. `nvfp4_gemm.py`
  already encodes this (0.5 B/elem is the lever; > ~1830 FLOP/byte to escape the
  wall). Every offloading paper's core trick — *prefetch to hide I/O* — optimizes a
  latency that the Spark has mostly designed away.
- **Therefore the only tensor movements that matter on the Spark are:** (a) **cold
  load** disk→pool once, and (b) **per-token bytes read by the GEMM** from the pool.
  (a) is amortized; (b) is the entire decode-speed game.

**This flips the optimization target.** On a discrete GPU you ask *"how do I move the
needed bytes in time?"* On the Spark you ask *"how do I read fewer bytes at all?"* —
because everything is already resident. That question is what the offloading SOTA
cannot answer (their bytes are pinned by the model; their win is *when*, not *how
many*), and it is exactly where Sophia's equivalence-proof discipline has an edge.

---

## 3. State of the art (for honesty about what's new)

| System | Core trick | Setting it assumes |
|---|---|---|
| AirLLM | Layer-by-layer stream, prefetch overlap, 4/8-bit blocks | Tiny VRAM + disk; I/O-bound; ~0.7 tok/s |
| MoE-SpeQ (arXiv:2511.14102) | Draft model predicts *future experts* → prefetch from host to hide PCIe I/O | Discrete GPU, expert offload over PCIe |
| SP-MoE (arXiv:2510.10302) | Spec-decoding-aware expert prefetch via draft/target structural correspondence | Discrete GPU, PCIe-bound |
| Hiding-Offload-Latency (arXiv:2508.21706) | Use spec decoding to overlap offload latency | Discrete GPU, PCIe-bound |
| LMCache / Nexus | KV-cache tiering / intra-GPU prefill-decode disaggregation | Datacenter serving |
| Deja Vu (contextual sparsity) | Predict which heads/MLP channels are active, skip the rest | Learned predictor, **no correctness certificate** |

**Two gaps no one above closes, both native to this repo:** (1) a *bandwidth-first,
no-transfer* cost model (unified memory), and (2) read-skipping that is **provably
lossless**, not just empirically tolerable. The idea below sits exactly in that gap.

---

## 4. The idea worth testing — **Governed Speculative Sparsity (GSS)**

> **One line.** Use a cheap draft pass to decide *which weights the target doesn't
> need to read this token*, then use the speculative accept/reject as an
> **equivalence certificate** that the skip didn't change the output distribution —
> turning "read fewer bytes" into a *provably lossless* operation on a
> bandwidth-bound, unified-memory machine.

### 4.1 The mechanism

1. **Draft = a 4-bit (NVFP4) self-pass** of the target model — the bytes
   `nvfp4_gemm.py` already streams (0.5 B/elem). It produces (a) candidate tokens
   *and* (b) a predicted **read-set**: the experts (`moe/router.py` top-k), and
   per-layer the channels/heads, that actually move the logits for *this* context.
2. **Target verifies on the pruned read-set.** The high-precision pass reads only the
   predicted read-set at full bits and treats the rest as their cheap/structured-zero
   contribution — a *contextually sparse* forward pass. Far fewer bytes cross the
   273 GB/s wall.
3. **The certificate.** Run the *standard speculative-decoding accept/reject* on the
   target's (pruned) distribution vs. the draft's proposals. Accepted tokens inherit
   speculative decoding's exact guarantee: **the realized output distribution equals
   the un-pruned target's.** A rejection triggers a *single* full-read correction
   step for that position. So the skip is never a silent approximation — it is
   lossless-by-verification, or it is corrected.

The novelty vs. §3: the SOTA uses the draft to decide *what to prefetch* (move bytes
earlier). GSS uses the draft to decide *what not to read at all* (move fewer bytes),
and unlike Deja Vu it carries a **correctness certificate** instead of a tolerance.
On unified memory those are different problems — and only the second one helps.

### 4.2 Why it is *Sophia's* idea, not a lab's

GSS is the [Governed-Scaling.md](Governed-Scaling.md) thesis made literal on the
metal — it instantiates all four governors at once:

- **Promote only what verifies** → a weight is *read at full precision* only when the
  draft's read-set selects it; the selection is the verification signal (the exact
  shape of `expert_offload.py`'s promote-on-route, now over *read bytes*).
- **Optimize only with an equivalence proof (the `flash == naive` bar)** → the
  accept/reject *is* the proof. GSS must clear "pruned == dense within ε," the same
  bar `tests/test_flash_attention.py` already enforces.
- **Make over-reliance measurable** → the per-prompt **acceptance rate is a live
  meter** of how compressible this context is. Feed it back to `moe/adapt.py`: where
  acceptance is high, *lower the bits / shrink the read-set further*; where it drops,
  *spend bytes*. Bit-depth becomes an online, governed control loop instead of an
  offline sensitivity table.
- **Bound drift** → the rejection rate bounds, per token, how far the cheap path
  strayed from trusted ground; it is a directly logged quantity, not an afterthought.

### 4.3 The cost model that decides if it wins (roofline-first)

On the Spark, decode time per token ≈ `bytes_read / 273 GB/s`. Let the target read
`B` bytes/token dense, with block size γ (drafted tokens per verify). GSS pays, per
**block**:

```
draft:   γ · 0.25·B   (γ *autoregressive* 4-bit self-passes; weights re-read each
                       token — drafting can't batch)
verify:      ρ·B      (ONE parallel target pass over the γ-token block; weights
                       amortized across the batch dimension)
over:        k        (tokens produced per block, k = (1−α^(γ+1))/(1−α))
```

Per produced token ≈ `B·(γ·0.25 + ρ)/k`, so GSS **wins iff `(γ·0.25 + ρ)/k < 1`**.
(An earlier sketch wrote `(0.25+ρ)/k`, collapsing the γ draft passes to one — that is
optimistic; the γ-aware form above is what `serving/gss_feasibility.py` computes and
the number to trust.) The knee is entirely (ρ, k, γ), all measurable offline before any
GPU spend. If they are poor, GSS provably can't beat dense and you don't ship it — the
cost model is itself fail-closed.

> **Lossless vs. aggressive — the distinction Tier 1 forces (`serving/gss.py`).** The
> `verify: ρ·B` term above is the *aggressive* version: it reads only the predicted
> read-set. Building the mechanism proved (to machine epsilon) that speculative
> accept/reject is exactly lossless **only when it verifies against the *dense* target**
> — verifying against a pruned target drifts the output by exactly `KL(dense ‖ pruned)`
> (`verify_drift`). So there are two honest numbers:
> - **Lossless GSS** = cheap 4-bit draft + **dense** verify ⇒ cost `(γ·0.25 + 1)/k`. On
>   the OLMoE result (γ=4, k≈4) that is **~2×**, *certified bit-exact*.
> - **Aggressive GSS** = pruned verify ⇒ cost `(γ·0.25 + ρ)/k`, the **~3.6× ceiling** —
>   but it carries a bounded error `KL(dense ‖ pruned)` and needs a periodic dense
>   correction to stay honest. Tier 0's ceiling is this aggressive bound.
>
> So: **~2× is the guaranteed-lossless win; ~3.6× is the aggressive ceiling with a
> measured error budget.** The mechanism makes the gap falsifiable instead of letting a
> headline hide it — exactly the `Governed-Scaling` equivalence bar.

**This is now built (Tier 0): [`serving/gss_feasibility.py`](../../serving/gss_feasibility.py).**
Its CI invariants demonstrate both regimes on synthetic activations: a concentrated
read-set + faithful draft → GO (ρ=0.06, k=3.8, cost_ratio=0.28, **3.6× ceiling**); a
diffuse read-set + poor draft → NO-GO (cost_ratio=1.66, the kill switch fires). Feed it
real `(contribs, target_probs, draft_probs)` arrays from a forward pass to get the
go/no-go for *your* model.

### First real-checkpoint measurement — OLMoE-1B-7B on a RunPod GPU

`tools/runpod_gss_probe.py` (dispatched via `.github/workflows/runpod-gss-probe.yml`) ran
the probe on **`allenai/OLMoE-1B-7B-0924`** (64 experts, top-8 routing, 16 layers) on a
rented GPU: full-precision pass vs a real **4-bit bitsandbytes** self-draft. Report in
[`agi-proof/benchmark-results/gss-allenai-OLMoE-1B-7B-0924.json`](../../agi-proof/benchmark-results/gss-allenai-OLMoE-1B-7B-0924.json):

| run | n (positions) | ρ (read-set) | α (4-bit accept) | k (γ=4) | cost_ratio | ceiling | verdict |
|---|---|---|---|---|---|---|---|
| short prompt | 10 | 0.0959 | 0.915 | 4.22 | 0.260 | 3.85× | GO |
| **~120-tok prompt** | **103** | **0.0960** | **0.883** | **3.96** | **0.277** | **3.61×** | **GO** |

Only ~9.6% of expert weights carry 90% of each token's output mass, and a 4-bit self-draft
agrees with FP16 ~88% of the time — exactly the (low ρ, high α) corner where GSS wins. The
structure GSS needs **is present in a real frontier-style MoE**, not just the toy. **ρ is
essentially identical (0.096) across both runs** — the read-set concentration is a stable
property of the model, not a small-sample artifact; α settles slightly lower on the larger,
more diverse sample (the honest direction). Both runs clear the gate with a ~3.6× ceiling.

**Honest caveats (this is *illustrative*, not registered):** still **first-party**, and the
two runs share one model and one prompt family. A registered result needs **≥3 runs across
varied prompts/seeds + CIs** per the no-overclaim gate (`RESULTS.md`) — and the real-bytes
roofline win (Tier 2) is unmeasured. This is a *feasibility* GO over the cost model: it
greenlights Tier 1, it is **not** a speedup claim. `canClaimAGI` stays `false`.

(A debugging note for reproducers: the 4-bit draft must be loaded **after** the
full-precision target is freed — holding both resident makes `device_map=auto` offload the
quantized model to CPU and bitsandbytes aborts. `tools/gss_probe.py` frees the target first.)

---

## 5. Test plan (falsifiable, weakest-cheapest first; mirrors the repo)

**Tier 0 — Does the structure even exist? (CPU, no GPU, decides go/no-go cheaply.)**
On a small open MoE/dense LM, measure offline: (a) **read-set stability** ρ — for each
token, what fraction of experts/channels carry the top-X% of the logit movement; (b)
**self-draft acceptance** k — accept/reject of a 4-bit self-pass vs FP16. These two
numbers *alone* decide via §4.3 whether GSS can win. Land as
`serving/gss_feasibility.py` with `offline_invariants()`. **Gate:** `(0.25+ρ)/k < 1`
on real activations, or GSS is honestly abandoned here — no GPU spent.

**Tier 1 — Equivalence invariant (CI, synthetic + the flash bar).**
`serving/gss.py`: the prune + accept/reject + single-step correction, pure-Python over
arrays. Falsifiable invariants: (1) **lossless** — with the corrector on, GSS output
distribution == dense within ε (the `flash == naive` bar); (2) a draft that mispredicts
the read-set is *caught and corrected*, never silently accepted (fail-closed); (3)
reported `bytes_read_ratio` and `acceptance_rate` always travel **together with** the
equivalence verdict (the `LowRamReport` rule). CI-gated, model-agnostic.

**Tier 2 — Real kernel, real bytes (single Spark/GPU, no rented cluster).**
Extend `nvfp4_gemm.py` with a **gather-on-read-set** variant: the GEMM reads only the
selected expert/channel tiles. A/B vs the dense NVFP4 kernel, both **rooflined against
273 GB/s** with ≥3 runs + dispersion (the `kernels/bench/roofline.py` discipline).
**Gate:** measured bytes-read reduction at `rel_err < 5e-2` vs the dense reference —
report **% of the Spark roofline**, never "Nx vs a strawman."

**Tier 3 — Gated end-to-end (RunPod source-of-record / Spark capability lane).**
Wire GSS through the low-RAM runtime (#221) and run the §4.3 loop live: dense vs GSS,
tokens/s + acceptance + bytes-read, **plus** the decisive correctness check from
[DSpark-Applicability.md](DSpark-Applicability.md) §4.2 — **the Sophia gate's
accept/abstain/block verdicts must be byte-identical** under GSS (lossless ⇒ identical
decisions). Apply the standard headline discipline (≥3 runs, CIs, equivalence gate
passes) before any number leaves *illustrative* status.

---

## 6. Honest boundaries

- **Novelty, stated precisely.** Draft-predicted *prefetch* is known (MoE-SpeQ,
  SP-MoE). The new, untested claims are: (a) reframing the target from *transfer
  latency* to *bytes-read* on **unified memory**, where prefetch is moot; and (b)
  making read-set pruning **lossless-by-verification** rather than tolerance-based
  (Deja Vu). Both are *hypotheses with a cheap Tier-0 kill switch*, not results.
- **Where it can fail.** If real read-sets aren't stable (ρ→1) or self-draft
  acceptance is low (k→1), §4.3 says GSS can't beat dense — and Tier 0 reveals that
  for the price of a CPU run. That is the design working, not failing.
- **Scope.** This is mechanism + a falsifiable path to measurement. No speedup is
  claimed here; `canClaimAGI` stays `false`. The deliverable is the certificate that
  any speed it *does* find cost no output fidelity — the repo's signature.

### Getting a real go/no-go on a checkpoint — `tools/gss_probe.py`

The feasibility meter consumes arrays; the probe harness extracts them from an actual
forward pass and calls the gate.

```bash
# Toy MoE, pure-numpy, no GPU — proves the extraction→gate pipeline (runs in CI):
python tools/gss_probe.py --backend moelm --experts 16 --tokens 64
#   → ρ=0.06  α=0.99  k=4.9  cost_ratio=0.21  ceiling=4.65×  → GO

# A real checkpoint (MoE → router contribs; dense → MLP-activation contribs):
python tools/gss_probe.py --backend hf --model Qwen/Qwen2-57B-A14B \
    --prompt "…" --draft bnb --out agi-proof/benchmark-results/gss-<model>.json
```

The `hf` backend uses `output_router_logits` for MoE read-sets and a 4-bit self-draft
(`--draft bnb` = `bitsandbytes` `load_in_4bit`, the `tools/train_lora.py` path; or
`fakequant` = device-agnostic int4 round-trip). It **skips cleanly (exit 0)** without
torch/transformers, like `kernels/src/nvfp4_gemm.py`. The verdict is *feasibility*, never
a speedup — a GO greenlights Tier 1, nothing more.

### Suggested build order
1. **Tier 0** `serving/gss_feasibility.py` — measure (ρ, k); cheapest possible go/no-go.
   ✅ **Built** — pure-numpy meter, CI-gated (`tests/test_gss_feasibility.py`), fail-closed;
   `tools/gss_probe.py` extracts (contribs, target, draft) from a real forward pass
   (numpy `MoELM` now; HF/bitsandbytes for real checkpoints).
2. **Tier 1** `serving/gss.py` — the prune + speculative accept/reject + equivalence gate.
   ✅ **Built** — the lossless core is proven exact (`speculative_realized == dense` to
   1e-12) and `verify_drift` makes the pruned-verify error falsifiable; CI-gated
   (`tests/test_gss.py`). Within/across-run CIs in `serving/gss_feasibility.py`
   (`feasibility_with_ci`, `aggregate_runs`).
3. **Tier 2** gather-on-read-set kernel in `kernels/src/` — the rooflined bandwidth win.
4. Fold the acceptance-rate meter back into `moe/adapt.py` as the online bit-depth loop.
