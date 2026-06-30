# Spark vs RunPod for the v5 QAT Adapter — Decision Doc

**Status:** decision doc, no capability claim; `canClaimAGI` stays `false`. Read with
`Cheap-Compute-Boundary.md`, `Spark-Cluster-Capacity.md`, and `OLMoE-NVFP4-Certification.md`.

> **Question.** The OLMoE NVFP4 low-RAM cert is NO-GO on `top1_agreement`. The v5 QAT-adapter
> lever is built but not yet run. Do we need RunPod (metered cloud GPU) to train and certify v5,
> or does the DGX Spark suffice?

---

## 1. Verdict — train v5 on the Spark

**Train the v5 QAT adapter on the DGX Spark. RunPod is NOT needed for capability.**

The Spark has already trained every adapter in this line — v1, v2, v3, and v4 — on its own
hardware, and the v5 lever (epochs 2→3 at the stable `λ=0.001`, output `olmoe-qat-spark-v5`,
plus the `--keep-suffixes` mixed-precision cert lever) is already coded and tested
(`tests/test_certify_keep_suffixes.py`, 5 pass). The only thing left is to run it.

**Evidence the Spark can do this — it already did it:**

| Run | Trained where | mean_kl | top1 | Note |
|---|---|---|---|---|
| v3 | DGX Spark | 0.0451 ✓ | 0.9062 | first valid expert-co-adapted run, val_loss 1.5012 |
| v4 | DGX Spark | 0.0537 | 0.9258 | 3 epochs, λ=0.01 — over-fit (`protected_max_kl` 0.71) |

(`agi-proof/benchmark-results/certify-lowram-olmoe-nvfp4-v3.json`, `...-v4.json`;
`OLMoE-NVFP4-Certification.md`.) Both ran end-to-end on the Spark: train (`tools/train_lora.py
--qat`) and certify (`tools/certify_lowram.py`) on the GB10. v5 is the *same workload* with one
more epoch — strictly within demonstrated capability.

**The fit is comfortable, not marginal.** The base is `allenai/OLMoE-1B-7B-0924-Instruct` —
6.92B total / ~1.3B active MoE. The Spark is a **128 GB** unified-memory box; per
`Spark-Cluster-Capacity.md` its 1-node 4-bit *fine-tune ceiling is ~70B* and *serve ceiling
~200B*. A ~7B sparse base + LoRA/QAT sits an order of magnitude under that ceiling. The same doc
names this exact run as a reference workload: "this repo's OLMoE-1B-7B QAT, 439 rows / 220 steps"
runs ~2.5–3 h on one Spark. There is no memory wall and no FLOP wall here.

**The cert blocker is the recipe, not the compute.** The cert fails on `top1_agreement` (0.906 v3 /
0.926 v4, bar ≥ 0.97), and the failure-ledger entries `nvfp4-v3-downproj-cert-t1-no-go-2026-06-30`
and `bench-b-cert-02` establish *why*: holding `down_proj` bf16, then `down_proj,gate_proj` bf16,
both left top1 pinned at **0.8945** — "Confirms the top1 gap is a **training/quant-depth** problem,
NOT a held-projection-count problem." That is a property of the **QAT recipe and the quant scheme**,
not of how much GPU you point at it. No amount of RunPod compute moves a quant-faithfulness ceiling.

---

## 2. When RunPod IS justified — deadline-bound recipe sweeps, and only then

RunPod buys exactly one thing the Spark cannot: **parallel breadth under a wall-clock deadline.**

The open levers for v5 are a small **recipe-search space**, not a single run:
- epochs (2 vs 3 vs more),
- `--qat-lambda` (v3 at 0.001 was best; v4 at 0.01 over-fit — the stable value is 0.001),
- mixed-precision keep-list (`--keep-suffixes`, e.g. hold `down_proj` and/or `gate_proj` bf16),
- target-module set / protected slice.

On **one** Spark these are explored **serially**. Each train+certify cycle is hours; the
no-overclaim gate then wants ≥3 seeds × ≥2 judge families (`RESULTS.md`), and
`Spark-Cluster-Capacity.md` notes that matrix "took this session *hours* serially." So:

**RunPod is justified when, and only when, a deadline forces the recipe sweep to finish in
parallel rather than serially.** RunPod (or a multi-Spark cluster) lets N recipe points run at
once — pure throughput. This is the same role `Spark-Cluster-Capacity.md` assigns to 4–8 Sparks
("8-wide parallel experiments," "the seed×family matrix in parallel") and to cloud GPU
(`tools/runpod_*.py` launchers): **parallel breadth, not single-run capability.**

If there is **no deadline**, the Spark runs the same sweep for free, serially. RunPod's only
deliverable over the Spark is *speed*, and speed is only worth metered dollars when the calendar,
not the hardware, is the binding constraint.

---

## 3. The honest ceiling — RunPod cannot fix a quant-scheme wall

This is the load-bearing caveat. **If the recipe sweep fails to reach `top1 ≥ 0.97`, the next move
is to change the quant scheme — and no GPU, Spark or RunPod, can do that for you.**

The evidence says the gap is in quant *depth*: full-NVFP4 v3 ≈ 0.906, and holding one or two
projections bf16 only nudged it to 0.8945 (`bench-b-cert-02`). When a recipe sweep (epochs / λ /
keep-list / seeds) cannot lift top1 over the floor, the remaining levers are **scheme-level**:
- **more bits** — a higher-precision served scheme than NVFP4 (e.g. wider mixed precision, INT8 on
  the sensitive projections), or
- **fewer quantized layers** — keep more of the sensitive projections in bf16 via the keep-list,
  trading the `mem_ratio` (currently ~3.3×) back down.

Both of these are **changes to the certification target**, i.e. to what "low-RAM served artifact"
*means*, not to the optimizer or the box it runs on. A quant scheme that loses too much next-token
agreement at NVFP4 will lose it on any GPU. **RunPod does not raise the NVFP4 faithfulness ceiling;
it only finds the best recipe under that ceiling faster.** Per `Cheap-Compute-Boundary.md`
Boundary 3, the claim that ships is a *measured-error-bounded* one — and if the scheme cannot meet
the bar, the honest output is NO-GO plus the aggregate fidelity claim ("served-quant retains BF16
next-token behavior to `mean_kl ≤ 0.05`"), never an overclaim. `canClaimAGI` stays `false`.

---

## 4. Cost note — Spark free, RunPod metered

- **Spark: free at the margin.** Owned hardware, no per-hour meter. The standing rule is *cheap
  adaptation* (`Cheap-Compute-Boundary.md`, Boundary 2) — the prior gate-disciplined adapter ran
  ~$0.67-equivalent of work; the v5 run is comparable. Running the full recipe sweep serially on
  the Spark costs only wall-clock time, not dollars.
- **RunPod: metered per GPU-hour.** Every recipe point in a parallel sweep is billed. RunPod's
  spend is justified *only* by the deadline value of finishing the sweep in one wall-clock pass
  (§2). Spending it to overcome a quant-scheme ceiling (§3) is wasted money — the ceiling does not
  move.

**Decision rule:** Train v5 on the Spark. Reach for RunPod (or multi-Spark) **only** to parallelize
a recipe sweep against a deadline — never as a fix for the `top1` ceiling, which is a quant-scheme
property no GPU count can buy past.

---

**Sources:** `docs/11-Platform/Cheap-Compute-Boundary.md` (Boundary 2 cheap-adaptation, Boundary 3
measured-error bar); `docs/11-Platform/Spark-Cluster-Capacity.md` (128 GB, ~70B 1-node fine-tune
ceiling, OLMoE-1B-7B QAT ~2.5–3 h reference, parallel-experiment value of more nodes, cloud-GPU
launchers for what the Spark can't do); `docs/11-Platform/OLMoE-NVFP4-Certification.md` (v3/v4
trained-on-Spark results, top1 < 0.97 failure); `agi-proof/benchmark-results/certify-lowram-olmoe-nvfp4-v3.json`,
`...-v4.json`; `agi-proof/failure-ledger.md` (`nvfp4-v5-mixed-precision-lever-built-2026-06-29`,
`nvfp4-v3-downproj-cert-t1-no-go-2026-06-30`, `bench-b-cert-02` — top1 0.8945 is a training/quant-depth
ceiling, not a held-projection-count problem).
