# Architecture notes — what the MoE toy gestures at

`moe.py` is a nano top-1 mixture-of-experts layer. It is a **teaching toy**, not a
reproduction. This note records the real designs it points at — the ones a DeepSeek
pretraining-architecture researcher works on — so the gap between toy and frontier is
explicit and the toy isn't mistaken for the real thing.

## DeepSeek-MoE (fine-grained experts + shared experts)

The production idea, in three parts:

1. **Fine-grained experts.** Instead of a few large experts, split the FFN into many small
   experts and route to top-`k` of them. Finer granularity lets the router compose
   specializations rather than pick one monolith — more effective capacity per active FLOP.
2. **Shared experts.** A small number of experts are *always on* for every token, capturing
   common structure, so the routed experts specialize on the residual. This reduces
   redundancy (every routed expert re-learning the basics) and stabilizes training.
3. **Load balancing.** Auxiliary balance signals (and, in later DeepSeek work, an
   *auxiliary-loss-free* bias-adjustment scheme) keep tokens from collapsing onto a few
   experts. Our toy uses a crude balance penalty in `MoELM.train_step`; the failure mode it
   targets — routing collapse — is real and is what `run_arch.py` measures via
   `load_balance()`.

What the toy keeps: top-1 routing, per-expert FFN blocks, ~constant active params as total
params grow, an explicit anti-collapse term. What it omits: shared experts, top-`k>1`,
expert-parallel sharding, the real auxiliary-loss-free balancing, and any attention stack.

## MLA — Multi-head Latent Attention

DeepSeek's other signature is **MLA**, which attacks the KV-cache bottleneck of inference.
Standard multi-head attention caches a key and value vector per head per token; at long
context and large batch this dominates memory and bandwidth. MLA **compresses keys/values
into a low-rank latent** that is cached instead, and reconstructs per-head K/V on the fly.
The cache shrinks by a large factor with little quality loss — a software/hardware
co-design win ("软硬件协同地设计强大和高效的模型结构").

This package does **not** implement attention (the nano model is a fixed-window MLP), so MLA
is documented, not coded. It is the natural next artifact if this line of work continued: a
nano attention block with a low-rank KV latent, measuring cache size vs perplexity — the
same fit-and-check methodology used in `scaling/`, applied to an architecture lever.

## Why document instead of overclaim

Naming MLA and DeepSeek-MoE here, and *not* pretending the toy implements them, is the
point. The honest deliverable is: a working sparse-routing experiment + a clear-eyed
account of the real designs and what it would take to actually study them at scale.

## References (for the reader, not fetched here)

- DeepSeek-MoE: fine-grained expert segmentation + shared-expert isolation.
- DeepSeek-V2 / V3: MLA, auxiliary-loss-free load balancing, MTP (multi-token prediction).

(Citations are by description; verify against the primary papers before relying on details.)
