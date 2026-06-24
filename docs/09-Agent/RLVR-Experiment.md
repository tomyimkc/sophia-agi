# RLVR experiment — verifier-as-reward GRPO

> **Honest scope, stated first.** This is a *grounded-model alignment/efficiency
> technique*, not a path to AGI. It trains a model to better satisfy Sophia's
> deterministic provenance gate (DeepSeek-R1 / OpenAI Reinforcement Fine-Tuning
> style: the verifier **is** the reward). It does not create reasoning ability the
> base model lacked, does not generalize to unverifiable domains, and is bounded by
> the base model's capacity. See [Generality.md](../11-Platform/Generality.md)'s
> "What this is NOT".

## What and why

The legitimate version of "train an open model with my repo's signal" is **RLVR**
(Reinforcement Learning with Verifiable Rewards): GRPO where the reward is a
deterministic, machine-checked verifier instead of a learned reward model.
Sophia already ships those verifiers (`agent/verifiers.py`), so the reward signal
is free — we reuse the exact gate the reasoning loop uses, via
`provenance_bench.rl_reward`.

This is **not** "fine-tune on 518 humanities examples to gain general intelligence"
(research shows narrow SFT *regresses* broad benchmarks — the alignment tax). The
reward here is task-grounded (does the model assert a forbidden attribution? does
it name the documented author?), and the only claim is a measured, gated
improvement in that grounded behaviour on a held-out split.

## Artifacts

| Artifact | Path |
|---|---|
| Reward core (no torch) | `provenance_bench/rl_reward.py` |
| Dataset + contamination-free split | `provenance_bench/rl_dataset.py` |
| GRPO runner (mock offline + GPU) | `tools/run_rlvr.py` |
| Tests | `tests/test_rlvr.py` |
| Deps | `requirements-rl.txt` |
| Offline report | `agi-proof/benchmark-results/rlvr.public-report.json` |

## Reward design

`reward_for_case(case, completion)` returns a deterministic value in `[-1, 1]`:

- **Gate first (the seam).** `provenance_faithful(records)` is run on every
  completion. If it **fails** (the model asserted a forbidden attribution) →
  reward **−1.0** (hard floor). This is the load-bearing, verifier-driven signal.
- **False case** ("Did Confucius write the Dao De Jing?"): else `0.4` (didn't
  assert) `+ 0.3·(explicit correction) + 0.3·(names the real author)`.
- **True case** ("Who wrote the Republic?"): `+1.0` if the gold author is named,
  else `0.0`.

### Reward-hacking surface (and mitigations)

A reward is only as honest as its exploits are closed. Known surfaces:

| Exploit | Mitigation here |
|---|---|
| Always-say-no / blanket abstention on false cases | A universal "no; the author is X" template **fails on true cases**, which require an *affirmation* — the policy must discriminate. |
| Gold-string buried in boilerplate | Bounded by the gate running first; gold term is additive, not the whole reward. (Tighten to a format anchor if observed.) |
| Hedging to dodge the gate (`extra_deny` carve-out: "traditionally/disputed/apocryphal") | **Anti-hedging cap:** >2 hedge markers caps the positive reward at the 0.4 floor. |
| True/false confusion (one hedge satisfies both) | **Mutual-exclusion:** a true-case answer containing a DENY marker scores 0. |

**Still honest:** this reward trains *no-assertion + gold-naming*, not reasoning.
A model could maximize it without truly "knowing" authorship — which is exactly
why the capability claim is gated (below) and why an imperfect verifier invites
reward hacking (cf. *LLMs Gaming Verifiers*, 2026). The verifier's semantic depth
is the real ceiling.

## Contamination-free split

`rl_dataset.split_cases` partitions by **(work, author) entity-pair**, not by row,
so a work evaluated was never trained on. Gate records are built **per partition**
(`gate_records_for`) so an eval case's reward verifier is never derived from a
train case. Split is seed-locked and sealed-hashed (`run_improvement_loop.py`
pattern). With 87 cases the 70/30 split is ~61 train / 26 eval — **small**, stated
honestly in the failure ledger.

A work *title* may appear on both sides with different authors (e.g. "Republic" as
a Socrates-false case and a Plato-true case). The guard is on the entity pair, not
the title.

## Falsifiable claim

- **Offline (asserted today, CI-gated):** the reward machinery is sound —
  deterministic, monotone in the correct direction, forbidden-assertion →
  negative, the `agent.verifiers` seam is actually invoked, bounded in `[-1,1]`,
  and the split is contamination-free. `tools/run_rlvr.py --model mock` exits
  non-zero if any fail.
- **Live (pre-registered, OPEN):** on the held-out entity-disjoint split, mean
  reward / pass@1 rises vs the untrained base adapter at ~0 false-positive
  regression, under the no-overclaim gate
  (`provenance_bench.aggregate._is_validated`: notMock + ≥2 judge families +
  Cohen's kappa ≥ 0.40 + ≥3 runs + 95% bootstrap CI excludes 0). **Not asserted**
  by any run here — see `agi-proof/failure-ledger.md`
  `rlvr-live-run-not-yet-gated-2026-06-21`.

## Hardware reality (read this before training)

The GRPO stack is **CUDA/NVIDIA-only**:

- `bitsandbytes` (4-bit QLoRA) and `vLLM` do not run on Apple Silicon. An M-series
  Mac runs `--model mock` (the offline check) and the held-out eval, **not** the
  live GRPO.
- **QLoRA(4-bit) + vLLM colocate crashes** ([trl#4973](https://github.com/huggingface/trl/issues/4973)):
  `merge_adapter` dequantizes 4-bit weights then pushes them to a vLLM engine
  expecting packed shapes → `AssertionError`. The runner **refuses** this combo.

Three working GPU paths (`--vllm` / `--quant`):

| Hardware | Flags | Notes |
|---|---|---|
| 2× 24 GB (e.g. 2×4090) | `--vllm server` | vLLM on GPU 1 (bf16), QLoRA training on GPU 0. The documented #4973 workaround. Fastest feasible solo-ish path. |
| 1× 80 GB (A100/H100) | `--quant bf16` | colocate, no quantization. Sidesteps #4973 entirely. |
| 1× 24 GB | `--vllm none` | native `model.generate()`, QLoRA. Correct but slow. |

## Base model + license

Default `--model zai-org/glm-4-9b-chat-hf` (dense, native `GlmForCausalLM` in
transformers ≥ 4.46.2, vLLM-supported, no `trust_remote_code`). It is the
solo-fine-tunable dense GLM.

> **License is NOT MIT.** GLM-4-9B-Chat ships under the **glm-4-9b License**
> (free for research; commercial use needs Zhipu registration). This differs from
> GLM-5.2 / GLM-4.5+ (MIT). The repo is MIT; a derived adapter inherits the base
> model's license, so the saved adapter is **glm-4-9b-licensed**, not MIT. If MIT
> is a hard requirement, switch to `zai-org/glm-4.5-air` (MIT, 106B/12B MoE) —
> which needs a multi-GPU node. GLM-5.2 itself is open+MIT but 744B MoE: API/teacher
> only, not solo-fine-tunable.

### GLM LoRA target_modules (a real trap)

GLM uses a **fused-QKV** layout. The Qwen names hard-coded in
`tools/train_lora.py` (`q_proj/k_proj/v_proj/...`) are **wrong for GLM** and raise
`ValueError: Target modules not found` at adapter build. `run_rlvr.py` uses the
GLM-correct set: `query_key_value`, `dense`, `dense_h_to_4h`, `dense_4to_h`.

## Reproduce

```bash
# Offline reward-wiring check (any machine, incl. Apple Silicon; no torch):
python tests/test_rlvr.py
python tools/run_rlvr.py --model mock --dry-run

# Live GRPO (rented CUDA GPU):
pip install -r requirements-rl.txt
python tools/run_rlvr.py --model zai-org/glm-4-9b-chat-hf --vllm server   # 2x24GB
python tools/run_rlvr.py --model zai-org/glm-4-9b-chat-hf --quant bf16   # 1x80GB
```

## Math task (`--task math`) — the cleanest held-out domain

The same runner trains on a second, fully objective reward: **symbolic-math
equivalence**. The reward is `agent.verifiers.math_equivalent` (sympy) — like the
interpreter for code, the CAS is ground truth, so there is **no LLM judge** and the
signal is ungameable. This is the contamination-safe, judge-free held-out domain the
self-extension rung wants (see the roadmap in
[Corpus-MathCode-Capability-Roadmap.md](../06-Roadmap/Corpus-MathCode-Capability-Roadmap.md)).

| Artifact | Path |
|---|---|
| Reward core (no torch) | `provenance_bench/math_reward.py` |
| Problem set (24, 6 families) | `provenance_bench/data/math_problems.json` |
| Family-disjoint split | `provenance_bench/math_dataset.py` |
| Offline invariants | `provenance_bench.math_reward.offline_invariants()` |
| Tests | `tests/test_math_rlvr.py` |
| Deps | `requirements-math.txt` (sympy) |

The split is **family-disjoint**: whole problem families (e.g. `factor`,
`simplify`) are held out of training, so a passing eval problem is a *new kind* of
problem, not a memorized instance — the stronger generalization test.

```bash
# Offline reward-wiring check (any machine; needs sympy, no torch/GPU):
pip install -r requirements-math.txt
python tests/test_math_rlvr.py
python tools/run_rlvr.py --task math --model mock         # full offline invariants

# Live GRPO on the math reward (rented CUDA GPU), everything below the line is the
# same GPU stack as the provenance task — only --task changes:
pip install -r requirements-rl.txt -r requirements-math.txt
python tools/run_rlvr.py --task math --model zai-org/glm-4-9b-chat-hf --vllm server   # 2x24GB
python tools/run_rlvr.py --task math --model zai-org/glm-4-9b-chat-hf --quant bf16    # 1x80GB
```

After training, the live capability claim is **held-out pass@1 on the eval families
rises vs the base adapter**, scored deterministically by `math_equivalent`
(no judge needed) — but it remains **Open** until a gated run (≥3 seeds, CI excludes
0) per `agi-proof/failure-ledger.md`. This run does not assert that claim.

## Budget (live run, rented GPU)

~60 train prompts, `num_generations=8`, ~50–100 GRPO steps, short completions →
~1–4 GPU-hours. 1×A100 80GB bf16-colocate ≈ \$1.5–2/hr → **~\$3–8**; or
2×RTX-4090 server-vLLM ≈ \$1.2/hr → **~\$2–5**. The held-out eval is cents via an
API model on your Mac.

## What this is not

- Not AGI, and not evidence of it. The verifier-gated loop only helps tasks whose
  correctness reduces to a checkable predicate.
- The reward does not transfer to unverifiable domains, and RLVR does not expand
  the base model's reasoning capacity (cf. *Limit of RLVR*, NeurIPS 2025: base
  models overtake at large pass@k). It raises pass@1 **within** the verifier's
  reach.
