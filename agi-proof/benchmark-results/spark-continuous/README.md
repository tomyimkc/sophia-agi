# Spark Continuous Benchmark Results

This directory holds **CONTINUOUS, AUTOMATED** Spark benchmark results.

Every artifact written here is stamped:

- `sparkIteration: true`
- `registeredResult: false`
- `canClaimAGI: false`

## What this is

These results are produced by the **spark-warm loop** running on the Spark box,
using the local **ollama `qwen2.5:7b`** model together with the **Lean kernel**.
The loop runs continuously and drops one artifact per iteration into this lane.

## Why these are NOT registered results

The Spark is an **ITERATION tier** machine. Its numerics differ from the x86
RunPod reference environment:

- Spark is **aarch64** and runs **bf16 + SDPA** attention.
- RunPod (the registration tier) is **x86** with the reference numerics stack.

Because the aarch64 bf16+sdpa numerics do not match the x86 RunPod reference,
results generated here are **NOT registered results** and **NOT headline claims**.
They exist purely for fast, continuous iteration and trend-watching.

## Rules

- Do **not** cite anything in this directory as a registered result.
- Do **not** use anything here as an AGI / headline claim (`canClaimAGI: false`).
- Registration and headline claims must come from the RunPod (x86) reference tier.
