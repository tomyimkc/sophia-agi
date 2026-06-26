# Spark Data Refinery — the always-on, gate-filtered data engine (P2)

> **Companion:** this is the concrete tool behind
> [`DGX-Spark-Maximization.md` §3.1](./DGX-Spark-Maximization.md) ("the always-on
> data refinery — highest leverage"). It inherits the provenance boundary from
> [`Spark-Local-GPU-Lane.md`](./Spark-Local-GPU-Lane.md) and implements roadmap
> phase **P2** of [`Training-Efficiency-Feasibility.md`](./Training-Efficiency-Feasibility.md).

Tool: [`tools/spark_data_refinery.py`](../../tools/spark_data_refinery.py).
Tests: [`tests/test_spark_data_refinery.py`](../../tests/test_spark_data_refinery.py).

## Why this is the highest-ROI Spark use

Our own [`Training-Efficiency-Feasibility.md`](./Training-Efficiency-Feasibility.md)
proves the capability lever is **data quality, not training speed**: the honest LoRA
speedup is ~2× (dynamic padding), and the headline "10–50×" is really the
corpus-shrink lever — a small, gate-clean corpus. The Spark's superpower —
**128 GB unified memory + always-on + free** — is therefore best spent not on faster
training but on *manufacturing gate-clean training data 24/7*: hold a 70B-class
teacher in NVFP4 in the unified pool and generate council-distillation / RFT targets
around the clock, each one run through the repo's intrinsic fail-closed gate before it
is allowed to become training signal. The work is latency-tolerant, so the Spark's
bandwidth wall doesn't hurt. **Do this first.**

## The intrinsic-gate-no-question rule (load-bearing)

Every candidate target is gated with:

```python
check_response(text, mode="advisor")["violations"]   # NO question
```

We call [`agent.gate.check_response`](../../agent/gate.py) **without the question on
purpose.** This is not an oversight — it is the central finding of Feasibility §4:

- With a question, `check_response` additionally runs the attribution **trap grader**
  — a *positive-expectation completeness* check ("expected discussion of socrates",
  "expected tradition context 'daoist'"). That check is about **wording**, not
  fabrication: a clean gold answer phrased differently fails it. Verified, it flagged
  **88/564 (16%) of clean, curated rows** — deleting good data over phrasing.
- Intrinsic-only (no question) checks the things that are wrong **regardless of the
  prompt**: a fabricated/nonexistent legal citation, false arithmetic, a
  forbidden-lineage attribution merge. Verified, it flags **0/439** curated rows while
  still catching genuine fabrication in synthetic teacher output.

So the refinery only ever uses the intrinsic path. This matches the existing
`--guard` filter in [`tools/train_lora.py`](../../tools/train_lora.py) and the `chosen`
gate-check in [`tools/wiki_to_training.py`](../../tools/wiki_to_training.py) — the
refinery is the *generation* front-end to that same fail-closed discipline.

Drops are **never silent**: the summary surfaces `candidates`, `kept`, and `dropped`.

## Mock vs. local teacher (pluggable)

The teacher is pluggable so the offline path never requires a GPU or model weights:

| Teacher | What it is | When |
| --- | --- | --- |
| `--teacher mock` (default) | Deterministic, **seeded**, pure-Python teacher. Same seed + input → same output, always. Never imports torch/transformers. | CI, tests, `--dry-run`, plumbing changes. |
| `--teacher local` | Hook for a real local model on the Spark (NVFP4 70B-class teacher, `--quant bf16 --vllm none` per [`config/inference.local.spark.json`](../../config/inference.local.spark.json)). Imported **lazily**. | Producing actual P2 data on the Spark. |

The `local` hook deliberately raises `NotImplementedError` until the Spark inference
backend is wired, so it can never silently degrade to an unfiltered or
non-deterministic path. The offline path is the default and the only path CI exercises.

## Fail-closed boundary

If the intrinsic gate cannot be imported, the refinery **refuses to emit** — it does
*not* fall back to passing candidates through unfiltered. The gate function is
injectable (so tests can supply a stub matching `check_response`'s contract), but the
**production default is the real gate** (`agent.gate.check_response`).

## Provenance boundary

Emitted rows are in the SFT / council-trace shape (system = advisor source-discipline
prompt, user = seed prompt, assistant = the gate-clean target) and carry provenance
metadata, the **same boundary** as the Spark-Local-GPU lane:

```json
"metadata": {
  "source": "spark-data-refinery",
  "teacher": "mock",
  "sparkIteration": true,
  "registeredResult": false,
  "gatePassed": true,
  "gateMode": "advisor",
  "gateIntrinsic": true,
  "gateViolations": []
}
```

`sparkIteration: true` / `registeredResult: false` means a Spark-refined row can never
be mistaken for a registered result downstream — exactly the discipline in
[`Spark-Local-GPU-Lane.md`](./Spark-Local-GPU-Lane.md). A refined row is *iteration
fuel*, not a registered number.

## How it feeds the Training-Efficiency roadmap (P2 → P3)

- **P2 — gate-filtered RFT data engine (this tool).** The teacher proposes; the
  intrinsic gate disposes. Only gate-clean targets reach SFT/RFT. The output JSONL
  folds straight into [`tools/train_lora.py`](../../tools/train_lora.py) (`--distill`),
  and re-passes its `--guard` intrinsic filter as a belt-and-braces safety net.
- **P3 — gate-as-reward GRPO.** The same intrinsic gate becomes the *reward* signal.
  The hard rule held fixed from P2 into P3: **abstention is reward-positive** — a
  correct "I can't verify that" must earn reward, never a penalty, or RL will train
  *out* the fail-closed behavior the gate exists to protect.

The verifier ceiling carries over too: passing the gate certifies **absence of
violation, not correctness**. Pair refined data with task-quality signals; never treat
gate-clean as gate-true.

## Usage

```bash
# Offline default — mock teacher end-to-end, prints {candidates, kept, dropped}, writes nothing:
python tools/spark_data_refinery.py --dry-run

# Refine from seed prompts and write gate-clean rows:
python tools/spark_data_refinery.py --seeds seeds.jsonl --out training/spark_refined.jsonl

# On the Spark with a real local teacher (once the backend hook is wired):
python tools/spark_data_refinery.py --teacher local --model <id> --seeds seeds.jsonl --out training/spark_refined.jsonl
```

Seed JSONL is one `{"id", "prompt"}` object per line. The output is ready for
`--distill` into the LoRA trainer.
