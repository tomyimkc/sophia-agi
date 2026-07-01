# Judge Pool — per-family replica lanes for the local judge farm

**design/infra; no capability claim; canClaimAGI stays false.** Code: `tools/judge_pool.py`,
`config/inference.local.judge-pool.json`. Wiring: `tools/judge_pilot_answers.py --judge-pool`,
`tools/cluster_schedule_sim.py --judge-pool/--mac-lanes`. Tests: `tests/test_judge_pool.py`,
`tests/test_judge_pool_identity.py`.

> This is a **throughput/routing** change only. It changes **how many lanes serve each judge
> family, never which families judge**. The 2-family VALIDATED gate (κ ≥ 0.40, ≥ 2 DISTINCT
> families, judge ≠ subject) is untouched. Replicas of the **same** model are the **same** family —
> adding lanes does **not** add families and does **not** change any verdict (a tested invariant,
> see `tests/test_judge_pool_identity.py`).

## The bottleneck this fixes

The throughput sim (`tools/cluster_schedule_sim.py --jobs forecast`) shows the judged queue plateaus
at **~1.36x** speedup regardless of Spark count, because ~7/10 jobs funnel through **ONE serialized
Mac judge**. Adding more Sparks does not help: the binding term is the single judge lane, not
compute.

```
nodes | speedup (1 judge lane) | speedup (4 judge lanes)
   1  |        1.00            |        1.00
   2  |        1.10            |        1.56
   4  |        1.10            |        1.93
   8  |        1.36            |        2.81   <-- the ROI of the pool
  16  |        1.36            |        2.81
```
(Forecast queue, `--jobs forecast`. PLANNING numbers — the sim schedules the owner's GPU-time
**estimates**; no hardware claim. `--mac-lanes 1` reproduces the old serialized Mac judge.)

## The design: per-family replica lanes

A `JudgePool` maps `family -> [endpoint specs]`. Each spec is the **existing**
`provider:model@http://host:port/v1` string that `agent.model.default_client` already builds an
endpoint client from. A family served by N replicas has **N lanes**; the bottleneck 70B family gets
the Mac **plus** 70Bs on spare Sparks, so judge requests distribute across lanes instead of queueing
on one box.

Routing (`judge_pool.next_endpoint`) is **least-loaded with a deterministic tie-break by spec
string — no random, no clock.** Each lane is its own client; a per-item request goes to the
least-loaded lane. Because a verdict depends only on the **model + prompt** (not the base_url),
spreading requests over replicas of the **same** model is timing/routing only.

## The family-key subtlety (get this right)

`families()` and `validate_pool()` key every spec via
`tools/run_lora_uplift_validation._family_key` — the **same** function the κ gate counts families
with — so the pool stays in lockstep with the gate. That keying has a trap:

- **Aggregator** providers (`vllm`/`sglang`/`llamacpp`/`openai`/`openrouter`) key by the model
  **vendor** prefix: `vllm:mlx-community/Llama-3.3-70B-Instruct-4bit` → family **`mlx-community`**.
- **Non-aggregator** engines (`mlx`/`ollama`) key by the **engine**:
  `mlx:mlx-community/Llama-3.3-70B-Instruct-4bit` → family **`mlx`**.

So the **same weights** served by two different engines key to two **different** families and would
silently inflate the family count. **Every lane of one family must use the same provider+vendor.**
`validate_pool` **enforces** this — it refuses a family label whose replicas key differently, and
refuses a pool with **< 2** distinct families (one family on many lanes is throughput, not a second
family).

## Deployment recipe

Serve the bottleneck 70B on each spare Spark via **vLLM with no api key** (the keyless `vllm`
preset; `api_key_default=EMPTY`), then point the pool's lanes at them:

```bash
# on each spare Spark (and on the Mac, as a vllm-compatible /v1 endpoint):
vllm serve mlx-community/Llama-3.3-70B-Instruct-4bit --port 8001
```

`config/inference.local.judge-pool.json` (worked example):
- family **`qwen`** = 1 lane (`vllm:Qwen/Qwen2.5-7B-Instruct@…:8000`) — not the bottleneck, 1 lane is enough.
- family **`mlx-community`** (the 70B) = **3 lanes**: the Mac (`169.254.26.171:8081`) + 2 spare
  Sparks (`:8001`). All three are `vllm:mlx-community/…` so they key to the **same** family
  `mlx-community` — **3 lanes, still 1 family**. Total: **2 families, 4 lanes**.

Validate before use:

```bash
python tools/judge_pool.py --config config/inference.local.judge-pool.json --validate
# -> {"families": ["mlx-community","qwen"], "lanesPerFamily": {...}, "totalLanes": 4}
```

## Using the pool

```bash
# Judge with each family fanned across its replica lanes (verdict-identical to single-endpoint):
python tools/judge_pilot_answers.py --answers ANSWERS.json \
  --judges vllm:Qwen/Qwen2.5-7B-Instruct@…,vllm:mlx-community/Llama-3.3-70B-Instruct-4bit@… \
  --judge-pool config/inference.local.judge-pool.json

# Report the real ROI of adding judge replicas in the sim:
python tools/cluster_schedule_sim.py --jobs forecast --nodes 1,2,4,8,16 \
  --judge-pool config/inference.local.judge-pool.json
```

Without `--judge-pool`, both tools are **unchanged** (single endpoint per family). The pool is
strictly opt-in.
