# Wisdom-internalization experiment — runbook

**Claim under test:** the provenance/abstain behavior is an *internalizable, distillable*
inductive bias — not just a runtime wrapper. Self-distill Sophia's own gated pipeline
into a small model's weights, then measure how much wisdom is now **intrinsic** (caught
with the deterministic gate turned OFF) vs. how much the gate still has to carry.

The deterministic gate **stays bolted on at serve time** regardless — internalization is
measured for evidence, not used to remove the fail-closed guarantee. The gap between
"student gate-off" and "student gate-on" is exactly the defense-in-depth residual.

## Hardware roles

See [`docs/HARDWARE.md`](HARDWARE.md) and [`config/devices.local.json`](../config/devices.local.json).
Trace-gen + ablation run on the **Mac Studio M3 Ultra** (mlx, bandwidth-bound generation);
training runs on the **DGX Spark** (hf/cuda, compute-bound FP4).

## Components

| File | Role |
|---|---|
| `tools/model_backends.py` | the single model seam: `make_generate(backend, model, adapter)` → `generate(system,user)->ModelResult`. backends: `mock` / `mlx` / `hf`. |
| `tools/gen_distill_traces.py` | teacher farm: harvest the gated arm, double-firewall re-check, gold-verify, passport-stamp, held-out seal. |
| `tools/train_lora.py` | existing trainer; `--guard --scaffold --distill <traces>` folds the gated traces in. |
| `tools/run_wisdom_ablation.py` | proof matrix + fabrication-vs-compute curve over the sealed held-out split. |
| `scripts/wisdom_internalization.sh` | stage driver: `trace` / `train` / `ablate` / `all`. |

## Quick smoke test (offline, no weights)

```bash
bash scripts/wisdom_internalization.sh all          # mock backend end-to-end
```

Expected: ~141 traces harvested (teacher gate delta ≈ 0.87, 0 false-positive cost);
ablation prints `intrinsicWisdom` with the headline `studentGateOffHalluc`.

## Real run (per device)

**0. Either device — ~10s smoke check first** (loads the model, does one real generation,
fails closed on an empty/not-ok result before you commit to the full pass):
```bash
python -m tools.model_backends --backend mlx --model Qwen/Qwen3-4B   # Mac
python -m tools.model_backends --backend hf  --model Qwen/Qwen3-4B   # Spark
```
The `trace` and `ablate` stages run this automatically before their full pass.

**1. Mac — generate traces**
```bash
pip install "mlx-lm>=0.20"
BACKEND=mlx BASE=Qwen/Qwen3-4B bash scripts/wisdom_internalization.sh trace
rsync -a training/council/distill_traces.jsonl spark:~/sophia-agi/training/council/
```

**2. Spark — distill the student**
```bash
pip install -r requirements-lora.txt
BACKEND=hf BASE=Qwen/Qwen3-4B bash scripts/wisdom_internalization.sh train
rsync -a models/sophia-4b-internalized/ mac:~/sophia-agi/models/sophia-4b-internalized/
```

**3. Mac — ablate + curve**
```bash
BACKEND=mlx BASE=Qwen/Qwen3-4B bash scripts/wisdom_internalization.sh ablate
# -> agi-proof/wisdom-internalization/ablation-*.json
```

## Reading the output

```jsonc
"intrinsicWisdom": {
  "baseGateOffHalluc": 0.95,        // raw base fabricates on ~95% of false attributions
  "studentGateOffHalluc": 0.31,     // <-- THE NUMBER: distilled weights, gate OFF
  "internalizedDrop": 0.64,         // wisdom that moved into the weights
  "residualCaughtByGateOnly": 0.05  // what the deterministic gate still has to catch
},
"fabricationVsCompute": [           // the honest "wisdom scaling law"
  {"step": 500,  "studentGateOffHalluc": 0.52, ...},
  {"step": 1500, "studentGateOffHalluc": 0.31, ...}   // falling toward zero
]
```

## Integrity guards (all fail-closed)

- **Held-out seal** — trace-gen writes a prompt-only digest manifest; the ablation
  asserts the live held-out is a subset of it (`HELD-OUT DRIFT` → abort). No training
  data can leak into the eval.
- **Double firewall** — a harvested trace must pass `agent.guarded.check_claim` *and*
  the gold verifier; confident-but-wrong teacher reasoning is dropped, not distilled.
- **Provenance by construction** — every row is `data_passport`-stamped with teacher,
  backend, gate SHA, verifier, seed, decode. API-origin rows are un-stampable → excluded.
- **Anti-gaming** — `reward_is_hackable` flags a student that only looks good behind the
  gate (train verifier) but not with it off (held-out verifier).
- **Calibration** — ECE + Brier reported per cell, so "confident" must track "correct".
