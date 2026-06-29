# aarch64 / GB10 (DGX Spark) Compatibility Patches for Phase-3 RLVR

Date: 2026-06-29
Platform: Linux aarch64, NVIDIA GB10, torch 2.12.1+cu130, trl ~1.7.x, transformers 5.12.x

## Context
- Used --vllm none (vllm wheels had ABI mismatch / undefined symbols with the CUDA torch build).
- trl GRPOConfig in this version lacked `max_prompt_length` (only `max_completion_length` and other fields).
- Model loading produced "parameters on the meta device" + device-side asserts during generation.
- Patches applied before full seeds to preserve the aarch64/GB10 fixes.

## 1. tools/run_rlvr.py (tracked, committed)

```diff
diff --git a/tools/run_rlvr.py b/tools/run_rlvr.py
index 9028cc5c..2f428cea 100644
--- a/tools/run_rlvr.py
+++ b/tools/run_rlvr.py
@@ -383,7 +383,7 @@ def _run_gpu(args: argparse.Namespace) -> int:
         train_rows = [{**r, "prompt": step_reward.STEP_INSTRUCTION + r["prompt"]} for r in train_rows]
     ds = Dataset.from_list(train_rows)  # columns kept: remove_unused_columns=False

-    model_init_kwargs: dict = {"trust_remote_code": False}
+    model_init_kwargs: dict = {"trust_remote_code": False, "device_map": "cuda"}
     if four_bit:
         model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
             load_in_4bit=True, bnb_4bit_quant_type="nf4",
@@ -398,7 +398,7 @@ def _run_gpu(args: argparse.Namespace) -> int:
         per_device_train_batch_size=args.batch_size,
         gradient_accumulation_steps=args.grad_accum,
         num_generations=args.num_generations,
-        max_prompt_length=args.max_prompt_len,
+        # max_prompt_length omitted for this trl version (handled by tokenizer / data prep for --vllm none)
         max_completion_length=args.max_completion_len,
         num_train_epochs=args.epochs,
         max_steps=args.max_steps,  # >0 bounds a smoke run (overrides epochs)
@@ -410,6 +410,9 @@ def _run_gpu(args: argparse.Namespace) -> int:
         model_init_kwargs=model_init_kwargs,
         use_vllm=use_vllm,
     )
+    # filter to only supported fields in this trl/GRPOConfig version
+    cfg_fields = set(getattr(GRPOConfig, "__dataclass_fields__", {}))
+    grpo_kwargs = {k: v for k, v in grpo_kwargs.items() if k in cfg_fields or k in ["output_dir", "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps", "num_train_epochs", "max_steps", "beta", "logging_steps", "save_strategy", "report_to", "remove_unused_columns", "model_init_kwargs", "use_vllm"]}
     if use_vllm:
         # vllm_mode (colocate/server selector) only exists in trl >= 0.17; older
         # pinned trl (0.16.x) does in-process colocate via use_vllm alone and
```

## 2. Site-packages patches (NOT in git - recorded here for reproducibility)

### 2.1 .venv/lib/python3.12/site-packages/trl/import_utils.py

**Exact change around line 113:**

```python
def is_vllm_available(min_version: str | None = None) -> bool:
    # Patched for --vllm none on this platform (vllm wheel ABI mismatch with torch)
    return False
```

(Original body that did `_is_package_available("vllm"...)` + version checks was replaced so that importing GRPOTrainer / GRPOConfig does not trigger vllm._C load.)

### 2.2 .venv/lib/python3.12/site-packages/trl/generation/vllm_generation.py

**Exact change:**

```python
# original
from .vllm_client import VLLMClient

# patched
try:
    from .vllm_client import VLLMClient
except Exception:
    VLLMClient = None  # patched for --vllm none on incompatible platform
```

This prevents unconditional import of the vllm client even when `is_vllm_available()` is forced False.

## How to re-apply
- For tracked: the commit below.
- For site-packages: re-run the exact sed / python patch snippets or copy the blocks above into the venv files after any `pip install -r requirements-rl.txt`.

## Notes
- device_map="cuda" prevents meta-device offload warnings and CUDA asserts on GB10.
- These are minimal surgical patches to make the official script run on the Spark aarch64 + current package resolution.
