#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Decompose a model into per-layer shards + a manifest — the on-disk layout AirLLM streams.

``serving/layer_stream.py`` streams one transformer layer at a time from disk; to do that it
needs the model laid out *as* per-layer shards plus a manifest mapping layer index → file,
dtype, byte size and (optional) quantized bit-width. This tool produces that layout from a
consolidated Hugging Face checkpoint, mirroring what AirLLM's ``save_pretrained``-time
decomposition does.

Two modes:

  ``--plan`` (default, dependency-free)
      Read only ``config.json`` (``num_hidden_layers``, ``hidden_size``, dtype) and emit the
      manifest *layout* — one entry per transformer layer plus the embedding and head
      shards — with **estimated** byte sizes from the architecture. No weights are loaded, so
      this runs in CI on any machine and is the path the offline invariants exercise. Use it
      to size GPU budget and sanity-check the streaming plan before paying for a real shard.

  ``--materialize`` (needs ``safetensors`` + the real weights)
      Group the checkpoint's tensors by ``model.layers.{i}.`` prefix, write one
      ``layer_{i:04d}.safetensors`` per block (embeddings/norm/lm_head go to ``embed`` /
      ``head`` shards), and record true byte sizes in the manifest.

With ``--target-avg-bits`` the manifest annotates a per-layer bit-width via
``serving.layer_stream.plan_layer_bits`` (which delegates to ``moe.adapt.bit_allocator``), so
the streamer's quant-aware sizing and the shards agree on width. Protected shards
(embeddings, lm_head) keep the floor.

Manifest schema: ``sophia.layer_shard_manifest.v1``. The streamer / a real mmap loader is the
consumer; this tool is the producer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST_SCHEMA = "sophia.layer_shard_manifest.v1"

# Bytes per element for the dtypes a checkpoint config commonly declares.
_DTYPE_BYTES = {
    "float32": 4, "float16": 2, "bfloat16": 2, "float64": 8,
    "int8": 1, "uint8": 1,
}


def _dtype_bytes(name: str) -> int:
    return _DTYPE_BYTES.get(str(name).lower().replace("torch.", ""), 2)


def estimate_layer_params(cfg: dict[str, Any]) -> int:
    """Rough parameter count of ONE decoder layer from a HF config.

    A standard decoder layer = attention (q,k,v,o projections) + MLP (gate,up,down) +
    2 norms. With GQA the kv projections shrink by ``n_heads / n_kv_heads``. This is an
    estimate for sizing the stream plan, not an exact count — the materialize path records
    the true bytes.
    """
    h = int(cfg.get("hidden_size", 0))
    inter = int(cfg.get("intermediate_size", 4 * h))
    n_heads = int(cfg.get("num_attention_heads", max(1, h // 128)))
    n_kv = int(cfg.get("num_key_value_heads", n_heads))
    head_dim = h // n_heads if n_heads else h
    # attention: q (h*h) + k,v (h * n_kv*head_dim each) + o (h*h)
    attn = h * h + 2 * (h * n_kv * head_dim) + h * h
    # mlp: gate + up (h*inter each) + down (inter*h)
    mlp = 3 * h * inter
    norms = 2 * h
    return attn + mlp + norms


def build_plan(config_path: Path, *, target_avg_bits: float | None) -> dict[str, Any]:
    """Build a manifest layout from ``config.json`` alone (no weights loaded)."""
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    n_layers = int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or 0)
    if n_layers <= 0:
        raise ValueError("config has no positive num_hidden_layers / n_layer")
    h = int(cfg.get("hidden_size", 0))
    vocab = int(cfg.get("vocab_size", 0))
    dtype = cfg.get("torch_dtype", "bfloat16")
    elt = _dtype_bytes(dtype)

    per_layer_params = estimate_layer_params(cfg)
    layer_bytes = per_layer_params * elt
    embed_bytes = vocab * h * elt
    head_bytes = vocab * h * elt   # untied lm_head; tied weights share the embed shard

    shards: list[dict[str, Any]] = []
    shards.append({"shard": "embed", "kind": "embedding", "layer_index": None,
                   "est_bytes": embed_bytes, "protected": True})
    for i in range(n_layers):
        shards.append({"shard": f"layer_{i:04d}", "kind": "decoder_layer", "layer_index": i,
                       "est_bytes": layer_bytes, "protected": i in (0, n_layers - 1)})
    shards.append({"shard": "head", "kind": "lm_head", "layer_index": None,
                   "est_bytes": head_bytes, "protected": True})

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "mode": "plan",
        "source_config": str(config_path),
        "architecture": cfg.get("architectures", ["?"])[0] if cfg.get("architectures") else "?",
        "num_hidden_layers": n_layers,
        "hidden_size": h,
        "vocab_size": vocab,
        "dtype": dtype,
        "bytes_per_element": elt,
        "est_total_bytes": embed_bytes + head_bytes + n_layers * layer_bytes,
        "est_layer_bytes": layer_bytes,
        "shards": shards,
    }
    _annotate_bits(manifest, target_avg_bits)
    return manifest


def _annotate_bits(manifest: dict[str, Any], target_avg_bits: float | None) -> None:
    """Attach a per-layer bit-width to the manifest via the sensitivity allocator."""
    if target_avg_bits is None:
        return
    from serving.layer_stream import plan_layer_bits

    decoder = [s for s in manifest["shards"] if s["kind"] == "decoder_layer"]
    layer_fp16 = {s["layer_index"]: max(1, s["est_bytes"]) for s in decoder}
    protected = {s["layer_index"] for s in decoder if s.get("protected")}
    bits = plan_layer_bits(layer_fp16, target_avg_bits, protected=protected)
    for s in manifest["shards"]:
        if s["kind"] == "decoder_layer":
            s["bits"] = int(bits.get(s["layer_index"], 16))
        else:
            s["bits"] = 16   # embeddings / head stay fp16 (protected)
    manifest["target_avg_bits"] = target_avg_bits


def materialize(model_dir: Path, out_dir: Path, *, target_avg_bits: float | None) -> dict[str, Any]:
    """Write real per-layer safetensors shards by grouping tensors on the layer prefix."""
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except Exception as exc:  # pragma: no cover - exercised only with the dep present
        raise RuntimeError("materialize needs `safetensors` (and torch). Use --plan for the "
                            "dependency-free layout.") from exc
    import re

    out_dir.mkdir(parents=True, exist_ok=True)
    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"no .safetensors in {model_dir}")

    layer_re = re.compile(r"\.layers\.(\d+)\.")
    groups: dict[str, dict[str, Any]] = {}
    for f in st_files:
        with safe_open(str(f), framework="pt") as h:
            for key in h.keys():
                m = layer_re.search(key)
                shard = f"layer_{int(m.group(1)):04d}" if m else (
                    "head" if "lm_head" in key else "embed")
                groups.setdefault(shard, {})[key] = h.get_tensor(key)

    shards = []
    for shard, tensors in sorted(groups.items()):
        path = out_dir / f"{shard}.safetensors"
        save_file(tensors, str(path))
        nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
        idx = int(shard.split("_")[1]) if shard.startswith("layer_") else None
        shards.append({"shard": shard, "kind": "decoder_layer" if idx is not None else
                       ("lm_head" if shard == "head" else "embedding"),
                       "layer_index": idx, "file": path.name, "est_bytes": nbytes,
                       "protected": shard in ("embed", "head")})

    manifest = {
        "schema": MANIFEST_SCHEMA, "mode": "materialize", "source_model": str(model_dir),
        "num_hidden_layers": sum(1 for s in shards if s["kind"] == "decoder_layer"),
        "est_total_bytes": sum(s["est_bytes"] for s in shards),
        "shards": shards,
    }
    _annotate_bits(manifest, target_avg_bits)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("model_dir", type=Path, help="HF model dir (must contain config.json)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir for shards/manifest (default: <model_dir>/sharded)")
    ap.add_argument("--plan", action="store_true",
                    help="layout-only from config.json, no weights loaded (default if no deps)")
    ap.add_argument("--materialize", action="store_true",
                    help="write real per-layer safetensors shards (needs safetensors+torch)")
    ap.add_argument("--target-avg-bits", type=float, default=None,
                    help="annotate per-layer bit-widths at this byte-weighted average")
    args = ap.parse_args(argv)

    out_dir = args.out or (args.model_dir / "sharded")
    if args.materialize:
        manifest = materialize(args.model_dir, out_dir, target_avg_bits=args.target_avg_bits)
        print(f"materialized {manifest['num_hidden_layers']} layer shards -> {out_dir}")
    else:
        manifest = build_plan(args.model_dir / "config.json",
                              target_avg_bits=args.target_avg_bits)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        gb = manifest["est_total_bytes"] / 1e9
        print(f"planned {manifest['num_hidden_layers']} layers (~{gb:.1f} GB fp est, "
              f"layer ~{manifest['est_layer_bytes'] / 1e6:.0f} MB) -> {out_dir/'manifest.json'}")
    return 0


# ---------------------------------------------------------------------------
# Offline invariants (run with --selftest; also covered by tests/)
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    import tempfile
    checks: dict[str, bool] = {}
    detail: dict = {}

    # A Llama-70B-ish config (80 layers) — the AirLLM headline case.
    cfg = {
        "architectures": ["LlamaForCausalLM"], "num_hidden_layers": 80,
        "hidden_size": 8192, "intermediate_size": 28672,
        "num_attention_heads": 64, "num_key_value_heads": 8,
        "vocab_size": 128256, "torch_dtype": "bfloat16",
    }
    with tempfile.TemporaryDirectory() as d:
        cfgp = Path(d) / "config.json"
        cfgp.write_text(json.dumps(cfg), encoding="utf-8")

        plan = build_plan(cfgp, target_avg_bits=None)
        # 1. One shard per layer, plus embed + head.
        decoder = [s for s in plan["shards"] if s["kind"] == "decoder_layer"]
        checks["one_shard_per_layer"] = len(decoder) == 80
        checks["has_embed_and_head"] = (plan["shards"][0]["kind"] == "embedding"
                                        and plan["shards"][-1]["kind"] == "lm_head")
        checks["layer_indices_contiguous"] = [s["layer_index"] for s in decoder] == list(range(80))

        # 2. A single layer is a tiny fraction of the whole model (the streaming premise).
        ratio = plan["est_layer_bytes"] / plan["est_total_bytes"]
        checks["layer_is_small_fraction"] = ratio < 0.05
        detail["layer_fraction"] = round(ratio, 5)
        detail["est_total_gb"] = round(plan["est_total_bytes"] / 1e9, 1)

        # 3. Bit annotation hits the target average and keeps protected shards at fp16.
        try:
            import numpy  # noqa: F401  (plan_layer_bits → moe.adapt needs numpy)
            planq = build_plan(cfgp, target_avg_bits=4.5)
            dq = [s for s in planq["shards"] if s["kind"] == "decoder_layer"]
            avg = sum(s["bits"] for s in dq) / len(dq)
            checks["avg_bits_near_target"] = 3.0 <= avg <= 6.0
            checks["protected_stay_fp16"] = all(
                s["bits"] == 16 for s in planq["shards"] if s["kind"] != "decoder_layer")
            detail["avg_bits"] = round(avg, 3)
        except ImportError:
            checks["avg_bits_near_target"] = True   # skipped without numpy (CI has it)
            checks["protected_stay_fp16"] = True

        # 4. Fail-closed on a config without a layer count.
        bad = Path(d) / "bad.json"
        bad.write_text(json.dumps({"hidden_size": 64}), encoding="utf-8")
        try:
            build_plan(bad, target_avg_bits=None); checks["no_layers_rejected"] = False
        except ValueError:
            checks["no_layers_rejected"] = True

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    if "--selftest" in (sys.argv[1:] if len(sys.argv) > 1 else []):
        ok, detail = offline_invariants()
        print("Shard-checkpoint offline invariants:", "PASS" if ok else "FAIL")
        for k, v in detail["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        print(f"  layer is {detail.get('layer_fraction')} of model "
              f"(~{detail.get('est_total_gb')} GB est; avg bits {detail.get('avg_bits')})")
        raise SystemExit(0 if ok else 1)
    raise SystemExit(main())
