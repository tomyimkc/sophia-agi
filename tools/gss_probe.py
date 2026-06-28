#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run a real forward pass and get a GSS go/no-go on a concrete checkpoint.

This is the harness that feeds :mod:`serving.gss_feasibility` (the Tier-0 meter) with
arrays extracted from an actual model, so you can answer — for *your* model — "can
Governed Speculative Sparsity beat a dense decode here?" before spending a GPU-hour.

It produces the three arrays the gate needs and calls it:
  - ``contribs``      (T, U): per-token, per-weight-unit contribution magnitude — the
                      read-set. For an **MoE**, the router gate probability per expert
                      (an expert the router doesn't pick is a weight you don't read);
                      for a **dense** model, the per-channel activation magnitude at the
                      MLP down-projection input.
  - ``target_probs``  (P, V): next-token distributions from the full-precision pass.
  - ``draft_probs``   (P, V): next-token distributions from a cheap **4-bit self-draft**
                      of the *same* model (the speculative draft).

Backends (``--backend``):
  - ``moelm`` (default): the in-repo pure-numpy top-1 MoE (`pretraining/architecture/moe.py`).
    Real forward pass, **no torch/GPU**, runs anywhere incl. CI — the reference that
    proves the extraction→gate pipeline end-to-end. Use it to sanity-check the harness.
  - ``hf``: a real HuggingFace checkpoint via ``transformers``. MoE models expose
    ``output_router_logits`` → router contribs; dense models use MLP-activation hooks.
    The 4-bit draft is ``bitsandbytes`` ``load_in_4bit`` (``--draft bnb``, the
    `tools/train_lora.py` path) or device-agnostic int4 fake-quant (``--draft fakequant``).
    Skips cleanly (prints why, exits 0) without torch/transformers — like
    ``kernels/src/nvfp4_gemm.py``.

Honest scope. This emits a *feasibility* verdict (a `GSSFeasibilityReport`), never a
speedup. A GO means "GSS is worth prototyping on this model" (Tier 1+), not "this model
is faster". The ``moelm`` toy is real but tiny — its numbers characterise the *harness*,
not a frontier model. ``canClaimAGI`` stays ``false``.

    python tools/gss_probe.py --backend moelm --experts 16 --tokens 64
    python tools/gss_probe.py --backend hf --model Qwen/Qwen2-57B-A14B --prompt "..." --draft bnb
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False

from serving.gss_feasibility import GSSFeasibilityGate  # noqa: E402


# ===========================================================================
# Pure-numpy helpers (no model, no torch) — the testable core.
# ===========================================================================

def softmax_np(z, axis: int = -1):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def gate_probs_to_contribs(router_logits, *, top_k: "int | None" = None):
    """Router logits (T, E) → non-negative read-set contributions (T, E).

    Softmax over experts gives each expert's share of the token. With ``top_k`` set, the
    non-selected experts are zeroed — encoding that a non-routed expert is a weight the
    decoder genuinely does not read (the honest read-set for top-k routing).
    """
    g = softmax_np(np.asarray(router_logits, dtype=np.float64), axis=1)
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        keep = min(top_k, g.shape[1])
        thresh = np.sort(g, axis=1)[:, -keep][:, None]   # kth-largest per row
        g = np.where(g >= thresh, g, 0.0)
    return g


def int4_group_quant_roundtrip(W, *, group: int = 32):
    """Symmetric int4 (signed, −8..7) group round-trip of a 2-D weight along its last axis.

    The cheap self-draft's weight proxy: quantize per ``group`` columns with a per-group
    scale, dequantize, return the lossy weight. Pure-numpy so it models the 4-bit draft
    on any backend. Returns a float64 array the same shape as ``W``.
    """
    A = np.asarray(W, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("W must be 2-D")
    if group < 1:
        raise ValueError("group must be >= 1")
    r, c = A.shape
    pad = (-c) % group
    if pad:
        A = np.concatenate([A, np.zeros((r, pad))], axis=1)
    blocks = A.reshape(r, A.shape[1] // group, group)
    amax = np.max(np.abs(blocks), axis=2, keepdims=True)
    scale = np.where(amax == 0, 1.0, amax / 7.0)         # int4 signed range 7
    q = np.clip(np.round(blocks / scale), -8, 7)
    deq = (q * scale).reshape(r, A.shape[1])
    return deq[:, :c]


# ===========================================================================
# Backend: in-repo numpy MoELM — a real forward pass with no heavy deps.
# ===========================================================================

def probe_moelm(*, n_experts: int, tokens: int, vocab: int = 32, context: int = 4,
                hidden: int = 16, train_steps: int = 200, group: int = 8, seed: int = 0):
    """Real top-1 MoE forward pass → (contribs, target_probs, draft_probs).

    Builds and briefly trains the in-repo :class:`MoELM` so routing isn't degenerate,
    then for ``tokens`` contexts extracts: router-gate contribs, the full-precision
    next-token distribution, and a 4-bit int4-fake-quant self-draft distribution.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    import random as _random
    from pretraining.architecture.moe import MoELM

    rng = _random.Random(seed)
    m = MoELM(vocab=vocab, context=context, hidden=hidden, n_experts=n_experts, seed=seed)
    # A tiny structured corpus (next token = (sum of ctx) % vocab) so the router learns
    # something real to route on — enough for a non-trivial acceptance number.
    def make_example():
        ctx = [rng.randrange(vocab) for _ in range(context)]
        tgt = sum(ctx) % vocab
        return ctx, tgt
    train = [make_example() for _ in range(max(0, train_steps))]
    for ctx, tgt in train:
        m.train_step(ctx, tgt, lr=0.05)

    examples = [make_example() for _ in range(tokens)]

    # contribs: per-token router gate probabilities (top-1 read-set).
    E = m.n_experts
    rlogits = np.zeros((tokens, E), dtype=np.float64)
    for t, (ctx, _t) in enumerate(examples):
        active = m._active_inputs(ctx)
        row = list(m.br)
        for p in active:
            wrow = m.Wr[p]
            for e in range(E):
                row[e] += wrow[e]
        rlogits[t] = row
    contribs = gate_probs_to_contribs(rlogits, top_k=1)   # top-1 routing

    # target distributions: MoELM's own full-precision forward.
    target = np.array([m.forward(ctx) for ctx, _t in examples], dtype=np.float64)

    # draft: clone, int4-fake-quant the big matrices (experts + router), forward again.
    mq = copy.deepcopy(m)
    for exp in mq.experts:
        exp["W1"] = int4_group_quant_roundtrip(exp["W1"], group=group).tolist()
        exp["W2"] = int4_group_quant_roundtrip(exp["W2"], group=group).tolist()
    mq.Wr = int4_group_quant_roundtrip(mq.Wr, group=group).tolist()
    draft = np.array([mq.forward(ctx) for ctx, _t in examples], dtype=np.float64)

    return contribs, target, draft


# ===========================================================================
# Backend: HuggingFace checkpoint (gated — skips cleanly without torch).
# ===========================================================================

def _contribs_and_logits(model, ids):
    """One forward pass → (contribs [T,U], logits [T,V]) for a loaded HF model.

    MoE models expose ``router_logits`` (top-k-masked → the true read-set); dense models
    fall back to per-channel |activation| at the MLP down-projection. Shared by the single
    probe and the campaign so both extract the read-set the same way.
    """
    rl_out = None
    try:
        rl_out = model(ids, output_router_logits=True, use_cache=False)
        rl = getattr(rl_out, "router_logits", None)
    except TypeError:
        rl = None
    if rl:
        top_k = getattr(model.config, "num_experts_per_tok", None)
        mats = [r.detach().float().cpu().numpy() for r in rl if r is not None]
        contribs = np.concatenate([gate_probs_to_contribs(mt, top_k=top_k) for mt in mats], axis=1)
        logits = rl_out.logits[0].detach().float().cpu().numpy()
        return contribs, logits
    # Dense fallback: hook the second MLP projection's input → per-channel |activation|.
    captured: list = []
    handles = []

    def hook(_m, inp, _o):
        captured.append(inp[0].detach().float().abs().mean(dim=0).cpu().numpy())
    for name, mod in model.named_modules():
        if any(kk in name for kk in ("down_proj", "fc2", "c_proj", "wo")) and hasattr(mod, "weight"):
            handles.append(mod.register_forward_hook(hook))
    out = model(ids, use_cache=False)
    for h in handles:
        h.remove()
    if not captured:
        raise RuntimeError("no MLP projections matched for dense contribs; unsupported arch")
    T = ids.shape[1]
    contribs = np.concatenate([np.tile(c, (T, 1)) for c in captured], axis=1)
    logits = out.logits[0].detach().float().cpu().numpy()
    return contribs, logits


def probe_hf(*, model_id: str, prompt: str, draft: str = "fakequant",
             device: str = "auto", max_positions: int = 256, group: int = 32):
    """Real HF forward pass → (contribs, target_probs, draft_probs). Heavy; GPU-friendly.

    Raises ``RuntimeError`` (caught by ``main`` → clean skip) if torch/transformers are
    unavailable, mirroring the repo's gated-GPU-path convention.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # pragma: no cover - exercised only off-CI
        raise RuntimeError(f"hf backend needs torch+transformers ({type(e).__name__}: {e})")

    tok = AutoTokenizer.from_pretrained(model_id)
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def _load(**extra):
        return AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device, **extra
        ).eval()

    model = _load()
    ids = ids.to(model.device)

    contribs, logits = _contribs_and_logits(model, ids)
    n = min(max_positions, logits.shape[0])
    target = softmax_np(logits[:n], axis=1)

    # Draft: a cheap 4-bit self-pass. The target's outputs are now captured as numpy, so
    # we must NOT keep two full models resident — that is what made bitsandbytes offload
    # to CPU and fail. For bnb, free the target first; for fakequant, mutate the resident
    # model in place (no second copy) since we no longer need the FP weights.
    import gc
    if draft == "bnb":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"--draft bnb needs bitsandbytes ({e})")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dmodel = _load(quantization_config=BitsAndBytesConfig(load_in_4bit=True))
        dout = dmodel(ids.to(dmodel.device), use_cache=False)
        dl = dout.logits[0].detach().float().cpu().numpy()
    else:  # fakequant: int4 round-trip every 2-D Linear weight in place on the resident model
        with torch.no_grad():
            for mod in model.modules():
                w = getattr(mod, "weight", None)
                if w is not None and w.dim() == 2:
                    q = int4_group_quant_roundtrip(w.detach().float().cpu().numpy(), group=group)
                    w.copy_(torch.tensor(q, dtype=w.dtype, device=w.device))
        dout = model(ids, use_cache=False)
        dl = dout.logits[0].detach().float().cpu().numpy()
    dn = min(n, dl.shape[0])
    draft_probs = softmax_np(dl[:dn], axis=1)
    return contribs[:n], target[:dn], draft_probs[:dn]


# Diverse prompts for the across-prompt CI campaign (science / code / narrative / dialogue
# / legal) — variety is what makes the run-to-run CI meaningful.
CAMPAIGN_PROMPTS = [
    "The mitochondrion is a double membrane bound organelle found in most eukaryotic cells. "
    "It generates most of the chemical energy needed to power the biochemical reactions of "
    "the cell, storing that energy in adenosine triphosphate. The number of mitochondria in "
    "a cell varies widely by organism, tissue, and cell type.",
    "def quicksort(items):\n    if len(items) <= 1:\n        return items\n    pivot = items[len(items)//2]\n"
    "    left = [x for x in items if x < pivot]\n    mid = [x for x in items if x == pivot]\n"
    "    right = [x for x in items if x > pivot]\n    return quicksort(left) + mid + quicksort(right)",
    "The old lighthouse keeper climbed the spiral stairs one last time. Below him the storm "
    "threw the sea against the rocks as it had for forty years, and he wondered who would "
    "tend the lamp when winter came and his hands could no longer hold the rail.",
    "Q: What is the capital of Australia, and why is it not Sydney? A: The capital is "
    "Canberra. It was chosen as a compromise between the rivals Sydney and Melbourne, and "
    "purpose-built in the early twentieth century to serve as the seat of federal government.",
    "Pursuant to the agreement, the party of the first part shall indemnify and hold harmless "
    "the party of the second part against any and all claims, losses, and liabilities arising "
    "from a breach of the warranties set forth in Section 4, except to the extent caused by "
    "the gross negligence of the indemnified party.",
]


def campaign_hf(*, model_id: str, prompts: "list[str]", draft: str = "bnb",
                device: str = "auto", gamma: int = 4, coverage: float = 0.9,
                max_positions: int = 256, group: int = 32):
    """Score K prompts on ONE model load → per-prompt GSS reports + an across-prompt CI.

    Loads the full-precision target once (scores every prompt's contribs+logits), frees it,
    loads the 4-bit draft once (scores every prompt), then builds a `GSSFeasibilityReport`
    per prompt and an `aggregate_runs` across-prompt 95% CI — the registered-result
    statement. One boot, one download: far cheaper than K separate pods.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"hf backend needs torch+transformers ({type(e).__name__}: {e})")
    import gc
    from serving.gss_feasibility import GSSFeasibilityGate, aggregate_runs

    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    def _load(**extra):
        return AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device, **extra).eval()

    id_list = [tok(p, return_tensors="pt")["input_ids"] for p in prompts]

    # Pass 1: full-precision target — contribs + logits per prompt.
    model = _load()
    per: list = []
    for ids in id_list:
        ids = ids.to(model.device)
        contribs, logits = _contribs_and_logits(model, ids)
        n = min(max_positions, logits.shape[0])
        per.append({"contribs": contribs[:n], "target": softmax_np(logits[:n], axis=1), "n": n})

    # Pass 2: the 4-bit draft (one model resident at a time — the load-order fix).
    if draft == "bnb":
        from transformers import BitsAndBytesConfig
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dmodel = _load(quantization_config=BitsAndBytesConfig(load_in_4bit=True))
        for i, ids in enumerate(id_list):
            dl = dmodel(ids.to(dmodel.device), use_cache=False).logits[0].detach().float().cpu().numpy()
            per[i]["dl"] = dl
    else:  # fakequant in place
        with torch.no_grad():
            for mod in model.modules():
                w = getattr(mod, "weight", None)
                if w is not None and w.dim() == 2:
                    q = int4_group_quant_roundtrip(w.detach().float().cpu().numpy(), group=group)
                    w.copy_(torch.tensor(q, dtype=w.dtype, device=w.device))
        for i, ids in enumerate(id_list):
            dl = model(ids.to(model.device), use_cache=False).logits[0].detach().float().cpu().numpy()
            per[i]["dl"] = dl

    gate = GSSFeasibilityGate(gamma=gamma, coverage=coverage)
    reports = []
    for i, d in enumerate(per):
        dn = min(d["n"], d["dl"].shape[0])
        rep = gate.evaluate(d["contribs"][:dn], d["target"][:dn],
                            softmax_np(d["dl"][:dn], axis=1)).as_dict()
        rep["prompt_index"] = i
        reports.append(rep)
    return reports, aggregate_runs(reports)


# ===========================================================================
# CLI
# ===========================================================================

def _run(args) -> "tuple[object, object, object]":
    if args.backend == "moelm":
        return probe_moelm(n_experts=args.experts, tokens=args.tokens, vocab=args.vocab,
                           context=args.context, hidden=args.hidden,
                           train_steps=args.train_steps, group=args.group, seed=args.seed)
    if args.backend == "hf":
        return probe_hf(model_id=args.model, prompt=args.prompt, draft=args.draft,
                        device=args.device, max_positions=args.tokens, group=args.group)
    raise ValueError(f"unknown backend {args.backend!r}")


def _run_campaign(args) -> int:
    """≥N-run campaign → per-run reports + an across-run 95% CI (registered statement)."""
    from serving.gss_feasibility import GSSFeasibilityGate, aggregate_runs
    n = int(args.campaign)
    try:
        if args.backend == "hf":
            prompts = [CAMPAIGN_PROMPTS[i % len(CAMPAIGN_PROMPTS)] for i in range(n)]
            reports, agg = campaign_hf(model_id=args.model, prompts=prompts, draft=args.draft,
                                       device=args.device, gamma=args.gamma, coverage=args.coverage,
                                       max_positions=args.tokens, group=args.group)
            model_name = args.model
        else:  # moelm: N independent seeds
            gate = GSSFeasibilityGate(gamma=args.gamma, coverage=args.coverage,
                                      draft_byte_frac=args.draft_byte_frac)
            reports = []
            for s in range(n):
                c, t, d = probe_moelm(n_experts=args.experts, tokens=args.tokens, vocab=args.vocab,
                                      context=args.context, hidden=args.hidden,
                                      train_steps=args.train_steps, group=args.group, seed=args.seed + s)
                rep = gate.evaluate(c, t, d).as_dict(); rep["seed"] = args.seed + s
                reports.append(rep)
            agg = aggregate_runs(reports)
            model_name = "MoELM(toy)"
    except RuntimeError as e:
        print(f"[gss_probe] skipped: {e}")
        return 0

    out = {"backend": args.backend, "model": model_name, "n_runs": len(reports),
           "per_run": reports, "aggregate": agg}
    print(f"\n[gss_probe] CAMPAIGN — {model_name} via {args.backend} ({len(reports)} runs)")
    for key in ("rho", "alpha", "k", "cost_ratio"):
        lo, hi = agg[f"{key}_ci95"]
        print(f"  {key:11s}= {agg[key]:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  → {'GO (CI excludes 1)' if agg['go_ci_excludes_1'] else 'GO (point)' if agg['go'] else 'NO-GO'}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"  wrote {args.out}")
    return 0


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["moelm", "hf"], default="moelm")
    p.add_argument("--model", default=None, help="HF model id (--backend hf)")
    p.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.",
                   help="prompt text (--backend hf)")
    p.add_argument("--draft", choices=["fakequant", "bnb"], default="fakequant",
                   help="4-bit self-draft source (--backend hf)")
    p.add_argument("--device", default="auto", help="HF device_map (--backend hf)")
    # GSS gate knobs
    p.add_argument("--coverage", type=float, default=0.9)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--draft-byte-frac", type=float, default=0.25)
    p.add_argument("--max-cost-ratio", type=float, default=1.0)
    # moelm knobs
    p.add_argument("--experts", type=int, default=16)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--vocab", type=int, default=32)
    p.add_argument("--context", type=int, default=4)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--campaign", type=int, default=0,
                   help="run N>0 trials (hf: N diverse prompts; moelm: N seeds) and report an "
                        "across-run 95%% CI — the registered-result statement")
    p.add_argument("--out", default=None, help="write the JSON report here")
    args = p.parse_args(argv)

    if not _HAVE_NUMPY:
        print("[gss_probe] skipped: numpy not installed.")
        return 0
    if args.backend == "hf" and not args.model:
        print("[gss_probe] --backend hf requires --model <hf-id>.")
        return 2

    if args.campaign and args.campaign > 0:
        return _run_campaign(args)

    try:
        contribs, target, draft = _run(args)
    except RuntimeError as e:
        print(f"[gss_probe] skipped: {e}")
        return 0   # clean skip keeps CI/orchestrator green, like nvfp4_gemm.py

    gate = GSSFeasibilityGate(gamma=args.gamma, coverage=args.coverage,
                              draft_byte_frac=args.draft_byte_frac,
                              max_cost_ratio=args.max_cost_ratio)
    report = gate.evaluate(contribs, target, draft)
    d = report.as_dict()
    d["backend"] = args.backend
    d["model"] = args.model if args.backend == "hf" else "MoELM(toy)"

    verdict = "GO — worth prototyping GSS (Tier 1+)" if report.go else \
              "NO-GO — GSS cannot beat dense here; do not spend GPU"
    print(f"\n[gss_probe] {d['model']} via {args.backend}")
    print(f"  ρ (read-set fraction)      : {d['rho']}")
    print(f"  α (self-draft acceptance)  : {d['alpha']}")
    print(f"  k (tokens / verify pass)   : {d['k']}")
    print(f"  cost_ratio (γ·{args.draft_byte_frac}+ρ)/k : {d['cost_ratio']}")
    print(f"  speedup ceiling            : {d['speedup_ceiling']}×")
    print(f"  temporal stability (diag)  : {d['temporal_stability']}")
    print(f"  → {verdict}")
    if report.reasons:
        print(f"  reasons: {report.reasons}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(d, indent=2))
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
