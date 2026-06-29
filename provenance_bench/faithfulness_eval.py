# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Held-out faithfulness eval — the measurement instrument for the reasoning core.

The training reward (``retrieval_faithfulness``) and loop (``faithfulness_grpo``) are
the *intervention*; this module is the *instrument* that decides whether the
intervention worked, under the repo's measurement contract (``measurement-thesis.md``):
pre-registered metric, uncertainty quantified, powered to an MDE, anytime-valid under
peeking. Per the contract, the instrument is built and validated BEFORE any training
claim — a reward you cannot measure cleanly yields a confident wrong verdict.

Primary metric — ``counterfactual_grounding_rate``:

    over all KNOWLEDGE claims a policy makes on the held-out set,
    the fraction that are SUPPORTED by a retrieved chunk AND DISAPPEAR when that
    chunk is dropped (survives_ablation == False).

A claim that is unsupported, or that survives its support being dropped (leaked from
the weights), does NOT count as grounded. This is exactly the reward's faithfulness
signal, lifted to a population metric with a CI. ``compare`` does the paired
base-vs-adapter contrast (``bootstrap_ci_paired`` + ``verdict_or_underpowered``) the
GO/NO-GO gate consumes.

Deterministic over a fixed trajectory set; the policy generation that produces the
trajectories is injected (mock offline; a live model/API for a real run). No capability
is claimed here — this ships the instrument. canClaimAGI:false.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_stats import (  # noqa: E402
    bootstrap_ci_paired,
    confidence_sequence_mean,
    fixed_n_ci_mean,
    mde_at_n,
)


def case_grounding(traj: dict) -> tuple[int, int]:
    """(grounded, total) over a trajectory's KNOWLEDGE claims. Grounded == supported and
    NOT surviving the citation-drop (genuinely dependent on retrieval)."""
    kclaims = [c for c in (traj.get("claims") or [])
               if str(c.get("kind", "knowledge")) == "knowledge"]
    grounded = sum(1 for c in kclaims
                   if str(c.get("verdict")) == "supported" and not c.get("survives_ablation"))
    return grounded, len(kclaims)


def counterfactual_grounding_rate(trajs: list) -> dict:
    """Micro-averaged grounding rate over all knowledge claims + the per-case macro
    list (each case's own rate) used for the CI. Also reports coverage (cases that made
    a claim) and abstention rate (fail-closed is not penalized as ungrounded)."""
    per_case, g_tot, c_tot, abstained = [], 0, 0, 0
    for t in trajs:
        if t.get("abstained"):
            abstained += 1
            continue
        g, c = case_grounding(t)
        g_tot += g
        c_tot += c
        if c:
            per_case.append(g / c)
    return {
        "rate": (g_tot / c_tot) if c_tot else None,
        "perCaseRates": per_case,
        "groundedClaims": g_tot,
        "knowledgeClaims": c_tot,
        "casesWithClaims": len(per_case),
        "abstentionRate": (abstained / len(trajs)) if trajs else None,
        "n": len(trajs),
    }


def evaluate(cases: list, *, generate: Callable, retrieve: Callable,
             extract_claims: Callable, verify_claim: Callable,
             check_correct: Callable | None = None) -> dict:
    """Run the policy over the held-out ``cases`` and report the grounding rate with a
    fixed-n CI and an anytime-valid confidence sequence (the workflow peeks/iterates, so
    the CS is the honest interval — measurement-thesis pillar 4)."""
    from provenance_bench.faithfulness_rollout import rollout

    seams = dict(retrieve=retrieve, extract_claims=extract_claims,
                 verify_claim=verify_claim, check_correct=check_correct)
    trajs = [rollout(c, generate=generate, **seams) for c in cases]
    agg = counterfactual_grounding_rate(trajs)
    vals = agg["perCaseRates"]
    agg["fixedNCI95"] = fixed_n_ci_mean(vals) if vals else None
    agg["anytimeValidCS95"] = confidence_sequence_mean(vals) if len(vals) >= 2 else None
    agg["mdeAtN"] = mde_at_n(len(vals)) if vals else None
    return {"aggregate": agg, "trajectories": trajs}


def compare(cases: list, *, base_generate: Callable, adapter_generate: Callable,
            retrieve: Callable, extract_claims: Callable, verify_claim: Callable,
            check_correct: Callable | None = None) -> dict:
    """Paired base-vs-adapter grounding-rate contrast on the SAME held-out cases. Returns
    the per-case paired diffs, a paired bootstrap 95% CI, and the aggregate rates — the
    inputs ``tools/claim_gate.py`` turns into a GO/NO-GO on a powered, CI-clean uplift."""
    base = evaluate(cases, generate=base_generate, retrieve=retrieve,
                    extract_claims=extract_claims, verify_claim=verify_claim,
                    check_correct=check_correct)
    adapter = evaluate(cases, generate=adapter_generate, retrieve=retrieve,
                       extract_claims=extract_claims, verify_claim=verify_claim,
                       check_correct=check_correct)
    # Pair per-case rates over cases where BOTH made a claim (others are non-comparable).
    b_traj, a_traj = base["trajectories"], adapter["trajectories"]
    diffs = []
    for bt, at in zip(b_traj, a_traj):
        bg, bc = case_grounding(bt)
        ag, ac = case_grounding(at)
        if bc and ac:
            diffs.append((ag / ac) - (bg / bc))
    return {
        "baseRate": base["aggregate"]["rate"],
        "adapterRate": adapter["aggregate"]["rate"],
        "pairedDiffs": diffs,
        "meanDiff": (sum(diffs) / len(diffs)) if diffs else None,
        "pairedBootstrapCI95": bootstrap_ci_paired(diffs) if diffs else None,
        "nPaired": len(diffs),
        "claimCeiling": "candidate_only; canClaimAGI:false",
    }


def make_hf_compare_policies(model_name: str, adapter_path: str):
    """Load a local HF model with a LoRA adapter ONCE and return
    ``(base_generate, adapter_generate, label)`` — two rollout ``generate`` seams that
    share the weights, toggling the adapter on/off (``disable_adapter()`` = the frozen
    base = the comparison baseline). Greedy decoding for eval reproducibility. This is the
    local-HF policy seam that lets the on-pod base-vs-adapter faithfulness eval run on the
    TRAINED adapter (not an API model). Torch/transformers/peft, CUDA — gated, not CI."""
    from contextlib import nullcontext

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from provenance_bench.faithfulness_grpo import _build_prompt

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda().eval()
    model = PeftModel.from_pretrained(base, adapter_path).eval()

    def _gen(query: str, context_chunks: list, *, use_base: bool) -> str:
        ids = tok(_build_prompt(query, context_chunks), return_tensors="pt").to(model.device)
        cm = model.disable_adapter() if use_base else nullcontext()
        with torch.no_grad(), cm:
            out = model.generate(**ids, max_new_tokens=160, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    base_generate = lambda q, c: _gen(q, c, use_base=True)       # noqa: E731
    adapter_generate = lambda q, c: _gen(q, c, use_base=False)   # noqa: E731
    return base_generate, adapter_generate, f"hf:{model_name}+adapter"


def offline_invariants() -> tuple[bool, dict]:
    """Validate the instrument (no torch/GPU/network): on the mock world, a retrieval-
    using (faithful) policy must score a higher grounding rate than a weights-leaking
    one, the paired contrast must favour the faithful policy, and the metric must be
    bounded in [0, 1]."""
    from provenance_bench import faithfulness_rollout as fr

    cases = [
        {"prompt": "Who wrote the Project Phoenix Charter?", "should_retrieve": True,
         "answerable": True, "gold": "founding committee"},
    ] * 6
    seams = dict(retrieve=fr._mock_retrieve, extract_claims=fr._mock_extract,
                 verify_claim=fr._mock_verify, check_correct=fr._check_correct)

    faithful = evaluate(cases, generate=fr._faithful_policy, **seams)
    leaky = evaluate(cases, generate=fr._leaky_policy, **seams)
    contrast = compare(cases, base_generate=fr._leaky_policy,
                       adapter_generate=fr._faithful_policy, **seams)

    f_rate = faithful["aggregate"]["rate"]
    l_rate = leaky["aggregate"]["rate"]
    checks = {
        "faithfulFullyGrounded": f_rate == 1.0,
        "leakyNotGrounded": l_rate == 0.0,
        "faithfulBeatsLeaky": f_rate > l_rate,
        "pairedContrastFavoursFaithful": (contrast["meanDiff"] or 0) > 0.0,
        "rateBounded": all(0.0 <= r <= 1.0 for r in (f_rate, l_rate)),
        "ciPresent": faithful["aggregate"]["fixedNCI95"] is not None,
    }
    detail = {
        "checks": checks,
        "faithfulRate": f_rate,
        "leakyRate": l_rate,
        "meanDiff": contrast["meanDiff"],
        "faithfulCI95": faithful["aggregate"]["fixedNCI95"],
        "note": "instrument validation only — no model/policy capability is claimed.",
    }
    return all(checks.values()), detail


__all__ = ["counterfactual_grounding_rate", "case_grounding", "evaluate", "compare",
           "make_hf_compare_policies", "offline_invariants"]
