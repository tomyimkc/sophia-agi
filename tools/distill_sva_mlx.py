#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""A2 — SVA-lite: salient-vocabulary-aligned multi-teacher distillation (MLX seam).

Implements the stage-3 mechanism of Agents-A1 (arXiv 2606.30616 §2.3) at Sophia
scale: a student LoRA on the frozen Qwen2.5-3B base is supervised by N domain
teachers that are themselves LoRA adapters ON THE SAME BASE — so the paper's
"vocabulary alignment" reduces to its clean core (identical tokenizer, top-k
support + renormalization) with no cross-tokenizer mapping.

Mechanism (their Eqs. 4–6, faithfully):
  * per position t on a STUDENT-generated prefix, take the teacher's top-k
    token support S, renormalize BOTH distributions on S, and take the
    truncated reverse KL  sum_S p̄_s(u) log(p̄_s(u)/p̄_t(u));
  * average over trainable positions (tool outputs / observations masked);
  * HARD domain routing: each sample supervised only by its domain's teacher;
  * two-level aggregation: average within each active domain, then across
    active domains (frequent/high-loss domains cannot dominate);
  * coverage rho = student mass on S is MONITORED, never optimized (their Eq. 5).

Honest seams (do not overclaim):
  * The math core below is pure Python and CI-tested. The MLX training step
    (build_mlx_sva_step) is lazy and fail-closed — it needs Apple-Silicon MLX
    and is validated on the Mac bench lane (mac-mlx-bench.yml), never here.
  * k is UNSPECIFIED in the paper — pre-registered sweep k in {8, 32, 128}.
  * No uplift claim: the B-SVA pre-registered experiment
    (agi-proof/agents-a1-horizon-scaling-2026-07-02/README.md) is the only
    place a claim can come from, and every trained arm must pass the
    post-training calibration/abstention re-audit. candidateOnly:true.

Usage:
  PYTHONPATH=. python3 tools/distill_sva_mlx.py --self-check          # offline invariants
  PYTHONPATH=. python3 tools/distill_sva_mlx.py --train ...           # Mac bench only
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any, Callable, Mapping, Sequence

DEFAULT_TOP_K = 32
SCHEMA = "sophia.sva_distill.v1"


# --------------------------------------------------------------------------- #
# Pure math core (CI-tested; no mlx required)
# --------------------------------------------------------------------------- #
def teacher_topk_support(p_teacher: Mapping[Any, float], k: int) -> list:
    """Top-k tokens under the TEACHER distribution (deterministic tie-break)."""
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    return [t for t, _ in sorted(p_teacher.items(), key=lambda kv: (-kv[1], str(kv[0])))[:k]]


def sva_position_loss(p_student: Mapping[Any, float], p_teacher: Mapping[Any, float],
                      *, k: int = DEFAULT_TOP_K, eps: float = 1e-12) -> "tuple[float, float]":
    """(truncated reverse KL on the teacher's top-k support, coverage rho).

    Both distributions are renormalized on the teacher-selected support before
    the KL; rho is the student's UN-renormalized mass on that support (Eq. 5).
    """
    support = teacher_topk_support(p_teacher, k)
    s_mass = sum(p_student.get(u, 0.0) for u in support)
    t_mass = sum(p_teacher.get(u, 0.0) for u in support)
    rho = min(1.0, max(0.0, s_mass))
    if s_mass <= eps or t_mass <= eps:
        # student has no mass on the teacher's salient support: maximal signal
        # is undefined under renormalization — fail closed to a large finite
        # penalty rather than dividing by ~0 (keeps training numerically sane).
        return math.log(1.0 / eps), rho
    kl = 0.0
    for u in support:
        ps = p_student.get(u, 0.0) / s_mass
        pt = p_teacher.get(u, 0.0) / t_mass
        if ps > eps:
            kl += ps * math.log(ps / max(pt, eps))
    return max(0.0, kl), rho


def sva_sequence_loss(positions: "Sequence[tuple[Mapping, Mapping, bool]]",
                      *, k: int = DEFAULT_TOP_K) -> "tuple[float, float, int]":
    """Average SVA loss over TRAINABLE positions (mask=False positions skipped).

    positions: iterable of (p_student, p_teacher, trainable). Returns
    (mean_loss, mean_rho, n_trainable); a sequence with no trainable positions
    contributes (0, 1, 0) — nothing to train on, not an error.
    """
    losses, rhos = [], []
    for p_s, p_t, trainable in positions:
        if not trainable:
            continue
        l, r = sva_position_loss(p_s, p_t, k=k)
        losses.append(l)
        rhos.append(r)
    if not losses:
        return 0.0, 1.0, 0
    return sum(losses) / len(losses), sum(rhos) / len(rhos), len(losses)


def domain_normalized_aggregate(sample_losses: "Sequence[tuple[str, float]]") -> float:
    """Their Eq. 6: average within each active domain, then across active domains."""
    by_domain: dict[str, list[float]] = {}
    for domain, loss in sample_losses:
        by_domain.setdefault(str(domain), []).append(float(loss))
    if not by_domain:
        return 0.0
    return sum(sum(v) / len(v) for v in by_domain.values()) / len(by_domain)


def route_teacher(domain: str, teachers: Mapping[str, Any]) -> Any:
    """HARD routing (their §2.3.2): the sample's domain teacher or fail-closed."""
    if domain not in teachers:
        raise KeyError(
            f"no teacher registered for domain {domain!r} (hard routing is fail-closed; "
            f"available: {sorted(teachers)})")
    return teachers[domain]


# --------------------------------------------------------------------------- #
# MLX training step (lazy, fail-closed; validated on the Mac bench lane only)
# --------------------------------------------------------------------------- #
def build_mlx_sva_step(base_model_id: str, teacher_adapters: Mapping[str, str],
                       student_adapter: str, *, k: int = DEFAULT_TOP_K):
    """Return an MLX loss closure for one (domain, token_ids, mask) sample batch.

    Teachers and student share ``base_model_id``; teachers are frozen LoRA
    adapters (logits computed under stop_gradient), the student is the
    trainable LoRA. Raises RuntimeError where MLX is unavailable so the pure
    core above stays the offline default.
    """
    try:
        import mlx.core as mx
        import mlx.nn as nn  # noqa: F401
        from mlx_lm import load as _load
    except Exception as e:  # pragma: no cover - exercised only without mlx
        raise RuntimeError(
            "SVA MLX step requires mlx/mlx_lm (Apple Silicon bench); the pure math "
            f"core remains usable offline. (import failed: {type(e).__name__}: {e})"
        ) from e

    student, tokenizer = _load(base_model_id, adapter_path=student_adapter)
    teachers = {d: _load(base_model_id, adapter_path=p)[0] for d, p in teacher_adapters.items()}

    def loss_fn(domain: str, token_ids: "list[int]", trainable_mask: "list[bool]"):
        teacher = route_teacher(domain, teachers)
        ids = mx.array([token_ids])
        t_logits = mx.stop_gradient(teacher(ids))[0]
        s_logits = student(ids)[0]
        total, rhos, n = mx.array(0.0), [], 0
        for t in range(1, len(token_ids)):
            if not trainable_mask[t]:
                continue
            t_probs = mx.softmax(t_logits[t - 1])
            s_probs = mx.softmax(s_logits[t - 1])
            top = mx.argpartition(-t_probs, k)[:k]
            ps, pt = s_probs[top], t_probs[top]
            ps_n = ps / mx.maximum(mx.sum(ps), 1e-12)
            pt_n = pt / mx.maximum(mx.sum(pt), 1e-12)
            total = total + mx.sum(ps_n * (mx.log(ps_n + 1e-12) - mx.log(pt_n + 1e-12)))
            rhos.append(float(mx.sum(ps)))
            n += 1
        loss = total / max(1, n)
        return loss, (sum(rhos) / len(rhos) if rhos else 1.0), n

    loss_fn.tokenizer = tokenizer  # type: ignore[attr-defined]
    return loss_fn


# --------------------------------------------------------------------------- #
# Offline invariants (CI-gated via tests/test_distill_sva_mlx.py)
# --------------------------------------------------------------------------- #
def offline_invariants(*, k: int = 3) -> "tuple[bool, dict]":
    t = {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}
    matched, rho_m = sva_position_loss(t, t, k=k)
    far = {"d": 0.7, "c": 0.2, "a": 0.05, "b": 0.05}
    off, rho_o = sva_position_loss(far, t, k=k)
    seq_loss, _, n = sva_sequence_loss(
        [(t, t, True), (far, t, True), (far, t, False)], k=k)
    agg = domain_normalized_aggregate(
        [("d1", 1.0), ("d1", 1.0), ("d1", 1.0), ("d2", 0.0)])
    try:
        route_teacher("missing", {"d1": object()})
        routing_fail_closed = False
    except KeyError:
        routing_fail_closed = True

    checks = {
        "identicalIsZero": abs(matched) < 1e-9 and abs(rho_m - (0.5 + 0.3 + 0.15)) < 1e-9,
        "mismatchPositive": off > 0.0,
        "rhoInUnitInterval": 0.0 <= rho_o <= 1.0,
        "maskExcluded": n == 2,
        "sequenceIsMeanOfTrainable": abs(seq_loss - (matched + off) / 2) < 1e-9,
        "domainNormalizedEqualWeight": abs(agg - 0.5) < 1e-9,  # not 0.75 (freq-weighted)
        "hardRoutingFailsClosed": routing_fail_closed,
        "badKFailsClosed": _raises(lambda: teacher_topk_support(t, 0)),
        "deterministic": sva_position_loss(far, t, k=k) == (off, rho_o),
    }
    detail = {"k": k, "losses": {"matched": matched, "off": off}, "checks": checks,
              "note": "math-core invariants only; the MLX step is Mac-bench-validated; "
                      "no uplift claim (candidateOnly, B-SVA is the pre-registered test)."}
    return all(checks.values()), detail


def _raises(fn: Callable) -> bool:
    try:
        fn()
        return False
    except ValueError:
        return True


def main(argv: "Sequence[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="A2 SVA-lite multi-teacher distillation")
    ap.add_argument("--self-check", action="store_true", help="run offline invariants (no mlx)")
    ap.add_argument("--train", action="store_true", help="MLX training (Apple Silicon bench only)")
    ap.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    args = ap.parse_args(argv)

    if args.train:
        print(json.dumps({"schema": SCHEMA, "ok": False, "candidateOnly": True,
                          "reason": "training loop is the Mac-bench seam; use "
                                    "build_mlx_sva_step from a bench script"}, indent=2))
        return 2
    ok, detail = offline_invariants(k=min(args.k, 3))
    print(json.dumps({"schema": SCHEMA, "ok": ok, "candidateOnly": True,
                      "level3Evidence": False, "canClaimAGI": False, **detail}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
