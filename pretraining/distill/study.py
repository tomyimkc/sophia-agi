# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Distillation-into-sparsity study — teacher quality at the *active* (low-RAM) cost.

*The question this study answers (falsifiably).* "Low RAM at release" via sparsity means a
model whose *total* params are large (so it is capable) but whose *active* params per token
are small (so its resident/compute footprint is small — the MoE thesis,
``docs/11-Platform/Cheap-Compute-Boundary.md`` Boundary 1, and the weight-streaming target of
``serving/expert_offload.py``). The obvious worry: does a sparse student trained at that small
*active* budget actually *learn* enough, or does sparsity just trade quality for the saving?
Distillation is the standard answer — let a high-capacity dense **teacher** supply the
training signal so the cheap student inherits its function. So the measurable question is:

  **At matched *active* compute, does a sparse MoE student distilled from a dense teacher
  beat a dense student of the same active size — i.e. does sparsity+distillation recover
  teacher-level quality at a fraction of the active (RAM-at-release) cost?**

Why the nano substrate makes this honest
----------------------------------------
On a real LLM the irreducible loss floor ``E`` is unknown, so you cannot tell whether a
student fell short because the *task* is hard (floor) or because *distillation/sparsity*
failed. The nano substrate's corpus is an order-``k`` Markov source whose conditional entropy
``E = source_entropy(source)`` is **closed-form** (``pretraining/nano/data.py``). So every
arm's loss is read against a *known* floor, and the distillation/sparsity effect is cleanly
attributable — the same identity the QAT study (``pretraining/qat/study.py``) and the
sparse-quant study (``pretraining/architecture/run_sparse_quant.py``) use.

The arms (all evaluated against the same held-out set and the same floor ``E``)
  - ``teacher``        : a *large-hidden* dense ``NanoLM`` trained on true labels. Its eval
                         loss ``L_teacher`` is the quality target (approaches ``E``).
  - ``dense_student``  : a *small-hidden* dense ``NanoLM`` (the matched active budget),
                         distilled from the teacher (trained on teacher-relabeled targets).
  - ``moe_student``    : a top-1 ``MoELM`` whose *active* params ≈ the dense student's, but
                         whose *total* params are larger (many experts), distilled the same way.

The falsifiable claim: if ``sparsity_helps`` holds across seeds, the MoE student reaches a
lower loss than the equally-active dense student *at the same active cost* — sparsity buys
real capacity that distillation can fill, which is the "large params / small active RAM"
thesis measured against ground truth. We do **not** claim nano reproduces frontier MoE; we
claim the *effect is measurable* against a known floor.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pretraining.architecture.moe import MoELM
from pretraining.nano.data import (
    make_source,
    sample_stream,
    source_entropy,
    to_examples,
)
from pretraining.nano.model import NanoLM, eval_loss
from pretraining.nano.train import train


# ---------------------------------------------------------------------------
# 1. Distillation: relabel the training contexts with the teacher's predictions
# ---------------------------------------------------------------------------

def teacher_relabel(teacher: NanoLM, examples) -> "list[tuple[list[int], int]]":
    """Hard-distillation targets: replace each example's label with the teacher's argmax.

    The student then learns the teacher's *learned function* over the input distribution,
    not the raw labels. Hard (argmax) distillation keeps the existing single-label train
    loops (``nano.train`` / ``MoELM.train_step``) usable unchanged; soft-label KL is the
    obvious extension but unnecessary for the measurable effect here.
    """
    out = []
    for ctx, _true in examples:
        _, probs = teacher.forward(ctx)
        pred = max(range(len(probs)), key=lambda k: probs[k])
        out.append((ctx, pred))
    return out


def _train_moe(model: MoELM, examples, *, epochs: int, lr: float, seed: int) -> float:
    """Train a MoELM over ``examples`` for ``epochs`` and return the final mean loss."""
    import random

    rng = random.Random(seed)
    order = list(range(len(examples)))
    last = float("nan")
    for _ in range(epochs):
        rng.shuffle(order)
        total = 0.0
        for j in order:
            ctx, t = examples[j]
            total += model.train_step(ctx, t, lr)
        last = total / max(1, len(examples))
    return last


def _eval_moe(model: MoELM, examples) -> float:
    if not examples:
        return float("nan")
    return sum(model.nll(ctx, t) for ctx, t in examples) / len(examples)


# ---------------------------------------------------------------------------
# 2. The study
# ---------------------------------------------------------------------------

def run_study(*, vocab: int = 6, context: int = 1,
              teacher_hidden: int = 48, student_hidden: int = 8,
              n_experts: int = 6, n_train: int = 500, n_eval: int = 250,
              epochs: int = 40, lr: float = 0.1, seed: int = 0) -> "dict[str, Any]":
    """Train teacher + two distilled students; measure quality vs the known floor.

    Returns a report with each arm's eval loss, gap-to-floor, the active/total param ratio
    of the MoE student (the RAM-at-release lever), and the two verdicts:
      - ``distill_grounded`` : the teacher actually learned (``L_teacher`` near ``E``), so the
        distillation signal is meaningful (a guard against a vacuous comparison).
      - ``sparsity_helps``   : the MoE student beats the equally-active dense student.
    """
    src = make_source(vocab, order=context, seed=seed)
    E = source_entropy(src)
    train_ex = to_examples(sample_stream(src, n_train, seed=seed), context)
    eval_ex = to_examples(sample_stream(src, n_eval, seed=seed + 999), context)

    # Teacher: high-capacity dense model on true labels.
    teacher = NanoLM(vocab, context, teacher_hidden, seed=seed)
    train(teacher, train_ex, epochs=epochs, optimizer="adam", lr=lr, seed=seed)
    L_teacher = eval_loss(teacher, eval_ex)

    # Distillation targets from the teacher.
    distilled_ex = teacher_relabel(teacher, train_ex)

    # Dense student: small-hidden dense model, distilled.
    dense_student = NanoLM(vocab, context, student_hidden, seed=seed + 1)
    train(dense_student, distilled_ex, epochs=epochs, optimizer="adam", lr=lr, seed=seed + 1)
    L_dense = eval_loss(dense_student, eval_ex)

    # MoE student: many experts, ~same active size, distilled.
    moe_student = MoELM(vocab, context, student_hidden, n_experts, seed=seed + 2)
    _train_moe(moe_student, distilled_ex, epochs=epochs, lr=lr, seed=seed + 2)
    L_moe = _eval_moe(moe_student, eval_ex)

    def gap(L: float) -> float:
        return L - max(E, L_teacher)

    active = moe_student.active_params()
    total = moe_student.num_params()
    return {
        "E": E,
        "arms": {
            "teacher": {"L_fp": L_teacher, "params": teacher.num_params(),
                        "active_params": teacher.num_params()},
            "dense_student": {"L_fp": L_dense, "params": dense_student.num_params(),
                              "active_params": dense_student.num_params(), "gap": gap(L_dense)},
            "moe_student": {"L_fp": L_moe, "params": total, "active_params": active,
                            "gap": gap(L_moe)},
        },
        "moe_active_fraction": active / total,           # the RAM-at-release lever (<1 = sparse)
        "moe_total_over_active": total / active,
        "distill_grounded": L_teacher < E * 1.5,         # teacher meaningfully learned
        "sparsity_helps": L_moe < L_dense,               # MoE beats equally-active dense
        "sparsity_margin": L_dense - L_moe,              # >0 means sparsity+distill won
        "config": {"vocab": vocab, "context": context, "teacher_hidden": teacher_hidden,
                   "student_hidden": student_hidden, "n_experts": n_experts,
                   "epochs": epochs, "lr": lr, "seed": seed},
        "honest_scope": (
            "Nano-scale methodology result only. Demonstrates the *measurability* of a "
            "distillation-into-sparsity effect against a known irreducible-loss floor — NOT "
            "that it scales to frontier MoE. A real low-active-RAM capability claim needs the "
            "same measurement on a real model to the no-overclaim gate (RESULTS.md)."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Offline invariants
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    detail: dict = {}

    # 1. teacher_relabel produces in-vocab labels and preserves contexts/length.
    src = make_source(8, order=2, seed=0)
    ex = to_examples(sample_stream(src, 60, seed=0), 2)
    t = NanoLM(8, 2, 16, seed=0)
    train(t, ex, epochs=3, optimizer="adam", lr=0.05, seed=0)
    rel = teacher_relabel(t, ex)
    checks["relabel_same_len"] = len(rel) == len(ex)
    checks["relabel_contexts_preserved"] = all(rel[i][0] == ex[i][0] for i in range(len(ex)))
    checks["relabel_labels_in_vocab"] = all(0 <= lbl < 8 for _, lbl in rel)

    # 2. The MoE student is genuinely sparse: active params strictly < total params.
    moe = MoELM(8, 2, 12, 6, seed=0)
    checks["moe_is_sparse"] = moe.active_params() < moe.num_params()
    detail["moe_active_fraction"] = round(moe.active_params() / moe.num_params(), 4)

    # 3. The study runs and reports all required fields, with a known positive floor.
    #    ctx=1 (order-1 Markov) is the regime where the nano teacher reliably reaches the
    #    floor, so the distillation signal is grounded (a strong teacher to distill from).
    rep = run_study(vocab=6, context=1, teacher_hidden=48, student_hidden=8,
                    n_experts=6, n_train=400, n_eval=200, epochs=40, lr=0.1, seed=0)
    required = {"E", "arms", "moe_active_fraction", "sparsity_helps", "honest_scope"}
    checks["study_complete"] = required.issubset(rep.keys())
    checks["study_has_floor"] = rep["E"] > 0.0
    checks["all_losses_finite"] = all(
        math.isfinite(a["L_fp"]) and a["L_fp"] > 0 for a in rep["arms"].values())

    # 4. The teacher learned (loss near the floor) — distillation signal is grounded.
    checks["teacher_grounded"] = rep["distill_grounded"]
    detail["E"] = round(rep["E"], 4)
    detail["L_teacher"] = round(rep["arms"]["teacher"]["L_fp"], 4)
    detail["L_dense"] = round(rep["arms"]["dense_student"]["L_fp"], 4)
    detail["L_moe"] = round(rep["arms"]["moe_student"]["L_fp"], 4)
    detail["sparsity_margin"] = round(rep["sparsity_margin"], 4)
    detail["sparsity_helps"] = rep["sparsity_helps"]

    # 5. The MoE student has strictly more total capacity than its active footprint
    #    (the "large params / small active RAM" identity the study is about).
    checks["moe_total_exceeds_active"] = rep["moe_total_over_active"] > 1.0
    detail["moe_total_over_active"] = round(rep["moe_total_over_active"], 3)

    # 6. The honest-scope caveat is present (no-overclaim escape hatch).
    checks["scope_present"] = "Nano-scale methodology result only" in rep["honest_scope"]

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Distill-study offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print(f"  floor E={detail.get('E')} | L_teacher={detail.get('L_teacher')} "
          f"L_dense={detail.get('L_dense')} L_moe={detail.get('L_moe')}")
    print(f"  sparsity_margin={detail.get('sparsity_margin')} (>0 = MoE wins) "
          f"| MoE total/active={detail.get('moe_total_over_active')}x "
          f"(active frac {detail.get('moe_active_fraction')})")
    raise SystemExit(0 if ok else 1)
