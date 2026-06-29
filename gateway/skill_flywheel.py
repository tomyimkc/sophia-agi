# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Self-improving + synthesized skills (P4) wired to strong verifier synthesis.

This module now uses :mod:`agent.verifier_synthesis`, not the toy substring stump
in ``selfextend/``.  That matters because the strong path supports:

  - disjoint fit / validation / test splits;
  - precision + recall promotion floors;
  - AST-sandboxed LLM-proposed predicates via ``propose_fn``;
  - explicit abstention when no candidate verifier clears validation.

Public API remains compatible with the previous ``[(text, bool)]`` examples.
The bool means "this text/output should be accepted".
"""

from __future__ import annotations

from typing import Callable

from agent.verifier_synthesis import SynthesisResult, compose, synthesize
from gateway.registry import ToolEntry


def _task_from_pairs(domain: str, examples: "list[tuple[str, bool]]") -> dict:
    return {
        "task_id": domain,
        "examples": [
            {"answer": str(text), "label": bool(label), "_idx": i}
            for i, (text, label) in enumerate(examples or [])
        ],
    }


def _summarize(res: SynthesisResult) -> dict:
    report = res.report()
    return {
        "abstained": res.abstained,
        "metaVerified": res.meta_verified,
        "admitted": report.get("admitted", []),
        "rejectedCount": report.get("rejectedCount", 0),
        "splits": report.get("splits", {}),
        "testStats": report.get("testStats"),
    }


def _promoted(res: SynthesisResult, *, threshold: float) -> bool:
    if res.abstained or not res.admitted:
        return False
    # Promotion is earned on validation via agent.verifier_synthesis.  If a test
    # split exists, require it not to contradict the validation claim.  Tiny test
    # splits with no positive/negative class can make recall vacuous, so accuracy
    # is a safer guard here.
    if res.test_stats is not None and res.test_stats.n >= 2:
        return res.test_stats.accuracy >= threshold
    return True


def synthesize_gate(
    domain: str,
    examples: "list[tuple[str, bool]]",
    *,
    threshold: float = 0.8,
    seed: int = 1,
    propose_fn: Callable | None = None,
) -> tuple[bool, SynthesisResult]:
    """Fit + meta-verify a gate from labelled examples.

    ``propose_fn`` may call an LLM to propose extra predicates, but those
    predicates are admitted only after the same AST sandbox and validation floors
    enforced by ``agent.verifier_synthesis``.
    """
    task = _task_from_pairs(domain, examples)
    res = synthesize(
        task,
        seed=seed,
        fit_frac=0.5,
        val_frac=0.25,
        min_precision=max(0.8, threshold),
        min_recall=threshold,
        compose_mode="any",  # one validated sufficient condition may accept
        propose_fn=propose_fn,
    )
    return _promoted(res, threshold=threshold), res


def improve_skill(
    gateway,
    skill_id: str,
    examples: "list[tuple[str, bool]]",
    *,
    threshold: float = 0.8,
    seed: int = 1,
    propose_fn: Callable | None = None,
) -> dict:
    """Synthesize + meta-verify a verifier from abstention/failure examples.

    If promoted, attach the harness-compatible synthesized verifier to the skill;
    otherwise leave the skill unchanged and keep abstaining/failing closed.
    """
    entry = gateway.registry.get(skill_id)
    if entry is None:
        return {"skill_id": skill_id, "improved": False, "reason": "unknown skill"}
    ok, res = synthesize_gate(skill_id, examples, threshold=threshold, seed=seed, propose_fn=propose_fn)
    if ok:
        gateway.attach_synthesized_verifier(skill_id, res.gate)
    return {"skill_id": skill_id, "improved": ok, **_summarize(res)}


def synthesize_skill(
    gateway,
    domain: str,
    examples: "list[tuple[str, bool]]",
    *,
    blp_level: str = "UNCLASSIFIED",
    threshold: float = 0.8,
    seed: int = 1,
    propose_fn: Callable | None = None,
) -> "dict":
    """Verifier-first skill creation.

    From labelled text examples, synthesize a classifier skill that returns
    ``{"answer": bool, "sources": ["skillforge://..."]}``.  The skill is
    registered only if its synthesized verifier clears validation/test gates.
    Otherwise no skill is shipped.
    """
    ok, res = synthesize_gate(domain, examples, threshold=threshold, seed=seed, propose_fn=propose_fn)
    if not ok or res.gate is None:
        return {"created": False, "domain": domain, "reason": "verifier failed validation", **_summarize(res)}

    # Reuse the admitted candidate predicates directly for classification.
    predicate = compose(res.admitted, mode="any")
    skill_id = f"skill.{domain}"

    def _program(args, _pred=predicate, _skill_id=skill_id):
        text = str(args.get("text") or args.get("query") or args.get("input") or "")
        return {"answer": bool(_pred(text)), "sources": [f"skillforge://{_skill_id}"]}

    entry = ToolEntry(
        id=skill_id,
        handler=_program,
        kind="skill",
        verifier_ref="grounding",
        blp_level=blp_level,
        side_effects="none",
        description=f"meta-verified synthesized classifier for {domain}",
    )
    gateway.register(entry)
    return {"created": True, "skill_id": skill_id, "domain": domain, **_summarize(res)}
