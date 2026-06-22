"""Self-improving + synthesized skills (P4) — the flywheel wired into the gateway.

  - improve_skill: from labeled examples a skill abstains on, synthesize + validate a
    verifier; if it clears the bar, upgrade the skill to a deterministic-verified one
    (its outputs now checked by the learned rule) — competence rises.
  - synthesize_skill: verifier-FIRST creation. Synthesize + validate a verifier for a new
    domain; only if promoted, register a classifier skill backed by it. Else abstain
    (return None) — never ship a skill whose checker could not be validated.

Reuses selfextend (synthesize/validate). Deterministic, offline.
"""

from __future__ import annotations

from gateway.registry import ToolEntry
from selfextend.verifier_synthesis import propose_and_validate, stratified_split, synthesize_verifier


def improve_skill(gateway, skill_id: str, examples: "list[tuple[str, bool]]", *,
                  threshold: float = 0.8) -> dict:
    """Synthesize+validate a verifier from examples; if promoted, attach it to the skill
    (verifier_ref -> 'synthesized') so the gateway enforces it. Returns the outcome."""
    entry = gateway.registry.get(skill_id)
    if entry is None:
        return {"skill_id": skill_id, "improved": False, "reason": "unknown skill"}
    train, heldout = stratified_split(examples)
    result = propose_and_validate(train, heldout, threshold=threshold)
    if result["promoted"]:
        rule = synthesize_verifier(train)
        gateway.attach_synthesized_verifier(skill_id, rule)   # interceptor support
    return {"skill_id": skill_id, "improved": result["promoted"],
            "heldoutAccuracy": result["heldoutAccuracy"], "rule": result.get("rule")}


def synthesize_skill(gateway, domain: str, examples: "list[tuple[str, bool]]", *,
                     blp_level: str = "UNCLASSIFIED", threshold: float = 0.8) -> "dict":
    """Verifier-first skill creation. If a verifier validates, register a classifier skill
    that predicts the concept; else abstain (no skill)."""
    train, heldout = stratified_split(examples)
    result = propose_and_validate(train, heldout, threshold=threshold)
    if not result["promoted"]:
        return {"created": False, "domain": domain, "reason": "verifier failed validation",
                "heldoutAccuracy": result["heldoutAccuracy"]}
    rule = synthesize_verifier(train)
    skill_id = f"skill.{domain}"

    def _program(args, _rule=rule):
        text = str(args.get("text", ""))
        return {"answer": bool(_rule.predict(text)), "sources": [f"synthesized:{skill_id}"]}

    entry = ToolEntry(id=skill_id, handler=_program, kind="skill", verifier_ref="grounding",
                      blp_level=blp_level, side_effects="none",
                      description=f"synthesized classifier for {domain}")
    gateway.register(entry)
    return {"created": True, "skill_id": skill_id, "domain": domain,
            "heldoutAccuracy": result["heldoutAccuracy"], "rule": result["rule"]}
