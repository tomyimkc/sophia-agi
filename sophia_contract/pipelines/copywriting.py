"""Role 5 — copywriting-in-a-bespoke-voice, end to end through the contract.

The loop the playbook describes, made real:

    brief note  ->  draft (Role/Data/Requirements/Format prompt)  ->  draft note in
    06_Review/  ->  VaultGate (record_claim + verify_claim, stamp frontmatter)  ->
    [accepted -> publishable]  or  [held(needs_human) -> founder approves -> publishable]

Offline-safe: the default drafter is the repo's model router (`agent.model`), which
falls back to a deterministic mock with no API key — so the whole loop runs in CI.
Confidential briefs escalate to human review (the approve-by-exception path); an
UNCLASSIFIED, sourced brief auto-accepts.
"""

from __future__ import annotations

from pathlib import Path

from okf import frontmatter
from sophia_contract.service import SophiaContract
from sophia_contract.vault import VaultGate

ROLE = "role_05_copywriting"

_SYSTEM = (
    "You are Sophia's copywriter. Write in the client's voice. Use only the supplied "
    "brief and brand facts; do NOT invent quotes, statistics, names, or citations — if "
    "a specific is unknown, leave it out. Source discipline over flourish."
)


def _default_drafter(system: str, user: str, *, model_spec: "str | None") -> str:
    from agent.model import complete

    return complete(system, user, spec=model_spec)


class CopywritingPipeline:
    def __init__(self, contract: "SophiaContract | None" = None, *,
                 vault_root: "str | Path | None" = None, model_spec: "str | None" = None,
                 drafter=None):
        self.contract = contract or SophiaContract()
        self.vault_root = Path(vault_root) if vault_root else Path(".")
        self.review_dir = self.vault_root / "06_Review"
        self.model_spec = model_spec
        self.gate = VaultGate(self.contract, vault_root=self.vault_root)
        self._drafter = drafter or (lambda s, u: _default_drafter(s, u, model_spec=model_spec))

    def build_prompt(self, brief: str, *, voice_guide: str = "", exemplars: "list[str]" = None,
                     requirements: "list[str]" = None, fmt: str = "") -> str:
        """The four-step template: Role / Data / Requirements / Format."""
        parts = ["## ROLE", _SYSTEM, "", "## DATA", f"Brief:\n{brief}"]
        if voice_guide:
            parts += ["", f"Voice guide:\n{voice_guide}"]
        if exemplars:
            parts += ["", "Exemplars (match this voice):", *[f"- {e}" for e in exemplars]]
        parts += ["", "## REQUIREMENTS"]
        parts += [f"- {r}" for r in (requirements or ["On-voice", "No fabricated specifics"])]
        parts += ["", "## FORMAT", fmt or "Return the finished copy only."]
        return "\n".join(parts)

    def draft(self, brief: str, **kw) -> str:
        return self._drafter(_SYSTEM, self.build_prompt(brief, **kw))

    def run(self, brief_path: "str | Path", *, role: str = ROLE,
            prompt_version: str = "PRM-copywriting-v1") -> dict:
        """Draft from a brief note, write a gated draft into 06_Review/, return the
        verdict + paths. Fail-closed: a draft is publishable only if accepted."""
        brief_path = Path(brief_path)
        meta, body = frontmatter.parse(brief_path.read_text(encoding="utf-8"))
        blp_level = str(meta.get("blp_level", "UNCLASSIFIED"))
        sources = meta.get("sources") or meta.get("source") or [f"brief:{brief_path.name}"]
        if not isinstance(sources, list):
            sources = [sources]

        copy = self.draft(
            body,
            voice_guide=str(meta.get("voice_guide", "")),
            exemplars=meta.get("exemplars") or [],
            requirements=meta.get("requirements") or None,
            fmt=str(meta.get("format", "")),
        )

        self.review_dir.mkdir(parents=True, exist_ok=True)
        draft_path = self.review_dir / f"{brief_path.stem}.draft.md"
        draft_meta = {
            "role": role,
            "blp_level": blp_level,
            "sources": sources,
            "parents": [str(meta.get("id", brief_path.stem))],
            "model": self.model_spec or "auto",
            "prompt_version": prompt_version,
            "status": "needs_review",
        }
        draft_path.write_text(frontmatter.serialize(draft_meta, copy), encoding="utf-8")

        verdict = self.gate.gate_note(draft_path, role=role, clearance=blp_level)
        return {
            "draft_path": str(draft_path),
            "verdict": verdict,
            "publishable": self.gate.is_publishable(draft_path),
        }

    def approve(self, draft_path: "str | Path", *, note: str = "", reviewer: str = "founder") -> dict:
        """Founder approves a held draft: record the human verdict, then re-gate so the
        preference feedback loop short-circuits it to accepted (publishable)."""
        draft_path = Path(draft_path)
        meta, _ = frontmatter.parse(draft_path.read_text(encoding="utf-8"))
        claim_id = meta.get("provenance_id")
        if not claim_id:
            return {"error": "draft has no provenance_id; run() it first"}
        self.contract.record_human_verdict(claim_id=claim_id, verdict="accepted",
                                           note=note, reviewer=reviewer)
        verdict = self.gate.gate_note(draft_path, role=meta.get("role", ROLE),
                                      clearance=str(meta.get("blp_level", "UNCLASSIFIED")))
        return {"verdict": verdict, "publishable": self.gate.is_publishable(draft_path)}

    def publish(self, draft_path: "str | Path", publish):
        """Publish only if accepted (the single choke point)."""
        return self.gate.publish_if_accepted(draft_path, publish)
