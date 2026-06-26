# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Schemas + validators for vertical pretraining data: agent, feedback, multimodal.

The data direction's 垂类数据研究: "多模态数据、agent 数据、用户反馈数据、高质量学术数据".
Sophia has agent/council traces and a feedback directory but no *typed, validated* record
formats for them, and no multimodal format at all. This defines minimal, provenance-aware
schemas and pure-stdlib validators so these data streams become first-class, checkable
training assets (every record carries a source + license, consistent with the data passport).

Three record types:
  * AgentTrajectory  — a tool-use episode: goal, steps (action/observation), outcome, reward.
  * FeedbackSignal   — a user-feedback datum: prompt, response, signal (rating/edit/accept).
  * MultimodalItem   — an image/audio reference + text, kept as a typed stub so the
    multimodal gap (see the eval matrix) has a landing spot instead of being ignored.

Validators return {"ok": bool, "errors": [...]}: fail-closed, explicit about what's missing.
"""
from __future__ import annotations

from typing import Any

# Shared provenance fields every vertical record must carry.
_PROVENANCE = ("source", "license")


def _check_provenance(rec: dict, errors: list) -> None:
    for f in _PROVENANCE:
        if not rec.get(f):
            errors.append(f"missing provenance field: {f}")


def validate_agent_trajectory(rec: dict) -> "dict[str, Any]":
    """An agent tool-use episode.

    Required: goal:str, steps:list[{action:str, observation:str}], outcome:str,
    reward:number in [0,1] (or null if unscored), + provenance.
    """
    errors: list[str] = []
    _check_provenance(rec, errors)
    if not isinstance(rec.get("goal"), str) or not rec["goal"].strip():
        errors.append("goal must be a non-empty string")
    steps = rec.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append("steps must be a non-empty list")
    else:
        for i, st in enumerate(steps):
            if not isinstance(st, dict) or "action" not in st or "observation" not in st:
                errors.append(f"step[{i}] needs 'action' and 'observation'")
    if not isinstance(rec.get("outcome"), str):
        errors.append("outcome must be a string")
    r = rec.get("reward", None)
    if r is not None and not (isinstance(r, (int, float)) and 0.0 <= r <= 1.0):
        errors.append("reward must be null or a number in [0,1]")
    return {"ok": not errors, "errors": errors, "type": "agent_trajectory"}


def validate_feedback_signal(rec: dict) -> "dict[str, Any]":
    """A user-feedback datum usable as a preference/reward signal.

    Required: prompt:str, response:str, signal:{kind in {rating,edit,accept,reject},
    value}, + provenance. Ratings carry a numeric value; edits carry corrected text.
    """
    errors: list[str] = []
    _check_provenance(rec, errors)
    if not isinstance(rec.get("prompt"), str) or not rec["prompt"].strip():
        errors.append("prompt must be a non-empty string")
    if not isinstance(rec.get("response"), str):
        errors.append("response must be a string")
    sig = rec.get("signal")
    kinds = {"rating", "edit", "accept", "reject"}
    if not isinstance(sig, dict) or sig.get("kind") not in kinds:
        errors.append(f"signal.kind must be one of {sorted(kinds)}")
    else:
        if sig["kind"] == "rating" and not isinstance(sig.get("value"), (int, float)):
            errors.append("rating signal needs numeric value")
        if sig["kind"] == "edit" and not isinstance(sig.get("value"), str):
            errors.append("edit signal needs corrected-text value")
    return {"ok": not errors, "errors": errors, "type": "feedback_signal"}


def validate_multimodal_item(rec: dict) -> "dict[str, Any]":
    """A multimodal training item (typed stub).

    Required: modality in {image,audio,video}, asset_ref:str (path/URI/hash),
    text:str (caption/transcript/instruction), + provenance. Content bytes are NOT
    embedded; this is a reference record so packs stay text-sized and auditable.
    """
    errors: list[str] = []
    _check_provenance(rec, errors)
    if rec.get("modality") not in {"image", "audio", "video"}:
        errors.append("modality must be one of image|audio|video")
    if not isinstance(rec.get("asset_ref"), str) or not rec["asset_ref"].strip():
        errors.append("asset_ref must be a non-empty string (path/URI/content-hash)")
    if not isinstance(rec.get("text"), str) or not rec["text"].strip():
        errors.append("text must be a non-empty string")
    return {"ok": not errors, "errors": errors, "type": "multimodal_item"}


VALIDATORS = {
    "agent_trajectory": validate_agent_trajectory,
    "feedback_signal": validate_feedback_signal,
    "multimodal_item": validate_multimodal_item,
}


def validate(record_type: str, rec: dict) -> "dict[str, Any]":
    v = VALIDATORS.get(record_type)
    if not v:
        return {"ok": False, "errors": [f"unknown record_type: {record_type}"],
                "type": record_type}
    return v(rec)


__all__ = [
    "validate", "validate_agent_trajectory", "validate_feedback_signal",
    "validate_multimodal_item", "VALIDATORS",
]
