# Generated Sophia verifiable skill.
from __future__ import annotations
from verifier import check_answer


def run(args: dict) -> dict:
    text = str(args.get("text") or args.get("query") or args.get("input") or "")
    return {"answer": bool(check_answer(text)), "sources": ["skillforge://" + __name__]}
