# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Hugging Face Space — interactive demo of the Sophia governance gate.

Lets anyone verify, in the browser, the core claim: the gate publishes only what it can
check, and abstains instead of fabricating. Runs offline (no API key) on deterministic
verifiers. Deploy: create a Gradio Space pointing at this repo with
``app_file: huggingface/space/app.py``.

The UI functions are importable without Gradio (so they are unit-testable); the Gradio
UI is built only when the library is present.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import Gateway, ToolEntry  # noqa: E402
from provenance_bench.calibration_score import score_answer  # noqa: E402
from sophia_contract import SophiaContract  # noqa: E402

_FIXED = lambda: "2026-01-01T00:00:00+00:00"  # noqa: E731


def verify_claim_ui(content: str, sources: str, clearance: str) -> str:
    """Verify a claim through the gate. Sources is a comma-separated list."""
    gw = Gateway(contract=SophiaContract(clock=_FIXED))
    src = [s.strip() for s in (sources or "").split(",") if s.strip()]
    v = gw.verify({"answer": content, "sources": src}, verifier_ref="grounding", clearance=clearance)
    publishable = v["verdict"] == "accepted"
    lines = [
        f"verdict      : {v['verdict']}" + (f"  (held_reason: {v.get('held_reason')})" if v.get("held_reason") else ""),
        f"publishable  : {publishable}   ← only 'accepted' may be published",
        f"confidence   : {v.get('confidence')}",
        f"provenance_id: {v.get('provenance_id')}",
        f"reasons      : {'; '.join(v.get('reasons', []))}",
    ]
    if not publishable and v.get("suggested_fix"):
        lines.append(f"suggested_fix: {v['suggested_fix']}")
    return "\n".join(lines)


def abstain_vs_fabricate_ui(answer: str) -> str:
    """Score an answer to an UNKNOWN-author question: abstaining is good, fabricating is 0."""
    case = {"epistemicLabel": "abstain",
            "fabricationMarkers": ["re:(written|authored|created) by [A-Z]", "the author was",
                                   "re:author is [A-Z]"], "scoring": {}}
    r = score_answer(answer, case)
    return (f"state : {r['state']}\ncalibration score : {r['score']}\n"
            f"({'honest abstention' if r['state']=='abstained' else 'fabricated a specific' if r['fabricated'] else 'dodged'})")


def gateway_call_ui(query: str) -> str:
    """A federated knowledge tool, gated: grounded answers pass; ungrounded are withheld."""
    gw = Gateway(contract=SophiaContract(clock=_FIXED))
    gw.register(ToolEntry(
        id="kb.lookup", verifier_ref="grounding", side_effects="read",
        handler=lambda a: ({"answer": "Laozi (Daoist tradition)", "sources": ["wiki://dao-de-jing"]}
                           if "dao" in (a.get("q", "").lower()) else {"answer": "unknown", "sources": []})))
    r = gw.call_tool("kb.lookup", {"q": query})
    res = "WITHHELD (fail-closed)" if r.get("result") is None else r["result"]
    return f"verdict: {r['verdict']}\nresult : {res}\nprovenance: {r.get('provenance_id')}"


def build_demo():  # pragma: no cover - requires gradio at runtime
    import gradio as gr

    with gr.Blocks(title="Sophia Governance Gate") as demo:
        gr.Markdown("# 🛡️ Sophia — the governance gate\n"
                    "Verify any claim, see abstain-vs-fabricate scoring, and call a gated tool. "
                    "Only what can be checked is published; the rest fails closed.")
        with gr.Tab("Verify a claim"):
            c = gr.Textbox(label="Claim / answer")
            s = gr.Textbox(label="Sources (comma-separated; leave blank to see it held)")
            cl = gr.Dropdown(["UNCLASSIFIED", "CONFIDENTIAL", "SECRET", "TOP_SECRET"],
                             value="UNCLASSIFIED", label="Caller clearance")
            gr.Button("Verify").click(verify_claim_ui, [c, s, cl], gr.Textbox(label="Verdict"))
        with gr.Tab("Abstain vs fabricate"):
            a = gr.Textbox(label="Answer to an unknown-author question",
                           value="The author of the Voynich Manuscript is unknown.")
            gr.Button("Score").click(abstain_vs_fabricate_ui, a, gr.Textbox(label="Calibration"))
        with gr.Tab("Gated tool call"):
            q = gr.Textbox(label="Query (try 'dao' vs anything else)", value="dao")
            gr.Button("Call tool").click(gateway_call_ui, q, gr.Textbox(label="Gated result"))
    return demo


if __name__ == "__main__":  # pragma: no cover
    build_demo().launch()
