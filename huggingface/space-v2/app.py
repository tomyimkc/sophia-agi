# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Hugging Face Space — Sophia, the Wisdom Gate (interactive, honest demo).

A high-conversion but HONEST demo: every claim shown here matches the repo's
no-overclaim gate and failure ledger. The gate runs live in your browser on
deterministic verifiers (no API key). Build: a Gradio SDK Space whose repo
contains the sophia-agi tree, with app_file: huggingface/space-v2/app.py.

UI logic is importable without Gradio (so it's unit-testable); the Gradio UI is
built only when the library is present.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Robust repo-root discovery (works in-repo regardless of nesting depth).
def _find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "gateway").is_dir() and (parent / "sophia_contract").is_dir():
            return parent
    return here.parents[2]

ROOT = _find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway import Gateway, ToolEntry  # noqa: E402
from provenance_bench.calibration_score import score_answer  # noqa: E402
from sophia_contract import SophiaContract  # noqa: E402

REPO = "https://github.com/tomyimkc/sophia-agi"
SPACE = "https://huggingface.co/spaces/tomyimkc/sophia-agi"
THESIS = "https://tomyimkc.github.io/sophia-agi/"
DATASET = "https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus"

_FIXED = lambda: "2026-01-01T00:00:00+00:00"  # noqa: E731

# In-session only (clearly labelled; NOT a fabricated global leaderboard).
_SESSION = {"attempts": 0, "held": 0}


# --------------------------------------------------------------------------- #
# Core gate demos (reused from the verified repo Space)
# --------------------------------------------------------------------------- #
def verify_claim_ui(content: str, sources: str, clearance: str) -> str:
    gw = Gateway(contract=SophiaContract(clock=_FIXED))
    src = [s.strip() for s in (sources or "").split(",") if s.strip()]
    v = gw.verify({"answer": content, "sources": src}, verifier_ref="grounding", clearance=clearance)
    publishable = v["verdict"] == "accepted"
    badge = {"accepted": "🟢 ACCEPTED", "held": "🟡 HELD", "rejected": "🔴 REJECTED"}.get(v["verdict"], v["verdict"])
    lines = [
        f"{badge}" + (f"   (reason: {v.get('held_reason')})" if v.get("held_reason") else ""),
        f"publishable  : {publishable}   ← only 'accepted' may ever be shown to a user",
        f"confidence   : {v.get('confidence')}",
        f"provenance_id: {v.get('provenance_id')}",
        f"reasons      : {'; '.join(v.get('reasons', []))}",
    ]
    if not publishable and v.get("suggested_fix"):
        lines.append(f"suggested_fix: {v['suggested_fix']}")
    return "\n".join(lines)


def abstain_vs_fabricate_ui(answer: str) -> str:
    case = {"epistemicLabel": "abstain",
            "fabricationMarkers": ["re:(written|authored|created) by [A-Z]", "the author was",
                                   "re:author is [A-Z]"], "scoring": {}}
    r = score_answer(answer, case)
    verdict = ("✅ honest abstention" if r["state"] == "abstained"
               else "❌ fabricated a specific" if r["fabricated"] else "⚠️ dodged")
    return f"{verdict}\nstate: {r['state']}   calibration score: {r['score']}"


def break_the_gate_ui(claim: str, sources: str) -> str:
    """Try to make the gate publish an UNSOURCED claim. If you can't, the gate held."""
    _SESSION["attempts"] += 1
    out = verify_claim_ui(claim, sources, "UNCLASSIFIED")
    held = "ACCEPTED" not in out.split("\n")[0]
    if held:
        _SESSION["held"] += 1
    tally = (f"\n\n— this session: {_SESSION['held']}/{_SESSION['attempts']} attempts held by the gate "
             f"(in-memory only; this is not a global leaderboard)")
    verdict = ("🛡️ the gate HELD — it refused to publish without resolvable sources."
               if held else "🟢 ACCEPTED — you supplied resolvable sources, so it published honestly.")
    return verdict + "\n\n" + out + tally


# --------------------------------------------------------------------------- #
# Moral Gate v2 (public standard + moral parliament) via the in-process MCP tools
# --------------------------------------------------------------------------- #
def moral_gate_ui(text: str) -> str:
    from sophia_mcp import tools_impl as t
    ps = t.public_standard_check_tool(text)
    mp = t.moral_parliament_tool(text)
    return (
        f"public-standard verdict : {ps.get('verdict')}\n"
        f"normative?             : {ps.get('isNormative')}\n"
        f"violations             : {ps.get('violations') or '—'}\n"
        f"unmet duties           : {ps.get('unmetDuties') or '—'}\n"
        f"gray zone              : {ps.get('grayZone')}\n"
        f"-- moral parliament (bounded moral-uncertainty aggregate) --\n"
        f"parliament verdict     : {mp.get('verdict')}\n"
        f"aggregate / variance   : {mp.get('aggregate')} / {mp.get('variance')}\n"
        f"reason                 : {mp.get('reason')}\n\n"
        f"(deterministic policy layer — labels, not moral cognition; candidate infrastructure, not AGI)"
    )


def _results_markdown() -> str:
    try:
        return (ROOT / "RESULTS.md").read_text(encoding="utf-8")
    except Exception:
        return "RESULTS.md not found in this deployment."


# --------------------------------------------------------------------------- #
# Honest copy blocks
# --------------------------------------------------------------------------- #
HERO = f"""
<div style="text-align:center;padding:18px;border-radius:14px;
background:linear-gradient(135deg,#0a0e1a 0%,#10233b 60%,#0a0e1a 100%);
border:1px solid #1f6feb;color:#e6edf3">
<h1 style="margin:.2em 0;font-size:2.0em">🛡️ Sophia — the Wisdom Gate</h1>
<p style="font-size:1.15em;color:#7ee787;margin:.2em">A verifiable foundation <i>toward</i> AGI — built by tomyimkc (sole author)</p>
<p style="color:#c9d1d9">The AI gate that <b>abstains instead of fabricating</b> — and publishes its own failure rate.</p>
<p style="color:#8b949e;font-size:.95em">
<b>Δ12.5%</b> attribution-hallucination cut <code>[5.6–19.4%]</code>, 2 independent judges ·
<b>0% fabrication</b> on the abstain pack <code>(95% CI 0–11%)</code> ·
Live Moral Gate v2 · <b>Not a claim of AGI</b> — a measured step toward one
</p>
</div>
"""

WHY = """
### Why this is different (and why honesty converts)
- **Measured, not claimed.** Only numbers that pass ≥2 independent judge families + confidence intervals headline. Everything else is labelled *illustrative* or *candidate*.
- **Fail-closed.** The gate publishes only what it can machine-check; otherwise it **holds**. ~23.6% of attributions still get through on the hardest 8B model — it's a *filter that reduces harm, not a guarantee*.
- **It publishes its own failures.** See the public failure ledger in the repo. That's the moat.
"""

FOOTER = f"""
---
<div style="text-align:center;color:#8b949e">
Built in 2026 by the sole author <b>tomyimkc</b> ·
Dual-licensed: Apache 2.0 code + reserved brand ·
<b>Not a claim of AGI</b> ·
<a href="{REPO}">⭐ Star the repo</a> to support open, <i>measured</i> AI honesty.
</div>
"""

_TWEET = ("I tested Sophia — the Wisdom Gate by @tomyimkc: an AI gate that abstains instead of "
          "fabricating, cuts attribution-hallucination Δ12.5%25 (95%25 CI 5.6–19.4%25, 2 independent "
          f"judges), and publishes its own failure ledger. The honest foundation AGI needs. {SPACE}")
_TWEET_URL = "https://twitter.com/intent/tweet?text=" + _TWEET.replace(" ", "%20")


def build_demo():  # pragma: no cover - requires gradio at runtime
    import gradio as gr

    with gr.Blocks(theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
                   title="Sophia — the Wisdom Gate", css="footer{visibility:hidden}") as demo:
        gr.HTML(HERO)

        with gr.Row():
            gr.Button("⭐ Star on GitHub", link=REPO)
            gr.Button("🐦 Share on X", link=_TWEET_URL)
            gr.Button("🎬 Watch the thesis", link=THESIS)
            gr.Button("🐞 Try to break the gate (issues)", link=f"{REPO}/issues")

        with gr.Tab("🎯 Challenge the gate"):
            gr.Markdown("Give the gate a claim. With **no resolvable sources it HOLDS** "
                        "(fail-closed); with real sources it publishes. Try to make it fabricate.")
            c = gr.Textbox(label="Claim / answer", value="The Dao De Jing is attributed to Laozi.")
            s = gr.Textbox(label="Sources (comma-separated; leave blank to see it HOLD)", value="wiki://dao-de-jing")
            out = gr.Textbox(label="Gate verdict + provenance trace", lines=8)
            gr.Button("Run the gate", variant="primary").click(break_the_gate_ui, [c, s], out)
            gr.Markdown("#### Abstain-vs-fabricate (calibration scorer)")
            a = gr.Textbox(label="Answer to an unknown-author question",
                           value="The author of the Voynich Manuscript is unknown.")
            ao = gr.Textbox(label="Calibration", lines=2)
            gr.Button("Score it").click(abstain_vs_fabricate_ui, a, ao)

        with gr.Tab("📊 Live benchmarks"):
            gr.Markdown("These are the repo's **real** numbers under the no-overclaim gate "
                        "(VALIDATED vs illustrative are labelled honestly):")
            gr.Markdown(_results_markdown())
            gr.Markdown(f"Full methodology + failure ledger in the [repo]({REPO}).")

        with gr.Tab("⚖️ Test the Moral Gate (v2)"):
            gr.Markdown("Screen a statement against the **public moral standard (overlapping "
                        "consensus)** + the **moral parliament**. Deterministic policy layer — "
                        "labels, not moral cognition.")
            m = gr.Textbox(label="Statement to screen",
                           value="We should deploy the model before the safety evidence is complete.")
            mo = gr.Textbox(label="Moral Gate v2 verdict", lines=10)
            gr.Button("Run Moral Gate v2", variant="primary").click(moral_gate_ui, m, mo)

        with gr.Tab("🔧 Fork & fine-tune"):
            gr.Markdown(f"""
**Everything here is Apache-2.0 and reproducible.**
- ⭐ **[Star / fork the repo]({REPO})** — the core is Apache-2.0 forever.
- 📦 **[HF dataset (528 bilingual examples)]({DATASET})**
- 🧪 Reproduce the gate locally:
```bash
git clone {REPO} && cd sophia-agi
python scripts/demo_gate.py            # offline, no keys
pip install -r requirements-lora.txt   # LoRA starter
```
- 📜 Governance contract: see `CONTRACT.md` · Honest limits: see "What Sophia cannot do (yet)" in the README.
""")

        gr.HTML(FOOTER)
    return demo


if __name__ == "__main__":  # pragma: no cover
    build_demo().launch()
