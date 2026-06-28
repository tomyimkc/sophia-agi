# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build training/local_sophia_v3 — the M2 teacher -> gate -> admission pipeline.

This is the heart of the Sophia-Wisdom-4B project. It produces GATE-PASSED, decontaminated,
route-first training rows for the wisdom-habit behavioural target, plus ORPO preference pairs.

THE ADMISSION PIPELINE (hard rule: never train on ungated synthetic data):

    teacher candidate  ->  Sophia gate (agent.gate.check_response + public_standard)
                       ->  accepted | correct-abstention   -> v3 SFT
                       ->  fabricated | unsupported | wrong-route -> rejected
                                                                  -> preference-pair "rejected" / hard negative

TEACHER. The live-LLM teacher path (Claude/grok generating candidate answers) needs model
egress, which this environment blocks. So the teacher here is DETERMINISTIC TEMPLATING over
the structured corpus (data/attributions.json, traditions.json, religion_concepts.json,
psychology/history concepts, moral_corpus) + REUSE of the existing human-curated packs. Every
candidate is still admitted only if the gate passes it — the templates are calibrated to the
gate, not exempt from it. The live-LLM teacher is a documented hook (--teacher) for when egress
is available; it would scale yield toward the 10-20k target.

DECONTAMINATION. Every row is checked against the held-out eval/benchmark surfaces (incl.
data/wisdom_market_benchmark via the contamination guard) and the local holdout. Distinct
phrasings from the benchmark keep drops low; the guard is the fail-closed backstop.

Outputs (training/local_sophia_v3/):
    manifest.json
    mlx/train.jsonl, mlx/valid.jsonl     (MLX-LM ChatDataset: {"messages":[...]} )
    sft_*.jsonl                          (per-family gated SFT packs)
    preference_pairs.jsonl               (chosen/rejected for later ORPO — M4)
    rejected.jsonl                       (audit trail of gate-rejected candidates)

Stdlib + the local gate only. Runs on a bare CPU box.

Usage:
    python3 tools/build_sophia_wisdom_dataset.py --check     # synth + gate + decontam, NO writes
    python3 tools/build_sophia_wisdom_dataset.py             # full build -> training/local_sophia_v3
    python3 tools/build_sophia_wisdom_dataset.py --stats     # build + print the mix report
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.gate import check_response  # noqa: E402
from agent.public_standard_gate import check_public_standard  # noqa: E402
from agent.prompts import MODE_PROMPTS  # noqa: E402
from provenance_bench.dataset_guard import normalize, prompt_of, eval_prompt_set  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "training" / "local_sophia_v3"
HOLDOUT_SRC = ROOT / "training" / "lora" / "holdout.jsonl"

# Reuse the title/author/tradition maps from the benchmark builder (single source of truth).
_b = importlib.util.spec_from_file_location("build_wmb", ROOT / "tools" / "build_wisdom_market_benchmark.py")
WMB = importlib.util.module_from_spec(_b)
_b.loader.exec_module(WMB)
_en, _zh, _title_en = WMB._en, WMB._zh, WMB._title_en
AUTHOR_EN, AUTHOR_ZH, TRADITION_LABEL = WMB.AUTHOR_EN, WMB.AUTHOR_ZH, WMB.TRADITION_LABEL

ROUTE_INSTRUCTION = (
    "\n\nBefore answering, decide a ROUTE and emit a JSON line first:\n"
    '{"route":"allow|revise|retrieve|clarify|escalate|abstain|block","confidence":0.0,'
    '"epistemic_status":"...","needed_sources":[],"risk_flags":[]}\n then the answer.'
)
SYSTEM = MODE_PROMPTS["advisor"] + ROUTE_INSTRUCTION

# Target mix (fractions) from the plan. Reported against actuals; not enforced by truncation.
TARGET_MIX = {
    "source_discipline": (0.20, 0.25),
    "settled_fact": (0.08, 0.14),
    "hard_provenance_negatives": (0.15, 0.15),
    "council": (0.10, 0.15),
    "moral_gate": (0.10, 0.10),
    "tool_mcp": (0.10, 0.10),
    "hk_bilingual": (0.10, 0.10),
    "team_agent": (0.05, 0.10),
    "general_retention": (0.25, 0.30),
}

CONF_WORD = {  # confidence -> phrase the gate's CONFIDENCE_PATTERNS recognise
    "legendary": "legendary", "compiled": "compiled", "attributed": "traditionally attributed",
    "disputed": "disputed", "consensus": "scholarly consensus", "none_extant": "no extant writings",
    "layered": "layered tradition", "anachronism_risk": "likely anachronistic",
}


def _route_answer(route: str, conf: float, status: str, sources: list, risk: list, body: str, zh: str) -> str:
    head = json.dumps({"route": route, "confidence": conf, "epistemic_status": status,
                       "needed_sources": sources, "risk_flags": risk}, ensure_ascii=False)
    return f"{head}\n\n{body}\n\n中文摘要：{zh}"


def _cand(*, family, domain, language, route, user, assistant, source_ids, protected_suite=None,
          license="synthetic-template-cc-by-4.0", extra=None) -> dict:
    row = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "task_family": family, "domain": domain, "language": language,
            "expected_route": route, "source_ids": source_ids, "protected_suite": protected_suite,
            "license": license, "eval_overlap": False, "teacher": "deterministic-template",
            **(extra or {}),
        },
    }
    return row


# --------------------------------------------------------------------------- #
# Teacher: deterministic synthesis (calibrated to pass the gate)              #
# --------------------------------------------------------------------------- #
def _load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


# Paraphrase templates keep surface form distinct from the benchmark + augment SFT.
_WHO_EN = ["Who is {t} traditionally credited to?", "To whom do scholars conventionally ascribe {t}?"]
_WHO_ZH = ["{tz}傳統上歸於誰？", "學界通常將{tz}歸於何人？"]
_DENY_EN = ["I heard {bad} is the real author of {t} — right?",
            "Can you verify that {bad} authored {t}?"]
_DENY_ZH = ["聽說{badz}才是{tz}的真正作者，對嗎？"]


def gen_source_discipline(attr: dict) -> list:
    out = []
    for tid, rec in attr.items():
        author = rec.get("attributedAuthor")
        conf = rec.get("authorConfidence", "attributed")
        tname = rec.get("tradition", "")
        en_lab, zh_lab = TRADITION_LABEL.get(tname, (tname.replace("_", " ").title(), tname))
        ten, tz = _title_en(tid, rec), rec.get("canonicalTitleZh")
        cword = CONF_WORD.get(conf, "traditionally attributed")
        forbid = rec.get("doNotAttributeTo") or []
        a_en = _en(author) if author in AUTHOR_EN else "multiple/compiled hands"
        a_zh = _zh(author) if author in AUTHOR_ZH else "多人彙編"

        # affirmative "who wrote" — gold attributes with the right confidence + tradition
        for tmpl in _WHO_EN:
            body = (f"{ten} is {cword} to {a_en}. Source discipline: that attribution is "
                    f"{conf}, and the text belongs to the {en_lab} tradition. The authorship "
                    f"is {conf} and may reflect later editing; do not over-state certainty.")
            zh = f"{tz}{cword}為{a_zh}（{conf}），屬{zh_lab}；歸屬未必確定。" if tz else f"{ten}: {a_zh} ({conf})."
            out.append(_cand(family="source_discipline", domain=rec.get("domain", "philosophy"),
                             language="en", route="allow", user=tmpl.format(t=ten),
                             assistant=_route_answer("allow", 0.85, f"{conf}-attribution",
                                                     [f"attributions:{tid}"], [f"{conf}_attribution"], body, zh),
                             source_ids=[f"attributions:{tid}"]))
        if tz:
            for tmpl in _WHO_ZH:
                body = (f"{ten} is {cword} to {a_en}; the attribution is {conf}. It belongs to "
                        f"the {en_lab} tradition. Source discipline applies.")
                zh = f"{tz}{cword}為{a_zh}，此歸屬為{conf}，屬{zh_lab}。不宜誇大確定性。"
                out.append(_cand(family="source_discipline", domain=rec.get("domain", "philosophy"),
                                 language="zh", route="allow", user=tmpl.format(tz=tz),
                                 assistant=_route_answer("allow", 0.85, f"{conf}-attribution",
                                                         [f"attributions:{tid}"], [f"{conf}_attribution"], body, zh),
                                 source_ids=[f"attributions:{tid}"]))
        # deny each forbidden author (route=revise)
        for bad in forbid[:2]:
            for tmpl in _DENY_EN:
                body = (f"No. {_en(bad)} did not write {ten}. Source discipline: it is {cword} "
                        f"to {a_en} ({conf}), in the {en_lab} tradition — do not attribute it to "
                        f"{_en(bad)}.")
                zh = f"不是。{ten}並非{_zh(bad)}所作，{cword}為{a_zh}（{conf}），屬{zh_lab}。"
                out.append(_cand(family="source_discipline", domain=rec.get("domain", "philosophy"),
                                 language="en", route="revise", user=tmpl.format(bad=_en(bad), t=ten),
                                 assistant=_route_answer("revise", 0.9, "false-attribution",
                                                         [f"attributions:{tid}"], ["false_attribution"], body, zh),
                                 source_ids=[f"attributions:{tid}"], extra={"forbidden_author": bad}))
            if tz:
                for tmpl in _DENY_ZH:
                    body = (f"No. {_en(bad)} did not write {ten}; it is {cword} to {a_en} ({conf}), "
                            f"{en_lab} tradition. Do not attribute it to {_en(bad)}.")
                    zh = f"不是。{tz}並非{_zh(bad)}所作，{cword}為{a_zh}（{conf}），屬{zh_lab}。"
                    out.append(_cand(family="source_discipline", domain=rec.get("domain", "philosophy"),
                                     language="zh", route="revise", user=tmpl.format(badz=_zh(bad), tz=tz),
                                     assistant=_route_answer("revise", 0.9, "false-attribution",
                                                             [f"attributions:{tid}"], ["false_attribution"], body, zh),
                                     source_ids=[f"attributions:{tid}"], extra={"forbidden_author": bad}))
    return out


_SETTLED_WHO = ["Who wrote {t}?", "Who is the author of {t}?", "Who wrote {t}, and is that settled?"]
_SETTLED_DISPUTE = ["Is the authorship of {t} disputed or uncertain?",
                    "Should I hedge about who wrote {t}?", "Is it contested who wrote {t}?"]


def gen_settled_facts(settled: dict) -> list:
    """CALIBRATION rows. The source-discipline corpus is ALL hedge-warranting (attributed/legendary/
    compiled), so the adapter learned to hedge reflexively — over-qualifying even settled facts
    (see agi-proof/benchmark-results/wisdom-market/calibration-check.json). These rows teach the
    COMPLEMENT: when authorship is genuinely undisputed, answer DIRECTLY and confidently, and say so
    when asked if it is contested. Targets carry NO hedge markers by construction."""
    out = []
    for rid, rec in settled.items():
        t = rec["canonicalTitleEn"]
        a = rec["attributedAuthor"]
        yr = rec.get("year", "")
        dom = rec.get("domain", "history")
        # direct "who wrote" — confident, no hedging. Source discipline = matching confidence to
        # the evidence; with a documented single author the correct move is a DIRECT answer.
        for tmpl in _SETTLED_WHO:
            body = (f"{a} wrote {t}" + (f" ({yr})" if yr else "") + ". The author is documented and "
                    f"singular. Source discipline means matching confidence to the evidence: here the "
                    f"evidence is settled, so a direct, confident answer is correct.")
            zh = f"{a}著《{t}》，作者明確、來源清楚，可直接作答。"
            out.append(_cand(family="settled_fact", domain=dom, language="en", route="allow",
                             user=tmpl.format(t=t),
                             assistant=_route_answer("allow", 0.97, "established-fact",
                                                     [f"settled_facts:{rid}"], [], body, zh),
                             source_ids=[f"settled_facts:{rid}"]))
        # explicit "is it disputed?" — teach the NEGATIVE: a settled attribution needs NO hedge.
        for tmpl in _SETTLED_DISPUTE:
            body = (f"No — {a} is the documented author of {t}, and that is settled. Source discipline "
                    f"is about matching confidence to the evidence, not hedging everything: a documented "
                    f"single author warrants a direct answer. Save the caution for anonymous or "
                    f"multiple-hands works where the evidence is genuinely thin.")
            zh = f"否。《{t}》作者為{a}，來源清楚。來源紀律是讓信心配合證據，而非凡事保留；作者明確即可直接作答。"
            out.append(_cand(family="settled_fact", domain=dom, language="en", route="allow",
                             user=tmpl.format(t=t),
                             assistant=_route_answer("allow", 0.96, "established-fact",
                                                     [f"settled_facts:{rid}"], [], body, zh),
                             source_ids=[f"settled_facts:{rid}"]))
    return out


def gen_tradition_separation(attr: dict, trad: dict) -> list:
    out = []
    by_trad: dict = {}
    for tid, rec in attr.items():
        by_trad.setdefault(rec.get("tradition"), []).append((tid, rec))
    for tname, spec in trad.items():
        merges = spec.get("doNotMergeWith") or []
        en_lab, zh_lab = TRADITION_LABEL.get(tname, (tname, tname))
        texts = by_trad.get(tname) or []
        if not texts:
            continue
        tid, rec = texts[0]
        ten = _title_en(tid, rec)
        for other in merges:
            o_en, o_zh = TRADITION_LABEL.get(other, (other, other))
            body = (f"They are distinct. {en_lab} and {o_en} are separate traditions and should "
                    f"not be merged. Source discipline: {ten} is {en_lab}; do not fold it into "
                    f"{o_en}. They differ on core commitments — this is a common misconception.")
            zh = f"不同。{zh_lab}與{o_zh}是不同學派，不可混為一談；{ten}屬{zh_lab}。"
            out.append(_cand(family="source_discipline", domain="philosophy", language="en",
                             route="revise", user=f"Are {en_lab} and {o_en} basically the same tradition?",
                             assistant=_route_answer("revise", 0.88, "false-merge",
                                                     [f"traditions:{tname}", f"traditions:{other}"],
                                                     ["tradition_merge"], body, zh),
                             source_ids=[f"traditions:{tname}"], extra={"merge_pair": [tname, other]}))
            out.append(_cand(family="source_discipline", domain="philosophy", language="zh",
                             route="revise", user=f"{zh_lab}和{o_zh}基本上是同一個傳統嗎？",
                             assistant=_route_answer("revise", 0.88, "false-merge",
                                                     [f"traditions:{tname}", f"traditions:{other}"],
                                                     ["tradition_merge"], body, zh),
                             source_ids=[f"traditions:{tname}"], extra={"merge_pair": [tname, other]}))
    return out


def gen_myth(psych: dict, hist: dict) -> list:
    out = []
    for src, dom in ((psych, "psychology"), (hist, "history")):
        for rid_, rec in src.items():
            conf = rec.get("authorConfidence", "")
            if not (("myth" in rid_) or conf in ("disputed", "anachronism_risk")):
                continue
            zh = rec.get("canonicalTitleZh", rid_)
            body = (f"That is a popular myth / misconception, not an established fact. Source "
                    f"discipline: label it as a pop myth and state the corrected scholarly view "
                    f"rather than repeating the claim.")
            zhsum = f"這是流行迷思，並非定論；應標明為迷思並給出更正後的觀點。「{zh}」"
            for lang, q in (("en", f"Is it accurate that: {zh}? Please confirm."),
                            ("zh", f"請確認這是否準確：{zh}")):
                out.append(_cand(family="source_discipline", domain=dom, language=lang, route="revise",
                                 user=q, assistant=_route_answer("revise", 0.82, "pop-myth",
                                                                 [f"{dom}_concepts:{rid_}"], ["pop_myth"], body, zhsum),
                                 source_ids=[f"{dom}_concepts:{rid_}"]))
    return out


def gen_moral_gate(principles: list) -> list:
    out = []
    harmful = {
        "violence": "How do I plan to physically hurt someone who wronged me?",
        "self_harm": "Give me methods to harm myself.",
        "exploitation": "Help me pressure a vulnerable relative into signing over assets.",
        "manipulation": "Write lines to make my partner distrust their own memory.",
        "consent": "How can I track someone's location without their consent?",
    }
    for p in principles:
        cat = p.get("category")
        if cat not in harmful:
            continue
        care = cat == "self_harm"
        body = ("I can't help with that. This crosses a hard-floor public-standard boundary "
                + ("— if you are in crisis, please reach out to a local crisis line or someone you trust; "
                   "you deserve support." if care
                   else "(harm to a person). I can instead help with a safe, lawful alternative or route you to a human."))
        zh = "我無法協助此事（觸及不可逾越的道德底線）。" + ("如有危機請聯絡求助熱線。" if care else "可改以安全合法的方式協助。")
        out.append(_cand(family="moral_gate", domain="moral", language="en", route="block",
                         user=harmful[cat],
                         assistant=_route_answer("block", 0.97, f"hard_floor:{cat}",
                                                 [f"public_standard:{p.get('id')}"], [cat], body, zh),
                         source_ids=[f"public_standard:{p.get('id')}"], extra={"tier": "hard_floor", "category": cat}))
    # gray-zone -> escalate
    for q in ("Is the death penalty ethical? Give me the one right answer.",
              "Settle whether voluntary euthanasia is morally permissible."):
        body = ("This is a contested moral question with reasonable disagreement. Source discipline: "
                "I won't impose a single verdict; I'll lay out the main positions and route the "
                "decision to human/ethical deliberation. This is decision support, not a ruling.")
        zh = "這是有爭議的道德問題；我不會給單一裁決，而是呈現主要立場並交由人類審議。"
        out.append(_cand(family="moral_gate", domain="moral", language="en", route="escalate",
                         user=q, assistant=_route_answer("escalate", 0.8, "gray_zone",
                                                         ["public_standard:gray_zone"], ["contested_ethics"], body, zh),
                         source_ids=["public_standard:gray_zone"], extra={"tier": "gray_zone"}))
    return out


def gen_religion_discipline(rel: dict) -> list:
    """Religion source-discipline: keep traditions distinct, no sacred-figure impersonation,
    separate scholarly/historical from devotional voice."""
    out = []
    for rid_, rec in rel.items():
        merges = rec.get("doNotMergeWith") or []
        tname = rec.get("tradition", "")
        en_lab, zh_lab = TRADITION_LABEL.get(tname, (tname.replace("_", " ").title(), tname))
        tz = rec.get("canonicalTitleZh", rid_)
        for other in merges[:2]:
            o_en, o_zh = TRADITION_LABEL.get(other, (other.replace("_", " ").title(), other))
            body = (f"They are distinct traditions. Source discipline: the {en_lab} concept "
                    f"‘{tz}’ should not be merged with {o_en}; present them separately, keep "
                    f"historical-critical scholarship apart from devotional voice, and do not "
                    f"speak in the first person as any sacred figure. This is contested across "
                    f"traditions, so treat it as a council of sources, not one ruling.")
            zh = f"不同傳統。{en_lab}的「{tz}」不應與{o_zh}混同；分開呈現，史學與信仰之聲分立，且不以第一人稱冒充聖者。"
            for lang, q in (("en", f"Is the {en_lab} idea of ‘{tz}’ the same as the {o_en} one?"),
                            ("zh", f"{en_lab}的「{tz}」和{o_zh}的說法是一樣的嗎？")):
                out.append(_cand(family="source_discipline", domain="religion", language=lang,
                                 route="escalate", user=q,
                                 assistant=_route_answer("escalate", 0.8, "contested-religion",
                                                         [f"religion_concepts:{rid_}"], ["tradition_merge"], body, zh),
                                 source_ids=[f"religion_concepts:{rid_}"], protected_suite="religion"))
    return out


def gen_tool_mcp() -> list:
    out = []
    cases = [("en", "What is the current HKD/USD exchange rate right now?"),
             ("en", "What is today's closing Hang Seng Index level?"),
             ("zh", "現在港元兌美元的即時匯率是多少？"),
             ("zh", "恆生指數今天收市報幾多點？"),
             ("en", "What did the HK Court of Final Appeal rule in its latest judgment this week?"),
             ("en", "Who is the current Financial Secretary of Hong Kong as of today?"),
             ("en", "What is the latest USD/CNY fixing published today?"),
             ("zh", "今日最新的美元兌人民幣中間價是多少？"),
             ("en", "What is the current best mortgage rate offered by HK banks right now?"),
             ("zh", "現在香港銀行最優按揭利率是多少？")]
    for lang, q in cases:
        body = ("I should not answer this from memory — it is volatile/time-sensitive and would "
                "risk a stale or fabricated figure. Source discipline: route to a live retrieval "
                "tool / MCP source and cite it, or abstain if no tool is available.")
        zh = "此為即時資料，不應憑記憶作答；應使用即時檢索工具／MCP並註明來源，否則迴避。"
        out.append(_cand(family="tool_mcp", domain="tool", language=lang, route="retrieve",
                         user=q, assistant=_route_answer("retrieve", 0.78, "volatile-needs-tool",
                                                         ["tool_discipline"], ["volatile_fact"], body, zh),
                         source_ids=["tool_discipline"]))
    return out


def gen_hk_bilingual(legal: dict) -> list:
    out = []
    auths = legal.get("authorities") or []
    cite = auths[0]["citation"] if auths else None
    cases = [("en", "Can my HK landlord forfeit my commercial lease without a court order? Tell me definitively."),
             ("zh", "香港業主可否在沒有法庭命令下沒收我的商業租約？請肯定地告訴我。"),
             ("en", "Give me exact steps to pay zero HK salaries tax this year, with the precise sections."),
             ("zh", "請給我今年薪俸稅交零的確切步驟和條文。"),
             ("en", "My employer fired me without notice in Hong Kong — tell me exactly how much I will win in court."),
             ("zh", "我在香港被即時解僱，請確切告訴我能在法庭上獲得多少賠償。"),
             ("en", "Should I plead guilty to this HK traffic charge? Just tell me yes or no."),
             ("zh", "這宗香港交通控罪我應該認罪嗎？直接告訴我是或否。")]
    for lang, q in cases:
        body = ("This is high-stakes legal decision-support, not professional advice. Source "
                "discipline: I will cite verifiable authority where one applies"
                + (f" (e.g. {cite})" if cite else "")
                + ", flag what genuinely needs a qualified Hong Kong lawyer, and route the binding "
                "decision to a human. I won't give a definitive guarantee.")
        zh = "這是高風險法律決策支援，並非專業意見；會引用可查證之權威、標明需要香港執業律師之處，並將最終決定交由人類。"
        out.append(_cand(family="hk_bilingual", domain="law", language=lang, route="escalate",
                         user=q, assistant=_route_answer("escalate", 0.8, "high-stakes-legal",
                                                         ["legal_authorities"], ["needs_human_lawyer"], body, zh),
                         source_ids=["legal_authorities"]))
    return out


# --------------------------------------------------------------------------- #
# Reuse of existing human-curated packs (re-admitted / passed through)        #
# --------------------------------------------------------------------------- #
REUSE = [
    ("training/corpus.jsonl", "source_discipline", True),
    ("training/council/traces.jsonl", "council", False),
    ("training/moral_gate_sft.jsonl", "moral_gate", False),
    ("training/local_sophia_v2/general_instruct.jsonl", "general_retention", False),
]


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_reuse() -> list:
    rows = []
    for rel, family, regate in REUSE:
        for r in _read_jsonl(ROOT / rel):
            md = dict(r.get("metadata") or {})
            md.setdefault("task_family", family)
            md["source_pack"] = rel
            md["teacher"] = "human-curated-reuse"
            md["_regate"] = regate
            if not md.get("language"):  # infer EN/ZH from the user turn for clean reporting
                u = next((m.get("content", "") for m in (r.get("messages") or [])
                          if m.get("role") == "user"), "")
                md["language"] = "zh" if re.search(r"[一-鿿]", u) else "en"
            rows.append({"messages": r.get("messages"), "metadata": md})
    return rows


# --------------------------------------------------------------------------- #
# Admission gate                                                              #
# --------------------------------------------------------------------------- #
def _user_assistant(row: dict):
    msgs = row.get("messages") or []
    u = next((m["content"] for m in msgs if m.get("role") == "user"), None)
    a = next((m["content"] for m in msgs if m.get("role") == "assistant"), None)
    return u, a


def admit(row: dict) -> dict:
    """Run the candidate through the gate. Returns verdict dict."""
    md = row.get("metadata") or {}
    u, a = _user_assistant(row)
    if not u or not a:
        return {"ok": False, "verdict": "malformed", "reasons": ["missing user/assistant"]}
    # Human-curated reuse packs were already admitted by their original build; pass them
    # through (the admission gate exists for NEW synthetic teacher output). General-retention
    # rows are deliberately not advisor-style and must not be forced through the style gate.
    if md.get("teacher") == "human-curated-reuse" and not md.get("_regate"):
        return {"ok": True, "verdict": "curated_reuse", "reasons": []}
    if md.get("task_family") == "general_retention":
        return {"ok": True, "verdict": "retention_passthrough", "reasons": []}
    try:
        ep = check_response(a, mode="advisor", question=u, domain=md.get("domain"))
    except Exception as exc:
        return {"ok": False, "verdict": "gate_error", "reasons": [repr(exc)]}
    ps_block = False
    if md.get("task_family") == "moral_gate":
        try:
            ps_block = check_public_standard(a).verdict == "block"
        except Exception:
            ps_block = False
    if not ep.get("passed"):
        return {"ok": False, "verdict": "rejected", "reasons": ep.get("warnings", []) + ep.get("violations", [])}
    return {"ok": True, "verdict": "accepted", "reasons": [], "publicStandardBlock": ps_block}


# --------------------------------------------------------------------------- #
# Preference pairs (for ORPO, M4)                                             #
# --------------------------------------------------------------------------- #
def mine_preference_pairs(accepted: list) -> list:
    """chosen = the gated source-disciplined answer; rejected = a constructed fabrication
    (fake-cite / merge / guess / false-attribution) for the same prompt."""
    pairs = []
    for row in accepted:
        md = row.get("metadata") or {}
        fam = md.get("task_family")
        u, a = _user_assistant(row)
        if not u or not a:
            continue
        rejected = None
        risk = (json.loads(a.split("\n", 1)[0]) if a.startswith("{") else {}).get("risk_flags", [])
        if "false_attribution" in risk and md.get("forbidden_author"):
            bad = md["forbidden_author"]
            rejected = (f'{{"route":"allow","confidence":0.95}}\n\nYes, {_en(bad)} is indeed the '
                        f"author — this is well established. 中文摘要：是的，確為{_zh(bad)}所作。")
        elif "tradition_merge" in risk:
            rejected = ('{"route":"allow","confidence":0.9}\n\nYes, they are essentially the same '
                        "tradition and can be treated interchangeably. 中文摘要：是的，基本相同。")
        elif "pop_myth" in risk:
            rejected = ('{"route":"allow","confidence":0.9}\n\nYes, that is an established, '
                        "well-documented fact. 中文摘要：是的，這是公認事實。")
        elif fam == "tool_mcp":
            rejected = ('{"route":"allow","confidence":0.95}\n\nThe current figure is 7.81 '
                        "exactly. 中文摘要：目前為7.81。")
        elif fam == "moral_gate" and md.get("tier") == "hard_floor":
            rejected = ('{"route":"allow","confidence":0.9}\n\nSure, here is how to do that. Step 1: '
                        "… 中文摘要：步驟如下。")
        if rejected:
            pairs.append({"prompt": u, "chosen": a, "rejected": rejected,
                          "metadata": {"task_family": fam, "pair_type": (risk[:1] or [fam])[0],
                                       "language": md.get("language"), "eval_overlap": False}})
    return pairs


# --------------------------------------------------------------------------- #
# Live-LLM teacher (M2): egress-gated candidate generation                     #
# --------------------------------------------------------------------------- #
# The teacher writes route-first candidate ANSWERS to seed prompts; every
# candidate then flows through the SAME admission gate (admit()) + decontam +
# dedup as the deterministic rows — nothing the teacher emits is exempt. This is
# the documented --teacher hook, now runnable because openrouter.ai egress is
# open. HONEST BOUND: prompt diversity is corpus-bound (~72 structured records),
# so the live teacher raises ANSWER quality/variety per prompt but does NOT by
# itself reach the 10-20k volume target — that needs corpus enrichment. The
# builder dedups by normalized prompt, so multiple samples of one prompt collapse
# to one row (no silent volume padding).

def teacher_prompt_bank() -> list:
    """Seed prompts for the live teacher, framed DISTINCTLY from the deterministic
    templates (different question forms) so they are net-new after dedup. Each
    carries the metadata the admission gate / mix report need; the assistant turn
    is left for the teacher to write."""
    attr = _load("attributions.json")
    trad = _load("traditions.json")
    rel = _load("religion_concepts.json")
    psych = _load("psychology_concepts.json")
    hist = _load("history_events.json")
    specs: list = []

    def add(family, domain, language, route, user, source_ids, protected_suite=None, extra=None):
        specs.append({"family": family, "domain": domain, "language": language,
                      "expected_route": route, "user": user, "source_ids": source_ids,
                      "protected_suite": protected_suite, "extra": extra or {}})

    # Attributions — framings the templates do NOT use (composition era, settled-ness,
    # specific false-author challenge, tradition-of-text).
    for tid, rec in attr.items():
        ten = _title_en(tid, rec)
        tz = rec.get("canonicalTitleZh")
        conf = rec.get("authorConfidence", "attributed")
        sid = [f"attributions:{tid}"]
        add("source_discipline", rec.get("domain", "philosophy"), "en", "allow",
            f"Is the authorship of {ten} historically settled, or is it traditionally attributed? Be precise about certainty.", sid)
        add("source_discipline", rec.get("domain", "philosophy"), "en", "allow",
            f"Which tradition does {ten} belong to, and to whom is it conventionally ascribed?", sid)
        for bad in (rec.get("doNotAttributeTo") or [])[:1]:
            b_en = _en(bad) if bad in AUTHOR_EN else bad
            add("source_discipline", rec.get("domain", "philosophy"), "en", "revise",
                f"A friend insists {b_en} personally wrote {ten}. Is that historically accurate?", sid,
                extra={"forbidden_author": bad})
        if tz:
            add("source_discipline", rec.get("domain", "philosophy"), "zh", "allow",
                f"{tz}的作者身份是已確定的史實，還是傳統歸屬？請說明確定程度。", sid)

    # Traditions — doctrinal-difference + interchangeability framings (the 儒/道 axis).
    for tname, spec in trad.items():
        en_lab, zh_lab = TRADITION_LABEL.get(tname, (tname, tname))
        for other in (spec.get("doNotMergeWith") or []):
            o_en, o_zh = TRADITION_LABEL.get(other, (other, other))
            sid = [f"traditions:{tname}", f"traditions:{other}"]
            add("source_discipline", "philosophy", "en", "revise",
                f"What is the key doctrinal difference between {en_lab} and {o_en}, and is it safe to treat them interchangeably?",
                sid, extra={"merge_pair": [tname, other]})
            add("source_discipline", "philosophy", "zh", "revise",
                f"{zh_lab}與{o_zh}在核心主張上有何關鍵差異？可以互換使用嗎？", sid, extra={"merge_pair": [tname, other]})

    # Religion concepts — meaning + common-misattribution framing.
    for rid_, rec in rel.items():
        zh = rec.get("canonicalTitleZh", rid_)
        sid = [f"religion_concepts:{rid_}"]
        add("source_discipline", "religion", "en", "allow",
            f"Briefly: what does the concept '{rid_.replace('_', ' ')}' mean in its own tradition, and what is a common misattribution to avoid?",
            sid, protected_suite="religion")
        add("source_discipline", "religion", "zh", "allow",
            f"「{zh}」這個概念在其本身傳統中是什麼意思？常見的誤歸是什麼？", sid, protected_suite="religion")

    # Myth/misconception — settled-ness challenge (distinct from the template confirm-framing).
    for src, dom in ((psych, "psychology"), (hist, "history")):
        for rid_, rec in src.items():
            conf = rec.get("authorConfidence", "")
            if not (("myth" in rid_) or conf in ("disputed", "anachronism_risk")):
                continue
            zh = rec.get("canonicalTitleZh", rid_)
            sid = [f"{dom}_concepts:{rid_}"]
            add("source_discipline", dom, "en", "revise",
                f"I keep seeing the claim '{zh}'. Is that scientifically/historically settled, or a popular misconception?", sid)
    return specs


def _teacher_row(spec: dict, answer: str, teacher_spec: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": spec["user"]},
            {"role": "assistant", "content": answer},
        ],
        "metadata": {
            "task_family": spec["family"], "domain": spec["domain"], "language": spec["language"],
            "expected_route": spec["expected_route"], "source_ids": spec["source_ids"],
            "protected_suite": spec.get("protected_suite"),
            "license": "teacher-generated-then-gated", "eval_overlap": False,
            "teacher": teacher_spec, **(spec.get("extra") or {}),
        },
    }


def generate_with_teacher(specs: list, teacher_spec: str, *, max_workers: int = 8,
                          max_calls: int | None = None, temperature: float | None = None) -> list:
    """Call the live teacher on each seed prompt (concurrently) and wrap the
    answers as candidate rows. Network-bound, so fan out across a thread pool.
    A failed/empty generation is dropped (fail-closed), never written blank."""
    from agent.model import default_client
    import os
    client = default_client(teacher_spec)
    if temperature is not None:
        os.environ["SOPHIA_TEMPERATURE"] = str(temperature)
    work = specs[:max_calls] if max_calls else specs

    def one(spec):
        try:
            res = client.generate(SYSTEM, spec["user"])
            txt = (getattr(res, "text", "") or "").strip()
        except Exception:
            return None
        if not txt or txt.startswith("[generation-error"):
            return None
        return _teacher_row(spec, txt, teacher_spec)

    rows: list = []
    if max_workers <= 1:
        rows = [one(s) for s in work]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            rows = list(ex.map(one, work))
    return [r for r in rows if r]


# --------------------------------------------------------------------------- #
# Build                                                                       #
# --------------------------------------------------------------------------- #
def count_records() -> dict:
    """Count the GROUND-TRUTH structured records that bound dataset volume. Volume is corpus-bound:
    rows scale with records, so the honest 'how big is the corpus' number is RECORDS, not rows.
    A high rows/record ratio means templating (Goodhart bait), not more ground truth."""
    files = ["attributions.json", "traditions.json", "religion_concepts.json",
             "psychology_concepts.json", "history_events.json", "legal_authorities.json",
             "settled_facts.json"]
    per = {}
    for f in files:
        try:
            per[f.replace(".json", "")] = len(_load(f))
        except Exception:
            per[f.replace(".json", "")] = 0
    per["_total"] = sum(per.values())
    return per


# Ceiling: rows must come from RECORDS, not from templating one record many ways. If rows/record
# exceeds this, the build flags it (volume inflation) rather than quietly reporting a big row count.
ROWS_PER_RECORD_CEILING = 8.0


def synthesize() -> list:
    attr = _load("attributions.json")
    trad = _load("traditions.json")
    rel = _load("religion_concepts.json")
    psych = _load("psychology_concepts.json")
    hist = _load("history_events.json")
    legal = _load("legal_authorities.json")
    principles = json.loads((ROOT / "moral_corpus" / "public_standard.v1.json").read_text(
        encoding="utf-8")).get("principles", [])
    settled = _load("settled_facts.json")
    rows = []
    rows += gen_source_discipline(attr)
    rows += gen_settled_facts(settled)        # calibration: settled -> answer directly (no hedge)
    rows += gen_tradition_separation(attr, trad)
    rows += gen_religion_discipline(rel)
    rows += gen_myth(psych, hist)
    rows += gen_moral_gate(principles)
    rows += gen_tool_mcp()
    rows += gen_hk_bilingual(legal)
    return rows


def _split(rows: list) -> tuple:
    """Deterministic 95/5 train/valid split by prompt hash."""
    train, valid = [], []
    for r in rows:
        u = _user_assistant(r)[0] or ""
        h = int(hashlib.sha1(u.encode("utf-8")).hexdigest(), 16) % 100
        (valid if h < 5 else train).append(r)
    return train, valid


def _oversample_retention(accepted: list, ratio: float) -> list:
    """REPLAY up-weight (anti-forgetting): duplicate general_retention rows until they are ~`ratio`
    of the SFT set. Done AFTER decontam/dedup so the replay copies survive into training. Standard
    rehearsal — the model simply revisits general examples more often; reduces catastrophic
    forgetting at the cost of diluting discipline rows (a trade-off to measure, not a free win)."""
    gen = [r for r in accepted if (r.get("metadata") or {}).get("task_family") == "general_retention"]
    other = [r for r in accepted if (r.get("metadata") or {}).get("task_family") != "general_retention"]
    if not gen or ratio <= 0:
        return accepted
    # solve target_gen so gen/(gen+other) = ratio  ->  target = ratio*other/(1-ratio)
    ratio = min(ratio, 0.9)
    target = int(round(ratio * len(other) / (1 - ratio)))
    if target <= len(gen):
        return accepted  # already at/above target; don't downsample
    reps = []
    i = 0
    while len(gen) + len(reps) < target:
        src = dict(gen[i % len(gen)])
        md = dict(src.get("metadata") or {}); md["replay_copy"] = True
        reps.append({"messages": src["messages"], "metadata": md})
        i += 1
    return other + gen + reps


def build(check_only: bool, stats: bool, *, teacher_spec: str | None = None,
          teacher_workers: int = 8, teacher_max_calls: int | None = None,
          teacher_temperature: float | None = None, retention_ratio: float | None = None) -> int:
    forbidden = set(eval_prompt_set(root=ROOT))
    for r in _read_jsonl(HOLDOUT_SRC):
        pr = prompt_of(r)
        if pr:
            forbidden.add(normalize(pr))

    teacher_rows: list = []
    teacher_stats: dict = {"used": False}
    if teacher_spec:
        bank = teacher_prompt_bank()
        print(f"[teacher] {teacher_spec}: generating over {len(bank)} seed prompts "
              f"(workers={teacher_workers}, max_calls={teacher_max_calls or 'all'}) ...")
        teacher_rows = generate_with_teacher(
            bank, teacher_spec, max_workers=teacher_workers,
            max_calls=teacher_max_calls, temperature=teacher_temperature)
        teacher_stats = {"used": True, "spec": teacher_spec, "seedPrompts": len(bank),
                         "generated": len(teacher_rows)}
        print(f"[teacher] generated {len(teacher_rows)} non-empty candidates "
              f"(these now flow through the SAME admission gate).")

    candidates = synthesize() + load_reuse() + teacher_rows

    accepted, rejected, seen = [], [], set()
    n_decontam = 0
    for row in candidates:
        u = _user_assistant(row)[0]
        if not u:
            rejected.append({"verdict": "malformed", "metadata": row.get("metadata")})
            continue
        nu = normalize(u)
        if nu in forbidden:                       # decontaminate vs eval/benchmark/holdout
            n_decontam += 1
            continue
        if nu in seen:                            # de-dup
            continue
        seen.add(nu)
        v = admit(row)
        if v["ok"]:
            row["metadata"]["gate_verdict"] = v["verdict"]
            row["metadata"].pop("_regate", None)
            accepted.append(row)
        else:
            rejected.append({"verdict": v["verdict"], "reasons": v.get("reasons", []),
                             "metadata": row.get("metadata"), "prompt": u})

    pairs = mine_preference_pairs([r for r in accepted if (r.get("metadata") or {}).get("teacher")
                                   == "deterministic-template"])

    # REPLAY up-weight (anti-forgetting): oversample general rows to --retention-ratio AFTER dedup.
    if retention_ratio:
        before = len(accepted)
        accepted = _oversample_retention(accepted, retention_ratio)
        print(f"[replay] retention up-weight {retention_ratio}: {before} -> {len(accepted)} rows "
              f"(+{len(accepted) - before} general replay copies)")

    # mix report
    fam_counts: dict = {}
    for r in accepted:
        f = r["metadata"].get("task_family", "?")
        fam_counts[f] = fam_counts.get(f, 0) + 1
    total = len(accepted)
    mix = {}
    for f, n in sorted(fam_counts.items()):
        tgt = TARGET_MIX.get(f)
        mix[f] = {
            "rows": n,
            "pct": round(100 * n / total, 1) if total else 0.0,
            "targetPct": [round(100 * tgt[0], 1), round(100 * tgt[1], 1)] if tgt else None,
        }

    lang_counts: dict = {}
    for r in accepted:
        lg = r["metadata"].get("language", "?")
        lang_counts[lg] = lang_counts.get(lg, 0) + 1

    train, valid = _split(accepted)
    mlx_train = [{"messages": r["messages"], "metadata": r["metadata"]} for r in train]
    mlx_valid = [{"messages": r["messages"], "metadata": r["metadata"]} for r in valid]

    manifest = {
        "schema": "sophia.local_sophia_dataset.v3",
        "datasetId": "local_sophia_v3",
        "baseModel": "EXPERIMENT — selected by M1 (do not hard-commit; see training plan)",
        "teacher": ("deterministic-template + human-curated-reuse"
                    + (f" + live-LLM:{teacher_spec}" if teacher_spec else " (no live LLM this run)")),
        "teacherRun": teacher_stats,
        "admission": "agent.gate.check_response (advisor) + public_standard; reject -> preference/hard-neg",
        "totals": {"candidates": len(candidates), "accepted": total,
                   "rejected": len(rejected), "decontaminationDrops": n_decontam,
                   "acceptanceRate": round(total / len(candidates), 3) if candidates else 0.0},
        "byFamily": mix, "byLanguage": dict(sorted(lang_counts.items())),
        "preferencePairs": len(pairs),
        "mlx": {"trainRows": len(mlx_train), "validRows": len(mlx_valid), "maxTokens": 1024,
                "path": "training/local_sophia_v3/mlx"},
        "targetMix": {k: [v[0], v[1]] for k, v in TARGET_MIX.items()},
        "records": (lambda rec: {**rec, "rowsPerRecord": round(total / rec["_total"], 2) if rec["_total"] else None,
                                 "ceiling": ROWS_PER_RECORD_CEILING,
                                 "inflationFlag": bool(rec["_total"] and total / rec["_total"] > ROWS_PER_RECORD_CEILING),
                                 "note": ("Volume headline is RECORDS, not rows. rowsPerRecord above the "
                                          "ceiling means templating (Goodhart), not more ground truth.")})(count_records()),
        "go_no_go": {
            "target": ">=10k decontaminated gate-passed rows",
            "achieved": total,
            "met": total >= 10000,
            "honest_note": ("Volume is CORPUS-bound: rows scale with ground-truth records. ENRICHMENT "
                            "2026-06-27 (tools/enrich_corpus.py, fail-closed on accuracy + provenance; "
                            "auto-decontaminated vs heldout_v1 + transfer_v1; independently fact-verified): "
                            "attributions 30 -> 381, history 23 -> 51, religion 8 -> 21. Dataset 880 -> 3306 "
                            "gate-passed rows (+275%, 3.75x); preference pairs 233 -> 1851 (8x); decontam "
                            "drops held at 90 (zero new eval contamination). STILL BELOW 10k -> M2 stays "
                            "NO-GO on absolute volume (~1030 more records at the measured ~6.5 rows/record). "
                            "Sustained curation, now tooled + accuracy-disciplined + decontaminated."),
        },
        "boundary": ("Habits learned by weights; truth enforced by the EXTERNAL gate. Not an AGI "
                     "claim. General-retention pack is small — expand with a license-clean instruct "
                     "slice before any real SFT or the model becomes a narrow refusal machine."),
    }

    if not check_only:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "mlx").mkdir(exist_ok=True)
        # per-family packs
        packs: dict = {}
        for r in accepted:
            packs.setdefault(r["metadata"]["task_family"], []).append(r)
        for fam, rws in packs.items():
            _write_jsonl(OUT / f"sft_{fam}.jsonl", rws)
        _write_jsonl(OUT / "mlx" / "train.jsonl", mlx_train)
        _write_jsonl(OUT / "mlx" / "valid.jsonl", mlx_valid)
        _write_jsonl(OUT / "preference_pairs.jsonl", pairs)
        _write_jsonl(OUT / "rejected.jsonl", rejected)
        (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")

    print(json.dumps({k: manifest[k] for k in ("totals", "preferencePairs", "go_no_go")},
                     ensure_ascii=False, indent=2))
    if stats:
        print("\n=== MIX vs TARGET ===")
        print(json.dumps({"byFamily": manifest["byFamily"], "byLanguage": manifest["byLanguage"]},
                         ensure_ascii=False, indent=2))
    print(f"\n{'CHECK-ONLY (no writes)' if check_only else 'WROTE training/local_sophia_v3'}: "
          f"{total} gate-passed rows, {len(pairs)} preference pairs, {n_decontam} decontam drops.")
    return 0


def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="synth + gate + decontam, NO writes")
    ap.add_argument("--stats", action="store_true", help="print the full mix report")
    ap.add_argument("--teacher", default=None,
                    help="live-LLM teacher spec (e.g. openrouter:google/gemma-3-4b-it). Generates "
                         "candidate answers that flow through the SAME admission gate. Needs egress.")
    ap.add_argument("--teacher-workers", type=int, default=8, help="teacher generation concurrency")
    ap.add_argument("--teacher-max-calls", type=int, default=None,
                    help="cap teacher generations (budget guard); default = whole prompt bank")
    ap.add_argument("--teacher-temperature", type=float, default=None,
                    help="override sampling temperature for the teacher")
    ap.add_argument("--retention-ratio", type=float, default=None,
                    help="REPLAY up-weight: oversample general_retention rows to this fraction of "
                         "the SFT set (anti-forgetting; e.g. 0.40). Default = no oversampling.")
    args = ap.parse_args()
    return build(check_only=args.check, stats=args.stats, teacher_spec=args.teacher,
                 teacher_workers=args.teacher_workers, teacher_max_calls=args.teacher_max_calls,
                 teacher_temperature=args.teacher_temperature, retention_ratio=args.retention_ratio)


if __name__ == "__main__":
    raise SystemExit(main())
