# Sophia AGI — Max Exposure Launch Copies (v0.7.42)

Ready-to-paste, updated for current state (0% fabrication gate proof, 527 examples, OKF provenance wiki, self-extending verifier flywheel, Grok-CLI 100%, thesis + HF + contract).

**Core numbers to lead with:**
- 0% fabrication (sophia-full gated) vs 17–25% raw models on unknown-answer traps (multi-judge, 3 runs, κ=0.74)
- Teacher / Grok-CLI reference: 100% across philosophy/psych/history/religion leaderboards
- 527 bilingual training examples
- Live thesis + interactive: https://tomyimkc.github.io/sophia-agi/
- HF: https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus
- Repo: https://github.com/tomyimkc/sophia-agi
- Key demo: `python scripts/demo_gate.py`

**Social preview banner:** Use the generated minimalist bronze/ivory image (Greek φ + Chinese 智 + glowing provenance neural net, text "Sophia AGI — Wisdom Before Intelligence"). Copy from the generation output and upload in GitHub Settings → Social preview.

---

## 1. Show HN (news.ycombinator.com/submit)

**URL:** https://tomyimkc.github.io/sophia-agi/

**Title:**
Show HN: Sophia AGI – Provenance-aware gate + corpus that forces LLMs to know "who wrote what" before reasoning (0% fabrication on traps)

**Body:**

LLMs routinely merge lineages and fabricate: Confucius wrote the Dao De Jing, Socrates wrote the Republic, pop-psych slogans as clinical facts.

Sophia AGI is the open **provenance-aware corpus + verifier gate + self-extending flywheel** that makes models abstain instead of lie.

**Validated:** On unknown-answer questions, the full gated Sophia fabricates 0% where raw models fabricate 17–25% (DeepSeek baseline; corroborated by GPT-4o + Claude judges, κ=0.74 across 3 runs). Teacher and Grok-CLI hit 100% on the attribution benchmark.

Live:
- Thesis site with leaderboards + "Ask Sophia" (advisor/repo/life)
- 527 bilingual (EN+中文) examples on HF
- OKF provenance wiki (machine-checkable belief graph)
- `python scripts/demo_gate.py` (30s offline gate demo)
- Verifier contract for any pipeline (fail-closed publish only on accepted)
- Self-extend loop that grows its own verifiers on held-out data

Not AGI — the honest machinery and no-overclaim gate that every safe AGI path needs first ("Wisdom before intelligence").

Feedback wanted: external model submissions on the benchmark, new traps, sector councils (law), replication.

Repo: https://github.com/tomyimkc/sophia-agi
Dataset: https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus
Results + proof: https://github.com/tomyimkc/sophia-agi/blob/main/RESULTS.md

---

## 2. Reddit (r/LocalLLaMA, r/MachineLearning, r/singularity, r/AGI, r/OpenAI)

**Title (r/LocalLLaMA example):**
[Project] Sophia AGI v0.7 — 0% fabrication gate on attribution traps (raw models 17-25%). Provenance corpus + OKF wiki + self-extend flywheel. 527 ex, 100% teacher/Grok.

**Body:**

LLMs keep confidently merging sources they shouldn't (Confucius + Dao De Jing, etc.). I built the open source discipline layer to measure and gate it.

**Current proof (gated, multi-judge validated):**
- Sophia full: **0% fabrication** on unknown-author/quote cases
- Raw model baseline: 16.7–25%
- Δ up to 28%+ with tools
- Reference teacher and Grok-CLI via CLI: 100% on 4-domain benchmark (philosophy 9/9, psych 9/9, history 8/8, religion 6/6 with council panel)

Features:
- Curated RAG + gate (no open web grounding)
- OKF/LLM-Wiki: versioned provenance belief graph + counterfactuals + retractions (new in 0.7)
- Self-extending verifier synthesis flywheel (abstain → synthesize verifier → validate on held-out → promote)
- Portable skill + full MCP (32 tools)
- Contract/governance: record_claim / verify_claim → only accepted outputs publish
- LoRA training path + council distillation for small models
- Live thesis + leaderboards

Try in 30s:
```
git clone https://github.com/tomyimkc/sophia-agi.git
cd sophia-agi
python scripts/demo_gate.py
```

Thesis: https://tomyimkc.github.io/sophia-agi/
HF: https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus
Repo + benchmarks + RESULTS.md: https://github.com/tomyimkc/sophia-agi

Happy to take PRs for new external evals (use the templates) and expansion.

---

## 3. X / Twitter thread (from @HaremKi61388779 or your handle)

1/ LLMs merge Confucius with the Dao De Jing and then lie about it. Most paths to AGI will ship the same hallucinated lineages at scale.

Sophia AGI fixes it at the corpus + gate level.

0% fabrication on the hard traps (vs raw 17-25%). 100% teacher/Grok. Open. Thread 🧵

[attach GIF of demo_gate.py abstain on Dao De Jing trap + provenance chain]

2/ Core idea: "Wisdom before intelligence"

- Source discipline first (who wrote what, tradition boundaries, confidence)
- Verifier gate that is machine-checkable (OKF wiki frontmatter)
- Abstain is success when uncertain
- Self-extend flywheel writes + validates its own new verifiers

3/ Validated numbers (RESULTS.md):
Sophia gated: 0% fab on unknown (deterministic scorer + GPT+Claude judges)
Raw models: 17-25%
Grok-CLI and teacher ref: 100% clean passes

4/ Live today
- Thesis + interactive leaderboards: https://tomyimkc.github.io/sophia-agi/
- HF corpus 527 bilingual ex: https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus
- Offline 30s demo: python scripts/demo_gate.py
- Full repo + contract + MCP + LoRA: https://github.com/tomyimkc/sophia-agi

5/ Why this matters for AGI
Every other approach risks lineage-merge hallucinations at the root. This is the missing "honesty infrastructure."

Maximum exposure = more contributors = richer corpus = stronger safe foundation.

Star if you agree wisdom must come before intelligence.

#AGI #OpenSourceAI #LLMSafety #AIAlignment #SourceDiscipline

Tag relevant: @xai @grok @anthropic @huggingface etc.

Post daily updates with benchmarks, new wiki pages, agent runs.

---

## 4. Other channels (ready text)

**Product Hunt / DevHunt launch blurb:**
Sophia AGI: The provenance gate that makes AI stop lying about who wrote what. 0% fabrication proof on attribution traps. Open corpus, OKF wiki, self-extend verifier flywheel, MCP contract. Wisdom before intelligence for the first safe AGI.

Link to thesis + repo.

**Dev.to / LessWrong / LinkedIn post title:**
How Source Discipline + a Verifier Gate Solves the Biggest Root Risk for AGI (lineage-merge hallucinations)

Lead with 0% vs 17-25% numbers + links + call for benchmark submissions.

**Discord / Alignment / LocalLLaMA / HF communities:**
Same Reddit body shortened + direct "try the 30s demo" + links. Offer to answer questions on the gate.

**Chinese communities (Xiaohongshu / WeChat / philosophy-AI groups):**
Bilingual summary: “智慧先于智能” — Sophia AGI 开源语料 + 验证门，0% 归因陷阱虚构。实测教师/Grok 100%。GitHub + HF + 论文站链接。欢迎贡献新领域与反例。

---

## 5. Additional actions (one-pager)

- Submit thesis https://tomyimkc.github.io/sophia-agi/ to Google Search Console. Add the meta already updated.
- PR into Awesome lists (search "awesome-llm", "awesome-ai-agents", "awesome-ai-safety"). Suggested description: "Provenance-aware benchmark + gate + OKF wiki. 0% fabrication on source traps. Wisdom before intelligence."
- Post in HF Discussions on the dataset page + tag trending models.
- 60s YouTube / Loom: screen record demo_gate + one agent query + thesis browse. Link everywhere.
- Set GitHub watch + Google Alert "sophia-agi" + track stars weekly.

---

**Next:** After these posts, run the repo agent again for follow-ups, and monitor with tools/run_benchmark.py style scripts.

This package + the GitHub polish (description, topics, banner, README hero) should deliver the 5-10x discoverability + viral burst.

Good luck — this is exactly the infrastructure the first real AGI needs.