# Reddit r/LocalLLaMA — post now

**Submit:** https://www.reddit.com/r/LocalLLaMA/submit  
**Flair:** Project (or Resource if available)

---

## Title

```
[Project] Sophia AGI v0.6.0 — provenance benchmark + curated RAG + LoRA at 87% (20/23)
```

---

## Body

```
LLMs keep merging lineages (Confucius → Dao De Jing, Freud → cognitive dissonance, nirvana → heaven). I built an open harness to measure and fix that.

**Sophia AGI v0.6.0** — 518 training examples, 23-case benchmark, epistemic gate, MCP tools

What's new:
- **Curated online RAG** (no open-web grounding) → Gemini/Vertex + gate
- **LoRA sophia-v1** (Qwen2.5-3B): 20/23 on held-out traps — philosophy & history perfect
- **RAG + Claude**: 22/23; fixes all 3 cases LoRA missed (stockholm pop_myth, religion council)
- Self-correcting loop proof + v2 train paraphrases

Domains: philosophy · psychology · history · religion (council panel format for religion)

Leaderboard contract: heuristic source-discipline markers + 中文 summary — same for teacher, Claude, LoRA, RAG.

Thesis site: https://tomyimkc.github.io/sophia-agi/
Repo: https://github.com/tomyimkc/sophia-agi
HF adapter: https://huggingface.co/tomyimkc/sophia-agi-lora-v1

Happy to take benchmark submissions (GPT/Gemini/Grok/Llama) — templates in benchmark/templates/.
```