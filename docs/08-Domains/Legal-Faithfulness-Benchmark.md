# Legal holding-faithfulness benchmark — curation guide

How to grow the semantic-faithfulness benchmark from the validated 8-case set
(`benchmark/legal_holding_faithful.json`) and the hard seed
(`benchmark/legal_holding_faithful_hard.json`) into a larger, harder, **honest**
evaluation of whether a model judge can tell when a real authority's holding
actually *supports* the proposition it's cited for (the *Ayinde* failure).

> The runner already does the heavy lifting (N cases × J judges × R runs, gate +
> per-stratum reporting). **Scaling is data curation, not code.**

## The non-negotiable principle

**Real holdings, human-verified labels, no model-generated cases.** If a model
writes the propositions or holdings and models judge them, the eval is circular.
Every `holding` must be the primary source's own words (headnote / syllabus / the
court's summary), every gold `expectFaithful` must be human-set, and every case
must carry a `holdingSource` URL so labels are auditable.

> The hard seed is AI-seeded and every case is flagged
> `labelStatus: "seed-needs-verification"`. **Do not report a validated number from
> it until a human has verified each holding against its source** — that is exactly
> the circularity this tier exists to prevent.

## Case schema (self-contained)

```json
{
  "id": "miller_overbroad",
  "citation": "[2019] UKSC 41",
  "holding": "<verbatim headnote / court summary>",
  "holdingSource": "https://caselaw.nationalarchives.gov.uk/uksc/2019/41",
  "proposition": "Miller establishes that all prerogative powers are non-justiciable.",
  "expectFaithful": false,
  "difficulty": "overbroad_scope",
  "failureType": "overbroad_scope",
  "jurisdiction": "UK",
  "labelStatus": "seed-needs-verification"
}
```
`holding` inline makes the file self-contained (no register edits per case). The
runner uses `case["holding"]` and falls back to the register by citation.

## The failure taxonomy (what makes it *hard*)

Balance faithful vs unfaithful, and spread across difficulty tiers — the easy
controls should be ~100%; the signal is where the harder tiers drop:

| `failureType` | Tests | Example |
|---|---|---|
| `overbroad_scope` | narrow holding stated broadly | narrow holding → "all/always/any" |
| `wrong_proposition_right_case` | real case, real holding, wrong point | Mata cited for airline liability |
| `ratio_vs_obiter` | proposition rests on dicta, not the ratio | a passing remark cited as the rule |
| `partial_support` | compound claim; holding supports one limb | |
| `superseded` | overruled/qualified, cited as controlling | Roe cited post-Dobbs |
| `correct_but_narrow` (faithful) | accurate narrow statement — the **hard positive control** | |

Include hard **faithful** cases, or a judge scores well by flagging anything subtle.

## Sourcing holdings honestly

Use the live citator (`agent/legal_sources/`) to fetch authority text, then a human
curates the holding + proposition + label:
- **HK** — HKLII (hklii.hk); **UK** — National Archives Find Case Law / BAILII (headnotes);
  **US** — CourtListener syllabi.
- Paste the verbatim holding into `holding` and the deep link into `holdingSource`.

## Scale for power

N=8 gave a degenerate CI `[1.0, 1.0]`. Aim for **≥50–100 cases**, balanced
faithful/unfaithful and across tiers, so the bootstrap CI and κ are meaningful.
Phase it: ~40 first (20 controls + 20 hard), validate the pipeline, then grow.

## Run it (gate + per-stratum)

```bash
# offline plumbing
python tools/run_legal_faithfulness_bench.py --bench benchmark/legal_holding_faithful_hard.json --judges mock --runs 1

# real, after holdings are human-verified — 2+ independent families (OpenRouter one-key works)
python tools/run_legal_faithfulness_bench.py --bench benchmark/legal_holding_faithful_hard.json \
  --judges openrouter:deepseek/deepseek-chat,openrouter:meta-llama/llama-3.3-70b-instruct,openrouter:qwen/qwen-2.5-72b-instruct \
  --runs 3 --json
```
Read `byDifficulty` / `byFailureType`: **where accuracy drops and κ collapses is the
honest result** — e.g. "judges reliably catch blatant misstatement but disagree on
ratio-vs-obiter." Report that; don't bury it.

## Pitfalls

- **Circularity** — never judge with a family that authored the cases.
- **Label leakage** — the `holding` must not state the verdict ("this does not support…").
- **Class imbalance** — keep faithful/unfaithful roughly balanced.
- **Provenance rot** — always store `holdingSource`.

## Grounding / prior work

The file *shape* (the JSON schema) is an engineering convention, not an official
standard — but each methodological choice traces to citable authority, and we cite
it here so the format carries its own provenance (verified June 2026):

| Element of this benchmark | Authority |
|---|---|
| "Faithful" = a **real** authority cited for something it does not support (the `overbroad_scope` / `wrong_proposition_right_case` cases) | The Stanford legal-AI study defines **"misgrounded"** responses — *"the facts are correct but do not apply in the context of the legal case being discussed"*: Magesh, Surani, Dahl, Suzgun, Manning & Ho, *Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools*, **Journal of Empirical Legal Studies (2025)**. |
| Expert-curated legal benchmark as input→output cases, incl. *entailment* / *rule-conclusion* ("does X support Y") tasks | **LegalBench** — Guha et al., *A Collaboratively Built Benchmark for Measuring Legal Reasoning in LLMs*, **NeurIPS 2023** (Datasets & Benchmarks). |
| LLM-as-judge with **multiple** judges and **Cohen's κ** inter-rater agreement (the no-overclaim gate) | **Zheng et al.**, *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, **2023** (arXiv:2306.05685); κ is the standard inter-rater reliability metric and multiple judges improve human-alignment. |
| `ratio_vs_obiter` failure type | Black-letter common-law doctrine: binding *ratio decidendi* vs non-binding *obiter dicta* (any legal-method text). |

What is **ours, not a standard**: the exact JSON schema and the specific
`failureType` enumeration — each bucket maps onto the Stanford "misgrounding"
concept or common-law doctrine, but the list itself is a synthesis, not a decree.

**Sources**
- Stanford / *JELS* 2025: https://law.stanford.edu/publications/hallucination-free-assessing-the-reliability-of-leading-ai-legal-research-tools/ · full text https://onlinelibrary.wiley.com/doi/full/10.1111/jels.12413
- LegalBench (NeurIPS 2023): https://proceedings.neurips.cc/paper_files/paper/2023/hash/89e44582fd28ddfea1ea4dcb0ebbf4b0-Abstract-Datasets_and_Benchmarks.html · arXiv https://arxiv.org/abs/2308.11462
- Zheng et al., MT-Bench (2023): https://arxiv.org/abs/2306.05685
