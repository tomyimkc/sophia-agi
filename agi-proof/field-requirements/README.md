# Field-Requirements Capability Proof

**Claim:** this repository has already developed, and can demonstrate with tested
code and measured artifacts, the *core engineering functions* that frontier AI
labs are hiring for — most directly the new **agent-evaluation** and
**specialized-domain data** functions. This directory makes that claim
**machine-checkable** rather than rhetorical.

## Why this exists

DeepSeek / High-Flyer's 2026 recruitment drive — ~33 roles across seven categories
— is an explicit pivot to **agentic AI plus owning the data/eval/infra stack**,
with three brand-new role titles: **Agent Data Evaluation Expert, Agent Deep
Learning Algorithm Researcher, Agent Infrastructure Engineer**, plus
specialized-domain data roles for **medicine and law**
([Bloomberg](https://www.bloomberg.com/news/articles/2026-03-24/deepseek-s-latest-job-postings-highlight-pivot-to-agentic-ai),
[SCMP](https://www.scmp.com/tech/big-tech/article/3358394/deepseek-hiring-spree-chinese-ai-firm-seeks-newcomers-it-pursues-agi),
[Digitimes](https://www.digitimes.com/news/a20260611PD231/deepseek-infrastructure-data-capacity-training.html)).

Those categories are a precise statement of **market demand**. This proof maps each
one to concrete artifacts already in the repo.

## How it is enforced (not just asserted)

- [`manifest.json`](manifest.json) — the machine-readable map: each capability lists
  its `modules`, `tests`, `evidence`, an honest `status`, and (for the strongest
  tier) a `measured` line.
- [`tools/verify_field_requirements.py`](../../tools/verify_field_requirements.py)
  — walks the manifest and checks every cited module exists **and compiles**, every
  test file exists, and every evidence artifact is present. Exits non-zero on any
  gap. Run it:

  ```bash
  python tools/verify_field_requirements.py            # PASS/FAIL table
  python tools/verify_field_requirements.py --import   # also try importing modules
  ```

- [`tests/test_field_requirements.py`](../../tests/test_field_requirements.py) —
  makes the verification a CI gate. Delete a cited module or report and this test
  fails, so the map cannot silently drift out of truth.

This is the same discipline as the rest of `agi-proof/`: a claim is only as good as
the artifact and test behind it.

## Status legend

| Status | Meaning |
| --- | --- |
| `demonstrated` | module(s) + passing tests, and — where the capability is a measurable claim — a committed measured artifact. |
| `candidate` | built and tested, but evidence is first-party / seed / not-yet-independently-validated, **or** the underlying research claim is explicitly unproven. |
| `interface-only` | a fail-closed interface with a toy reference implementation (see [AGI-Missing-Pillars](../../docs/11-Platform/AGI-Missing-Pillars.md)). |

`demonstrated` is a statement about the **engineering** — the function exists, is
tested, and where applicable produces a measured number under the repo's
no-overclaim gate. It is **never** a claim that the broader research goal (AGI,
beating a frontier lab on a hidden eval) is achieved. Those remain `candidate`, by
design, with the gaps tracked in the [failure ledger](../failure-ledger.md).

## The map (summary)

| Field requirement (market) | Repo capability | Status |
| --- | --- | --- |
| **Agent Data Evaluation Expert** (new) | `agent/trajectory_eval.py` + `provenance_bench/agent_faithfulness.py` — step-by-step trajectory faithfulness + a deterministic benchmark | **demonstrated** |
| Specialized-domain data — **law** | `agent/legal_faithfulness.py` — fake-citation + misstated-authority detection, judged under the gate | **demonstrated** |
| Specialized-domain data — **medicine** | `agent/medical_faithfulness.py` — PMID/DOI/guideline existence + faithfulness | candidate |
| Specialized-domain data — **non-English** | bilingual corpus + `agent/cantonese.py` | candidate |
| **Model data strategy** / pre-training data | `tools/validate_attribution.py`, `agent/poison_resistant_ingestion.py`, sealed held-out splits | **demonstrated** |
| **AI core-system R&D** | the verifier/gate kernel + MCP server (`agent/gate.py`, `proof_carrying_reasoning.py`, `formal_verifier.py`, `sophia_mcp/server.py`) | **demonstrated** |
| **Deep-learning research (AGI)** | RSI/continual scaffolds, pre-registered thresholds, public failure ledger | candidate |
| **Agent Infrastructure Engineer** (new) / supercomputing | RunPod orchestration (`tools/runpod_*.py`) + CI workflows | candidate |
| **Server-side development** | MCP server + governance gateway | **demonstrated** |

Full detail, evidence paths, measured numbers, and per-capability limits are in
[`manifest.json`](manifest.json). The single headline measured result behind the
core kernel — attribution-hallucination **36.1% → 23.6%** (Δ12.5%, 95% CI
[5.6%, 19.4%]) at 0% false-positive cost — is the one number that clears the
no-overclaim gate; see [RESULTS.md](../../RESULTS.md).

## Honest reading

The repo is **strongest exactly where the new money is**: agent evaluation and
domain (law/medicine) data discipline. It is **candid where it is weaker**:
multilingual breadth, owned-cluster infrastructure, and the AGI research claim
itself are `candidate`, not `demonstrated`. The proof is built so that an outside
reader can run one command and see which is which.
