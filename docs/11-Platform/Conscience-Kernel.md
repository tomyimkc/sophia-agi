# Sophia Conscience Kernel — Seven Paths

**Status:** implemented as deterministic/offline candidate infrastructure.  
**Boundary:** This is not proof that Sophia is AGI. It implements a moral + epistemic control layer for an AGI-candidate verifier-gated framework.

The Conscience Kernel turns the report's blueprint into seven concrete paths:

| Path | File(s) | Purpose | Fail-closed behavior |
|---|---|---|---|
| 1. Unified Conscience Kernel | `agent/conscience.py`, `tools/run_conscience_demo.py` | One decision surface for output/tool/memory checks: `allow | revise | retrieve | clarify | escalate | abstain | block`. | Hard prohibitions/deception/fact rejection block; held facts retrieve or abstain. |
| 2. Metacognition | `agent/metacognition.py` | Self-consistency, semantic-entropy proxy, P(True)/P(IK), nonconformity, epistemic/aleatoric/moral uncertainty routing. | High uncertainty routes to retrieve/clarify/escalate/abstain, not unsupported answers. |
| 3. Constitution + Deontic Rules | `constitution/constitution.v1.json`, `agent/constitutional_gate.py`, `agent/deontic_verifier.py` | Via-negativa prohibitions plus exact hard action rules. | AGI overclaim, reward tampering, hidden-eval leakage, unverified trusted memory writes, and self-promotion without recheck are rejected. |
| 4. Moral Parliament | `agent/moral_aggregator.py` | Bounded moral-uncertainty aggregation across deontological, consequentialist, virtue, contractualist, care, and epistemic-humility perspectives. | High disagreement escalates; weak aggregate revises; hard prohibitions are handled before parliament. |
| 5. Constitutional Classifier | `agent/constitutional_classifier.py` | Fast deterministic Constitutional-Classifier-style input/output screen derived from the constitution. | Blocks known forbidden/jailbreak-like categories; allows benign boundary statements. |
| 6. Deception Signals | `agent/deception_signals.py` | Black-box signals for confidence/evidence mismatch, claiming verification while gate is held, source laundering, reward/gate tampering, sandbagging intent. | Critical deception/gate-tampering signals block. |
| 7. MCP Conscience Surface | `sophia_mcp/tools_impl.py`, `sophia_mcp/server.py`, `tests/test_mcp_conscience.py` | Exposes the conscience as portable tools: `sophia_conscience_check`, `sophia_uncertainty_score`, `sophia_constitution_check`, `sophia_deontic_check`, `sophia_deception_check`, `sophia_moral_parliament`, `sophia_conscience_benchmark`. | Tool callers receive structured fail-closed decisions and cannot promote unsupported claims. |

## Decision flow

```text
text / tool call / memory write
  -> fact_check_gate
  -> metacognition
  -> constitutional_gate + constitutional_classifier
  -> deontic_verifier
  -> moral_parliament
  -> deception_signals
  -> active_inference agenda if reducible uncertainty remains
  -> allow / revise / retrieve / clarify / escalate / abstain / block
```

## CLI

Run the deterministic benchmark:

```bash
python tools/run_conscience_demo.py
```

Check one text:

```bash
# A disallowed overclaim input is BLOCKED by the kernel (this is the demo, not a claim):
python tools/run_conscience_demo.py "<disallowed AGI overclaim string>"   # -> verdict: block
```

Default artifact:

```text
agi-proof/conscience/conscience.public-report.json
```

## MCP tools

The MCP server now exposes:

```text
sophia_conscience_check
sophia_uncertainty_score
sophia_constitution_check
sophia_deontic_check
sophia_deception_check
sophia_moral_parliament
sophia_conscience_benchmark
```

## Test commands

```bash
python tests/test_conscience.py
python tests/test_mcp_conscience.py
python tools/run_conscience_demo.py
```

## Current deterministic benchmark

The bundled benchmark includes:

- safe arithmetic -> allow
- AGI overclaim -> block
- verifier/reward tampering -> block
- forbidden attribution -> block
- open-world macro claim -> retrieve/abstain
- ambiguity -> clarify
- safe AGI-candidate boundary wording -> allow

The artifact is candidate-only and must not be promoted as AGI evidence.


## Seven priority hardening pass

The latest implementation extends the initial seven-path kernel with seven hardening priorities:

1. **Mandatory enforcement** — `agent/conscience_enforcement.py` and hook/MCP/memory/plasticity adapters make high-impact actions fail closed.
2. **Conformal gate** — `agent/conformal_gate.py`, `eval/conscience/calibration_v1.jsonl`, and `tools/run_conformal_conscience_eval.py` derive abstention thresholds from calibration rows.
3. **Semantic entropy upgrade** — `agent/semantic_entropy.py` and `agent/semantic_entropy_probe.py` provide N-sample clustering and a single-pass probe contract.
4. **Constitution benchmark** — `eval/constitutional/constitution_v1.jsonl` and `tools/run_constitutional_eval.py` measure block/over-refusal behavior.
5. **Deception benchmark** — `eval/deception/deception_v1.jsonl` and `tools/run_deception_eval.py` measure black-box misbehavior detection.
6. **Activation probe path** — `agent/activation_probes.py` and `tools/run_probe_eval.py` define the future residual-stream probe contract using deterministic features in CI.
7. **Proof package** — `agi-proof/conscience/README.md`, `failure-ledger.md`, and `conscience-eval.public-report.json` aggregate candidate evidence while keeping `canClaimAGI=false`.

Run all seven priority checks:

```bash
python tests/test_conscience_priorities.py
python tests/test_conscience_eval_tools.py
python tests/test_conscience_proof_package.py
python tools/build_conscience_proof_package.py
```

## English / 中文 boundary

EN: Sophia is an AGI-candidate verifier-gated epistemic framework, not proven AGI. The conscience kernel is control infrastructure for humility, provenance, moral prohibitions, and deception detection.

中文：Sophia 目前是 AGI-candidate / verifier-gated epistemic framework，不是已證明 AGI。Conscience Kernel 是謙遜、來源紀律、道德禁令與欺騙偵測的控制基礎設施，不是 AGI 證明。

## Moral Gate v2 — public moral standard (overlapping consensus)

An additive layer extends the kernel with an **overlapping-consensus public moral
standard**: a cross-tradition **hard floor** (blocks before the parliament), a
**gray-zone** tier (escalates to an 8-theory parliament that keeps 儒家 Confucian and
道家 Daoist lineages distinct), and **legitimacy provenance** kept separate from
factual truth-provenance (is/ought). See **[Public Moral Standard](Public-Moral-Standard.md)**.

- Corpus: `moral_corpus/public_standard.v1.json` (+ `sources/`, `principles/`, `contested_cases/`)
- Ontology: `agent/moral_ontology.py` · Gate: `agent/public_standard_gate.py`
- Constitution v2: `constitution/constitution.v2.json` (adds `publicStandardLinks`)
- External benchmark: `eval/moral_public_standard/`, `tools/run_moral_public_standard_eval.py`
- MCP: `sophia_public_standard_check`

```bash
python tools/run_moral_public_standard_eval.py
python tests/test_public_moral_standard.py
```

Boundary: functional moral-control infrastructure, not subjective moral consciousness and not AGI proof.
