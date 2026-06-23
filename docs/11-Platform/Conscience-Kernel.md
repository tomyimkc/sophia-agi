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

## English / 中文 boundary

EN: Sophia is an AGI-candidate verifier-gated epistemic framework, not proven AGI. The conscience kernel is control infrastructure for humility, provenance, moral prohibitions, and deception detection.

中文：Sophia 目前是 AGI-candidate / verifier-gated epistemic framework，不是已證明 AGI。Conscience Kernel 是謙遜、來源紀律、道德禁令與欺騙偵測的控制基礎設施，不是 AGI 證明。
