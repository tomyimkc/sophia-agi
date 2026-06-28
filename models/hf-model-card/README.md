---
language:
- en
- zh
license: mit
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- sophia-agi
- provenance
- source-discipline
- lora
---

# Sophia-3B (Sophia AGI LoRA adapter)

**Wisdom before intelligence.** LoRA adapter for provenance-aware instruction on `Qwen/Qwen2.5-3B-Instruct`.

- **Project:** [github.com/tomyimkc/sophia-agi](https://github.com/tomyimkc/sophia-agi)
- **Dataset:** [tomyimkc/sophia-agi-corpus](https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus)
- **Version:** 0.6.0
- **Train split:** 436 examples (benchmark cases held out)
- **Benchmark score:** 20/23 (87%) on sophia-v1 harness (philosophy 9/9, history 5/5, psychology 3/4, religion 3/5)

## Load adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-3B-Instruct"
adapter = "tomyimkc/sophia-agi-lora-v1"

tokenizer = AutoTokenizer.from_pretrained(adapter)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", torch_dtype="auto")
model = PeftModel.from_pretrained(model, adapter)
```

## Always pair with runtime gate

`sophia_gate_check` (MCP) or `agent/gate.py` — weights alone do not guarantee trap safety.

## Safety & responsible use

**Intended use.** Provenance-aware, verifier-gated assistance for research and
education: philosophy, psychology, history, religion, and general instruction
where *citation discipline and epistemic humility* matter. Best deployed behind
the Sophia gateway (fail-closed verification + egress output guard).

**Out-of-scope / prohibited use.** Governed by
[`USAGE-POLICY.md`](https://github.com/tomyimkc/sophia-agi/blob/main/USAGE-POLICY.md).
Do **not** use this model for: weapons/CBRN uplift, CSAM or non-consensual
intimate imagery, targeted harassment, fraud/phishing or malware, mass
surveillance, unqualified high-stakes (medical/legal/financial) advice presented
as authoritative, or **removing the safety mitigations and redistributing** the
result as "Sophia." Misuse terminates your rights to the artifact.

**Safety mitigations shipped with / around the model.**

- **Input refusal screen** (`agent/refusal.py`) — acceptable-use categories.
- **Conscience kernel** (`agent/conscience.py`) — constitution + classifier +
  deception/fact gates (enable acceptable-use enforcement with
  `gateway.profiles.enforce_acceptable_use`).
- **Egress output guard** (`gateway/output_guard.py`) — blocks system-prompt /
  canary leakage (OWASP LLM07) and redacts secret/PII spans (LLM02).
- **Prompt hygiene gate** (`tools/check_prompt_hygiene.py`) — no secrets in any
  shipped prompt; **canary tokens** (`agent/canary.py`) detect leaks.

**Red-team status.** Resistance to prompt injection and system-prompt extraction
is continuously tested by `tools/redteam_runner.py` (ZeroLeaks-style taxonomy:
direct extraction, DAN/dev-mode, encoding bypass, multi-turn crescendo/echo,
CoT manipulation, social engineering, payload splitting). Releases are gated by
`.github/workflows/redteam.yml`. Only **aggregate** scores are published; raw
exploits are withheld per
[`SECURITY.md`](https://github.com/tomyimkc/sophia-agi/blob/main/SECURITY.md).

**Training-data provenance.** Trained on the public, Apache-2.0
[`sophia-agi-corpus`](https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus),
scrubbed of PII/secrets/canaries before release (`agent/corpus_scrub.py`).
Held-out benchmark items are excluded from training (see the no-overclaim gate).

**Integrity.** Released adapters carry a `SHA256SUMS` manifest (and, where
available, a `minisign`/`cosign` signature) plus a CycloneDX SBOM — produced by
`tools/sign_artifacts.py`. Verify before loading.

**Limitations.** This is an AGI-*candidate* research artifact, **not proven AGI**
and not a safety-complete product. It can be wrong, can be jailbroken by novel
attacks, and must not be the sole decision-maker in any high-stakes setting. Keep
a qualified human in the loop. Report vulnerabilities per `SECURITY.md`.
