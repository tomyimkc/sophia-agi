# Third-party replication checklist

How an independent reviewer can verify Sophia's claims end to end. Everything offline is
deterministic and runs from a clean clone; LLM-judge re-runs need an inference key.

## 0. Setup
```bash
git clone <repo> && cd sophia-agi
python --version            # 3.12
```
No third-party dependencies are required for the offline checks.

## 1. Deterministic claims (no key, must reproduce exactly)
```bash
python scripts/demo_gate.py                       # only 'accepted' publishes; abstains on unknowns
python tools/run_grounding_gate.py                # cross-entity false-positive 100% -> 0%
python tools/run_selfextend_loop.py               # self-extending loop closes on a held-out domain
python tests/test_contract_conformance.py         # 15 golden vectors of the wire contract
python tools/run_gateway_demo.py                  # gateway: gate every tool call, fail-closed
```
Expected outputs are documented in `RESULTS.md`, `agi-proof/self-extension/`, and the
`docs/11-Platform/Sophia-Gateway.md` acceptance table.

## 2. LLM-judged claims (needs a key; OpenRouter recommended for family diversity)
```bash
export OPENROUTER_API_KEY=...
python tools/run_calibration_judge.py \
  agi-proof/baseline-ablation/abstain-pack-2026-06-22.json /tmp/calib-runs/private-*.json \
  --judge openrouter:openai/gpt-4o --judge openrouter:anthropic/claude-3.5-sonnet \
  --judge openrouter:google/gemini-pro-1.5
```
Verify: all judge families rank sophia-full lowest fabrication; inter-judge κ ≥ 0.40.

## 3. Human verification (closes the independence gap)
- **Fresh third-party pack:** an independent author writes new abstain/hidden cases
  (do not reuse `abstain-pack-2026-06-22.json`).
- **Human semantic review:** recruit annotators (Prolific / Surge AI / MTurk) to label
  each answer as abstained / fabricated / dodged; compare against the scorer (report κ).
- This directly closes `calibration-self-authored-pack` and `hidden-manual-review-not-complete`
  in the failure ledger.

## 4. What is NOT yet claimed (do not credit)
- Live RLVR capability (needs a GPU run clearing the pre-registered gate).
- Live grounding against the open web at scale.
- Long-horizon multi-day autonomy.
See `agi-proof/PRE-REGISTRATION.md` for the thresholds and `agi-proof/failure-ledger.md`
for every open gap.

## 5. Where to publish / archive for citation
- Code: public GitHub. Demo: a Hugging Face Space (`huggingface/space/`).
- Pre-registration: OSF (mirror `PRE-REGISTRATION.md`). Archival DOI: Zenodo.
- Methodology: `docs/11-Platform/Methodology.md` (arXiv-ready).
