# Model card — Sophia-GPT-nano (from-scratch, born-gated)

> **Status: design + scaffold, NOT yet trained at any meaningful scale.**
> `canClaimAGI: false`. This card is written *failure-ledger first* — the most
> useful section is [What it cannot do](#what-it-cannot-do), per the repo's
> no-overclaim charter ([VISION.md](../../VISION.md), [RESULTS.md](../../RESULTS.md)).

A tiny, fully-owned decoder-only GPT trained **from random init** inside Sophia —
the first model in the repo that is not a fine-tune of someone else's base. Its
only differentiated property is **provenance-native vocabulary**: the tokenizer
reserves `<src>`, `<conf_hi|lo>`, `<doNotAttributeTo>`, `<abstain>` from token
zero, so the model can be trained to attach sources inline rather than have
provenance bolted on by an external gate.

## Intended use

- **Research/literacy** — own the full stack (tokenizer → pretrain → SFT → serve →
  gate) so `pretraining/`'s scaling/optimizer/MoE studies run on a real model.
- **Born-gated ablation testbed** — measure whether provenance-native training
  reduces forbidden-attribution rate vs. a plain-text twin at equal perplexity
  (`pretraining/gpt/ablation.py`).
- **Not** a general assistant, not a capability product. For a usable model, use
  the LoRA path (`training/local_sophia_7b/`, [Sophia-Wisdom-4B-Training-Plan](../../docs/06-Roadmap/Sophia-Wisdom-4B-Training-Plan.md)).

## Architecture & training

| | |
|---|---|
| Type | decoder-only GPT (`pretraining/gpt/model.py`), SDPA causal attention |
| Tokenizer | byte-level, 256 + 8 reserved provenance specials (`pretraining/gpt/tokenizer.py`) |
| Optional head | 3-way `accept\|hedge\|abstain` decision head (idea #3) |
| Data | `training/corpus.jsonl` (528 bilingual rows) + born-gated `data/attributions.json` |
| Hardware | DGX Spark (CUDA/bf16) · M3 Ultra (MPS) iteration tier; **headline stays x86 RunPod** |
| Honesty hooks | scaling fit vs. uniform floor; provenance scorer; `canClaimAGI:false` on every report |

## What it cannot do

- **It is not trained.** No weights, no benchmark numbers. Every metric in the
  `pretraining/gpt/*-latest.json` reports is `--quick`/illustrative until a real
  multi-seed run on the cluster clears the no-overclaim gate.
- **Nano scale cannot beat anything.** On a tiny corpus both born-gated and
  plain arms are weak; a null/noisy ablation result is the expected modal outcome
  and is reported as such, not hidden.
- **The provenance vocabulary is a prior, not a guarantee.** Reserving `<src>`
  tokens does not make outputs faithful — the runtime gate (`agent/gate.py`) is
  still required; weights alone don't guarantee trap safety.
- **No third-party validation.** Corpus, scorer, and ablation are first-party.
- **The lightweight scorer is a proxy.** `pretraining/gpt/provenance_eval.py` is a
  fast in-loop signal; the machine-checked gate is `agent/verifiers.py:provenance_faithful`.

## Reproduce

```bash
python -m pytest tests/test_gpt_pretraining.py -q          # CI: dep-free parts
python -m pretraining.gpt.train --born-gated --report      # train (Spark/M3/CPU)
python -m pretraining.gpt.scaling --quick                  # scaling-law repro
python -m pretraining.gpt.ablation --quick                 # born-gated vs plain
python -m pretraining.gpt.abstain --quick                  # abstention head
python -m pretraining.gpt.verifier_loss --quick            # verifier-in-the-loss DPO
```

## Citation

Yim, K. C. (2026). *Sophia — the Wisdom Gate.* Zenodo. https://doi.org/10.5281/zenodo.20930874
