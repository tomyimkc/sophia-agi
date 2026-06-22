---
title: Sophia Governance Gate
emoji: 🛡️
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: huggingface/space/app.py
pinned: false
license: mit
---

# Sophia — the governance gate (interactive demo)

Verify, in your browser, the core claim: **Sophia publishes only what it can check, and
abstains instead of fabricating.** Runs offline on deterministic verifiers — no API key.

- **Verify a claim** — only `accepted` is publishable; an unsourced claim is `held`.
- **Abstain vs fabricate** — an honest "unknown" scores 1.0; a confident fabricated
  specific scores 0.
- **Gated tool call** — a federated knowledge tool whose ungrounded output is withheld.

## Deploy
Create a **Gradio** Space pointing at this repository with the metadata above
(`app_file: huggingface/space/app.py`). The Space imports `sophia_contract`, `gateway`,
and `provenance_bench` from the repo root.

`requirements.txt` lists only `gradio`; everything else is standard-library and in-repo.

Source & methodology: see the repo `README.md`, `CONTRACT.md`, and
`docs/11-Platform/Methodology.md`.
