# RunPod real-GPU training evidence

Artifacts from `.github/workflows/train-runpod.yml` — real-GPU, gate-disciplined QLoRA
training + eval-ladder runs. **Candidate-only / illustrative evidence. Not a capability
claim. `canClaimAGI` = false.**

## Eval-ladder runs

### `qwen2.5-3b-lora-1ep-seed0.eval-ladder.public-report.json`

First successful real-GPU P6 run — QLoRA (4-bit) fine-tune of `Qwen/Qwen2.5-3B-Instruct`
(1 epoch, seed 0), evaluated base-vs-adapter × with/without the Sophia provenance gate on
a 32-item held-out provenance set (philosophy / psychology / history / religion).

**Headline (honest null):** the adapter is **capability-neutral on held-out content** —
base and adapter both score **23/32 = 71.9%** content. It redistributed within domains
(philosophy ↑ 6→8, psychology ↓ 6→5, religion content ↓ 5→4) for a net-zero content
change. The +6.2pt on the *combined* (format∧content) channel is format-driven, not a
content gain. The provenance gate left content unchanged at every rung.

**Scope:** single-judge, 32-item, 1-epoch — well below the VALIDATED bar (κ≥0.40 /
2 judge families / ≥3 runs / 95% CIs excluding zero). It validates the GitHub-Actions →
RunPod → eval-ladder path end-to-end and records a legitimate directional null to build on
(more epochs / larger curated data / the `moe/adapt` allocation), not a result to overclaim.

Provenance (run URL, head SHA, GPU pool, artifact id + sha256) is in the report's
`provenance` block. The eval-ladder numbers are transcribed from the workflow's own step-7
report printed in the run log; the W2 `promotion.public-report.json` verdict lives inside
the run artifact.

The `*.log` files in this directory are earlier SFT / SSH-smoke run logs.
