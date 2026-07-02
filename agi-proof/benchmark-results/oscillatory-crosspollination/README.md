# O1–O5 oscillatory cross-pollination — live benchmark (2026-07-02)

Took the five fail-closed instruments on `feat/oscillatory-crosspollination` from "passes unit
tests" to **measured against real data**, honestly. Every number below was reproduced and
adversarially re-derived (independent refutation attempts). `canClaimAGI=false` throughout.

## Bottom line

**None of the five oscillatory-coherence gates are met on the real data tested.** The Kuramoto
convergence/coherence readout does not add a verification/abstention signal beyond sophia's
existing gates. All five failure-ledger rows stay **Open**.

| Dir | Gate | Verdict | Key real-data number |
|---|---|---|---|
| **O1** consensus gate | consensus beats self-consistency, paired-AURC CI>0, ≥2 seeds | **NOT MET** | ties baseline; AURC-delta CI contains 0 all 3 seeds; `r` even slightly higher on *wrong* answers |
| **O2** energy verifier | hidden-state energy, AUROC/ECE + CI, goodhartGap≤0.15 | **NOT MET** | featurizer now real (Qwen MLX); LODO OOF AUROC 0.489; ECE 0.76; goodhartGap 0.26 |
| **O3** fixed-point | F1 beats admission arm at matched coverage | **NOT MET** | separation weak (AUROC 0.75) & mostly a `[ENTAILS]` label-token leak; full gate blocked (py3.11 + unbuilt coverage-F1) |
| **O4** adaptive compute | ≥25% fewer samples at no AURC loss, wired into long_horizon | **NOT MET** | 60% raw savings but a 2-oscillator k=2 saturation artifact; "no loss" vacuous (no selective skill); seam absent |
| **O5** substrate | (hardware — out of scope for software) | **CONFIRMED sim-only** | `simulationOnly:true`, `hardwareClaim:false`; gate correctly stays Open |

## What was genuinely built (real, reusable)

- **Semantic embedder seam** — `oscillator_core.hash_embed` now delegates to a cached
  all-MiniLM-L6-v2 under `OSC_EMBED_BACKEND=minilm` (offline, no cloud key), preserving both
  test invariants; `active_embed_backend()` + `embedBackend`/`hashEmbedSeam` surface it honestly.
- **Real MLX hidden-state featurizer for O2** — `build_hidden_state_featurizer` driven by a
  loaded `Qwen/Qwen2.5-3B-Instruct` (no stub). This also unlocks the W1/W5 seam.
- **The missing O1/O4 adapter** — `tools/gen_simpleqa_consensus_pack.py` re-generates the raw
  `{samples, correct}` texts the SimpleQA runner discarded (DeepSeek subject, grok grader).
- **The missing O2 gate metric** — `tools/eval_o2_energy_hidden.py` adds AUROC + paired-bootstrap
  CI with a leave-one-domain-out (no in-sample optimism) protocol.

## Why the theory failed (honest diagnosis)

The Kuramoto order parameter `r` **saturates**: for ≤2 short answers it locks toward 1.0
regardless of agreement, and for k≈6 short factual answers it carries no correctness-separation
direction beyond exact-string majority agreement — which is already a strong baseline. The energy
head's hidden-state signal does not survive leave-one-domain-out even when supervised on ground
truth. The one apparent O3 signal is a data-construction artifact. This is a clean negative result
for "coherence-as-verification" on these tasks, not a wiring failure — the instruments ran on real
backends, a real semantic embedder, and real labelled data.

## Reproduce

Backends: DeepSeek (`DEEPSEEK_API_KEY`) subject + grok grader (non-mock asserted); embedder
all-MiniLM-L6-v2 (offline); hidden states Qwen2.5-3B via MLX. See `O1-O5.public-report.json` for
per-tool commands, CIs, and artifact sha256s. Raw run artifacts + the generated datasets are in
this directory and `data/`.

## O2 deepening (2026-07-02) — the negative is robust

Given O2's fairest shot — a proper L2 logistic probe over the same real Qwen MLX hidden states,
the combined largest dataset (factcheck-full-r1 + fact-check-live, n=122, more domains), and both
supervision labels (`accepted` and `correct`) — the leave-one-domain-out AUROC still does not clear
chance (0.38–0.43; several arms below 0.5 = anti-correlated on unseen domains). So O2's failure is
robust to head + data + label, not an artifact of the weak centroid stand-in. See
`o2_probe_sweep.report.json`.

**Unlocked follow-ons (not executed here):** the now-real `build_hidden_state_featurizer` is the
shared seam for **W1** (process-reward model) and **W5** (probe-as-loss). O2's *named* dataset
source (`agent/memory/contract/verified_traces.jsonl`) is empty and would need a full RLVR run to
populate — the practical substitute used here is the factcheck verifier-labelled packs.
