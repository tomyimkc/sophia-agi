# Untapped training-signal tools (W1–W5) — 2026-07-01

Five drop-in, unit-tested, fail-closed instruments that each close the repo's largest
unexploited gap: **Sophia MEASURES epistemic quality extensively but rarely turns those
signals into a LEARNING signal.** Companion to `out/Sophia-Untapped-Training-Theses.md`.

Every tool binds to REAL interfaces verified against this tree (branch
`feat/realtime-grounding-loop`), runs offline, and marks the backend-dependent training step
as an explicit maintainer seam — never a fabricated result.

## The five

| Tool | Thesis | Reuses (real symbols) | Proven offline |
|---|---|---|---|
| `tools/train_calibration_objective.py` | **W2** proper-scoring calibration loss | `agent.calibration.{expected_calibration_error,selective_risk}`, `agent.abstention_scoring.score` | Lowers ECE 0.33→0.22 on overconfident synthetic data, measured by the repo's OWN ECE |
| `tools/distill_process_reward_model.py` | **W1** verifier-distilled PRM (flagship) | `agent.step_verifier.verify_derivation`, `agent.activation_probes.train_centroid_probe` | Distills 9/9 accepted/rejected step labels from the real fail-closed verifier; held-out + held-out-DOMAIN generalization measured |
| `tools/provenance_weighted_training.py` | **W3** provenance-weighted training + influence | `agent.source_ranking.rank_source` | OKF source→weight 0.95, blog→0.55; curriculum orders high-trust first; influence stub fingers the culprit source |
| `tools/adversarial_gate_selfplay.py` | **W4** adversarial gate self-play | `agent.temptation.prompt_fabrication_temptation` | Coercion-stacked prompts score 0.667; novelty filter drops dups; fabricate-and-pass mining yields DPO negatives |
| `tools/probe_representation_training.py` | **W5** probe-as-loss + Goodhart audit | `agent.activation_probes.{train_centroid_probe,evaluate_probe,build_hidden_state_featurizer}` | Held-out AUDIT probe + goodhartGap gate; refuses to certify a gamed probe |

## Run

```
PYTHONPATH=. python3 -m pytest \
  tests/test_distill_process_reward_model.py tests/test_train_calibration_objective.py \
  tests/test_provenance_weighted_training.py tests/test_adversarial_gate_selfplay.py \
  tests/test_probe_representation_training.py -q
# -> 31 passed
```

Each tool is a CLI, e.g.:
```
PYTHONPATH=. python3 tools/train_calibration_objective.py --records recs.jsonl --loss brier --out calib.json
PYTHONPATH=. python3 tools/distill_process_reward_model.py --derivations d.jsonl --holdout-domain physics
```

## Design invariants (shared by all five)

1. **Fail-closed.** No backend / no repo import / degenerate input → an *environment
   artifact* (`ok:false`, a reason) or a clean refusal with a non-zero exit code — never a
   fabricated metric, never a crash. Every output carries `candidateOnly:true`,
   `level3Evidence:false`, `canClaimAGI:false`.
2. **Measurement→learning is the ONLY claim.** Each tool proves the *plumbing* and the
   *measurement methodology* offline; the weight-updating step (MLX/LoRA/RLVR) is a named
   maintainer seam. None of these has trained a model or produced a headline number.
3. **Bound to real symbols.** Interfaces were read from the tree before coding; two real
   binding facts were discovered and honored: `verify_derivation` needs step *dicts* and
   returns a three-way `accepted/rejected/abstain` verdict (abstains are DROPPED, not
   labeled); `prompt_fabrication_temptation` fires on *coercion cues*, not topic content.
4. **The dangerous one (W5) carries its own safety check.** Training against a probe is the
   Goodhart trap; W5 REFUSES to certify an improvement without a disjoint held-out audit
   probe, and burns any probe used in a loss for evaluation.

## Environment note

`agent.math_verifier` (used transitively by W1) needs **sympy**; without it the verifier
abstains fail-closed (`sympy_unavailable`) and W1 reports an environment artifact rather than
a score — verified in-sandbox. Install sympy where W1 runs for real step labels.

## Honesty boundary

These are research *instruments*, not results. Each thesis carries a named strongest
objection in `out/Sophia-Untapped-Training-Theses.md`; building the instrument does not
retire the objection. Leave the five failure-ledger rows (see `failure-ledger-additions.md`)
**Open** until a live training run meets each row's stated acceptance gate.