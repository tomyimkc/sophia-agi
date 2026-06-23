# Activation-Steering Experiment (Spec B)

**The falsifiable claim, offline (CI-gated):** the steering *machinery* is
arithmetically correct — `register_forward_hook` adds `alpha·v̂` surgically,
difference-of-means recovers a planted direction, composition orthogonalizes,
and the SSA verdict abstains fail-closed. Verified by `tests/test_steering.py`
(pure stdlib) + `tests/test_personality_steering.py` (toy torch hook, skip-guarded
in CI) + `python tools/run_steering.py --model mock --dry-run`.

**The pre-registered live claim (OPEN until a gated run):** SSA — for N≥8 personas,
Level-3 steering produces a residualized OCEAN shift strictly larger than Spec A's
Level-1 persona baseline, behavior-corroborated (≥2 judge families distinct from
the subject, κ≥0.40) and capability-preserving. `SSA = 0/N` is a legitimate result.
Subject = local Phi-3.5-mini (fallback chain in `tools/run_steering.py`); judges =
local Ollama `qwen2.5:3b` + `llama3.2:3b`. Determinism is best-effort on MPS/Ollama;
only the pure scorer is bitwise-deterministic.

**Two-channel cross-validation:** the self-report channel reuses Spec A's
`measure_ocean`/`score_items`; the behavioral channel (`agent/personality_behavioral.py`)
judges open-ended, trait-name-free output. A self-report shift without a behavioral
shift → ABSTAIN ("claims, does not enact").

Run: `python tools/run_steering.py --model mock --dry-run` (offline) ·
`python tools/run_steering.py --model phi3.5` (gated real run, downloads ~7.6 GB).
