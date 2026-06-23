# Sophia AGI-candidate infrastructure plan

Sophia's credible path is not a bigger slogan; it is an eval-gated operating
system for growing competence safely. This runbook maps that path to concrete
repo infrastructure.

## Core invariant

Every new capability must flow through:

```text
task -> verifier -> skill/tool -> eval suite -> held-out validation -> registry -> gateway exposure -> monitoring
```

A capability is not promoted because an LLM proposed it. It is promoted because
it passed a held-out verifier/eval gate and did not regress protected knowledge.

## AGI verification gate

Run the non-human claim-readiness gate after every evidence-producing run:

```bash
python tools/run_agi_verification_gate.py --run-local-smoke --allow-open
python tools/run_agi_verification_gate.py --target level4 --allow-open
```

The gate writes:

```text
agi-proof/agi-verification/agi-verification-report.json
```

It intentionally keeps `canClaimAGI=false`; the repo can machine-check evidence lanes, but it cannot philosophically self-certify AGI. Stronger wording is blocked until the gate reports the relevant level as passed and external review accepts the evidence.

### Non-human verification lanes the gate enforces

The gate maps each capability in `agi-proof/definition.md` to a machine-checkable
lane (criteria: `agi-proof/agi-verification/criteria.json`):

- `provenance_validation`, `published_provenance_delta`, `calibration_abstention` (level 2)
- `grounded_gate`, `coding_eval`, `memory_eval` (level 3, deterministic smoke)
- `verifier_synthesis_integrity` — self-extension is non-circular: a synthesized
  verifier is trusted ONLY after meta-verification; the without-meta ablation must fail
- `cross_domain_transfer` — generalization (Legg-Hutter / Chollet): memorized rules
  must not transfer to unseen entities; grounding is required for low-false-positive transfer
- `hidden_full_comparison`, `self_extension_loop`, `distribution_shift`, `long_horizon_30m`,
  `rlvr_live_training` (level 3, artifact)
- `rlvr_adapter_eval`, `external_benchmarks`, `long_horizon_2h_1d`,
  `third_party_machine_replication` (level 4, artifact)

`verifier_synthesis_integrity` and `cross_domain_transfer` were added because the
operational definition lists skill-acquisition integrity and cross-domain transfer
as required capabilities; without them the gate would have let those stay unmeasured
prose claims.

## Local CI / Mac lane

Run these on every serious change:

```bash
python tools/validate_attribution.py
python tools/run_rlvr.py --model mock --dry-run
python tools/eval_rlvr_adapter.py --mode mock
python tools/run_coding_eval.py
python tools/run_memory_eval.py
python tools/run_grounding_gate.py --runs 3 --min-cases 40
```

Purpose:

- provenance rules still hold;
- RLVR reward/eval plumbing is sound without a GPU;
- executable code verifier lane works;
- memory writes are append-only and gated;
- grounded retrieval gate stays statistically useful.

## GPU lane

Run only after local lane is green:

```bash
python tools/runpod_rlvr.py \
  --api-key-file /private/tmp/runpod_api_key \
  --yes --source git --remote-mode live \
  --quant bf16 --vllm none --epochs 1.0 \
  --no-remote-delete-watchdog
```

The RunPod workflow must copy back:

```text
*.rlvr.public-report.json
*.rlvr.offline-report.json
*.repo-head.txt
*.sophia-rlvr-v1.tar.gz
*.sophia-rlvr-v1.tar.gz.sha256
```

Then evaluate the adapter on held-out cases:

```bash
python tools/eval_rlvr_adapter.py \
  --mode real \
  --model zai-org/glm-4-9b-chat-hf \
  --adapter training/rlvr/checkpoints/sophia-rlvr-v1 \
  --out agi-proof/benchmark-results/rlvr.adapter-eval.real.json
```

Do **not** close the RLVR capability claim until repeated held-out evals clear the
pre-registered no-overclaim gate.

## Hidden eval lane

Generate or collect responses privately, then aggregate with:

```bash
python tools/run_hidden_eval_full.py \
  --pack private/hidden-evals/PACK.json \
  --mode raw=private/hidden-evals/raw.responses.json \
  --mode raw_tools=private/hidden-evals/raw_tools.responses.json \
  --mode rag_only=private/hidden-evals/rag.responses.json \
  --mode gate_only=private/hidden-evals/gate.responses.json \
  --mode sophia_full=private/hidden-evals/sophia_full.responses.json \
  --out agi-proof/hidden-reviewer-packs/results/full-aggregate.json \
  --manual-review-out agi-proof/hidden-reviewer-packs/results/manual-review.md
```

Publish only aggregates and manual-review conclusions, not hidden prompts.

## Skill Forge promotion lane

```bash
python tools/sophia_skill_forge.py SPEC.json --register-smoke
```

Promoted skills are indexed in:

```text
skills/registry/forge_index.json
```

Gateway should expose only accepted registry entries by default. Rejected entries
stay useful as failure evidence.

## Long-horizon lane

For each timed autonomy run, store:

```text
agi-proof/long-horizon-runs/YYYY-MM-DD-<duration>/
  task.md
  agent_log.jsonl
  tool_calls.jsonl
  interventions.jsonl
  diff.patch
  test_results.txt
  postmortem.md
```

Pass condition is not "the agent said it succeeded". Pass condition is an
external verifier: tests, eval suite, no protected-knowledge regression, and low
human intervention.

## Infrastructure upgrades recommended

1. **Dedicated GPU artifact bucket**: store checkpoint tarballs outside ephemeral
   Pods immediately; treat reports without checkpoints as incomplete.
2. **CI matrix**: run local lane on Python 3.10/3.11/3.12; keep Python 3.9 only if
   repo syntax remains compatible.
3. **Secrets discipline**: use `/private/tmp` or secret manager files; never pass
   long-lived API keys into Pod env unless a self-delete watchdog is essential.
4. **Immutable eval manifests**: hash every hidden pack, RLVR split, source pack,
   and skill eval suite.
5. **Registry-driven gateway**: gateway loads accepted skill/tool/verifier records
   with risk, verifier, eval evidence, and retirement status.
6. **Result promotion bot**: a script should update RESULTS.md only when a JSON
   artifact has `validated=true` or an explicit `claimStatus` boundary.
7. **Observability**: store Langfuse/OpenTelemetry traces or JSONL equivalents for
   every agent action: prompt, tool call, verifier, repair, abstention, cost.

## What blocks stronger AGI language

- hidden third-party pack not cleared;
- long-horizon autonomy logs not complete;
- distribution-shift pack too small;
- RLVR adapter held-out improvement not gated;
- external benchmarks not run;
- independent clean-clone replication not complete.

Until those close, use: **AGI-candidate verifier-gated epistemic agent framework**.
