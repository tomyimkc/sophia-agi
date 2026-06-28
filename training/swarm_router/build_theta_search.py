#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build θ_search — the first V3 dual-use adapter (recommended first GPU step).

Per docs/11-Platform/Swarm-Variants-V3-V4-Spec.md (V3 "minimal first step"): train ONE
real ``search`` specialist adapter and show it works at both altitudes —
  * as an in-weights MoE expert (``DualUseAdapter.as_expert_fn`` → ``MoERouter.forward``), and
  * as a spawnable agent seat (``DualUseAdapter.as_team`` → ``agent.subagent``),
then gate it through ``agent.continual_plasticity`` before binding it into the catalogue.

Two paths, same discipline as the rest of the repo:

  --offline (default)  Build the deterministic DualUseAdapter REFERENCE (gain=0 identity
                       or a supplied gain), run the promotion gate on a supplied
                       before/after, and emit the decision. No GPU, fully reproducible —
                       proves the seam end to end before renting a box.

  --train              GUARDED real LoRA training via tools/train_lora.py on the search
                       corpus, then wrap → gate → (on promote) write the adapter binding
                       so TEAMS["search"].adapter_id points at the trained θ_search.
                       Loud-fails with install guidance if torch/peft are absent.

Rent + auto-terminate a GPU with tools/runpod_rlvr.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_plasticity import EvalMetric  # noqa: E402
from agent.dual_use_adapter import DualUseAdapter  # noqa: E402

BINDING_PATH = ROOT / "training" / "swarm_router" / "theta_search.binding.json"


def offline(args) -> int:
    """Build + gate the reference adapter with no GPU."""
    adapter = DualUseAdapter(id=args.adapter_id, team_name="search", gain=args.gain)

    # Demonstrate both altitudes exist off one artifact.
    team = adapter.as_team()
    expert_fn = adapter.as_expert_fn()
    sample = expert_fn([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])

    # Gate it exactly like any weight update (before/after are supplied; in --train these
    # come from the real search-recall eval pre/post adapter).
    decision = adapter.gate(
        target_suite="search_recall",
        before=args.before,
        after=args.after,
        verifier_artifacts=tuple(args.artifacts),
        protected=(EvalMetric("attribution_traps", args.protected_before, args.protected_after, protected=True),),
    )
    report = {
        "adapterId": adapter.id,
        "team": team.name,
        "boundAdapterId": team.adapter_id,
        "altitudes": {
            "inWeightsExpert": "DualUseAdapter.as_expert_fn → MoERouter.forward",
            "asAgentSeat": "DualUseAdapter.as_team → agent.subagent",
        },
        "expertSampleOut": sample,
        "promotion": decision.to_dict(),
    }
    print(json.dumps(report, indent=2))
    if decision.verdict == "promote" and args.write_binding:
        BINDING_PATH.write_text(json.dumps(
            {"team": "search", "adapterId": adapter.id, "gain": adapter.gain,
             "promotion": decision.to_dict()}, indent=2) + "\n")
        print(f"\nWrote binding → {BINDING_PATH} (load it to set TEAMS['search'].adapter_id)")
    print(f"\nVERDICT: {decision.verdict} ({'; '.join(decision.reasons)})")
    return 0 if decision.verdict == "promote" else 1


def train(args) -> int:
    """Guarded real LoRA training, then wrap → gate → bind."""
    try:
        import torch  # noqa: F401
        import peft  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(
            f"--train needs torch + peft ({type(exc).__name__}: {exc}).\n"
            "  pip install -r requirements-lora.txt\n"
            "Prove the seam with NO GPU first:  python training/swarm_router/build_theta_search.py --offline\n"
            "To rent + auto-terminate a GPU, see tools/runpod_rlvr.py.",
            flush=True,
        )
        return 2

    # The real run delegates to the repo's existing LoRA trainer on the search corpus,
    # then evaluates search-recall pre/post and feeds the deltas into adapter.gate().
    import subprocess

    out_dir = ROOT / "training" / "swarm_router" / "adapter_search"
    cmd = [
        sys.executable, str(ROOT / "tools" / "train_lora.py"),
        "--data", str(args.data), "--output", str(out_dir),
        "--base-model", args.base_model,
    ]
    print("Running:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"LoRA training exited {rc}", flush=True)
        return rc
    # TODO(live): run search-recall eval before/after on a held-out, decontaminated pack
    # and pass the measured scores here instead of the placeholders.
    print("Training done. Now evaluate search-recall pre/post and run the gate:",
          flush=True)
    print("  python training/swarm_router/build_theta_search.py --offline "
          "--gain <fit> --before <pre> --after <post> --write-binding", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true", help="guarded real LoRA training (needs torch+peft)")
    ap.add_argument("--offline", action="store_true", help="build + gate the reference adapter, no GPU (default)")
    ap.add_argument("--adapter-id", default="theta-search-v1")
    ap.add_argument("--gain", type=float, default=0.0, help="trained strength (0 = identity)")
    ap.add_argument("--before", type=float, default=0.60, help="search-recall before adapter")
    ap.add_argument("--after", type=float, default=0.71, help="search-recall after adapter")
    ap.add_argument("--protected-before", type=float, default=0.90)
    ap.add_argument("--protected-after", type=float, default=0.90)
    ap.add_argument("--artifacts", nargs="*", default=["recall_eval.json", "decontam.json"])
    ap.add_argument("--write-binding", action="store_true", help="on promote, write the team binding json")
    ap.add_argument("--data", type=Path, default=ROOT / "training" / "council" / "traces.jsonl")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    args = ap.parse_args()

    if args.train:
        return train(args)
    return offline(args)


if __name__ == "__main__":
    raise SystemExit(main())
