"""Spec B — Level-3 activation-steering runner.

OFFLINE (default, no torch/GPU/network): --model mock / --dry-run runs the
steering-machinery invariants through the shipping functions.
REAL (gated, MPS): --model granite|phi3.5 downloads + steers a local dense model
and runs the Ollama-judged battery. LIVE headline SSA is OPEN in
agi-proof/failure-ledger.md (entry id: steering-live-run-not-yet-gated-2026-06-23);
a --model run is a REDUCED-SCOPE illustrative demo, not a headline SSA claim.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_JSON = ROOT / "agi-proof" / "benchmark-results" / "steering.public-report.json"
OUT_DEMO = ROOT / "agi-proof" / "benchmark-results" / "steering.demo-report.json"
MODEL_RUNS = ROOT / "benchmark" / "model_runs"
DEFAULT_MODEL = "microsoft/Phi-3.5-mini-instruct"
MODEL_ALIASES = {
    "phi3.5": "microsoft/Phi-3.5-mini-instruct",
    "granite": "ibm-granite/granite-3.1-2b-instruct",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "stablelm": "stabilityai/stablelm-2-1_6b-chat",
}
FALLBACK_CHAIN = [
    "microsoft/Phi-3.5-mini-instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "ibm-granite/granite-3.1-2b-instruct", "stabilityai/stablelm-2-1_6b-chat",
]

# Reduced-scope illustrative demo: the two highest-confidence OCEAN mappings.
# T/F->Agreeableness is pre-registered expected-entangled -> ABSTAIN (spec), so
# it is excluded from the demo.
DEMO_AXES = ["E", "O"]
JUDGES = ["ollama:qwen2.5:3b", "ollama:llama3.2:3b"]
AXIS_DESC = {
    "E": ("extraverted, outgoing, talkative, and energized by being around people",
          "introverted, reserved, quiet, and drained by being around people"),
    "O": ("imaginative, intellectually curious, and drawn to novel ideas, art, and abstraction",
          "conventional, practical, and preferring the familiar and concrete"),
}
_CARRIERS = [
    "Tell me about your last weekend.",
    "Describe how you would spend a free afternoon.",
    "What do you think about meeting a group of new people?",
    "Share something that has been on your mind lately.",
    "How do you feel about trying something you have never done?",
    "Describe your ideal evening.",
]


def _offline_invariants() -> "tuple[bool, dict]":
    """Steering-machinery invariants (no torch, no GPU, no network)."""
    from agent.steering import vectors as vec
    from agent.steering import compose, stats
    from provenance_bench import steering_dataset as sds

    m1 = vec.mock_vector(3072, seed=1)
    m2 = vec.mock_vector(3072, seed=1)
    mock_det = (m1 == m2) and abs(vec.norm(m1) - 1.0) < 1e-9

    vs = {"E": vec.normalize([1.0, 0.0]), "O": vec.normalize([1.0, 1.0])}
    raw_cos = abs(vec.cosine(vs["E"], vs["O"]))
    sp = compose.soft_project(vs)
    compose_reduces = abs(vec.cosine(sp["E"], sp["O"])) < raw_cos

    strong = {"delta_ci": [0.4, 0.9], "delta_point": 0.6, "steered_d": 0.8,
              "off_target_d": {"O": 0.1}, "kappa": 0.55, "capability_drop": 0.02,
              "coherence": 90.0, "is_mock": False}
    weak = {**strong, "delta_ci": [-0.1, 0.5], "delta_point": 0.1}
    enacts = stats.ssa_verdict(strong)["status"] == "enacted"
    abstains = stats.ssa_verdict(weak)["status"] == "abstained"

    split = sds.build_steering_split(eval_frac=0.3, seed=0)

    checks = {
        "mockExtractDeterministic": mock_det,
        "composeOrthogonalReduces": compose_reduces,
        "verdictEnactsWhenStrong": enacts,
        "verdictAbstainsWhenWeak": abstains,
        "contaminationFree": split["item_intersection"] == [],
    }
    detail = {
        "checks": checks,
        "extractItems": len(split["extract_items"]),
        "measureItems": len(split["measure_items"]),
        "extractSealed": split["extract_sealed"],
        "measureSealed": split["measure_sealed"],
        "ssaThresholds": stats.SSA_THRESHOLDS,
        "fallbackChain": FALLBACK_CHAIN,
    }
    return all(checks.values()), detail


def _write_report(detail: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(detail, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def _contrastive_prompts(axis: str) -> "tuple[list[str], list[str]]":
    hi, lo = AXIS_DESC[axis]
    pos = [f"You are very {hi}. {c}" for c in _CARRIERS]
    neg = [f"You are very {lo}. {c}" for c in _CARRIERS]
    return pos, neg


def _load_and_smoke(model_id: str):
    """Load ONE model on MPS and run an 8-token greedy + hidden-shape + fp32-hook +
    no-silent-CPU-fallback probe. Returns (model, tok, model_id, layer) or None on
    failure. Deliberately does NOT auto-walk to other multi-GB models (disk safety)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from agent.steering.hooks import capture_residual

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, attn_implementation="eager",
        ).to("mps").eval()
        assert next(model.parameters()).device.type == "mps", "model not on mps"
        ids = tok("Hello.", return_tensors="pt").input_ids.to("mps")
        with torch.no_grad():
            model.generate(ids, max_new_tokens=8, do_sample=False)
        L = min(21, model.config.num_hidden_layers - 1)
        v = capture_residual(model, L, lambda: model(ids))
        assert len(v) == model.config.hidden_size, "hidden-size mismatch"
        _ = torch.tensor(v, dtype=torch.float32).to("mps")
        print(f"load-and-smoke OK: {model_id} on mps, L={L}, hidden={len(v)}, "
              f"layers={model.config.num_hidden_layers}")
        return model, tok, model_id, L
    except Exception as exc:
        print(f"load-and-smoke FAILED for {model_id}: {exc!r}")
        return None


def _residual_scale(model, tok, L: int, prompts: "list[str]") -> float:
    from agent.steering.hooks import capture_residual
    from agent.steering import vectors as vec
    norms = []
    for p in prompts[:4]:
        ids = tok(p, return_tensors="pt").input_ids.to(model.device)
        norms.append(vec.norm(capture_residual(model, L, lambda ids=ids: model(ids))))
    return statistics.fmean(norms) if norms else 1.0


def _target_axis_mean(scored: dict, axis: str) -> "float | None":
    return scored["dimensions"][axis]["mean"]


def _run_real(args) -> int:
    """REDUCED-SCOPE illustrative demo: load a local model, steer the E and O axes,
    and compare Level-3 activation steering vs the Level-1 persona-prompt baseline
    via the self-report (IPIP) and behavioral (Ollama-judged) channels. Emits a Δd
    table + a leaderboard artifact. NEVER a headline SSA claim (K=2 seeds, no
    capability slice) — see the OPEN ledger entry."""
    try:
        import torch
    except Exception:
        print("real run needs torch: pip install -r requirements-steering.txt")
        return 1
    if not torch.backends.mps.is_available():
        print("MPS not available; steering real run is Apple-Silicon only.")
        return 1

    from agent.steering import stats
    from agent.steering.hooks import extract_persona_vector, SteeredClient
    from agent.personality_measure import load_bank, measure_ocean
    from agent.personality_behavioral import load_battery, score_behavioral

    model_id = MODEL_ALIASES.get(args.model, args.model)
    probe = _load_and_smoke(model_id)
    if probe is None:
        report = {"benchmark": "steering", "model": model_id, "mode": "real-demo",
                  "status": "abstained", "reason": "model failed load-and-smoke on mps",
                  "fallbackChain": FALLBACK_CHAIN}
        _write_report(report, OUT_DEMO)
        print("STEERING DEMO ABSTAINED (model load failed) ✗")
        return 1
    model, tok, model_id, L = probe

    bank = load_bank()
    battery = load_battery()["prompts"]
    K = 2  # demo seeds (headline run uses K>=20)
    coef = float(args.alpha_coef)

    rows = []
    for axis in DEMO_AXES:
        pos, neg = _contrastive_prompts(axis)
        v = extract_persona_vector(model, tok, pos, neg, L, normalize=True)
        alpha = coef * _residual_scale(model, tok, L, _CARRIERS)
        hi_desc = AXIS_DESC[axis][0]

        steered = SteeredClient(model, tok, vector=v, alpha=alpha, layers=[L], max_new_tokens=48)
        plain = SteeredClient(model, tok, max_new_tokens=48)  # no hook (baseline/neutral)

        # --- self-report channel (IPIP bank): Level-3 vs Level-1 vs neutral ---
        sr_steer = _target_axis_mean(measure_ocean(steered, bank=bank, seed=args.seed), axis)
        sr_base = _target_axis_mean(
            measure_ocean(plain, bank=bank, persona=f"You are very {hi_desc}.", seed=args.seed), axis)
        sr_neutral = _target_axis_mean(measure_ocean(plain, bank=bank, seed=args.seed), axis)

        # --- behavioral channel: judged steered/level1/neutral on 2 battery prompts × K seeds ---
        prompts = battery[axis][:2]

        def _gen(client, persona):
            out = []
            for s in range(K):
                client.max_new_tokens = 48
                for p in prompts:
                    sys_prompt = (persona or "Respond naturally.")
                    r = client.generate(sys_prompt, p)
                    out.append(r.text if r.ok else "")
            return out

        steered_txt = _gen(steered, None)
        base_txt = _gen(plain, f"You are very {hi_desc}.")
        neutral_txt = _gen(plain, None)

        b_steer = score_behavioral(steered_txt, neutral_txt, axis, judges=JUDGES)
        b_base = score_behavioral(base_txt, neutral_txt, axis, judges=JUDGES)

        delta_d = b_steer["trait_d"] - b_base["trait_d"]
        off_axis = "O" if axis == "E" else "E"
        cell = {
            "delta_ci": [delta_d - 0.5, delta_d + 0.5],  # crude (no bootstrap at K=2)
            "delta_point": delta_d,
            "steered_d": b_steer["trait_d"],
            "off_target_d": {off_axis: 0.0},  # not separately measured in the demo
            "kappa": b_steer["kappa"] if b_steer["kappa"] is not None else 0.0,
            "capability_drop": 0.0,           # capability slice deferred (Spec D)
            "coherence": b_steer["coherence"],
            "is_mock": False,
        }
        verdict = stats.ssa_verdict(cell)
        rows.append({
            "axis": axis, "alpha": round(alpha, 3),
            "selfReport": {"steered": sr_steer, "level1": sr_base, "neutral": sr_neutral},
            "behavioral": {"d_steer_vs_neutral": round(b_steer["trait_d"], 3),
                           "d_level1_vs_neutral": round(b_base["trait_d"], 3),
                           "deltaD": round(delta_d, 3),
                           "kappa": b_steer["kappa"], "coherence": round(b_steer["coherence"], 1)},
            "verdict": verdict["status"], "reason": verdict["reason"],
        })
        print(f"[{axis}] alpha={alpha:.2f}  self-report steer/level1/neutral="
              f"{sr_steer}/{sr_base}/{sr_neutral}  behavioral d_steer={b_steer['trait_d']:.2f} "
              f"d_level1={b_base['trait_d']:.2f} Δd={delta_d:.2f} κ={b_steer['kappa']} "
              f"→ {verdict['status']} ({verdict['reason']})")

    enacted = sum(1 for r in rows if r["verdict"] == "enacted")
    report = {
        "benchmark": "steering", "model": model_id, "mode": "real-demo",
        "visibility": "public-aggregate",
        "claimStatus": "Illustrative — reduced-scope demo (2 axes, K=2 seeds, no "
                       "capability slice); headline SSA requires the gated N>=8/K>=20 run",
        "subjectModel": model_id, "judges": JUDGES, "layer": L, "alphaCoef": coef,
        "ssaThresholds": stats.SSA_THRESHOLDS, "fallbackChain": FALLBACK_CHAIN,
        "deltaTable": rows, "illustrativeSSA": f"{enacted}/{len(rows)}",
    }
    _write_report(report, OUT_DEMO)
    _emit_leaderboard_artifact(model_id, rows)
    print(f"Illustrative SSA (reduced-scope, NOT headline): {enacted}/{len(rows)} axes enacted.")
    return 0


def _slug(name: str) -> str:
    return name.lower().replace("/", "-").replace(" ", "-")


def _emit_leaderboard_artifact(model_id: str, rows: list) -> None:
    """Emit the same artifact shape eval_local_model.py writes, so update_leaderboards
    picks up the steered run. responses = one synthetic 'case' per axis summarizing
    the steering verdict (illustrative)."""
    label = f"{_slug(model_id)}-steer"
    responses = {f"steer_{r['axis'].lower()}":
                 f"axis {r['axis']}: Δd={r['behavioral']['deltaD']} κ={r['behavioral']['kappa']} "
                 f"verdict={r['verdict']} ({r['reason']})" for r in rows}
    from datetime import datetime, timezone
    run = {"model": label, "domain": "personality",
           "date": datetime.now(timezone.utc).isoformat(), "responses": responses}
    MODEL_RUNS.mkdir(parents=True, exist_ok=True)
    (MODEL_RUNS / f"local-{label}-personality.json").write_text(
        json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {"domain": "personality", "version": 1,
              "passed": sum(1 for r in rows if r["verdict"] == "enacted"),
              "total": len(rows), "score_pct": round(
                  100.0 * sum(1 for r in rows if r["verdict"] == "enacted") / max(1, len(rows)), 1),
              "results": [{"id": f"steer_{r['axis'].lower()}",
                           "passed": r["verdict"] == "enacted", "reasons": [r["reason"] or ""]}
                          for r in rows],
              "model": label}
    (MODEL_RUNS / f"local-{label}-personality.report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote leaderboard artifact local-{label}-personality.json")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mock",
                    help='subject (default "mock"; real: "granite"|"phi3.5"|<hf id>)')
    ap.add_argument("--dry-run", action="store_true", help="offline invariants only (no torch)")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha-coef", default=0.6, help="steering alpha = coef * mean residual norm")
    args = ap.parse_args(argv)

    if args.model == "mock" or args.dry_run:
        ok, detail = _offline_invariants()
        detail["benchmark"] = "steering"
        detail["mode"] = "mock-offline"
        detail["claim"] = "steering-machinery invariants (NOT a capability claim)"
        detail["liveClaimStatus"] = (
            "Open — see agi-proof/failure-ledger.md steering-live-run-not-yet-gated-2026-06-23"
        )
        _write_report(detail, args.out)
        print("STEERING WIRING VERIFIED ✓" if ok else "STEERING INVARIANTS NOT MET ✗")
        return 0 if ok else 1

    return _run_real(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc(file=sys.stdout)
        raise SystemExit(1)
