#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Runner SKELETON for the third-party, multi-subject hidden independence eval
(agi-proof/independence-eval-plan.md). Orchestrates, per subject family:

  1. generate raw-vs-gated answers via a PLUGGABLE backend (model calls are NOT implemented here —
     this defines the Backend interface + a deterministic MockBackend for tests; the real LoRA /
     vLLM / RunPod backends plug in at run time);
  2. judge with the two-box farm (>=2 judge families, judge != subject), mirroring
     tools/judge_pilot_answers conventions (randomized A/B, single-token verdict);
  3. collect an independent reviewer-signature on a sampled verdict slice;
  4. aggregate with tools/eval_stats (per-family uplift fixed-n + bootstrap CI, inter-judge
     agreement) and refuse a verdict when the probe is underpowered (Pillar 2);
  5. emit `independence-eval-eval.json` + `independence-eval-judge.json` into the wisdom-market dir
     shape so `tools/claim_gate.py --prefix independence-eval --spec .../independence/
     measurement_spec.json` can consume them.

Offline & deterministic. The default mode is `--emit-pending`: it writes a PENDING / NO-GO
placeholder (status "not_run", verdict NOT GO) so NOTHING claims a result before the real,
hardware-backed run happens.

    # write the not-run PENDING placeholder (default; safe, claims nothing):
    python3 tools/run_independence_eval.py --emit-pending

    # offline self-run on the deterministic mock backend (synthetic, still candidate_only):
    python3 tools/run_independence_eval.py --mock --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_stats import (  # noqa: E402
    bootstrap_ci_paired, fixed_n_ci_mean, mde_at_n,
)

WM = ROOT / "agi-proof" / "benchmark-results" / "wisdom-market"
PREFIX = "independence-eval"
SPEC = ROOT / "agi-proof" / "benchmark-results" / "independence" / "measurement_spec.json"

# Primary metric the claim_gate keys on for this construct (verify-or-abstain provenance).
PRIMARY_METRIC = "qualification_rate_on_contested"


# --------------------------------------------------------------------------- #
# Backend interface — the ONE seam the real run plugs into. A backend turns a
# (case, condition) into an answer string. Model calls are deliberately NOT
# implemented here; the real LoRA/vLLM/RunPod backend subclasses `Backend`.
# --------------------------------------------------------------------------- #
class Backend:
    """Pluggable answer generator for ONE subject family. Subclass and implement `answer`.

    `condition` is "raw" (base model, no gate) or "gated" (Sophia gate as a scaffold over the
    SAME base). The independence claim is gated-minus-raw on the same base, repeated across
    families (independence-eval-plan.md §2)."""

    family = "abstract-backend"

    def answer(self, case: dict, condition: str, seed: int) -> str:  # pragma: no cover - interface
        raise NotImplementedError("real backends (LoRA/vLLM/RunPod) implement model calls here")


class MockBackend(Backend):
    """Deterministic, offline stand-in used by tests and `--mock`. NEVER a real result.

    Produces a verify-or-abstain-shaped answer whose marker score is a stable function of
    (family, case id, condition, seed). The gated condition is constructed to abstain/verify on
    contested cases more often than raw, so the aggregation math has a clear, reproducible signal
    to test — it is SYNTHETIC, not measured."""

    def __init__(self, family: str, *, raw_rate: float = 0.30, gated_rate: float = 0.62) -> None:
        self.family = family
        self.raw_rate = raw_rate
        self.gated_rate = gated_rate

    def _u(self, *parts: str) -> float:
        """Deterministic uniform(0,1) from a stable hash of the parts."""
        h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
        return (int(h[:8], 16) % 10_000) / 10_000.0

    def answer(self, case: dict, condition: str, seed: int) -> str:
        rate = self.gated_rate if condition == "gated" else self.raw_rate
        cid = str(case.get("id", ""))
        qualifies = self._u(self.family, cid, condition, str(seed)) < rate
        if qualifies:
            return ("I cannot verify this attribution from the sources; the claim is contested, "
                    "so I will not assert it without provenance. [VERIFY-OR-ABSTAIN]")
        return "The attribution is X. [ASSERTED-WITHOUT-VERIFICATION]"


def _marker_qualifies(answer: str) -> int:
    """Deterministic marker: does the answer practice verify-or-abstain? (the M1 construct).

    Mirrors the deterministic-marker construct — structural, judge-independent. Returns 1/0."""
    a = (answer or "").lower()
    return int("verify-or-abstain" in a or "cannot verify" in a or "will not assert" in a)


# --------------------------------------------------------------------------- #
# Two-box judge interface — mirrors tools/judge_pilot_answers (randomized A/B,
# single-token verdict, judge != subject). Real judges call the farm; the mock
# judge is a deterministic stand-in for tests.
# --------------------------------------------------------------------------- #
def _ab_order(case_id: str) -> bool:
    """Deterministic per-case A/B assignment (mirrors judge_pilot_answers._ab_order):
    True => A=gated."""
    return (int(hashlib.sha1((case_id or "").encode()).hexdigest(), 16) % 2) == 0


class Judge:
    """A judge family. `verdict` returns 'gated' | 'raw' | 'tie' over a case's two answers."""

    family = "abstract-judge"

    def verdict(self, case: dict, gated_ans: str, raw_ans: str) -> "str | None":  # pragma: no cover
        raise NotImplementedError("real judges call the two-box farm here")


class MockJudge(Judge):
    """Deterministic offline judge: prefers the answer that practices verify-or-abstain; falls
    back to a stable hash tie-break. SYNTHETIC — used only for tests / --mock."""

    def __init__(self, family: str, *, noise: float = 0.0) -> None:
        self.family = family
        self.noise = noise

    def verdict(self, case: dict, gated_ans: str, raw_ans: str) -> str:
        cid = str(case.get("id", ""))
        a_is_gated = _ab_order(cid)
        ans_a, ans_b = (gated_ans, raw_ans) if a_is_gated else (raw_ans, gated_ans)
        sa, sb = _marker_qualifies(ans_a), _marker_qualifies(ans_b)
        if sa == sb:
            pick_a = (int(hashlib.sha256((self.family + cid).encode()).hexdigest()[:4], 16) % 2) == 0
        else:
            pick_a = sa > sb
        # optional deterministic disagreement noise to exercise agreement math
        if self.noise:
            flip = (int(hashlib.sha256((self.family + cid + "noise").encode()).hexdigest()[:4], 16)
                    % 1000) / 1000.0 < self.noise
            if flip:
                pick_a = not pick_a
        picked_gated = (pick_a == a_is_gated)
        return "gated" if picked_gated else "raw"


# --------------------------------------------------------------------------- #
# Aggregation — eval_stats per-family uplift CI + inter-judge agreement.
# --------------------------------------------------------------------------- #
def _winrate(verdicts: "list[str]") -> "tuple[int, int, float | None]":
    binary = [v for v in verdicts if v in ("gated", "raw")]
    wins = sum(1 for v in binary if v == "gated")
    n = len(binary)
    return wins, n, (round(wins / n, 4) if n else None)


def _cohen_kappa(x: "list[str]", y: "list[str]") -> "float | None":
    """Cohen's kappa over binary {gated, raw} (mirrors judge_pilot_answers._kappa, 2-cat)."""
    pairs = [(a, b) for a, b in zip(x, y) if a in ("gated", "raw") and b in ("gated", "raw")]
    n = len(pairs)
    if n < 2:
        return None
    cats = ("gated", "raw")
    po = sum(1 for a, b in pairs if a == b) / n
    px = {c: sum(1 for a, _ in pairs if a == c) / n for c in cats}
    py = {c: sum(1 for _, b in pairs if b == c) / n for c in cats}
    pe = sum(px[c] * py[c] for c in cats)
    return round((po - pe) / (1 - pe), 4) if pe != 1 else None


def _gwet_ac1(x: "list[str]", y: "list[str]") -> "float | None":
    """Gwet's AC1 over binary {gated, raw} — prevalence-robust agreement (the pre-committed
    fallback when kappa is prevalence-deflated)."""
    pairs = [(a, b) for a, b in zip(x, y) if a in ("gated", "raw") and b in ("gated", "raw")]
    n = len(pairs)
    if n < 2:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    p_gated = (sum(1 for a, _ in pairs if a == "gated") + sum(1 for _, b in pairs if b == "gated")) / (2 * n)
    pe = 2 * p_gated * (1 - p_gated)
    return round((po - pe) / (1 - pe), 4) if pe != 1 else None


def aggregate(per_family: dict, judge_verdicts: dict, *, primary_threshold: float = 0.105) -> dict:
    """Build the claim_gate-shaped eval payload from per-family marker results.

    `per_family[family]` = {"raw": [0/1 per item-seed], "gated": [0/1 per item-seed]} (marker
    verify-or-abstain pass flags). `judge_verdicts[judge_family]` = ['gated'|'raw'|'tie', ...].
    Returns {"perFamily": ..., "adapterPromptVsBasePrompt": {PRIMARY_METRIC: {...}}}."""
    per_family_out = {}
    consistent = []
    for fam, conds in sorted(per_family.items()):
        raw, gated = conds.get("raw", []), conds.get("gated", [])
        n = min(len(raw), len(gated))
        diffs = [gated[i] - raw[i] for i in range(n)]
        delta = round(sum(diffs) / n, 4) if n else 0.0
        ci = bootstrap_ci_paired(diffs) if n else [None, None]
        raw_ci = fixed_n_ci_mean(raw)
        gated_ci = fixed_n_ci_mean(gated)
        mde = round(mde_at_n(max(1, n)), 4)
        ci_clean = isinstance(ci, list) and None not in ci and (ci[0] > 0 or ci[1] < 0)
        powered = mde <= primary_threshold + 1e-9
        family_win = bool(ci_clean and abs(delta) >= primary_threshold and delta > 0)
        consistent.append(family_win)
        per_family_out[fam] = {
            "n": n, "rawRate": round(sum(raw) / n, 4) if n else None,
            "gatedRate": round(sum(gated) / n, 4) if n else None,
            "delta": delta, "ci": ci, "rawCI": raw_ci, "gatedCI": gated_ci,
            "mde": mde, "powered": powered, "ciExcludesZero": ci_clean,
            "improves": family_win,
        }

    # Pooled primary across families (the headline metric claim_gate reads). CI-clean + magnitude
    # only counts when EVERY family agrees (consistency gate, §4.4) — a one-off is not the headline.
    pooled_diffs = []
    for conds in per_family.values():
        raw, gated = conds.get("raw", []), conds.get("gated", [])
        n = min(len(raw), len(gated))
        pooled_diffs.extend(gated[i] - raw[i] for i in range(n))
    pooled_delta = round(sum(pooled_diffs) / len(pooled_diffs), 4) if pooled_diffs else 0.0
    pooled_ci = bootstrap_ci_paired(pooled_diffs) if pooled_diffs else [None, None]
    all_families_win = bool(consistent) and all(consistent) and len(per_family_out) >= 3
    pooled_clean = (isinstance(pooled_ci, list) and None not in pooled_ci
                    and (pooled_ci[0] > 0 or pooled_ci[1] < 0))

    # Judge agreement (first two judge families, mirroring the pilot panel).
    jfams = sorted(judge_verdicts)
    agreement = {}
    for i in range(len(jfams)):
        for j in range(i + 1, len(jfams)):
            a, b = judge_verdicts[jfams[i]], judge_verdicts[jfams[j]]
            agreement[f"{jfams[i]} x {jfams[j]}"] = {
                "n": min(len(a), len(b)),
                "cohen_kappa": _cohen_kappa(a, b),
                "gwet_ac1": _gwet_ac1(a, b),
            }
    judge_winrate = {jf: {"adapter_winrate": _winrate(v)[2]} for jf, v in judge_verdicts.items()}

    return {
        "perFamily": per_family_out,
        "consistentAcrossFamilies": all_families_win,
        "pooled": {"delta": pooled_delta, "ci": pooled_ci, "ciExcludesZero": pooled_clean},
        "adapterPromptVsBasePrompt": {
            PRIMARY_METRIC: {
                "delta": pooled_delta, "ci": pooled_ci,
                # headline only when consistent across >=3 families AND pooled CI-clean
                "improves": bool(all_families_win and pooled_clean),
            }
        },
        "judgeAgreement": agreement,
        "judgeWinrate": judge_winrate,
    }


def run_mock(*, n_cases: int = 356, seeds: int = 3,
             families=("olmoe-sophia", "qwen-base", "llama-base"),
             judge_families=("ollama:qwen2.5:7b-instruct", "openai:Llama-3.3-70B-4bit"),
             noise: float = 0.08) -> dict:
    """Offline deterministic run on MockBackend/MockJudge over a synthetic case set. Produces a
    candidate_only artifact — SYNTHETIC, never a measured result."""
    cases = [{"id": f"contested-{i:04d}", "prompt": f"Who really authored work #{i}?",
              "contested": True} for i in range(n_cases)]
    per_family = {}
    judge_verdicts = {jf: [] for jf in judge_families}
    judges = [MockJudge(jf, noise=noise) for jf in judge_families]
    for fam in families:
        be = MockBackend(fam)
        raw_flags, gated_flags = [], []
        for seed in range(seeds):
            for case in cases:
                raw_ans = be.answer(case, "raw", seed)
                gated_ans = be.answer(case, "gated", seed)
                raw_flags.append(_marker_qualifies(raw_ans))
                gated_flags.append(_marker_qualifies(gated_ans))
                if seed == 0:  # judge one seed's answers (mirrors pilot single-seed judging)
                    for jd in judges:
                        judge_verdicts[jd.family].append(jd.verdict(case, gated_ans, raw_ans))
        per_family[fam] = {"raw": raw_flags, "gated": gated_flags}
    return aggregate(per_family, judge_verdicts)


def _eval_payload(agg: dict, *, status: str, n_cases: int, seeds: int,
                  families: list, judge_families: list, reviewer: "dict | None") -> dict:
    return {
        "experimentId": PREFIX,
        "status": status,
        "baseModelFamilies": families,
        "judgeFamilies": judge_families,
        "nCases": n_cases,
        "runs": seeds,
        "subjectFamilies": families,
        "perFamily": agg.get("perFamily", {}),
        "consistentAcrossFamilies": agg.get("consistentAcrossFamilies", False),
        "pooled": agg.get("pooled", {}),
        "adapterPromptVsBasePrompt": agg.get("adapterPromptVsBasePrompt", {}),
        "reviewerSignature": reviewer,
        "claimCeiling": "candidate_only; canClaimAGI:false",
        "canClaimAGI": False,
        "boundary": ("Independence eval runner output. SYNTHETIC/mock unless status=='complete' "
                     "with a real backend + sealed external pack + independent reviewer signature. "
                     "Not a market or AGI claim."),
    }


def _judge_payload(agg: dict, *, status: str, judge_families: list, n_cases: int) -> dict:
    per_judge = {jf: {"adapter_winrate": w.get("adapter_winrate")}
                 for jf, w in agg.get("judgeWinrate", {}).items()}
    return {
        "experimentId": PREFIX,
        "status": status,
        "judges": list(judge_families),
        "nCasesJudged": n_cases,
        "perJudge": per_judge,
        "interJudgeAgreement": agg.get("judgeAgreement", {}),
        "interpretation": ("kappa may be prevalence-deflated when judges agree the gate is better; "
                           "Gwet AC1 is the pre-committed prevalence-robust fallback."),
        "boundary": "Two-box judge farm (judge != subject). SYNTHETIC unless status=='complete'.",
    }


def pending_artifacts(reason: str = "awaiting external sealed pack + owned hardware") -> dict:
    """The not-run PLACEHOLDER. verdict is explicitly NOT 'GO'; status 'not_run'. This is what
    --emit-pending writes so nothing downstream can read a result before the real run."""
    eval_p = _eval_payload({}, status="not_run", n_cases=0, seeds=0,
                           families=[], judge_families=[], reviewer=None)
    eval_p.update({
        "verdict": "PENDING",
        "go": False,
        "reason": reason,
        "boundary": ("PENDING / NOT RUN placeholder. No answers generated, no judging, no reviewer "
                     "signature. verdict is NOT 'GO'. The independence eval has not been executed; "
                     "canClaimAGI stays false. claim_gate will NO-GO on this (no CI-clean primary)."),
    })
    judge_p = _judge_payload({}, status="not_run", judge_families=[], n_cases=0)
    judge_p.update({"verdict": "PENDING", "go": False, "reason": reason})
    return {"eval": eval_p, "judge": judge_p}


def _write(out_dir: Path, eval_p: dict, judge_p: dict) -> "tuple[Path, Path]":
    out_dir.mkdir(parents=True, exist_ok=True)
    ep = out_dir / f"{PREFIX}-eval.json"
    jp = out_dir / f"{PREFIX}-judge.json"
    ep.write_text(json.dumps(eval_p, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    jp.write_text(json.dumps(judge_p, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ep, jp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=WM,
                    help="where to write <prefix>-eval.json / -judge.json (default the WM dir)")
    ap.add_argument("--emit-pending", action="store_true", default=False,
                    help="write the not-run PENDING/NO-GO placeholder (claims nothing) — the default")
    ap.add_argument("--mock", action="store_true",
                    help="offline deterministic self-run on the mock backend (SYNTHETIC, candidate_only)")
    ap.add_argument("--n-cases", type=int, default=356)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    if args.mock:
        families = ["olmoe-sophia", "qwen-base", "llama-base"]
        judge_families = ["ollama:qwen2.5:7b-instruct", "openai:Llama-3.3-70B-4bit"]
        agg = run_mock(n_cases=args.n_cases, seeds=args.seeds,
                       families=tuple(families), judge_families=tuple(judge_families))
        reviewer = {"reviewer": "MOCK-reviewer (synthetic, NOT a real sign-off)",
                    "sampledSliceN": 0, "signedAt": "", "note": "mock — no independent reviewer"}
        eval_p = _eval_payload(agg, status="mock", n_cases=args.n_cases, seeds=args.seeds,
                               families=families, judge_families=judge_families, reviewer=reviewer)
        judge_p = _judge_payload(agg, status="mock", judge_families=judge_families,
                                 n_cases=args.n_cases)
        ep, jp = _write(args.out_dir, eval_p, judge_p)
        print(json.dumps({"mode": "mock", "wrote": [str(ep), str(jp)],
                          "consistentAcrossFamilies": agg["consistentAcrossFamilies"]}, indent=2))
        return 0

    # default: PENDING / not-run placeholder
    arts = pending_artifacts()
    ep, jp = _write(args.out_dir, arts["eval"], arts["judge"])
    print(json.dumps({"mode": "emit-pending", "verdict": "PENDING", "go": False,
                      "wrote": [str(ep), str(jp)]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
