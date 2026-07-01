#!/usr/bin/env python3
"""
run_arc_agi_sophia.py — run ARC-AGI tasks through Sophia's fail-closed case
pipeline and write a no-overclaim public report (TODO item 11: external benchmark).

WHAT IT DOES
  Loads ARC-AGI task JSON (the official {"train":[{input,output}], "test":[{input}]}
  grid format) from a --tasks directory, formats each task into a prompt, runs it
  through Sophia's model adapter (lazy import; ABSTAIN if no backend), applies the
  fail-closed gate so an ungrounded/unsupported grid is an ABSTENTION rather than a
  guess, scores EXACT-GRID-MATCH against the held-out solution when provided, and
  writes an ARC public-report carrying the repo's honesty fields + responseHealth.

  ARC is an EXTERNAL, un-gameable benchmark: this converts scaffolding into a number
  no first-party pack can — subject to the same fail-closed discipline.

WHY EXACT-MATCH + ABSTENTION
  ARC is graded by exact output-grid reconstruction. Sophia's contribution is not a
  higher raw ARC score (it will likely be LOW) — it is an HONEST one: the abstention
  rate is reported alongside accuracy, so "solved", "wrong", and "declined" are three
  distinct outcomes. A high-abstention / low-wrong profile is itself the measured
  claim (fail-closed under novelty), not a failure to hide.

FAIL-CLOSED
  No backend -> "environment artifact, not a score" report, exit 0. Missing --tasks
  dir -> exit non-zero (cannot benchmark against data we don't have). Never vendors
  the dataset; never fabricates a solved grid.

HONEST BOUND
  ARC accuracy here measures raw novel-reasoning capability of the wired base model
  under a fail-closed gate; it is NOT evidence the gate improves ARC (that needs the
  raw-vs-gated ablation) and NOT an AGI claim. candidateOnly:true level3Evidence:false
  canClaimAGI:false.

USAGE
  # place official tasks at ./arc-data/  (NOT vendored here)
  python3 tools/run_arc_agi_sophia.py --tasks arc-data/evaluation \
      --adapter deepseek --out agi-proof/benchmark-results/arc-agi.public-report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

Grid = list[list[int]]


# ---------------------------------------------------------------- adapter
def load_generator(spec: str | None = None, *, allow_mock: bool = False) -> Callable[[str], str] | None:
    """generate(prompt)->text bound to the real agent/model.py adapter, or None.

    Uses resolve_config + default_client(spec).generate(system, user); guards the
    mock provider (auto-selected when no API key) so a keyless environment
    fail-closes to None instead of scoring against fabricated text.
    """
    try:
        from agent.model import default_client, resolve_config
    except Exception:
        return None
    try:
        cfg = resolve_config(spec)
    except Exception:
        return None
    if getattr(cfg, "kind", None) == "mock" and not allow_mock:
        return None
    try:
        client = default_client(spec)
    except Exception:
        return None

    def _gen(prompt: str) -> str:
        res = client.generate("", prompt)
        if not getattr(res, "ok", False):
            raise RuntimeError(f"backend failure: {getattr(res, 'error', None)}")
        return res.text

    return _gen


def load_gate() -> Callable[[str, str], bool] | None:
    """Return grounded_ok(answer, prompt)->bool from the repo gate, or None.

    A True verdict means the answer cleared the fail-closed epistemic gate; a
    False verdict (or a parse failure) is treated as an ABSTENTION downstream.
    """
    try:
        from agent.gate_reward import reward as gate_reward
    except Exception:
        return None
    return lambda answer, prompt: gate_reward(answer, question=prompt) >= 0.0


# ---------------------------------------------------------------- formatting
def grid_to_text(g: Grid) -> str:
    return "\n".join(" ".join(str(c) for c in row) for row in g)


def format_prompt(task: dict[str, Any]) -> str:
    parts = ["You are solving an ARC abstract-reasoning task. Infer the rule from the "
             "input->output examples and produce ONLY the output grid for the test input, "
             "as space-separated rows of integers. If you cannot determine the rule with "
             "confidence, reply exactly: I DON'T KNOW.\n"]
    for i, pair in enumerate(task.get("train", []), 1):
        parts.append(f"Example {i} input:\n{grid_to_text(pair['input'])}")
        parts.append(f"Example {i} output:\n{grid_to_text(pair['output'])}")
    test_in = task["test"][0]["input"]
    parts.append(f"Test input:\n{grid_to_text(test_in)}")
    parts.append("Test output:")
    return "\n\n".join(parts)


def parse_grid(text: str) -> Grid | None:
    """Extract the output grid, robust to prose preambles and code fences.

    Review Q-C(1): the old parser stopped at the FIRST non-grid line, so a leading
    prose sentence or a ``` fence aborted at line 0 and scored a present, correct
    grid as WRONG — a formatting mismatch masquerading as a reasoning failure. We
    instead: strip fences, then find the LONGEST maximal run of consecutive
    all-integer lines of equal width (the grid block), ignoring surrounding prose.
    """
    # drop code fences but keep their contents
    cleaned = "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("```"))

    best: Grid | None = None
    run: Grid = []

    def _flush(run: Grid, best: Grid | None) -> Grid | None:
        if run and len({len(r) for r in run}) == 1:
            if best is None or len(run) > len(best):
                return [r[:] for r in run]
        return best

    for line in cleaned.splitlines():
        toks = line.split()
        if toks and all(_is_int(t) for t in toks):
            run.append([int(t) for t in toks])
        else:
            best = _flush(run, best)
            run = []
    best = _flush(run, best)
    return best


def _is_int(tok: str) -> bool:
    try:
        int(tok)
        return True
    except ValueError:
        return False


ABSTAIN_MARKERS = ("i don't know", "i dont know", "cannot determine", "unsure")


def is_abstention(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in ABSTAIN_MARKERS)


# ---------------------------------------------------------------- decontam note
def decontam_ok(tasks_dir: Path, manifest: Path | None) -> str:
    """ARC is held-out by construction; we still assert the tasks dir is not the
    training manifest. Returns a status string for the report."""
    if manifest and manifest.exists():
        names = set(json.loads(manifest.read_text()).get("sources", []))
        if str(tasks_dir) in names:
            sys.exit(f"FAIL-CLOSED: {tasks_dir} appears in training manifest {manifest}")
        return "checked-against-manifest"
    return "arc-heldout-by-construction (no manifest provided)"


# ---------------------------------------------------------------- run
def score_task(task: dict[str, Any], gen, gate) -> dict[str, Any]:
    prompt = format_prompt(task)
    try:
        raw = gen(prompt)
    except Exception as e:
        return {"status": "backend_error", "error": f"{type(e).__name__}: {e}"}
    if is_abstention(raw):
        return {"status": "abstained"}
    grounded = gate(raw, prompt) if gate else True
    grid = parse_grid(raw)
    if grid is None or not grounded:
        return {"status": "abstained", "reason": "unparseable-or-ungrounded"}
    solution = task.get("test", [{}])[0].get("output")
    if solution is None:
        return {"status": "answered", "match": None}  # blind test set
    return {"status": "answered", "match": grid == solution}


def run(tasks_dir: Path, gen, gate, decontam: str) -> dict[str, Any]:
    files = sorted(tasks_dir.glob("*.json"))
    results, backend_failures = [], 0
    for f in files:
        task = json.loads(f.read_text())
        r = score_task(task, gen, gate)
        r["task"] = f.stem
        if r["status"] == "backend_error":
            backend_failures += 1
        results.append(r)

    answered = [r for r in results if r["status"] == "answered" and r.get("match") is not None]
    correct = sum(1 for r in answered if r["match"])
    abstained = sum(1 for r in results if r["status"] == "abstained")
    wrong = sum(1 for r in answered if not r["match"])
    n = len(results)
    return {
        "benchmark": "ARC-AGI",
        "taskCount": n,
        "responseHealth": {"backendFailureCount": backend_failures},
        "decontam": decontam,
        "accuracyExactMatch": (correct / len(answered)) if answered else None,
        "counts": {"correct": correct, "wrong": wrong, "abstained": abstained,
                   "scored": len(answered)},
        "abstentionRate": (abstained / n) if n else None,
        "perTask": results,
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
        "honestNote": "Raw base-model ARC under a fail-closed gate. Low accuracy expected; "
                      "the reported claim is the (correct, wrong, abstained) split, not a headline "
                      "score. Gate-vs-raw improvement requires the ablation arm.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def env_artifact(reason: str) -> dict[str, Any]:
    return {"environmentArtifact": True, "score": None, "reason": reason,
            "benchmark": "ARC-AGI", "candidateOnly": True, "level3Evidence": False,
            "canClaimAGI": False, "timestamp": datetime.now(timezone.utc).isoformat()}


# ---------------------------------------------------------------- two-arm ablation
def evaluate_raw(task: dict[str, Any], gen, gate) -> dict[str, Any]:
    """Generate ONCE and record the gate-agnostic facts to score BOTH arms.

    Review Q-C(3): a single headline number is uninformative without the raw-vs-gated
    ablation. Both arms score the SAME generation — the gate is a SELECTIVE classifier
    that can only turn an answer into an abstention, never change the grid. So gate-off
    and gate-on share identical predictions and differ only in coverage. Generating once
    (not twice) makes the comparison exact and halves cost.
    """
    prompt = format_prompt(task)
    try:
        raw = gen(prompt)
    except Exception as e:
        return {"status": "backend_error", "error": f"{type(e).__name__}: {e}"}
    grid = parse_grid(raw)
    solution = task.get("test", [{}])[0].get("output")
    return {
        "status": "ok",
        "rawAbstained": is_abstention(raw),
        "parsed": grid is not None,
        "match": (grid == solution) if (grid is not None and solution is not None) else None,
        "gatePass": bool(gate(raw, prompt)) if gate else True,
        "rawChars": len(raw),
    }


def _arm_answered(rec: dict[str, Any], *, use_gate: bool) -> bool:
    """A task is ANSWERED in this arm iff the model did not self-abstain, the grid
    parsed, and (gate-on only) the epistemic gate passed."""
    if rec["rawAbstained"] or not rec["parsed"]:
        return False
    return rec["gatePass"] if use_gate else True


def _aggregate_arm(records: list[dict[str, Any]], *, use_gate: bool, n_ok: int) -> dict[str, Any]:
    scored = [r for r in records if r["status"] == "ok" and _arm_answered(r, use_gate=use_gate)
              and r["match"] is not None]
    correct = sum(1 for r in scored if r["match"])
    wrong = len(scored) - correct
    abstained = n_ok - len(scored)
    return {
        "counts": {"correct": correct, "wrong": wrong, "abstained": abstained, "scored": len(scored)},
        "accuracyExactMatch": (correct / len(scored)) if scored else None,
        "coverage": (len(scored) / n_ok) if n_ok else None,
        "abstentionRate": (abstained / n_ok) if n_ok else None,
        "selectiveRisk": (wrong / len(scored)) if scored else None,
    }


def run_two_arms(tasks_dir: Path, gen, gate, decontam: str, *, limit: int | None = None,
                 spec: str | None = None) -> dict[str, Any]:
    """Gate-off vs gate-on on the SAME generations (accuracy-at-matched-coverage)."""
    all_files = sorted(tasks_dir.glob("*.json"))
    files = all_files[:limit] if limit else all_files
    records, backend_failures = [], 0
    for f in files:
        rec = evaluate_raw(json.loads(f.read_text()), gen, gate)
        rec["task"] = f.stem
        if rec["status"] == "backend_error":
            backend_failures += 1
        records.append(rec)
    n_ok = sum(1 for r in records if r["status"] == "ok")
    gate_off = _aggregate_arm(records, use_gate=False, n_ok=n_ok)
    gate_on = _aggregate_arm(records, use_gate=True, n_ok=n_ok)
    filtered = gate_off["counts"]["scored"] - gate_on["counts"]["scored"]
    return {
        "benchmark": "ARC-AGI-1",
        "modelSpec": spec,
        "taskCount": len(files),
        "scoredResponses": n_ok,
        "subset": None if limit is None else {"ran": len(files), "of": len(all_files)},
        "responseHealth": {"backendFailureCount": backend_failures},
        "decontam": decontam,
        "arms": {"gateOff": gate_off, "gateOn": gate_on},
        "selectivePrediction": {
            "answersFilteredByGate": filtered,
            "note": "gate-on scores a SELECTIVE subset of the SAME generations (the gate only "
                    "converts an answer to an abstention). Compare accuracy at each arm's coverage; "
                    "a real gate win is lower selectiveRisk (wrong/scored) at gate-on's coverage.",
        },
        "perTask": [{k: r.get(k) for k in ("task", "status", "rawAbstained", "parsed", "match",
                                           "gatePass", "error")} for r in records],
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
        "honestNote": "Raw wired-model ARC-AGI-1 under a fail-closed gate, exact-grid-match. Low "
                      "accuracy expected. The claim is the per-arm (correct, wrong, abstained) split "
                      "and the gate's selective-prediction effect, NOT a headline score, NOT an AGI "
                      "claim. ARC-AGI-3 (interactive) is out of scope for exact-match by construction.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", required=True, help="dir of ARC-AGI task .json files (NOT vendored)")
    ap.add_argument("--adapter", default=None, help="model spec; omit -> fail-closed env artifact")
    ap.add_argument("--manifest", default=None, help="training manifest for decontam check")
    ap.add_argument("--gate", choices=["on", "off", "both"], default="both",
                    help="both -> gate-off vs gate-on ablation on the same generations (default)")
    ap.add_argument("--limit", type=int, default=None, help="cap number of tasks (bounded pilot; logged)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks_dir = Path(args.tasks)
    if not tasks_dir.exists():
        sys.exit(f"FAIL-CLOSED: tasks dir {tasks_dir} not found. Place the official ARC-AGI "
                 "files there (this tool does not vendor the dataset).")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    decontam = decontam_ok(tasks_dir, Path(args.manifest) if args.manifest else None)

    gen = load_generator(args.adapter)
    if gen is None:
        out.write_text(json.dumps(env_artifact("no model backend available"), indent=2))
        print("FAIL-CLOSED (env artifact): no backend; wrote", out)
        return 0

    gate = load_gate()
    if args.gate == "both":
        report = run_two_arms(tasks_dir, gen, gate, decontam, limit=args.limit, spec=args.adapter)
        out.write_text(json.dumps(report, indent=2))
        off, on = report["arms"]["gateOff"], report["arms"]["gateOn"]
        print(f"OK: ARC two-arm report -> {out}  tasks={report['taskCount']} "
              f"scored={report['scoredResponses']} backendFailures={report['responseHealth']['backendFailureCount']}\n"
              f"    gate-OFF: acc={off['accuracyExactMatch']} cover={off['coverage']} risk={off['selectiveRisk']}\n"
              f"    gate-ON : acc={on['accuracyExactMatch']} cover={on['coverage']} risk={on['selectiveRisk']} "
              f"(filtered {report['selectivePrediction']['answersFilteredByGate']})")
        return 0

    use_gate = gate if args.gate == "on" else None
    report = run(tasks_dir, gen, use_gate, decontam)
    out.write_text(json.dumps(report, indent=2))
    print(f"OK: ARC report ({args.gate}) -> {out}  tasks={report['taskCount']}  "
          f"acc={report['accuracyExactMatch']}  abstain={report['abstentionRate']}  "
          f"backendFailures={report['responseHealth']['backendFailureCount']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
