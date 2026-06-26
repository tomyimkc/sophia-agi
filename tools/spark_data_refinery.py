#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Gate-filtered data refinery for the DGX Spark (roadmap phase P2).

The Spark's superpower is **128 GB unified memory + always-on + free**: ideal for
running a teacher locally 24/7 to generate training targets. But raw teacher output
is not training data — the capability lever proven in
``docs/11-Platform/Training-Efficiency-Feasibility.md`` is **data quality**, not
training speed. So every candidate the teacher emits is passed through the repo's
**intrinsic, fail-closed** gate and dropped on any violation before it can enter
SFT/RFT.

The intrinsic-gate rule (load-bearing, see Feasibility §4 / §6 P2):

    check_response(text, mode="advisor")["violations"]   # NO question

We call the gate WITHOUT the question on purpose. Passing a question additionally
invokes the attribution *trap grader* — a positive-expectation completeness check
("expected discussion of socrates") that flagged 88/564 (16%) of clean, curated rows
over **wording, not fabrication**. Intrinsic-only checking flagged 0/439 curated rows
(verified) while still catching genuine fabricated citations / false arithmetic /
forbidden-lineage merges in synthetic targets. This refinery honors that exactly.

Teachers are PLUGGABLE:

  * ``--teacher mock`` — a deterministic, seeded, offline teacher (no model, no torch).
    This is the CI / test / ``--dry-run`` default. It runs end-to-end with no GPU.
  * ``--teacher local`` — a hook that would call a local model on the Spark
    (NVFP4 70B-class teacher in the 128 GB pool). Imported lazily; the offline path
    never imports torch/transformers.

Provenance boundary: emitted rows are annotated ``sparkIteration=true`` /
``registeredResult=false`` (the same boundary as the Spark-Local-GPU lane), plus the
teacher id and the gate verdict, so a Spark-refined row can never be mistaken for a
registered result downstream.

Fail-closed throughout: if the gate cannot be imported, the refinery REFUSES to emit
(it does not pass candidates through unfiltered).

    python tools/spark_data_refinery.py --dry-run        # offline mock teacher, prints summary
    python tools/spark_data_refinery.py --seeds seeds.jsonl --out training/spark_refined.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The advisor source-discipline system prompt is the SFT/council-trace shape's
# system turn. Imported here (pure-Python, no GPU) so emitted rows match the corpus.
try:  # pragma: no cover - exercised indirectly; trivial fallback for odd checkouts
    from agent.prompts import MODE_PROMPTS  # noqa: E402

    ADVISOR_SYSTEM = MODE_PROMPTS["advisor"]
except Exception:  # noqa: BLE001
    ADVISOR_SYSTEM = (
        "You are a source-disciplined council advisor. State each finding with a "
        "source where one is relied on; if you cannot verify a needed authority or "
        "figure, ABSTAIN and route to a human rather than guess. End with a 中文摘要."
    )


# --------------------------------------------------------------------------- #
# The intrinsic gate (the load-bearing rule)
# --------------------------------------------------------------------------- #

def _real_gate(text: str) -> list[str]:
    """Call the repo's intrinsic fail-closed gate and return its violations.

    INTRINSIC: ``check_response`` is called with ``mode="advisor"`` and **no
    question** — passing a question invokes the attribution trap-grader, which is a
    positive-expectation completeness check that wrongly deletes clean rows over
    wording (Feasibility §4). We import here (not at module load) so the import error
    is surfaced at refine time and the caller can fail closed.
    """
    from agent.gate import check_response

    return list(check_response(text, mode="advisor")["violations"])


# A gate callable: text -> list of violation strings ([] == clean). Injectable so
# tests can stub the contract; the PRODUCTION DEFAULT is the real gate above.
GateFn = Callable[[str], list]


# --------------------------------------------------------------------------- #
# Pluggable teachers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Candidate:
    """A teacher's answer to one seed, before the gate has judged it."""

    seed_id: str
    prompt: str
    text: str
    teacher: str


def mock_teacher(seeds: Iterable[dict], *, seed: int = 0) -> list[Candidate]:
    """Deterministic, seeded, offline teacher — no model, no torch.

    Produces a stable answer per seed (same input + seed -> same output, always), so
    CI and tests are reproducible. A seed may carry an ``inject`` field to force a
    specific candidate string (used by tests to exercise the gate's drop path with a
    fabricated citation / false arithmetic); otherwise the mock emits a clean,
    source-disciplined, gate-passing answer.
    """
    out: list[Candidate] = []
    for idx, s in enumerate(sorted(seeds, key=lambda r: str(r.get("id", "")))):
        sid = str(s.get("id", f"seed-{idx}"))
        prompt = str(s.get("prompt", "")).strip()
        if "inject" in s:
            text = str(s["inject"])
        else:
            # Deterministic clean answer: abstains rather than fabricates, carries the
            # source-discipline framing + a 中文 summary so it passes the intrinsic gate.
            tag = (seed + idx) % 1000
            text = (
                f"On the question '{prompt}': the council finds no verifiable authority "
                f"to cite, so it ABSTAINS rather than guess (source discipline; trace {tag}). "
                f"This is not professional advice. 中文摘要：無可查證依據，故棄答並轉交人工。"
            )
        out.append(Candidate(seed_id=sid, prompt=prompt, text=text, teacher="mock"))
    return out


def local_teacher(seeds: Iterable[dict], *, seed: int = 0, model: str | None = None) -> list[Candidate]:
    """Hook for a real local model on the Spark (NVFP4 70B-class teacher).

    Lazy import only — the offline path never reaches here, so torch/transformers are
    never required for CI/tests/``--dry-run``. This intentionally raises until the
    Spark inference backend is wired, so it can never silently degrade to an unfiltered
    or non-deterministic path.
    """
    raise NotImplementedError(
        "local teacher requires the Spark inference backend (config/inference.local.spark.json, "
        "--quant bf16 --vllm none). Use --teacher mock for the offline/CI path."
    )


TEACHERS: dict[str, Callable[..., list[Candidate]]] = {
    "mock": mock_teacher,
    "local": local_teacher,
}


# --------------------------------------------------------------------------- #
# Refinery
# --------------------------------------------------------------------------- #

@dataclass
class RefineResult:
    rows: list = field(default_factory=list)
    candidates: int = 0
    kept: int = 0
    dropped: int = 0
    teacher: str = "mock"

    def summary(self) -> dict:
        return {
            "candidates": self.candidates,
            "kept": self.kept,
            "dropped": self.dropped,
            "teacher": self.teacher,
        }


def _row(cand: Candidate, *, gate_passed: bool, violations: list) -> dict:
    """Emit one gate-clean row in the SFT/council-trace shape with provenance.

    Shape mirrors ``training/council/traces.jsonl``: system (advisor) + user + the
    gate-clean assistant target. ``metadata`` carries the provenance boundary so the
    row can never be mistaken for a registered result.
    """
    return {
        "messages": [
            {"role": "system", "content": ADVISOR_SYSTEM},
            {"role": "user", "content": cand.prompt},
            {"role": "assistant", "content": cand.text},
        ],
        "metadata": {
            "source": "spark-data-refinery",
            "seedId": cand.seed_id,
            "teacher": cand.teacher,
            "kind": "refined-target",
            "labelStatus": "teacher-trace",
            # Provenance boundary — identical discipline to the Spark-Local-GPU lane.
            "sparkIteration": True,
            "registeredResult": False,
            # Gate verdict (intrinsic, no question).
            "gatePassed": gate_passed,
            "gateMode": "advisor",
            "gateIntrinsic": True,
            "gateViolations": list(violations),
        },
    }


def refine(
    seeds: list[dict],
    *,
    teacher: str = "mock",
    seed: int = 0,
    gate: GateFn | None = None,
    model: str | None = None,
) -> RefineResult:
    """Generate candidates from ``teacher``, gate-filter them, emit clean rows.

    Fail-closed: if no gate is supplied and the real gate cannot be imported, this
    RAISES rather than emit unfiltered rows. ``gate`` is injectable so tests can pass a
    stub matching ``check_response``'s contract; the production default is the real
    intrinsic gate (``_real_gate``).
    """
    teacher_fn = TEACHERS.get(teacher)
    if teacher_fn is None:
        raise ValueError(f"unknown teacher {teacher!r}; choose from {sorted(TEACHERS)}")

    if gate is None:
        # Resolve the real gate now. If it cannot be imported, FAIL CLOSED — never
        # fall back to a pass-through filter.
        try:
            gate = _real_gate
            gate("warmup: source discipline 中文")  # surface import errors eagerly
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "intrinsic gate unavailable (agent.gate.check_response); refusing to "
                "emit unfiltered data (fail-closed)"
            ) from exc

    candidates = teacher_fn(seeds, seed=seed, model=model) if teacher == "local" \
        else teacher_fn(seeds, seed=seed)

    result = RefineResult(teacher=teacher)
    for cand in candidates:
        result.candidates += 1
        violations = list(gate(cand.text))
        if violations:
            result.dropped += 1
            continue
        result.kept += 1
        result.rows.append(_row(cand, gate_passed=True, violations=violations))
    return result


# --------------------------------------------------------------------------- #
# Seeds I/O + CLI
# --------------------------------------------------------------------------- #

DEFAULT_SEEDS: list[dict] = [
    {"id": "seed-hk-lease", "prompt": "In Hong Kong, can a landlord forfeit a commercial lease without a court order?"},
    {"id": "seed-hk-pdpo", "prompt": "Does Hong Kong's PDPO restrict transferring personal data outside Hong Kong?"},
    {"id": "seed-hk-notice", "prompt": "What notice must a HK employer give to terminate a continuous contract?"},
]


def load_seeds(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate-filtered data refinery for the DGX Spark (P2)."
    )
    parser.add_argument("--seeds", type=Path, default=None,
                        help="JSONL of seed prompts ({id, prompt}); default = built-in offline seeds")
    parser.add_argument("--teacher", choices=sorted(TEACHERS), default="mock",
                        help="mock = deterministic offline (default); local = Spark model hook")
    parser.add_argument("--model", default=None, help="local-teacher model id (ignored by mock)")
    parser.add_argument("--seed", type=int, default=0, help="mock-teacher seed (deterministic)")
    parser.add_argument("--out", type=Path, default=None,
                        help="write gate-clean rows here (JSONL); omitted with --dry-run")
    parser.add_argument("--dry-run", action="store_true",
                        help="offline default: run the mock teacher end-to-end, print a summary, write nothing")
    args = parser.parse_args(argv)

    seeds = load_seeds(args.seeds) if args.seeds else list(DEFAULT_SEEDS)

    result = refine(seeds, teacher=args.teacher, seed=args.seed, model=args.model)

    if args.out and not args.dry_run:
        write_jsonl(args.out, result.rows)

    summary = result.summary()
    summary["dryRun"] = bool(args.dry_run)
    summary["out"] = str(args.out) if (args.out and not args.dry_run) else None
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
