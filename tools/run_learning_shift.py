#!/usr/bin/env python3
"""Learning-under-distribution-shift experiment for the Sophia AGI-candidate proof.

Implements the protocol in agi-proof/learning-under-shift/README.md as a single
orchestrated, publishable experiment:

  1. Pre-test on a hidden new-domain pack (before learning).
  2. Append-only learning: write promoted candidate records (source, confidence,
     reviewer note) to an append-only memory file; a promotion gate keeps
     unreviewed records out.
  3. Post-test on a *fresh* new-domain pack (cases not seen during learning).
  4. Old-benchmark stability: re-score a frozen old pack to show prior knowledge
     did not regress.
  5. Contamination audit + protected-knowledge hash proof: confirm post-test
     cases were not memorized from the learning records or training corpus and
     that protected old records were not silently overwritten.

Scoring reuses the same run_case / score_pack path as the hidden runner, so the
learning experiment and the hidden eval share one code path. Model calls need a
backend; the pure gate/audit/hash logic is independently testable.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.hidden_eval_protocol import score_pack, validate_pack  # noqa: E402
from tools.run_hidden_eval_sophia import (  # noqa: E402
    SOPHIA_FULL,
    DEFAULT_GROK_CWD,
    RunConfig,
    backend_preflight,
    protected_hashes,
    run_case,
    sha256_file,
)

SHIFT_MEMORY_FILE = ROOT / "agent" / "memory" / "learning_shift.jsonl"
TRAINING_DIR = ROOT / "training"


def apply_promotion_gate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Only records explicitly promoted (reviewed) may be appended to memory."""
    promoted = [r for r in records if r.get("promoted") is True]
    rejected = [r for r in records if r.get("promoted") is not True]
    return promoted, rejected


def append_learning_records(promoted: list[dict[str, Any]], memory_file: Path = SHIFT_MEMORY_FILE) -> dict[str, Any]:
    """Append promoted records to the append-only learning memory; never rewrite."""
    old_hash = sha256_file(memory_file)
    protected_before = protected_hashes()
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    appended_ids: list[str] = []
    with memory_file.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(promoted, 1):
            record_id = record.get("recordId") or f"shift_record_{index}"
            entry = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "recordId": record_id,
                "domain": record.get("domain"),
                "text": record.get("text", ""),
                "source": record.get("source", ""),
                "confidence": record.get("confidence", ""),
                "reviewerNote": record.get("reviewerNote", ""),
                "mode": "append-only",
            }
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            appended_ids.append(record_id)
    new_hash = sha256_file(memory_file)
    protected_after = protected_hashes()
    try:
        memory_label = str(memory_file.relative_to(ROOT))
    except ValueError:
        memory_label = str(memory_file)
    return {
        "memoryFile": memory_label,
        "oldHash": old_hash,
        "newHash": new_hash,
        "appended": old_hash != new_hash if promoted else False,
        "appendedRecordIds": appended_ids,
        "protectedKnowledgeUnchanged": protected_before == protected_after,
    }


def _training_corpus_text() -> str:
    parts: list[str] = []
    if TRAINING_DIR.exists():
        for path in sorted(TRAINING_DIR.glob("*.json")):
            try:
                parts.append(path.read_text(encoding="utf-8"))
            except OSError:
                continue
    return "\n".join(parts).lower()


def _shingles(text: str, n: int = 8) -> set[str]:
    """Word n-grams for robust verbatim-leakage detection (avoids matching on
    short generic prompts like 'what is 2 + 2')."""
    words = text.lower().split()
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _answer_tokens(case: dict[str, Any]) -> list[str]:
    """Specific answer strings the post-test requires (mustInclude match terms)."""
    tokens: list[str] = []
    for item in case.get("scoring", {}).get("mustInclude", []):
        if isinstance(item, str):
            tokens.append(item)
        elif isinstance(item, dict) and item.get("match"):
            tokens.append(str(item["match"]))
    return [t for t in tokens if len(t) >= 3]


def contamination_audit(
    pre_pack: dict[str, Any],
    post_pack: dict[str, Any],
    learning_records: list[dict[str, Any]],
    *,
    training_text: str | None = None,
) -> dict[str, Any]:
    """Confirm the post-test measures *learning*, not memorization.

    Flags: (1) pre/post case-id overlap; (2) a post-test prompt that appears
    verbatim (8-word shingle) in the learning records or training corpus — the
    model would regurgitate rather than generalize; (3) a post-test ANSWER token
    that already exists in the TRAINING corpus — then any gain is not attributable
    to the append-only learning. Note: answer tokens appearing in the LEARNING
    records are expected (that is the teaching) and are NOT flagged.
    """
    issues: list[str] = []
    pre_ids = {c["id"] for c in pre_pack.get("cases", [])}
    post_ids = {c["id"] for c in post_pack.get("cases", [])}
    overlap = pre_ids & post_ids
    if overlap:
        issues.append(f"pre/post share case ids: {sorted(overlap)}")

    record_text = "\n".join(f"{r.get('text', '')}\n{r.get('source', '')}" for r in learning_records)
    record_shingles = _shingles(record_text)
    corpus = _training_corpus_text() if training_text is None else training_text.lower()
    corpus_shingles = _shingles(corpus)

    for case in post_pack.get("cases", []):
        prompt = str(case.get("prompt", "")).strip()
        if not prompt:
            continue
        prompt_shingles = _shingles(prompt)
        if prompt_shingles and prompt_shingles & record_shingles:
            issues.append(f"post-test prompt for {case['id']} appears verbatim in learning records")
        if prompt_shingles and prompt_shingles & corpus_shingles:
            issues.append(f"post-test prompt for {case['id']} appears verbatim in training corpus")
        for token in _answer_tokens(case):
            if corpus and token.lower() in corpus:
                issues.append(
                    f"post-test answer token '{token}' for {case['id']} already in training corpus "
                    f"(gain not attributable to learning)"
                )

    return {
        "clean": not issues,
        "issues": issues,
        "preCaseCount": len(pre_ids),
        "postCaseCount": len(post_ids),
        "method": "id-overlap + 8-word prompt shingles vs records/corpus + answer-token vs training corpus",
    }


def run_pack(pack: dict[str, Any], config: RunConfig) -> dict[str, Any]:
    """Score a pack through the sophia-full pipeline (same path as hidden eval)."""
    responses: dict[str, str] = {}
    tool_logs: dict[str, Any] = {}
    memory_diffs: dict[str, Any] = {}
    for index, case in enumerate(pack["cases"], 1):
        print(f"    [{index}/{len(pack['cases'])}] {case['id']} ({case['domain']})", flush=True)
        result = run_case(case, pack["packId"], config=config, ablation=SOPHIA_FULL)
        responses[case["id"]] = result["answer"]
        tool_logs[case["id"]] = result["toolLog"]
        memory_diffs[case["id"]] = result["memoryDiff"]
    private = score_pack(pack, {"responses": responses, "toolLogs": tool_logs, "memoryDiffs": memory_diffs})
    return {
        "passed": private["passed"],
        "totalCases": private["totalCases"],
        "score": private["score"],
        "maxScore": private["maxScore"],
        "scorePct": private["scorePct"],
    }


def build_report(
    spec: dict[str, Any],
    pre: dict[str, Any],
    post: dict[str, Any],
    old: dict[str, Any] | None,
    memory_diff: dict[str, Any],
    promotion: dict[str, Any],
    audit: dict[str, Any],
    backend: str,
) -> dict[str, Any]:
    improvement = round(post["scorePct"] - pre["scorePct"], 2)
    old_delta = None
    baseline = spec.get("oldBenchmarkBaselineScorePct")
    if old is not None and isinstance(baseline, (int, float)):
        old_delta = round(old["scorePct"] - float(baseline), 2)
    # Stability is "ok" only if no old pack was requested, or an old pack AND a
    # baseline were supplied and the delta is within tolerance. Supplying an old
    # pack with no baseline is NOT a silent pass — it is unverifiable.
    if old is None:
        stability_ok = True
        stability_evaluable = "not-requested"
    elif old_delta is None:
        stability_ok = False
        stability_evaluable = "requested-but-no-baseline"
    else:
        stability_ok = old_delta >= -5.0
        stability_evaluable = "evaluated"
    passing_signal = bool(
        improvement > 0
        and memory_diff.get("protectedKnowledgeUnchanged")
        and audit.get("clean")
        and stability_ok
    )
    return {
        "experimentId": spec.get("experimentId", "learning-shift"),
        "runAt": datetime.now().isoformat(timespec="seconds"),
        "backend": backend,
        "visibility": "public-aggregate-no-prompts",
        "preTest": pre,
        "postTest": post,
        "improvementDeltaPct": improvement,
        "oldBenchmarkStability": old,
        "oldBenchmarkDeltaPct": old_delta,
        "stabilityEvaluable": stability_evaluable,
        "memoryDiff": {
            "memoryFile": memory_diff.get("memoryFile"),
            "appended": memory_diff.get("appended"),
            "appendedRecordIds": memory_diff.get("appendedRecordIds"),
            "protectedKnowledgeUnchanged": memory_diff.get("protectedKnowledgeUnchanged"),
        },
        "promotionGate": promotion,
        "contaminationAudit": audit,
        "passingSignal": passing_signal,
        "passingSignalRule": (
            "post>pre improvement AND protected knowledge unchanged AND clean "
            "contamination audit AND old-benchmark stability within 5 points."
        ),
        "notes": [
            "Scores are auto keyword/regex screens; two-pass manual semantic review "
            "still required before strong per-case quality claims.",
            "A flat or negative improvement is reported honestly per the failure ledger.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a learning-under-distribution-shift experiment")
    parser.add_argument("spec", type=Path, help="Experiment spec JSON")
    parser.add_argument("--backend", choices=["anthropic", "grok", "deepseek", "adapter"], default="grok")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--memory-file", type=Path, default=SHIFT_MEMORY_FILE)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--preflight-timeout-sec", type=int, default=45)
    parser.add_argument("--grok-cwd", type=Path, default=DEFAULT_GROK_CWD)
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    for key in ("preTestPack", "postTestPack"):
        errors = validate_pack(spec[key])
        if errors:
            print(json.dumps({"ok": False, "pack": key, "errors": errors}, indent=2, ensure_ascii=False))
            return 1
    if spec.get("oldBenchmarkPack"):
        errors = validate_pack(spec["oldBenchmarkPack"])
        if errors:
            print(json.dumps({"ok": False, "pack": "oldBenchmarkPack", "errors": errors}, indent=2, ensure_ascii=False))
            return 1

    config = RunConfig(backend=args.backend, timeout_sec=args.timeout_sec, grok_cwd=args.grok_cwd)

    if not args.skip_preflight:
        print(f"[preflight] checking {args.backend} backend")
        health = backend_preflight(backend=args.backend, timeout_sec=args.preflight_timeout_sec, grok_cwd=args.grok_cwd)
        if not health.get("ok"):
            print(json.dumps({"ok": False, "stage": "backend-preflight", "backendHealth": health}, indent=2, ensure_ascii=False))
            return 2

    print("[phase 1] pre-test (before learning)")
    pre = run_pack(spec["preTestPack"], config)

    print("[phase 2] append-only learning + promotion gate")
    records = spec.get("learningRecords", [])
    promoted, rejected = apply_promotion_gate(records)
    memory_diff = append_learning_records(promoted, args.memory_file)
    promotion = {
        "candidateCount": len(records),
        "promotedCount": len(promoted),
        "rejectedCount": len(rejected),
        "rejectedRecordIds": [r.get("recordId") for r in rejected],
    }

    print("[phase 3] post-test (fresh cases, after learning)")
    post = run_pack(spec["postTestPack"], config)

    old = None
    if spec.get("oldBenchmarkPack"):
        print("[phase 4] old-benchmark stability re-test")
        old = run_pack(spec["oldBenchmarkPack"], config)

    print("[phase 5] contamination audit")
    audit = contamination_audit(spec["preTestPack"], spec["postTestPack"], records)

    report = build_report(spec, pre, post, old, memory_diff, promotion, audit, args.backend)

    out_path = args.out or (
        ROOT / "agi-proof" / "learning-under-shift" / f"shift-result-{datetime.now().date().isoformat()}.public-report.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
