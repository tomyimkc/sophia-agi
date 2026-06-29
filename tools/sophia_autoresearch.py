#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophia-gated AutoResearch controller — the brakes and odometer for an autonomous loop.

Karpathy's ``autoresearch`` runs an agent overnight: edit ``train.py`` -> train 5 min ->
keep iff ``val_bpb`` dropped -> else ``git reset`` -> repeat. It is a powerful engine with
**no brakes and no odometer**: ``program.md`` itself adds *no* safeguard against overfitting,
validation-set leakage, or cheating, and treats the eval as ground truth the agent could edit.

This module is the missing trust layer. It keeps the autoresearch *architecture* (one editable
surface, fixed budget, one metric, git-as-trail, loop-until-interrupt) but replaces the greedy
point-estimate keep/discard with Sophia's discipline:

  1. **Reward-hacking firewall** — the agent may edit policy / data / hyperparameters, NEVER the
     verifier, gate, eval, reward, or constitution. A diff touching a protected path is rejected
     as tampering (the deontic "no reward/verifier tampering" rule, mechanised).
  2. **Evaluation isolation** — a result that failed decontamination (eval leakage) is discarded.
  3. **Power-gated improvement** — a change is kept only if the metric improves with a 95% CI
     that EXCLUDES zero on the improving side (not a single 5-minute number).
  4. **Protected-regression block** — religion / history (and any registered protected behaviour)
     must not regress, even for a metric win.
  5. **Honest trail** — every discard/reject yields a failure-ledger record (kept changes stay
     CANDIDATE until a real multi-seed run clears the project's κ≥0.40 / CI gate).

Pure-Python, deterministic, no torch, no GPU, no network: the controller is CI-testable; the GPU
training step plugs in behind it as the ``experiments`` iterator. canClaimAGI stays false.

    python tools/sophia_autoresearch.py --self-test
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Paths the optimiser must NEVER edit — editing what scores you is reward-hacking. Mirrors the
# constitution's reward/verifier-tampering prohibition and the repo's protected domains.
DEFAULT_PROTECTED_PATTERNS: "tuple[str, ...]" = (
    r"agent/.*verifier.*\.py",
    r"agent/gate\.py",
    r"agent/verifiers\.py",
    r"agent/benchmark_checks\.py",
    r"provenance_bench/",            # the reward definitions
    r"tools/claim_gate\.py",
    r"tools/eval_stats\.py",
    r"tools/assert_decontam\.py",
    r"tools/lint_claims\.py",
    r"constitution/",
    r".*/eval/.*\.jsonl",           # eval packs / validation sets
    r".*holdout.*\.jsonl",
    r".*heldout.*\.jsonl",
    r"data/religion_concepts\.json",  # PROTECTED domain
    r"data/history_events\.json",     # PROTECTED domain
)

# Entry points of the SCORING PATH. Everything reachable by import from these is, in
# principle, reward-hack surface (editing a non-listed file the gate imports can flip a
# verdict). But the full closure (~69 files) would make the editable surface unusably
# small and would flag legitimate refactors as fatal. So we SPLIT the closure:
#   • FATAL_PROXIES — the specific data-feed + routing + dispatch files that DETERMINISTICALLY
#     change scoring verdicts (the proven reward-hack surface from the red-team). Editing
#     any of these is reject_tamper.
#   • REVIEW_PROXIES — the broader transitive closure. Editing one is not auto-fatal but the
#     change is discarded pending manual review (manual_review must be set to keep).
# The split is named, auditable, and grows by explicit addition — never silently relaxed.
SCORING_PATH_ROOTS: "tuple[str, ...]" = (
    "agent/gate.py",                 # the runtime provenance gate
    "agent/math_verifier.py",        # verifier-family roots (representative)
    "provenance_bench/swarm_rl.py",  # the reward definition
    "tools/claim_gate.py",           # the promotion gate
)

# FATAL proximity: editing any of these deterministically changes what the gate/verifier
# returns, so it is reward-hacking even though it is not a hand-listed "verifier" path.
# Each entry was verified to be (a) imported by the scoring path and (b) on the verdict-
# producing path (data feed / routing / dispatch), not a utility. See the red-team tests.
FATAL_PROXIES: "tuple[str, ...]" = (
    r"agent/entity_aliases\.py",      # feeds attribution markers -> gate verdicts
    r"agent/claim_router\.py",        # decides WHICH verifier scores each claim
    r"agent/benchmark_checks\.py",    # the attribution checks the gate calls
    r"agent/dataflow/.*\.py",         # the dataflow firewall/taint the gate depends on
    r"agent/legal_citations\.py",     # feeds the legal-citation verifier
    r"agent/legal_faithfulness\.py",  # the legal-citation check logic
)

# Content signatures of runtime reward-hacking that never touch a protected path on disk.
# An agent can mutate the gate's behaviour at import time by monkeypatching sys.modules or
# rebinding symbols in a NON-protected file; the path firewall cannot see this, so the file
# *contents* must be scanned too. These are fatal in any editable file.
TAMPER_CONTENT_PATTERNS: "tuple[str, ...]" = (
    r"sys\.modules\s*\[",            # sys.modules poisoning / shadow-import
    r"\.verifier.*\s*=\s*",          # rebinding a verifier symbol at runtime
    r"\bgate\b.*\.check_response\s*=",  # monkeypatching the gate entrypoint
    r"import\s+\*\s*from\s+agent\.gate",  # star-import then overwrite (rare; defensive)
)

Z95 = 1.959963984540054  # two-sided 95% normal quantile


def _inrepo_imports(pyfile: str, root: "Path | None" = None) -> "list[str]":
    """Parse ``pyfile`` for in-repo imports (``agent``/``provenance_bench``/``tools``/
    ``skills``/``sophia_contract``) and return their resolved paths. No execution."""
    import ast
    import os
    base = (root or ROOT)
    try:
        tree = ast.parse((base / pyfile).read_text(encoding="utf-8"))
    except Exception:
        return []
    pkgs = ("agent", "provenance_bench", "tools", "skills", "sophia_contract")
    out: list[str] = []
    for node in ast.walk(tree):
        mod = None
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in pkgs:
            mod = node.module
        elif isinstance(node, ast.Import):
            for n in node.names:
                if n.name.split(".")[0] in pkgs:
                    mod = n.name
        if mod:
            for cand in (mod.replace(".", "/") + ".py", mod.replace(".", "/") + "/__init__.py"):
                if (base / cand).exists():
                    out.append(cand)
    return out


def transitive_protected_files(roots: "Iterable[str]" = SCORING_PATH_ROOTS, *,
                               root: "Path | None" = None) -> "set[str]":
    """BFS the in-repo import graph from the scoring-path roots and return every .py file
    reachable from them — because editing ANY of those can change the gate/verifier/reward
    behaviour (indirect tampering / data poisoning). Derived from the live tree, not assumed.
    """
    base = root or ROOT
    seen: set[str] = set()
    frontier = [r for r in roots if (base / r).exists()]
    while frontier:
        cur = frontier.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for nxt in _inrepo_imports(cur, base):
            if nxt not in seen:
                frontier.append(nxt)
    return seen


def content_tamper_indicators(file_contents: "dict[str, str]",
                              patterns: "Iterable[str]" = TAMPER_CONTENT_PATTERNS) -> "dict[str, list[str]]":
    """Scan the *contents* of edited files for runtime reward-hacking signatures
    (sys.modules poisoning, symbol rebinding, gate monkeypatching). Returns
    ``{path: [matched_pattern, ...]}`` for files with hits.

    The path firewall cannot see runtime tricks; this content layer is the second wall.
    """
    compiled = [re.compile(p) for p in patterns]
    hits: "dict[str, list[str]]" = {}
    for path, text in (file_contents or {}).items():
        found = [p for p in patterns if re.compile(p).search(text or "")]
        if found:
            hits[path] = found
    return hits


@dataclass(frozen=True)
class Measurement:
    """A metric measured as PAIRED deltas (candidate − baseline) over seeds / held-out items —
    never a single number. ``lower_is_better`` sets the improving direction (val_bpb -> True)."""

    metric: str
    deltas: "tuple[float, ...]"
    lower_is_better: bool = True

    @property
    def n(self) -> int:
        return len(self.deltas)

    @property
    def mean(self) -> float:
        return sum(self.deltas) / self.n if self.n else 0.0

    def ci95(self) -> "tuple[float, float]":
        """Normal-approx 95% CI on the mean paired delta. Deterministic (no sampling)."""
        n = self.n
        if n < 2:
            return (self.mean, self.mean)
        m = self.mean
        var = sum((d - m) ** 2 for d in self.deltas) / (n - 1)
        se = math.sqrt(var / n)
        return (m - Z95 * se, m + Z95 * se)

    def improves(self) -> bool:
        """True iff the CI excludes zero ON THE IMPROVING SIDE (power-gated, not a point estimate)."""
        lo, hi = self.ci95()
        return hi < 0.0 if self.lower_is_better else lo > 0.0


@dataclass(frozen=True)
class Experiment:
    """One candidate change proposed by the autonomous agent."""

    experiment_id: str
    touched_files: "tuple[str, ...]"        # files the agent SELF-REPORTS its diff changed
    measurement: Measurement
    decontaminated: bool = True             # result of tools/assert_decontam (supplied)
    protected_regressions: "tuple[str, ...]" = ()  # protected behaviours that regressed
    # --- hardening fields (red-team close: vectors 2, 4, 5) ---
    file_contents: "dict[str, str] | None" = None       # post-edit contents, for content tamper scan
    attested_changed_files: "tuple[str, ...] | None" = None  # git-diff attestation (independent of self-report)
    held_out_sealed: bool = True            # were the measurement deltas taken on a SEALED held-out split?
    manual_review: bool = False             # a human signed off on review-flagged edits


@dataclass
class Decision:
    experiment_id: str
    verdict: str                            # "keep" | "discard" | "reject_tamper"
    reasons: "list[str]" = field(default_factory=list)
    improved: bool = False
    ci95: "tuple[float, float]" = (0.0, 0.0)
    ledger_entry: "dict | None" = None

    @property
    def kept(self) -> bool:
        return self.verdict == "keep"

    def to_dict(self) -> dict:
        return {
            "experimentId": self.experiment_id, "verdict": self.verdict,
            "reasons": list(self.reasons), "improved": self.improved,
            "ci95": [round(self.ci95[0], 5), round(self.ci95[1], 5)],
        }


def firewall_violations(touched_files: "Iterable[str]",
                        patterns: "Iterable[str]" = DEFAULT_PROTECTED_PATTERNS,
                        *, fatal_proxies: "Iterable[str] | None" = FATAL_PROXIES,
                        root: "Path | None" = None) -> "list[str]":
    """Files the optimiser is forbidden to edit (verifier / gate / eval / reward / protected).

    Two layers: (1) the hand-listed ``patterns`` (verifier/gate/eval/reward/constitution);
    (2) the FATAL_PROXIES — the specific data-feed + routing + dispatch files whose edits
    deterministically change scoring verdicts (the proven reward-hack surface). Editing
    either is ``reject_tamper``. The broader transitive closure is handled by
    :func:`review_flags`, which discards rather than rejects (legit refactors are possible).
    """
    compiled = list(patterns) + list(fatal_proxies or [])
    cre = [re.compile(p) for p in compiled]
    files = list(touched_files)
    return [f for f in files if any(c.search(f) for c in cre)]


def review_flags(touched_files: "Iterable[str]",
                 patterns: "Iterable[str]" = DEFAULT_PROTECTED_PATTERNS,
                 fatal_proxies: "Iterable[str]" = FATAL_PROXIES,
                 *, transitive_roots: "Iterable[str]" = SCORING_PATH_ROOTS,
                 root: "Path | None" = None) -> "list[str]":
    """Files in the broader transitive scoring-path closure but NOT already fatal. Editing
    one is not auto-fatal (a legit refactor is possible) but the change must be discarded
    pending manual review — it is reward-hack-adjacent and a reviewer should confirm it
    does not alter scoring. Returns the review-set, ``touched_files`` order preserved."""
    files = list(touched_files)
    try:
        closure = transitive_protected_files(transitive_roots, root=root)
    except Exception:
        closure = set()
    fatal = set(firewall_violations(files, patterns, fatal_proxies=fatal_proxies, root=root))
    return [f for f in files if f in closure and f not in fatal]


def decide(exp: Experiment, *, patterns: "Iterable[str]" = DEFAULT_PROTECTED_PATTERNS,
           root: "Path | None" = None) -> Decision:
    """Gated keep/discard. Firewall (path + content + attestation) first — tampering is fatal.

    Five red-team holes closed here:
      • indirect tampering / data poisoning  → transitive-import firewall (above)
      • runtime monkeypatch / sys.modules    → content scan of ``exp.file_contents``
      • self-reported diff under-reporting   → cross-check vs ``exp.attested_changed_files``
      • overfit/memorization passing the CI  → require ``exp.held_out_sealed``
      • eval leakage                         → existing ``decontaminated`` flag (unchanged)
    """
    reasons: list[str] = []
    ci = exp.measurement.ci95()
    improved = exp.measurement.improves()

    # --- firewall layer 1: protected paths + transitive scoring-path closure -------------
    tamper = firewall_violations(exp.touched_files, patterns, root=root)
    if tamper:
        reasons.append(f"reward-hacking firewall: edited protected/scoring-path path(s) {tamper}")

    # --- firewall layer 2: diff attestation — the agent cannot under-report its diff -------
    if exp.attested_changed_files is not None:
        attested = {f for f in exp.attested_changed_files}
        reported = {f for f in exp.touched_files}
        hidden = attested - reported          # files the git diff sees that the agent hid
        if hidden:
            extra_tamper = firewall_violations(sorted(hidden), patterns, root=root)
            if extra_tamper:
                reasons.append(
                    f"reward-hacking firewall: agent under-reported protected edits {extra_tamper} "
                    f"(present in attested diff, absent from self-report)")
            else:
                reasons.append(
                    f"diff-attestation mismatch: {sorted(hidden)} in attested diff but not "
                    f"self-reported (integrity failure)")

    # --- firewall layer 3: content scan — runtime reward-hacking without touching paths ----
    if exp.file_contents:
        c_hits = content_tamper_indicators(exp.file_contents)
        if c_hits:
            reasons.append(
                f"reward-hacking firewall: runtime-tamper signatures in file contents {c_hits}")

    # Tampering (any layer) is FATAL — reject before any metric is read.
    if any("firewall" in r or "tamper" in r for r in reasons):
        d = Decision(exp.experiment_id, "reject_tamper", reasons, improved, ci)
        d.ledger_entry = _ledger_entry(exp, d)
        return d

    # --- firewall layer 4: review-flagged proximity (non-fatal, blocks auto-keep) ---------
    # The broader transitive closure of the scoring path. Editing one of these is
    # reward-hack-ADJACENT (could subtly change scoring) but not provably fatal, so a
    # human sign-off (manual_review=True) can still keep it; without sign-off it discards.
    flagged = review_flags(exp.touched_files, patterns, root=root)
    if flagged and not exp.manual_review:
        reasons.append(
            f"review-required: edits to scoring-path-adjacent files {flagged} need manual_review "
            f"(reward-hack-adjacent; not auto-fatal but not auto-keep)")

    if not exp.decontaminated:
        reasons.append("evaluation isolation: result failed decontamination (eval leakage)")
    if not exp.held_out_sealed:
        # An unsealed hold-out means the deltas could be measured on data the optimiser has
        # seen — a memorization/overfit win that still passes the CI sign check.
        reasons.append(
            "hold-out integrity: measurement deltas not attested on a sealed held-out split "
            "(memorization/overfit cannot be distinguished from a real gain)")
    if exp.protected_regressions:
        reasons.append(f"protected-regression: {list(exp.protected_regressions)} regressed")
    if not improved:
        reasons.append(f"no powered improvement: 95% CI {tuple(round(x, 4) for x in ci)} "
                       f"does not exclude zero on the improving side")

    verdict = "keep" if not reasons else "discard"
    if verdict == "keep":
        reasons.append(f"powered improvement: 95% CI {tuple(round(x, 4) for x in ci)} excludes zero; "
                       f"decontaminated; sealed hold-out; no protected regression (CANDIDATE until multi-seed gate)")
    d = Decision(exp.experiment_id, verdict, reasons, improved, ci)
    if verdict != "keep":
        d.ledger_entry = _ledger_entry(exp, d)
    return d


def _ledger_entry(exp: Experiment, d: Decision) -> dict:
    """A failure-ledger-shaped record for a discarded/rejected experiment (honest trail)."""
    return {
        "id": f"autoresearch-{exp.experiment_id}-{d.verdict}",
        "status": "OPEN",
        "verdict": d.verdict,
        "metric": exp.measurement.metric,
        "meanDelta": round(exp.measurement.mean, 5),
        "ci95": [round(d.ci95[0], 5), round(d.ci95[1], 5)],
        "reasons": list(d.reasons),
        "claimImpact": "none — change was not kept; recorded for the research trail",
    }


def run_loop(
    experiments: "Iterable[Experiment]",
    *,
    on_keep: "Callable[[Experiment, Decision], None] | None" = None,
    on_discard: "Callable[[Experiment, Decision], None] | None" = None,
    patterns: "Iterable[str]" = DEFAULT_PROTECTED_PATTERNS,
    max_iters: "int | None" = None,
) -> dict:
    """Drive the gated loop over an experiment stream (the GPU step is the iterator). ``on_keep``
    advances the git branch; ``on_discard`` resets it — both supplied by the caller. Loops until
    the stream ends, ``max_iters``, or KeyboardInterrupt (graceful stop, autoresearch-style)."""
    decisions: list[Decision] = []
    ledger: list[dict] = []
    kept = 0
    it: Iterator[Experiment] = iter(experiments)
    i = 0
    try:
        for exp in it:
            if max_iters is not None and i >= max_iters:
                break
            i += 1
            d = decide(exp, patterns=patterns)
            decisions.append(d)
            if d.ledger_entry:
                ledger.append(d.ledger_entry)
            if d.kept:
                kept += 1
                if on_keep:
                    on_keep(exp, d)
            elif on_discard:
                on_discard(exp, d)
    except KeyboardInterrupt:
        pass
    return {
        "evaluated": len(decisions),
        "kept": kept,
        "discarded": sum(1 for d in decisions if d.verdict == "discard"),
        "rejectedTamper": sum(1 for d in decisions if d.verdict == "reject_tamper"),
        "ledger": ledger,
        "decisions": [d.to_dict() for d in decisions],
    }


def offline_invariants() -> "tuple[bool, dict]":
    """Falsifiable, deterministic invariants for the gated controller (no GPU, no network)."""
    checks: dict[str, bool] = {}

    # 1. A genuine, powered improvement is kept.
    good = Experiment("genuine-win", ("train.py",),
                      Measurement("val_bpb", tuple([-0.05] * 12), lower_is_better=True))
    checks["genuine_win_kept"] = decide(good).verdict == "keep"

    # 2. An improvement that is NOT statistically separable from zero is discarded (anti-greedy).
    noisy = Experiment("noisy", ("train.py",),
                       Measurement("val_bpb", (-0.5, 0.4, -0.3, 0.45, -0.4, 0.5), lower_is_better=True))
    checks["noisy_point_estimate_discarded"] = decide(noisy).verdict == "discard"

    # 3. Editing a verifier/gate/eval path is rejected as tampering, even with a "great" metric.
    cheat = Experiment("cheater", ("agent/gate.py", "agent/math_verifier.py"),
                       Measurement("val_bpb", tuple([-0.9] * 12), lower_is_better=True))
    dc = decide(cheat)
    checks["tamper_rejected"] = dc.verdict == "reject_tamper" and not dc.kept

    # 4. Eval leakage (failed decontam) is discarded even with a powered win.
    leaky = Experiment("leaky", ("train.py",),
                       Measurement("val_bpb", tuple([-0.2] * 12), lower_is_better=True),
                       decontaminated=False)
    checks["leakage_discarded"] = decide(leaky).verdict == "discard"

    # 5. A protected-domain regression blocks the keep even on a metric win.
    regress = Experiment("regressor", ("train.py",),
                         Measurement("val_bpb", tuple([-0.2] * 12), lower_is_better=True),
                         protected_regressions=("religion-attribution",))
    checks["protected_regression_blocked"] = decide(regress).verdict == "discard"

    # 6. higher-is-better metric path works (e.g. verified hallucination-Δ). Uses train.py
    #    (a genuinely editable surface), NOT a scoring-path-adjacent file — the hardened
    #    firewall review-flags agent/swarm_router.py now, so it is no longer an innocent example.
    halluc = Experiment("halluc-up", ("train.py",),
                        Measurement("verified_halluc_delta", tuple([0.08] * 12), lower_is_better=False))
    checks["higher_is_better_kept"] = decide(halluc).verdict == "keep"

    # 7. Every non-keep produces a ledger entry (honest trail); a keep does not.
    checks["discard_logs_ledger"] = decide(noisy).ledger_entry is not None
    checks["keep_no_ledger"] = decide(good).ledger_entry is None

    # 8. Loop accounting reconciles and the firewall count is surfaced.
    summary = run_loop([good, noisy, cheat, leaky, regress, halluc])
    checks["loop_reconciles"] = (
        summary["evaluated"] == 6
        and summary["kept"] == 2
        and summary["rejectedTamper"] == 1
        and summary["evaluated"] == summary["kept"] + summary["discarded"] + summary["rejectedTamper"]
    )

    ok = all(checks.values())
    return ok, {"checks": checks, "summary": run_loop([good, noisy, cheat, leaky, regress, halluc])}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="run the deterministic offline invariants")
    args = ap.parse_args(argv)
    if args.self_test:
        ok, detail = offline_invariants()
        print("Sophia-gated AutoResearch controller invariants:", "PASS" if ok else "FAIL")
        for k, v in detail["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        s = detail["summary"]
        print(f"  loop: evaluated={s['evaluated']} kept={s['kept']} "
              f"discarded={s['discarded']} rejectedTamper={s['rejectedTamper']}")
        return 0 if ok else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
