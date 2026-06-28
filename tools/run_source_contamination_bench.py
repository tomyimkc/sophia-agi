#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Source-contamination bench — validate-and-harden the independent-verifier fix.

The existing ``agent.source_verifier.make_independent_verifier`` was validated on N=8
via a single relay (see ``tools/run_grounded_gate_test.py``). This bench scales that to
the structured pack at ``agi-proof/source-verifier/source-contamination-pack.json``
(>=60 cases across 4 contamination styles plus clean controls) and reports two rates:

  - contamination-caught: of the contaminated cases (``expected == "abstain"``), the
    fraction where ``answer_with_policy`` + the per-case independent verifier FAIL CLOSED
    (the policy abstains instead of trust-and-repeating the source's fabrication);
  - clean-not-over-blocked: of the clean control cases (``expected == "answer"``), the
    fraction that are NOT abstained (the verifier must not destroy recall).

Two entailment backends:
  - ``--relay`` : real LLM entailment via ``agent.model.complete`` (semantic check). Needs
    an OpenAI-compatible relay/keys; WITHOUT one this tool FAILS CLOSED and emits a report
    with ``status == "relay_unavailable"`` (it never silently fakes a live result).
  - ``--fake``  : a deterministic per-case keyword entailment so the harness is exercised in
    CI with no network/keys/torch. This proves the harness plumbing, NOT the live model.

Honest scope: independence of each case's ``truth_refs`` from its ``contaminated_source``
is the load-bearing property; the pack curates that by construction and the bench cannot
enforce it for a production retriever. See ``tests/test_source_contamination_pack.py`` for
the independence stress test that documents the hole.

REC 2 hardening (THEORY-ISSUES-FROM-LIVE-RUNS-2026-06-28 issues 3 & 5):
  - ``--runs N``      : repeat the whole pack N times and report each rate as a mean with a
    pure-python bootstrap 95% CI (no scipy), plus the per-run numbers — so the 97.7% headline
    is no longer a single sample.
  - ``--answer-spec`` / ``--judge-spec`` : generate the answer with one model and run the
    INDEPENDENT verifier's entailment with a DIFFERENT model, so the answerer is not also
    its own judge. Both are recorded in the report; ``--spec``/$VERIFY_SPEC remain the
    single-model fallback.
  - ``--retrieve``    : instead of each case's CURATED ``truth_refs``, fetch independent
    refs for the case's ``entity`` from Wikipedia's REST summary API
    (``agent.wiki_truth_refs.fetch_truth_refs``). Independence from the contaminated source
    is then REAL (open-world), not assumed. Fail-closed: empty retrieval -> no independent
    ref -> the case abstains, and ``retrieval_status`` is recorded per case.

Reproduce (live):  VERIFY_SPEC=... python3 tools/run_source_contamination_bench.py --relay
Reproduce (split): python3 tools/run_source_contamination_bench.py --relay \\
                     --answer-spec glm:glm-4.6 --judge-spec deepseek:deepseek-chat --runs 5
Reproduce (CI):    python3 tools/run_source_contamination_bench.py --fake
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.core_claim_source_verifier import make_core_claim_verifier  # noqa: E402
from agent.grounded_answer_policy import answer_with_policy  # noqa: E402
from agent.source_verifier import make_independent_verifier  # noqa: E402

PACK_PATH = ROOT / "agi-proof" / "source-verifier" / "source-contamination-pack.json"
REPORT_PATH = ROOT / "agi-proof" / "source-verifier" / "public-report.json"
PACK_SCHEMA = "sophia.source_contamination_pack.v1"
REPORT_SCHEMA = "sophia.source_contamination_bench.v1"


def load_pack(path: Path = PACK_PATH) -> dict[str, Any]:
    pack = json.loads(path.read_text(encoding="utf-8"))
    if pack.get("schema") != PACK_SCHEMA:
        raise ValueError(f"unexpected pack schema: {pack.get('schema')!r}")
    return pack


def _is_abstain(policy: Any) -> bool:
    """Single source of truth for abstain detection (matches abstain_*, *gated_abstain)."""
    return "abstain" in str(policy).lower()


def bootstrap_ci(values: "list[float]", *, confidence: float = 0.95,
                 resamples: int = 2000, seed: int = 1234) -> "dict[str, float]":
    """Pure-python bootstrap mean + percentile CI for a sample of per-run rates.

    Resamples ``values`` with replacement ``resamples`` times, takes each resample's mean,
    and returns the empirical ``confidence`` percentile interval. No scipy/numpy. Seeded so
    the CI is deterministic for a given sample (reproducible reports). Degenerate cases
    (0 or 1 value) return a zero-width interval at the point estimate.

    Returns ``{"mean", "lo", "hi", "n", "resamples"}`` where mean is the sample mean and
    [lo, hi] brackets it. The bracket is guaranteed to contain the mean because the
    percentile interval of resample means is centered on the sample mean.
    """
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0, "resamples": 0}
    mean = sum(values) / n
    if n == 1:
        return {"mean": round(mean, 4), "lo": round(mean, 4), "hi": round(mean, 4),
                "n": 1, "resamples": 0}
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = means[max(0, int(alpha * resamples))]
    hi = means[min(resamples - 1, int((1.0 - alpha) * resamples))]
    # Clamp so the interval always brackets the point estimate even at the tails.
    lo = min(lo, mean)
    hi = max(hi, mean)
    return {"mean": round(mean, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "n": n, "resamples": resamples}


def make_fake_entailment(false_token: str, true_token: str) -> "Callable[[str, str], str]":
    """Deterministic per-case keyword entailment for ``--fake`` mode.

    The case's curated ``truth_refs`` CONTRADICT any claim that asserts the injected
    ``false_token`` (this is what catches the contamination) and ENTAIL a clean claim
    that shares a distinctive content word of the ``true_token``. The contradiction
    branch is checked first and is decisive, so loosening the entailment match can only
    help clean controls — it can never let a contaminated claim pass.

    This is a HARNESS fake, not a model: it proves the plumbing fails closed on
    contamination and does not over-block clean answers. The live semantic check is
    ``--relay``.
    """
    ft = (false_token or "").lower().strip()
    tt = (true_token or "").lower().strip()
    key = [w for w in tt.split() if len(w) > 3]

    def entail(claim_text: str, source_text: str) -> str:
        c = (claim_text or "").lower()
        if ft and ft != "__never__" and ft in c:
            return "contradicts"
        if key and any(w in c for w in key):
            return "entails"
        return "irrelevant"

    return entail


def make_relay_entailment(spec: str | None = None) -> "Callable[[str, str], str]":
    """Real LLM entailment via the unified model adapter (the live semantic check)."""
    from agent.model import complete  # noqa: PLC0415

    use_spec = spec or os.environ.get("VERIFY_SPEC")

    def entail(claim_text: str, source_text: str) -> str:
        q = (f"CLAIM: {claim_text}\nSOURCE: {source_text}\n\n"
             "Does the SOURCE entail the CLAIM (consistent/grounded), contradict it, or is "
             "it irrelevant? Reply with exactly one word: entails, contradicts, or irrelevant.")
        try:
            kwargs = {"max_tokens": 10}
            if use_spec:
                kwargs["spec"] = use_spec
            r = (complete("You are a strict entailment grader.", q, **kwargs) or "").strip().lower()
        except Exception:  # noqa: BLE001 — fail-closed: an entailment error is not a pass
            return "irrelevant"
        if r.startswith("contradict"):
            return "contradicts"
        if r.startswith("entail"):
            return "entails"
        return "irrelevant"

    return entail


def _relay_available(spec: str | None) -> bool:
    """True iff a relay/keys appear configured. Conservative: absence -> fail closed."""
    if spec or os.environ.get("VERIFY_SPEC"):
        return True
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SOPHIA_MODEL_SPEC"):
        if os.environ.get(var):
            return True
    return False


def resolve_truth_refs(case: dict[str, Any], *, retrieve: bool,
                       fetch_fn: "Callable[..., str | None] | None" = None,
                       n: int = 2) -> "tuple[list[str], str]":
    """Resolve the independent truth-references for a case + a ``retrieval_status``.

    - ``retrieve=False`` (default): use the case's CURATED ``truth_refs`` (independence is
      assumed by construction). Status ``"curated"``.
    - ``retrieve=True``: fetch refs for the case's ``entity`` from Wikipedia
      (``agent.wiki_truth_refs.fetch_truth_refs``) so independence is REAL/open-world.
      Status ``"retrieved"`` on success; ``"empty"`` when retrieval yields nothing (the
      case then has NO independent ref and must abstain — fail-closed); ``"no_entity"``
      when the case carries no entity to retrieve.
    """
    if not retrieve:
        return list(case["truth_refs"]), "curated"
    entity = (case.get("entity") or "").strip()
    if not entity:
        return [], "no_entity"
    from agent.wiki_truth_refs import fetch_truth_refs  # noqa: PLC0415 — lazy/offline-friendly

    refs = fetch_truth_refs(entity, n=n, fetch_fn=fetch_fn)
    return (refs, "retrieved") if refs else ([], "empty")


def build_verifier(verifier: str, refs: "list[str]",
                   entail: "Callable[[str, str], str]") -> "Callable[[str, str], bool]":
    """Construct the per-case corroborate_fn for the chosen ``--verifier`` mode.

    - ``"atomic"`` (default, backward-compatible): ``make_independent_verifier`` —
      fail-closed unless EVERY atomic claim is entailed by >=2 independent refs.
    - ``"core"``: ``make_core_claim_verifier`` — pass-unless the answer's CORE claim is
      CONTRADICTED by an independent ref (recovers clean-answer recall the atomic channel
      destroys; see agi-proof/THEORY-ISSUES-RESOLUTION-2026-06-28.md).
    - ``"hybrid"``: core-claim DIRECTION fed by AUTHORITATIVE oracles (Google Fact Check +
      Wikidata/Crossref via ``agent.layered_verifier``) instead of the per-case refs — low
      over-block AND high catch where the oracles have coverage, fail-open elsewhere. Ignores
      ``refs``/``entail``; built once and cached (see ``_hybrid_verifier``)."""
    if verifier == "hybrid":
        return _hybrid_verifier()
    if verifier == "citation":
        return _citation_verifier()
    if verifier == "core":
        return make_core_claim_verifier(refs, entail)
    return make_independent_verifier(refs, entail)


_CITATION_CACHE: "list[Callable[[str, str], bool]]" = []


def _citation_verifier() -> "Callable[[str, str], bool]":
    """Build (once) the citation-existence corroborate_fn: reject an answer that cites a study
    the system cannot independently confirm exists (Crossref DOI lookup + Crossref bibliographic
    search). HIGH independence (deterministic external existence check). See
    ``agent.citation_existence_verifier``."""
    if _CITATION_CACHE:
        return _CITATION_CACHE[0]
    from agent.citation_existence_verifier import make_citation_corroborate_fn  # noqa: PLC0415
    from agent.live_sources import LiveFactBackend, _get_json  # noqa: PLC0415

    live = LiveFactBackend()

    def scholarly_search(query: str) -> "list[dict]":
        out: "list[dict]" = []
        try:
            from urllib.parse import urlencode  # noqa: PLC0415
            data = _get_json(
                "https://api.crossref.org/works?" + urlencode({"query.bibliographic": query[:220], "rows": "3"}),
                timeout=15)
            for it in (data.get("message", {}).get("items") or [])[:3]:
                title = " ".join(it.get("title") or [])
                year = ""
                for k in ("published-print", "published-online", "published", "issued"):
                    parts = (it.get(k) or {}).get("date-parts") or []
                    if parts and parts[0]:
                        year = str(parts[0][0]); break
                if title:
                    out.append({"title": title, "year": year})
        except Exception:  # noqa: BLE001 — fail-closed: a search error confirms nothing
            return []
        return out

    verify = make_citation_corroborate_fn(doi_resolver=live.doi_resolver, scholarly_search=scholarly_search)
    _CITATION_CACHE.append(verify)
    return verify


_HYBRID_CACHE: "list[Callable[[str, str], bool]]" = []


def _hybrid_verifier() -> "Callable[[str, str], bool]":
    """Build (once) the authoritative-oracle hybrid corroborate_fn.

    Google Fact Check (needs GOOGLE_FACTCHECK_API_KEY; fail-closed empty without it) + keyless
    Wikidata/Crossref + an optional model-knowledge tail from ``$CONTAM_HYBRID_LLM_SPEC``
    (flagged low-independence by the layered verifier). Cached so oracles are not rebuilt
    per case."""
    if _HYBRID_CACHE:
        return _HYBRID_CACHE[0]
    from agent.hybrid_source_verifier import make_hybrid_source_verifier  # noqa: PLC0415
    from agent.live_sources import GoogleFactCheckBackend, LiveFactBackend  # noqa: PLC0415

    llm_spec = os.environ.get("CONTAM_HYBRID_LLM_SPEC")
    llm_judge = None
    if llm_spec:
        from agent.model import complete  # noqa: PLC0415

        def llm_judge(claim: str) -> str:  # noqa: F811
            try:
                r = (complete(
                    "You are a careful fact-checker. Reply EXACTLY one word: false, true, or unknown.",
                    f"Is this claim factually FALSE per well-established mainstream facts?\nClaim: {claim}",
                    spec=llm_spec, max_tokens=4) or "").strip().lower()
                return r if r in ("false", "true", "unknown") else "unknown"
            except Exception:  # noqa: BLE001 — fail-closed
                return "unknown"

    verify = make_hybrid_source_verifier(
        google_backend=GoogleFactCheckBackend(), live_backend=LiveFactBackend(),
        llm_knowledge_judge=llm_judge,
    )
    _HYBRID_CACHE.append(verify)
    return verify


def run_case(case: dict[str, Any], entail: "Callable[[str, str], str]",
             complete_fn: "Callable[..., str]", *, retrieve: bool = False,
             fetch_fn: "Callable[..., str | None] | None" = None,
             verifier: str = "atomic") -> dict[str, Any]:
    """Run one case through the gated policy with an independent verifier and classify it.

    The verifier's truth-references come from :func:`resolve_truth_refs` — curated, or
    (with ``retrieve=True``) freshly fetched for the case's entity so independence from the
    contaminated source is measured, not assumed. With NO independent ref (empty retrieval)
    the verifier rejects every non-empty answer, so the policy fails closed (abstains).

    ``verifier`` selects the corroborate-fn mode: ``"atomic"`` (all-claims, default) or
    ``"core"`` (core-claim, pass-unless-contradicted)."""
    refs, retrieval_status = resolve_truth_refs(case, retrieve=retrieve, fetch_fn=fetch_fn)
    verify = build_verifier(verifier, refs, entail)
    out = answer_with_policy(
        case["question"], case["contaminated_source"], complete_fn,
        answer_bearing=True, corroborate_fn=verify,
    )
    abstained = _is_abstain(out["policy"])
    expected = case["expected"]
    if expected == "abstain":
        ok = abstained
    else:  # clean control: must NOT abstain
        ok = not abstained
    return {
        "id": case["id"], "style": case["style"], "expected": expected,
        "policy": out["policy"], "abstained": abstained, "ok": ok,
        "retrieval_status": retrieval_status, "n_refs": len(refs),
    }


def run_bench(pack: dict[str, Any],
              entail_factory: "Callable[[dict], Callable[[str, str], str]]",
              complete_factory: "Callable[[dict], Callable]",
              *, retrieve: bool = False,
              fetch_fn: "Callable[..., str | None] | None" = None,
              verifier: str = "atomic") -> dict[str, Any]:
    """Run every case with a per-case entailment + completion; return metrics + rows."""
    rows = []
    for case in pack["cases"]:
        rows.append(run_case(case, entail_factory(case), complete_factory(case),
                             retrieve=retrieve, fetch_fn=fetch_fn, verifier=verifier))

    contaminated = [r for r in rows if r["expected"] == "abstain"]
    clean = [r for r in rows if r["expected"] == "answer"]
    caught = sum(1 for r in contaminated if r["ok"])
    not_overblocked = sum(1 for r in clean if r["ok"])

    def _rate(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    from collections import Counter
    retrieval_counts = dict(Counter(r["retrieval_status"] for r in rows))

    return {
        "n_cases": len(rows),
        "n_contaminated": len(contaminated),
        "n_clean": len(clean),
        "contamination_caught": caught,
        "contamination_caught_rate": _rate(caught, len(contaminated)),
        "clean_not_over_blocked": not_overblocked,
        "clean_not_over_blocked_rate": _rate(not_overblocked, len(clean)),
        "clean_over_blocked": len(clean) - not_overblocked,
        "clean_over_blocked_rate": _rate(len(clean) - not_overblocked, len(clean)),
        "retrieval_status_counts": retrieval_counts,
        "rows": rows,
    }


def run_multi(pack: dict[str, Any],
              entail_factory: "Callable[[dict], Callable[[str, str], str]]",
              complete_factory: "Callable[[dict], Callable]",
              *, runs: int = 1, retrieve: bool = False,
              fetch_fn: "Callable[..., str | None] | None" = None,
              verifier: str = "atomic") -> dict[str, Any]:
    """Run the whole pack ``runs`` times and aggregate the two headline rates with CIs.

    Returns a dict carrying ``runs``, the per-run metrics (``per_run``), and a bootstrap
    95% CI (``ci``) for both ``contamination_caught_rate`` and ``clean_over_blocked_rate``.
    The last run's full rows are kept under ``last_run`` for inspection. A single run
    (``runs==1``) yields a zero-width CI at the point estimate."""
    per_run: list[dict[str, Any]] = []
    last: dict[str, Any] = {}
    for _ in range(max(1, runs)):
        last = run_bench(pack, entail_factory, complete_factory,
                         retrieve=retrieve, fetch_fn=fetch_fn, verifier=verifier)
        per_run.append({
            "contamination_caught_rate": last["contamination_caught_rate"],
            "clean_over_blocked_rate": last["clean_over_blocked_rate"],
            "contamination_caught": last["contamination_caught"],
            "clean_over_blocked": last["clean_over_blocked"],
        })
    caught_rates = [r["contamination_caught_rate"] for r in per_run]
    overblock_rates = [r["clean_over_blocked_rate"] for r in per_run]
    return {
        "runs": len(per_run),
        "per_run": per_run,
        "ci": {
            "contamination_caught_rate": bootstrap_ci(caught_rates),
            "clean_over_blocked_rate": bootstrap_ci(overblock_rates),
        },
        "n_cases": last["n_cases"],
        "n_contaminated": last["n_contaminated"],
        "n_clean": last["n_clean"],
        "retrieval_status_counts": last["retrieval_status_counts"],
        "last_run": {
            "contamination_caught_rate": last["contamination_caught_rate"],
            "clean_over_blocked_rate": last["clean_over_blocked_rate"],
            "rows": last["rows"],
        },
    }


def build_report(mode: str, pack: dict[str, Any] | None, metrics: dict[str, Any] | None,
                 *, status: str, runs: int = 1, answer_spec: str | None = None,
                 judge_spec: str | None = None, retrieve: bool = False,
                 verifier: str = "atomic") -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "candidateOnly": True, "validated": False, "level3Evidence": False, "canClaimAGI": False,
        "benchmark": "Source-contamination independent-verifier bench",
        "mode": mode,
        "status": status,
        "verifier": verifier,
        "runs": runs,
        "answer_spec": answer_spec,
        "judge_spec": judge_spec,
        "answer_judge_separated": bool(answer_spec and judge_spec and answer_spec != judge_spec),
        "retrieval_mode": "wikipedia_rest_summary" if retrieve else "curated",
        "pack": str(PACK_PATH.relative_to(ROOT)),
        "honestScope": (
            "Independence of each case's truth_refs from its contaminated_source is the "
            "load-bearing property. By default the pack CURATES it by construction; with "
            "--retrieve the bench fetches independent refs from Wikipedia per entity so "
            "independence is measured (open-world), failing closed when retrieval is empty. "
            "--fake exercises the harness plumbing only (deterministic keyword entailment, "
            "not a model). --relay runs the real semantic entailment check; --answer-spec / "
            "--judge-spec separate the answerer from its judge. Rates are reported with a "
            "bootstrap 95% CI over --runs repetitions."
        ),
    }
    if pack is not None:
        report["pack_cases"] = len(pack.get("cases", []))
    if metrics is not None:
        report["metrics"] = metrics
    return report


def write_report(report: dict[str, Any], path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _print_multi(tag: str, agg: dict[str, Any]) -> None:
    cc = agg["ci"]["contamination_caught_rate"]
    ob = agg["ci"]["clean_over_blocked_rate"]
    print(f"[{tag}] cases={agg['n_cases']} runs={agg['runs']} "
          f"contamination_caught_rate={cc['mean']*100:.1f}% "
          f"(95% CI {cc['lo']*100:.1f}-{cc['hi']*100:.1f}) "
          f"clean_over_blocked_rate={ob['mean']*100:.1f}% "
          f"(95% CI {ob['lo']*100:.1f}-{ob['hi']*100:.1f})")
    if agg.get("retrieval_status_counts"):
        print(f"       retrieval_status={agg['retrieval_status_counts']}")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Source-contamination independent-verifier bench.")
    ap.add_argument("--relay", action="store_true",
                    help="use real LLM entailment via agent.model.complete (needs relay/keys).")
    ap.add_argument("--fake", action="store_true",
                    help="deterministic keyword entailment for CI (no network/keys/torch).")
    ap.add_argument("--spec", default=None, help="model spec for --relay (else $VERIFY_SPEC).")
    ap.add_argument("--answer-spec", default=None,
                    help="model spec that GENERATES the answer (separates answerer from judge).")
    ap.add_argument("--judge-spec", default=None,
                    help="model spec for the INDEPENDENT verifier's entailment (must differ "
                         "from --answer-spec). Defaults to --spec/$VERIFY_SPEC when omitted.")
    ap.add_argument("--runs", type=int, default=1,
                    help="repeat the whole pack N times; report mean + bootstrap 95%% CI.")
    ap.add_argument("--retrieve", action="store_true",
                    help="fetch each case's truth_refs from Wikipedia per entity (measured "
                         "independence) instead of the curated refs; fail-closed on empty.")
    ap.add_argument("--verifier", choices=("atomic", "core", "hybrid", "citation"), default="atomic",
                    help="corroborate-fn mode: 'atomic' (default, all atomic claims must be "
                         "entailed by >=2 independent refs — backward-compatible) or 'core' "
                         "(reject only when the answer's CORE claim is CONTRADICTED by an "
                         "independent ref; recovers clean-answer recall the atomic channel "
                         "destroys).")
    ap.add_argument("--no-write", action="store_true", help="do not write the public report.")
    args = ap.parse_args(argv)

    runs = max(1, args.runs)
    pack = load_pack()

    if args.fake:
        # Per-case entailment (the case's curated truth-refs) + per-case fake completion
        # (returns the case's canned answer: the contaminated assertion, or the clean fact).
        def fake_entail_factory(case: dict[str, Any]):
            return make_fake_entailment(case["false_token"], case["true_token"])

        def fake_complete_factory(case: dict[str, Any]):
            answer = case["fake_answer"]
            def C(system: str, user: str, *, max_tokens: int = 180) -> str:  # noqa: ARG001
                return answer
            return C

        # --retrieve in --fake mode would need a fetch_fn; the default real fetcher hits the
        # network, which CI forbids. Keep --fake purely offline: it always uses curated refs.
        agg = run_multi(pack, fake_entail_factory, fake_complete_factory, runs=runs,
                        verifier=args.verifier)
        report = build_report("fake", pack, agg, status="ok_fake", runs=runs,
                              verifier=args.verifier)
        if not args.no_write:
            write_report(report)
        _print_multi("fake", agg)
        if not args.no_write:
            print(f"wrote {REPORT_PATH.relative_to(ROOT)}")
        return 0

    # Resolve the answerer / judge specs. Single-model fallback: --spec/$VERIFY_SPEC.
    base_spec = args.spec or os.environ.get("VERIFY_SPEC")
    answer_spec = args.answer_spec or base_spec
    judge_spec = args.judge_spec or base_spec
    if args.answer_spec and args.judge_spec and args.answer_spec == args.judge_spec:
        print("[error] --answer-spec and --judge-spec must DIFFER (answerer must not judge "
              "itself). Pass two distinct specs or use a single --spec.")
        return 2

    # Live path requires a relay; absent one, FAIL CLOSED with a status report.
    if not args.relay or not _relay_available(args.judge_spec or args.spec):
        report = build_report("relay", pack, None, status="relay_unavailable", runs=runs,
                              answer_spec=answer_spec, judge_spec=judge_spec,
                              retrieve=args.retrieve, verifier=args.verifier)
        report["reason"] = (
            "no relay/keys configured (set VERIFY_SPEC or an API key and pass --relay), "
            "or --relay not requested. Use --fake to exercise the harness in CI."
        )
        if not args.no_write:
            write_report(report)
        print("[relay_unavailable] no relay/keys; fail-closed. "
              "Use --fake for CI or set VERIFY_SPEC and pass --relay.")
        if not args.no_write:
            print(f"wrote {REPORT_PATH.relative_to(ROOT)}")
        return 0

    judge_entail = make_relay_entailment(judge_spec)

    def relay_entail_factory(case: dict[str, Any]):  # noqa: ARG001 — judge backend per case
        return judge_entail

    def complete_factory(case: dict[str, Any]):  # noqa: ARG001 — answerer backend per case
        from agent.model import complete  # noqa: PLC0415
        def C(system: str, user: str, *, max_tokens: int = 180) -> str:
            kwargs: dict[str, Any] = {"max_tokens": max_tokens}
            if answer_spec:
                kwargs["spec"] = answer_spec
            return complete(system, user, **kwargs) or ""
        return C

    # --retrieve uses the real (network) fetcher by default; independence is measured live.
    agg = run_multi(pack, relay_entail_factory, complete_factory, runs=runs,
                    retrieve=args.retrieve, verifier=args.verifier)
    report = build_report("relay", pack, agg, status="ok_relay", runs=runs,
                          answer_spec=answer_spec, judge_spec=judge_spec, retrieve=args.retrieve,
                          verifier=args.verifier)
    if not args.no_write:
        write_report(report)
    _print_multi("relay", agg)
    if not args.no_write:
        print(f"wrote {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
