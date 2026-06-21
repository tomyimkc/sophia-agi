"""Multi-judge consensus — the no-overclaim gate.

A single LLM-judge is unreliable: in our own audit two judges agreed only ~76%
of the time and the single-judge hallucination rate was ~2x the consensus. So no
number should be published as a *validated* headline unless it survives a
majority vote of >=2 independent judges, with the inter-judge agreement reported
alongside.

``make_consensus_judge`` returns a ``JudgeFn`` that majority-votes N judges and
attaches the per-judge votes to the ``Judgment`` (so agreement can be computed).
``percent_agreement`` summarises how often the judges agreed.
"""

from __future__ import annotations

from provenance_bench.judge import JudgeFn, Judgment


def make_consensus_judge(specs: "list[str] | None" = None, *, judge_fns: "list | None" = None) -> JudgeFn:
    """Majority vote over N judges.

    ``specs`` builds independent LLM judges (each must differ from the model under
    test). ``judge_fns`` injects ready-made judges (used by tests). Requires >=2.
    """
    if judge_fns is None:
        if not specs or len(specs) < 2:
            raise ValueError("consensus needs >=2 judge specs (or injected judge_fns)")
        from provenance_bench.llm_judge import make_llm_judge

        judge_fns = [make_llm_judge(s) for s in specs]
        labels = list(specs)
    else:
        if len(judge_fns) < 2:
            raise ValueError("consensus needs >=2 judge_fns")
        labels = list(specs) if specs else [f"judge{i}" for i in range(len(judge_fns))]
    n = len(judge_fns)

    def judge(answer: str, case) -> Judgment:
        votes = [fn(answer, case) for fn in judge_fns]

        def majority(attr: str) -> bool:
            return sum(1 for v in votes if getattr(v, attr)) * 2 > n  # strict majority

        return Judgment(
            abstained=majority("abstained"),
            hallucinated=majority("hallucinated"),
            affirmed_gold=majority("affirmed_gold"),
            method=f"consensus:{n}",
            votes=[{"judge": labels[i], "hallucinated": bool(votes[i].hallucinated)} for i in range(n)],
        )

    return judge


def cohen_kappa(a: "list[int]", b: "list[int]") -> "float | None":
    """Cohen's kappa for two binary label sequences (chance-corrected agreement)."""
    n = len(a)
    if n == 0:
        return None
    po = sum(int(x == y) for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if pe >= 1.0:
        return 1.0  # both unanimous and identical
    return round((po - pe) / (1 - pe), 4)


def percent_agreement(vote_lists: "list[list[dict]]", key: str = "hallucinated") -> "dict | None":
    """Agreement on ``key`` across judges over many cases: raw pairwise % AND the
    chance-corrected mean pairwise Cohen's kappa (the honest stat for a rare class).

    ``vote_lists`` is the list of ``Judgment.votes`` arrays (one per judged item).
    """
    import itertools

    rows = [vl for vl in vote_lists if vl and len(vl) >= 2]
    if not rows:
        return None
    pair_total = pair_agree = 0
    for vl in rows:
        vals = [bool(v.get(key)) for v in vl]
        for i, j in itertools.combinations(range(len(vals)), 2):
            pair_total += 1
            pair_agree += int(vals[i] == vals[j])

    # per-judge label sequences for kappa (only judges present in every row)
    judges = [v["judge"] for v in rows[0]]
    seqs: dict = {j: [] for j in judges}
    usable = True
    for vl in rows:
        labels = {v["judge"]: int(bool(v.get(key))) for v in vl}
        if set(labels) != set(judges):
            usable = False
            break
        for j in judges:
            seqs[j].append(labels[j])
    kappas = []
    if usable:
        for j1, j2 in itertools.combinations(judges, 2):
            k = cohen_kappa(seqs[j1], seqs[j2])
            if k is not None:
                kappas.append(k)
    return {
        "items": len(rows),
        "judges": len(rows[0]),
        "pairwiseAgreement": round(pair_agree / pair_total, 4) if pair_total else None,
        "meanPairwiseKappa": round(sum(kappas) / len(kappas), 4) if kappas else None,
    }
