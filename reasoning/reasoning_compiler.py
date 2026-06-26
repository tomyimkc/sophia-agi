#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""A reasoning *compiler*: lower a belief graph through optimization passes — and prove
the passes are semantics-preserving. An offline, falsifiable test of thesis feature #3.

Thesis under test (docs/06-Roadmap/Reasoning-As-Compute.md, feature #3):

  "Treat a goal as something you LOWER into a typed plan-IR, run PASSES over, then emit.
   Sophia's belief/justification graph IS a graph IR, so the passes write themselves:
   CSE = claim dedup, dead-code elimination = prune unsupported branches, type-checking =
   provenance/consistency verification before emit. As in a real compiler, optimization
   must PRESERVE SEMANTICS — the grounded conclusion is invariant."

The IR mirrors okf/graph.py exactly: a node is a claim with an ``author_confidence`` rank
and ``derives_from`` edges; effective confidence is the **min over the derivesFrom chain**
(the weakest-link rule `okf.graph.propagate_confidence` uses). A claim is *grounded* iff its
effective confidence is > 0 and its chain roots in a source (no dangling link drops it to 0).

Passes (each a classic compiler analogue):
  * canonicalize / CSE     — merge claims with the same normalized statement (verify once)
  * dead-code elimination  — keep only the live cone backward-reachable from the goal
  * type-check             — contradictions (X and ¬X both live) + confidence-laundering
  * confidence propagation — min-over-chain, identical to okf.graph

The central, falsifiable claim is the compiler-correctness property:

  H1  cost down:        CSE + DCE reduce verification cost (# distinct live claims) ...
  H2  ... semantics preserved: ... while the goal's effective confidence AND grounded
                        status are INVARIANT under the passes (optimization changes cost,
                        never output).
  H3  fail-closed:      when a contradiction is in the goal's live cone, the compiler
                        REFUSES to emit (never reasons on a contradicted premise); on clean
                        graphs it raises zero false contradictions.

Pure stdlib, seeded, offline — no GPU/keys. ``run_experiment`` builds many synthetic graphs
with planted ground truth (duplicates, dead branches, contradictions, laundered claims) and
checks the hypotheses against that ground truth.

    python reasoning/reasoning_compiler.py --run        # experiment + verdict
    python reasoning/reasoning_compiler.py --self-test   # assert the invariants
    python reasoning/reasoning_compiler.py --run --json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field


# --------------------------------------------------------------------------------------
# IR
# --------------------------------------------------------------------------------------
@dataclass
class Claim:
    cid: str
    statement: str
    kind: str                       # 'source' | 'derived' | 'goal'
    author_confidence: int          # rank 0..3 (mirrors okf confidence_rank)
    derives_from: list[str] = field(default_factory=list)


@dataclass
class ReasoningGraph:
    claims: dict[str, Claim]
    goal: str

    def copy(self) -> "ReasoningGraph":
        return ReasoningGraph({c.cid: Claim(c.cid, c.statement, c.kind, c.author_confidence,
                                            list(c.derives_from)) for c in self.claims.values()},
                              self.goal)


def _norm(stmt: str) -> str:
    return " ".join(stmt.strip().lower().split())


# --------------------------------------------------------------------------------------
# Semantics — min-over-derivesFrom-chain, identical to okf.graph.propagate_confidence.
# --------------------------------------------------------------------------------------
def effective_confidence(g: ReasoningGraph, cid: str) -> int:
    memo: dict[str, int] = {}

    def eff(nid: str, stack: frozenset[str]) -> int:
        if nid in memo:
            return memo[nid]
        node = g.claims.get(nid)
        if node is None:
            return 0  # dangling reference -> weakest link drops the chain to 0
        best = node.author_confidence
        for dep in node.derives_from:
            if dep in stack:
                continue
            best = min(best, eff(dep, stack | {nid}))
        memo[nid] = best
        return best

    return eff(cid, frozenset())


def is_grounded(g: ReasoningGraph, cid: str) -> bool:
    """Effective confidence > 0 AND the chain actually roots in a source claim."""
    if effective_confidence(g, cid) <= 0:
        return False
    seen: set[str] = set()

    def roots_in_source(nid: str) -> bool:
        if nid in seen:
            return False
        seen.add(nid)
        node = g.claims.get(nid)
        if node is None:
            return False
        if node.kind == "source":
            return True
        return any(roots_in_source(d) for d in node.derives_from)

    return roots_in_source(cid)


def live_cone(g: ReasoningGraph) -> set[str]:
    """Claims backward-reachable from the goal (the live set; everything else is dead)."""
    live: set[str] = set()
    stack = [g.goal]
    while stack:
        nid = stack.pop()
        if nid in live or nid not in g.claims:
            continue
        live.add(nid)
        stack.extend(g.claims[nid].derives_from)
    return live


# --------------------------------------------------------------------------------------
# Passes
# --------------------------------------------------------------------------------------
def pass_canonicalize(g: ReasoningGraph) -> ReasoningGraph:
    """CSE: merge claims sharing a normalized statement into one canonical id, rewiring
    edges. Picks the lexicographically-smallest id as canonical for determinism."""
    by_norm: dict[str, list[str]] = {}
    for c in g.claims.values():
        by_norm.setdefault(_norm(c.statement), []).append(c.cid)
    remap: dict[str, str] = {}
    for ids in by_norm.values():
        canon = sorted(ids)[0]
        for cid in ids:
            remap[cid] = canon
    new_claims: dict[str, Claim] = {}
    for c in g.claims.values():
        canon = remap[c.cid]
        if canon not in new_claims:
            new_claims[canon] = Claim(
                canon, c.statement if c.cid == canon else g.claims[canon].statement,
                g.claims[canon].kind, g.claims[canon].author_confidence, [])
        # rewire edges to canonical targets, dedup, drop self-loops
        tgt = new_claims[canon]
        for dep in g.claims[canon].derives_from:
            rd = remap.get(dep, dep)
            if rd != canon and rd not in tgt.derives_from:
                tgt.derives_from.append(rd)
    return ReasoningGraph(new_claims, remap[g.goal])


def pass_dead_code_elimination(g: ReasoningGraph) -> ReasoningGraph:
    live = live_cone(g)
    return ReasoningGraph({cid: g.claims[cid] for cid in live if cid in g.claims}, g.goal)


def pass_type_check(g: ReasoningGraph) -> dict:
    """Find contradictions (X and ¬X both live) and confidence-laundering in the live cone."""
    live = live_cone(g)
    norms = {cid: _norm(g.claims[cid].statement) for cid in live}
    present = set(norms.values())
    contradictions: list[tuple[str, str]] = []
    for cid, s in norms.items():
        neg = s[4:] if s.startswith("not ") else "not " + s
        if neg in present:
            a, b = sorted((s, neg))
            if (a, b) not in [tuple(sorted(p)) for p in contradictions]:
                contradictions.append((a, b))
    laundered: list[str] = []
    for cid in live:
        node = g.claims[cid]
        if node.kind != "source" and node.author_confidence > effective_confidence(g, cid):
            laundered.append(cid)
    return {"contradictions": contradictions, "laundered": sorted(laundered)}


# --------------------------------------------------------------------------------------
# Compile
# --------------------------------------------------------------------------------------
@dataclass
class CompileResult:
    cost_before: int
    cost_after: int
    cost_reduction: float
    goal_confidence_before: int
    goal_confidence_after: int
    goal_grounded_before: bool
    goal_grounded_after: bool
    semantics_preserved: bool
    contradictions: list
    laundered: list
    emittable: bool                 # fail-closed: False if a contradiction taints the goal

    def to_dict(self) -> dict:
        return asdict(self)


def compile_graph(g: ReasoningGraph) -> CompileResult:
    conf_before = effective_confidence(g, g.goal)
    grounded_before = is_grounded(g, g.goal)
    cost_before = len(g.claims)  # naive: verify every claim

    opt = pass_dead_code_elimination(pass_canonicalize(g))
    diags = pass_type_check(opt)

    conf_after = effective_confidence(opt, opt.goal)
    grounded_after = is_grounded(opt, opt.goal)
    cost_after = len(opt.claims)  # verify only distinct live claims

    semantics = (conf_before == conf_after) and (grounded_before == grounded_after)
    emittable = grounded_after and not diags["contradictions"]

    result = CompileResult(
        cost_before=cost_before,
        cost_after=cost_after,
        cost_reduction=(cost_before - cost_after) / cost_before if cost_before else 0.0,
        goal_confidence_before=conf_before,
        goal_confidence_after=conf_after,
        goal_grounded_before=grounded_before,
        goal_grounded_after=grounded_after,
        semantics_preserved=semantics,
        contradictions=diags["contradictions"],
        laundered=diags["laundered"],
        emittable=emittable,
    )

    # Verified-trace hook (observer-only): emit one fact+logic-stamped trace per
    # compile. The logic stamp IS the compiler's own type-check (emittable /
    # contradictions / laundered / semanticsPreserved); the fact stamp reuses the
    # grounded-conclusion provenance verdict. A logger fault can never break a
    # compile (``emit`` swallows exceptions per the repo's audit convention), and
    # the compiler's fail-closed behaviour is untouched.
    try:
        from agent.verified_trace import VerifiedTrace, emit, _trace_id
        emit(VerifiedTrace(
            traceId=_trace_id(f"reasoning_compiler:{id(g)}:{g.goal}"),
            runId="reasoning_compiler",
            phase="benchmark",
            stepIdx=0,
            claimText=g.claims[g.goal].statement,
            claimKind="goal",
            fact={
                "verdict": "allow" if grounded_after else "abstain",
                "source": "propagate_confidence",
                "authorConfidence": "compiled",
                "effectiveConfidenceRank": conf_after,
                "sources": [],
            },
            logic={
                "emittable": emittable,
                "contradictions": diags["contradictions"],
                "laundered": diags["laundered"],
                "semanticsPreserved": semantics,
            },
        ))
    except Exception:  # noqa: BLE001 - observer-only: never break a compile
        pass

    return result


# --------------------------------------------------------------------------------------
# Synthetic graphs with planted ground truth.
# --------------------------------------------------------------------------------------
@dataclass
class GroundTruth:
    live_statements: set      # normalized statements that are actually live (for DCE check)
    has_live_contradiction: bool


def make_graph(rng: random.Random, *, plant_contradiction: bool) -> tuple[ReasoningGraph, GroundTruth]:
    claims: dict[str, Claim] = {}
    n = 0

    def add(stmt: str, kind: str, conf: int, deps: list[str]) -> str:
        nonlocal n
        cid = f"c{n}"
        n += 1
        claims[cid] = Claim(cid, stmt, kind, conf, list(deps))
        return cid

    # A small grounded chain: 2-3 sources -> 2-3 derived -> goal.
    n_src = rng.randint(2, 3)
    sources = [add(f"src fact {i}", "source", rng.randint(2, 3), []) for i in range(n_src)]
    mids = []
    for j in range(rng.randint(2, 3)):
        deps = rng.sample(sources, k=rng.randint(1, len(sources)))
        mids.append(add(f"derived step {j}", "derived", rng.randint(2, 3), deps))
    goal_deps = rng.sample(mids, k=rng.randint(1, len(mids)))
    goal = add("conclusion", "goal", 3, goal_deps)

    # Plant CSE targets: duplicate some live claims (identical statement + confidence).
    for _ in range(rng.randint(1, 3)):
        victim = claims[rng.choice(mids + sources)]
        add(victim.statement, victim.kind, victim.author_confidence, list(victim.derives_from))

    # Plant dead code: claims not reachable from the goal.
    for _ in range(rng.randint(2, 4)):
        add(f"dead claim {rng.random():.5f}", "derived", rng.randint(1, 3),
            rng.sample(sources, k=1))

    g = ReasoningGraph(claims, goal)
    live = live_cone(g)
    live_statements = {_norm(claims[cid].statement) for cid in live}

    has_contra = False
    if plant_contradiction:
        # Negate a claim the goal GENUINELY depends on (a live mid), so both X and ¬X are
        # in the goal's live cone — a real live contradiction, not a dangling one.
        base = claims[rng.choice(goal_deps)]
        neg = add("not " + base.statement, "derived", rng.randint(2, 3),
                  rng.sample(sources, k=1))
        claims[goal].derives_from.append(neg)
        g = ReasoningGraph(claims, goal)
        live = live_cone(g)
        live_statements = {_norm(claims[cid].statement) for cid in live}
        has_contra = True

    return g, GroundTruth(live_statements, has_contra)


# --------------------------------------------------------------------------------------
# Experiment.
# --------------------------------------------------------------------------------------
def run_experiment(graphs: int = 400, seed: int = 2026, contradiction_frac: float = 0.5) -> dict:
    rng = random.Random(seed)
    n_contra = int(graphs * contradiction_frac)
    plan = [True] * n_contra + [False] * (graphs - n_contra)
    rng.shuffle(plan)

    cost_reductions = []
    semantics_ok = 0
    dce_exact = 0                  # DCE kept exactly the live ground-truth statements
    contra_detected = 0           # graphs with planted live contradiction -> caught
    contra_total = 0
    false_contra = 0              # clean graphs flagged as contradictory (must be 0)
    clean_total = 0
    failclosed_ok = 0             # contradicted graphs -> emittable == False

    for plant in plan:
        g, gt = make_graph(rng, plant_contradiction=plant)
        res = compile_graph(g)
        cost_reductions.append(res.cost_reduction)
        if res.semantics_preserved:
            semantics_ok += 1
        # DCE correctness: optimized live statements == ground-truth live statements.
        opt = pass_dead_code_elimination(pass_canonicalize(g))
        opt_statements = {_norm(opt.claims[c].statement) for c in opt.claims}
        if opt_statements == gt.live_statements:
            dce_exact += 1
        if gt.has_live_contradiction:
            contra_total += 1
            if res.contradictions:
                contra_detected += 1
            if not res.emittable:
                failclosed_ok += 1
        else:
            clean_total += 1
            if res.contradictions:
                false_contra += 1

    mean_cost_red = sum(cost_reductions) / len(cost_reductions)
    return {
        "graphs": graphs,
        "seed": seed,
        "mean_cost_reduction": mean_cost_red,
        "semantics_preserved_rate": semantics_ok / graphs,
        "dce_exact_rate": dce_exact / graphs,
        "contradiction_recall": (contra_detected / contra_total) if contra_total else None,
        "failclosed_rate": (failclosed_ok / contra_total) if contra_total else None,
        "false_contradiction_rate": (false_contra / clean_total) if clean_total else None,
        "contra_total": contra_total,
        "clean_total": clean_total,
    }


def format_report(r: dict) -> str:
    L = [
        f"Reasoning-compiler experiment  (graphs={r['graphs']}, seed={r['seed']})",
        "Each graph: a grounded source->derived->goal chain with planted duplicates (CSE),",
        "dead branches (DCE), and — in half — a live contradiction (¬X wired into the goal).\n",
        f"  mean verification-cost reduction (CSE+DCE):  {r['mean_cost_reduction']:.1%}",
        f"  semantics preserved (goal conf + grounded):  {r['semantics_preserved_rate']:.1%}",
        f"  DCE kept exactly the live ground-truth set:  {r['dce_exact_rate']:.1%}",
        f"  contradiction recall (caught in live cone):  {r['contradiction_recall']:.1%}"
        f"   over {r['contra_total']} planted",
        f"  fail-closed on contradicted goals:           {r['failclosed_rate']:.1%}",
        f"  FALSE contradictions on clean graphs:        {r['false_contradiction_rate']:.1%}"
        f"   over {r['clean_total']} clean",
        "",
        "THEORY VERDICT",
        f"  H1 cost down:            {r['mean_cost_reduction']:.1%} fewer claims to verify  -> "
        f"{'CONFIRMED' if r['mean_cost_reduction'] > 0 else 'REFUTED'}",
        f"  H2 semantics preserved:  {r['semantics_preserved_rate']:.0%} invariant goal  -> "
        f"{'CONFIRMED' if r['semantics_preserved_rate'] > 0.999 else 'REFUTED'}",
        f"  H3 fail-closed + clean:  {r['failclosed_rate']:.0%} refuse-on-contradiction, "
        f"{r['false_contradiction_rate']:.0%} false alarms  -> "
        f"{'CONFIRMED' if r['failclosed_rate'] > 0.999 and r['false_contradiction_rate'] < 1e-9 else 'REFUTED'}",
    ]
    return "\n".join(L)


# --------------------------------------------------------------------------------------
# CLI + self-test
# --------------------------------------------------------------------------------------
def _self_test() -> int:
    # Hand-built graph: a goal grounded via one chain, with a duplicate + a dead claim.
    claims = {
        "s0": Claim("s0", "fact A", "source", 3, []),
        "s1": Claim("s1", "fact B", "source", 2, []),
        "d0": Claim("d0", "step P", "derived", 3, ["s0", "s1"]),
        "d0dup": Claim("d0dup", "step P", "derived", 3, ["s0", "s1"]),  # CSE target
        "dead": Claim("dead", "irrelevant", "derived", 1, ["s0"]),     # DCE target
        "g": Claim("g", "conclusion", "goal", 3, ["d0", "d0dup"]),
    }
    g = ReasoningGraph(claims, "g")
    res = compile_graph(g)
    # weakest link: goal -> step P -> min(3, fact A=3, fact B=2) = 2
    assert res.goal_confidence_before == 2, res.goal_confidence_before
    assert res.semantics_preserved, res
    assert res.cost_after < res.cost_before, (res.cost_before, res.cost_after)
    assert res.goal_grounded_after and res.emittable, res
    # 'dead' and one duplicate of step P must be gone from the optimized graph.
    opt = pass_dead_code_elimination(pass_canonicalize(g))
    stmts = sorted(_norm(opt.claims[c].statement) for c in opt.claims)
    assert "irrelevant" not in stmts, stmts
    assert stmts.count("step p") == 1, stmts

    # Contradiction must be caught and block emission.
    claims2 = dict(g.claims)
    claims2["neg"] = Claim("neg", "not step P", "derived", 3, ["s0"])
    claims2["g"] = Claim("g", "conclusion", "goal", 3, ["d0", "neg"])
    res2 = compile_graph(ReasoningGraph(claims2, "g"))
    assert res2.contradictions, "contradiction missed"
    assert not res2.emittable, "must fail closed on a contradicted goal"

    # Aggregate experiment invariants.
    r = run_experiment(graphs=120, seed=1)
    assert r["semantics_preserved_rate"] > 0.999, r
    assert r["mean_cost_reduction"] > 0.0, r
    assert r["failclosed_rate"] == 1.0, r
    assert r["false_contradiction_rate"] == 0.0, r
    print(f"self-test OK: cost-red={r['mean_cost_reduction']:.1%}, "
          f"semantics={r['semantics_preserved_rate']:.0%}, "
          f"fail-closed={r['failclosed_rate']:.0%}, false-contra={r['false_contradiction_rate']:.0%}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true", help="run the experiment + verdict")
    p.add_argument("--self-test", action="store_true", help="assert the invariants and exit")
    p.add_argument("--json", action="store_true", help="emit raw results as JSON")
    p.add_argument("--graphs", type=int, default=400)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args(argv)
    if args.self_test:
        return _self_test()
    if args.run or args.json:
        r = run_experiment(graphs=args.graphs, seed=args.seed)
        print(json.dumps(r, indent=2) if args.json else format_report(r))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
