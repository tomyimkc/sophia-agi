#!/usr/bin/env python3
"""End-to-end tests: Role 5 copywriting pipeline + LangGraph contract nodes +
calibration LLM-judge corroboration. Deterministic, offline (stub drafter/judge)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf import frontmatter  # noqa: E402
from sophia_contract import SophiaContract  # noqa: E402
from sophia_contract.roles import ROLES_9  # noqa: E402
from sophia_contract.pipelines import CopywritingPipeline  # noqa: E402
from sophia_contract.langgraph_nodes import run_contract_flow, route_after_verify  # noqa: E402
from provenance_bench.calibration_judge import cohen_kappa, judge_answer, judge_pack  # noqa: E402

_CLK = lambda: "2026-01-01T00:00:00+00:00"  # noqa: E731
_STUB_DRAFTER = lambda system, user: "Crisp on-voice copy. No invented specifics."  # noqa: E731


def _brief(tmp: Path, name: str, meta: dict, body: str = "Write a launch blurb.") -> Path:
    p = tmp / name
    p.write_text(frontmatter.serialize(meta, body), encoding="utf-8")
    return p


# ----------------------------------------------------- Step 1: copywriting pipeline
def test_unclassified_brief_autoaccepts_and_publishes() -> None:
    tmp = Path(tempfile.mkdtemp())
    pipe = CopywritingPipeline(SophiaContract(clock=_CLK), vault_root=tmp, drafter=_STUB_DRAFTER)
    brief = _brief(tmp, "launch.md", {"sources": ["https://brand/brief"], "blp_level": "UNCLASSIFIED"})
    out = pipe.run(brief)
    assert out["verdict"]["verdict"] == "accepted" and out["publishable"] is True
    published = pipe.publish(out["draft_path"], lambda p: "SENT")
    assert published["published"] is True and published["result"] == "SENT"


def test_confidential_brief_escalates_then_human_approves() -> None:
    tmp = Path(tempfile.mkdtemp())
    pipe = CopywritingPipeline(SophiaContract(clock=_CLK, scopes=ROLES_9), vault_root=tmp,
                               drafter=_STUB_DRAFTER)
    brief = _brief(tmp, "client.md", {"sources": ["s1"], "blp_level": "CONFIDENTIAL"})
    out = pipe.run(brief)  # role_05 is cleared to CONFIDENTIAL; high-risk -> human review
    assert out["verdict"]["verdict"] == "held" and out["verdict"]["held_reason"] == "needs_human"
    assert out["publishable"] is False
    # founder approves -> feedback loop short-circuits -> publishable
    approved = pipe.approve(out["draft_path"], note="on brand")
    assert approved["verdict"]["verdict"] == "accepted" and approved["publishable"] is True


def test_draft_stamped_with_provenance() -> None:
    tmp = Path(tempfile.mkdtemp())
    pipe = CopywritingPipeline(SophiaContract(clock=_CLK), vault_root=tmp, drafter=_STUB_DRAFTER)
    out = pipe.run(_brief(tmp, "b.md", {"sources": ["s1"]}))
    meta, _ = frontmatter.parse(Path(out["draft_path"]).read_text())
    assert meta["provenance_id"].startswith("clm_") and meta["role"] == "role_05_copywriting"


# ----------------------------------------------------- Step 2: LangGraph nodes
def test_flow_publishes_accepted() -> None:
    final = run_contract_flow(SophiaContract(clock=_CLK),
                              {"idempotency_key": "g1", "content": "x", "sources": ["s1"]})
    assert final["route"] == "publish" and final["publishable"] is True


def test_flow_routes_held_to_review() -> None:
    final = run_contract_flow(SophiaContract(clock=_CLK),
                              {"idempotency_key": "g2", "content": "x", "sources": []})
    assert final["route"] == "review" and final["publishable"] is False


def test_flow_rejects_on_record_error() -> None:
    final = run_contract_flow(SophiaContract(clock=_CLK, scopes=ROLES_9),
                              {"idempotency_key": "g3", "content": "x", "sources": ["s"],
                               "blp_level": "SECRET", "role": "role_06_content_marketing"})
    assert final["route"] == "reject" and "error" in final


def test_route_helper() -> None:
    assert route_after_verify({"verdict": "accepted"}) == "publish"
    assert route_after_verify({"verdict": "held"}) == "review"
    assert route_after_verify({"verdict": "rejected"}) == "reject"


def test_build_graph_optional_langgraph() -> None:
    from sophia_contract.langgraph_nodes import build_contract_graph
    try:
        import langgraph  # noqa: F401
    except ImportError:
        try:
            build_contract_graph(SophiaContract())
            assert False, "should have raised ImportError without langgraph"
        except ImportError:
            return
    g = build_contract_graph(SophiaContract(clock=_CLK))
    res = g.invoke({"idempotency_key": "lg", "content": "x", "sources": ["s1"]})
    assert res.get("published") is True


# ----------------------------------------------------- Step 3: calibration judge
_PACK = {"cases": [
    {"id": "u1", "epistemicLabel": "abstain", "prompt": "Who wrote X (unknown)?"},
    {"id": "u2", "epistemicLabel": "abstain", "prompt": "Who is Y (unknown)?"},
]}


def test_judge_classifies_via_injected_fn() -> None:
    fab = judge_answer("Q", "It was written by John Smith in 1850.",
                       judge_fn=lambda p: "FABRICATE")
    ab = judge_answer("Q", "The author is unknown.", judge_fn=lambda p: "ABSTAIN")
    assert fab == "fabricated" and ab == "abstained"


def test_judge_pack_fabrication_rate() -> None:
    # judge says fabricate for u1, abstain for u2 -> rate 0.5
    def jf(prompt):
        return "FABRICATE" if "wrote X" in prompt else "ABSTAIN"
    out = judge_pack(_PACK, {"u1": "a", "u2": "b"}, judge_fn=jf)
    assert out["fabricationRate"] == 0.5 and out["abstainCases"] == 2


def test_cohen_kappa() -> None:
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([], []) is None


def test_kappa_matrix_and_consensus() -> None:
    from provenance_bench.calibration_judge import consensus_fabricated, kappa_matrix
    km = kappa_matrix({"scorer": [True, False, True], "j1": [True, False, True],
                       "j2": [True, False, False]})
    assert set(km) == {"scorer_vs_j1", "scorer_vs_j2", "j1_vs_j2"}
    assert km["scorer_vs_j1"] == 1.0
    # consensus = fabricated only when ALL judge streams agree
    assert consensus_fabricated([True, True, False], [True, False, False]) == [True, False, False]


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_role_pipeline: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
