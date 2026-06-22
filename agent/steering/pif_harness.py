"""C3 — PIF/SSA headline harness (pure stdlib). The build_cells_from_scores seam
takes pre-computed per-seed score arrays → fully CI-testable, no model. A near-null
headline confirming Spec B 0/2 is the expected, pre-registered result."""
from __future__ import annotations

import statistics

from agent.steering.stats import (bootstrap_diff_ci, bootstrap_diff_p, benjamini_hochberg,
                                   cohen_d, residualized_d, ssa_verdict)


def build_cells_from_scores(scores: dict, grid: "list[dict]") -> "list[dict]":
    cells = []
    for g in grid:
        cid, tgt = g["cell_id"], g["target_axis"]
        s = scores[cid]
        steer_t, base_t = s[tgt]["steer"], s[tgt]["base"]
        seed = g.get("seed", 0)
        delta_ci = bootstrap_diff_ci(steer_t, base_t, seed=seed)
        delta_point = statistics.fmean([a - b for a, b in zip(steer_t, base_t)]) if steer_t else 0.0
        p_raw = bootstrap_diff_p(steer_t, base_t, seed=seed)
        steered_d = abs(residualized_d(steer_t, {ax: s[ax]["steer"] for ax in g["off_target_axes"]}))
        off_target_d = {ax: cohen_d(s[ax]["steer"], s[ax]["neutral"]) for ax in g["off_target_axes"]}
        cell = {"cell_id": cid, "delta_ci": delta_ci, "delta_point": delta_point,
                "steered_d": steered_d, "off_target_d": off_target_d, "kappa": s["kappa"],
                "capability_drop": s["capability_drop"], "coherence": s["coherence"],
                "is_mock": g["is_mock"], "p_raw": p_raw}
        cell["verdict"] = ssa_verdict({k: cell[k] for k in (
            "delta_ci", "delta_point", "steered_d", "off_target_d",
            "kappa", "capability_drop", "coherence", "is_mock")})
        cells.append(cell)
    return cells


def headline(cells: "list[dict]", *, q: float = 0.05) -> dict:
    """A cell counts 'enacted' only if ssa_verdict=='enacted' AND survives BH at q."""
    pvals = [c.get("p_raw", 1.0) for c in cells]
    sig = benjamini_hochberg(pvals, q)
    enacted = sum(1 for c, ok in zip(cells, sig)
                  if c["verdict"]["status"] == "enacted" and ok)
    total = len(cells)
    return {"enacted": enacted, "total": total,
            "enacted_over_total": f"{enacted}/{total}",
            "bh_significant": sig}
