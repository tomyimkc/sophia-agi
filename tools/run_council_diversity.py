"""C1 driver — personality-diverse council A/B → ΔQ. NULL ΔQ is a legitimate result."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "agi-proof" / "benchmark-results" / "council-diversity.public-report.json"
# 4 distinct OCEAN profiles spanning poles (pre-registered).
PROFILES = [("O+C-", {"O": 0.9, "C": 0.1}), ("O-C+", {"O": 0.1, "C": 0.9}),
            ("E+A-", {"E": 0.9, "A": 0.1}), ("E-A+", {"E": 0.1, "A": 0.9})]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mock", help='council base client spec (mock|ollama:qwen2.5:3b|…)')
    ap.add_argument("--domain", default="philosophy")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    from agent.model import default_client
    from agent.council_personas import council_diversity
    client = default_client(args.model)
    res = council_diversity(args.domain, client=client, profiles=PROFILES)
    res["model"] = args.model
    res["headline"] = ("PASS: diversity helps" if res["dq"] > 0 and res["dq_ci"][0] > 0
                       else "NULL: trait diversity does not improve council quality on this slice")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"[{args.domain}] single={res['single']['passrate']:.3f} "
          f"homo={res['homogeneous']['passrate']:.3f} diverse={res['diverse']['passrate']:.3f} "
          f"ΔQ={res['dq']} CI={res['dq_ci']} → {res['headline']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
