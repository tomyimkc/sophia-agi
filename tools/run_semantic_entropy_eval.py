#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from agent.semantic_entropy import semantic_entropy
from agent.semantic_entropy_probe import write_probe_report
DEFAULT_OUT=ROOT/"agi-proof"/"conscience"/"semantic-entropy.public-report.json"
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=str(DEFAULT_OUT)); args=ap.parse_args()
    stable=["Jane Austen wrote Pride and Prejudice.","Pride and Prejudice was authored by Jane Austen.","Jane Austen wrote Pride and Prejudice."]
    unstable=["Jane Austen wrote Pride and Prejudice.","Douglas Adams wrote the book.","I do not know who wrote it."]
    report={"schema":"sophia.semantic_entropy_eval.v1","candidateOnly":True,"level3Evidence":False,"stable":semantic_entropy(stable),"unstable":semantic_entropy(unstable)}
    report["ok"]=report["unstable"]["entropy"] > report["stable"]["entropy"]
    p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    write_probe_report([stable[0], unstable[-1], "This is maybe unknown and unclear without evidence."], p.with_name("semantic-entropy-probe.public-report.json"))
    print(json.dumps({"ok":report["ok"],"out":args.out},indent=2)); return 0 if report["ok"] else 1
if __name__=="__main__": raise SystemExit(main())
