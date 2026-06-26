#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from provenance_bench.dataset import build_cases
from provenance_bench.tool_use_benchmark import balance_counts, content_hash
TARGET=40; UNANS=[f"Who wrote unknown lost text #{i}? Answer briefly." for i in range(10)]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--check",action="store_true"); args=ap.parse_args()
    if args.check:
        from provenance_bench.tool_use_benchmark import verify_manifest
        r=verify_manifest(root=ROOT); print(json.dumps(r,indent=2)); return 0 if r["ok"] and r["nCases"]>=120 else 1
    all_c=build_cases(); false=sorted([c for c in all_c if c.label=="false"],key=lambda c:c.id)
    true=sorted([c for c in all_c if c.label=="true"],key=lambda c:c.id); rows=[]
    for i,c in enumerate(false[:TARGET]):
        rows.append({"id":f"should-call-{i:03d}","prompt":c.prompt,"registry":["check_claim","wiki_search","belief"],
            "label":{"decision":"call","tool_id":"check_claim","gold_answer":f"No, {c.claimed_author} did not write {c.work}; it is by {c.gold_author}.","gold_citation":None},
            "trap":"none","wouldHelp":True,"sourceCaseId":c.id})
    for i,c in enumerate(true[:TARGET]):
        rows.append({"id":f"shouldnt-call-{i:03d}","prompt":c.prompt,"registry":["check_claim","wiki_search","belief"],
            "label":{"decision":"answer_direct","tool_id":None,"gold_answer":f"{c.work} was written by {c.gold_author}.","gold_citation":None},
            "trap":"already_known","wouldHelp":False,"sourceCaseId":c.id})
    specs=[("wrong_tool_output",c.id) for c in false[TARGET:TARGET+10]]+[("empty_tool_output",c.id) for c in false[TARGET+10:TARGET+20]]+[("unanswerable",f"u{i}") for i in range(10)]
    while len(specs)<TARGET: specs.append(("unanswerable","u0"))
    specs=specs[:TARGET]; by_id={c.id:c for c in all_c}
    for i,(tt,sid) in enumerate(specs):
        if sid.startswith("u"):
            rows.append({"id":f"trap-{i:03d}","prompt":UNANS[int(sid[1:])],"registry":["check_claim","wiki_search","belief"],
                "label":{"decision":"abstain","tool_id":None,"gold_answer":None,"gold_citation":None},"trap":"unanswerable","wouldHelp":False,"sourceCaseId":sid})
        else:
            c=by_id[sid]; rows.append({"id":f"trap-{i:03d}","prompt":c.prompt,"registry":["check_claim","wiki_search","belief"],
                "label":{"decision":"abstain","tool_id":"check_claim","gold_answer":None,"gold_citation":None},"trap":tt,"wouldHelp":True,"sourceCaseId":sid})
    out=ROOT/"data/tool_use_benchmark"; out.mkdir(parents=True,exist_ok=True)
    (out/"heldout_v1.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    m={"schema":"sophia.tool_use_benchmark.v1","version":"heldout_v1","nCases":len(rows),"contentHash":content_hash(rows),
       "balance":balance_counts(rows),"candidateOnly":True,"canClaimAGI":False,"sealed":True}
    (out/"manifest.json").write_text(json.dumps(m,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"nCases":len(rows),"balance":m["balance"],"contentHash":m["contentHash"]},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
