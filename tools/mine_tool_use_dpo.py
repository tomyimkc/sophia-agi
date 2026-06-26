#!/usr/bin/env python3
import argparse, json, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from provenance_bench.dataset import build_cases
from provenance_bench.dataset_guard import normalize, tool_use_benchmark_prompt_set

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=ROOT/"training/tool_use/dpo_pairs.jsonl"); ap.add_argument("--max-pairs",type=int,default=600); args=ap.parse_args()
    pool=[c for c in build_cases() if normalize(c.prompt) not in tool_use_benchmark_prompt_set(root=ROOT)]
    random.Random(42).shuffle(pool); train=pool[max(1,int(len(pool)*0.15)):]
    pairs=[]; counts={}
    for case in train:
        if len(pairs)>=args.max_pairs: break
        direct=(f"{case.work} was written by {case.gold_author}." if case.label=="true" else f"No, {case.claimed_author} did not write {case.work}; it is by {case.gold_author}.")
        for rt,rej in [("over_call",f"[needless call] {direct}"),("ignored_error","parrot error"),("spurious_extra",f"{direct}\n[extra]"),("schema_invalid",'{"bad":1}')]:
            pairs.append({"prompt":case.prompt,"chosen":direct,"rejected":rej,"metadata":{"rejected_type":rt,"caseId":case.id}}); counts[rt]=counts.get(rt,0)+1
        if case.label=="false":
            for rt,rej in [("mis_ground",f"Yes, {case.claimed_author} wrote {case.work}."),("wrong_tool",f"Wiki: {case.claimed_author}")]:
                pairs.append({"prompt":case.prompt,"chosen":direct,"rejected":rej,"metadata":{"rejected_type":rt,"caseId":case.id}}); counts[rt]=counts.get(rt,0)+1
    pairs=pairs[:args.max_pairs]; args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text("".join(json.dumps(p,ensure_ascii=False)+"\n" for p in pairs),encoding="utf-8")
    print(json.dumps({"pairs":len(pairs),"byRejectedType":counts},indent=2)); return 0 if pairs else 1
if __name__=="__main__": raise SystemExit(main())
