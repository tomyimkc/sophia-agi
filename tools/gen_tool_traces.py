#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from agent.tool_use.verifier import trace_passes, verify_trace
from provenance_bench.dataset import build_cases
from provenance_bench.dataset_guard import normalize, tool_use_benchmark_prompt_set
from provenance_bench.local_agent import TOOL_SYSTEM, dispatch_tool, tool_loop, ScriptedClient

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",default="mock"); ap.add_argument("--max-traces",type=int,default=80)
    ap.add_argument("--out",type=Path,default=ROOT/"training/tool_use/sft_traces.jsonl"); args=ap.parse_args()
    pool=[c for c in build_cases() if normalize(c.prompt) not in tool_use_benchmark_prompt_set(root=ROOT)]
    client=ScriptedClient(pool); kept=[]; dropped={"verify_fail":0}
    for case in pool:
        if len(kept)>=args.max_traces: break
        text,log=tool_loop(client,case); tcs=[]; trs=[]; turns=[{"role":"user","content":case.prompt}]
        label=({"decision":"call","tool_id":"check_claim","gold_answer":f"No, {case.claimed_author} did not write {case.work}; it is by {case.gold_author}."} if case.label=="false" else {"decision":"answer_direct","tool_id":None,"gold_answer":f"{case.work} was written by {case.gold_author}."})
        if log:
            claim=f"{case.claimed_author} wrote {case.work}" if case.claimed_author else case.prompt[:80]
            tc={"name":"check_claim","arguments":{"text":claim}}; tcs=[tc]; out=dispatch_tool("check_claim",{"text":claim})
            trs=[{"name":"check_claim","args":{"text":claim},"result":out}]; turns.append({"role":"assistant","tool_calls":[tc]})
        turns.append({"role":"assistant","content":text,"final":True})
        if case.label=="true" and tcs: dropped["verify_fail"]+=1; continue
        if not trace_passes(verify_trace(answer=text,tool_calls=tcs,label=label,tool_results=trs,trace_turns=turns)): dropped["verify_fail"]+=1; continue
        msgs=[{"role":"system","content":TOOL_SYSTEM},{"role":"user","content":case.prompt}]
        if tcs: msgs+=[{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"check_claim","arguments":json.dumps(tcs[0]["arguments"])}}]},{"role":"tool","tool_call_id":"c1","content":json.dumps(trs[0]["result"],ensure_ascii=False)}]
        msgs.append({"role":"assistant","content":text}); kept.append({"messages":msgs,"metadata":{"caseId":case.id,"verified":True,"mode":args.mode}})
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in kept),encoding="utf-8")
    print(json.dumps({"kept":len(kept),"dropped":dropped},indent=2)); return 0 if kept else 1
if __name__=="__main__": raise SystemExit(main())
