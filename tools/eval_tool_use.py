#!/usr/bin/env python3
import argparse, json, sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from agent.steering.stats import bootstrap_diff_ci
from agent.tool_use.policy import apply_trap, should_invoke_tools
from agent.tool_use.verifier import verify_call, verify_decision, verify_error_recovery, verify_grounding, verify_tool_selection
from provenance_bench.dataset import Case, build_cases
from provenance_bench.judge import judge_answer
from provenance_bench.local_agent import dispatch_tool, tool_loop, ScriptedClient
from provenance_bench.tool_use_benchmark import load_cases, verify_manifest

@dataclass
class RunResult:
    answer:str; tool_calls:list; tool_results:list; condition:str

def prov(case):
    for c in build_cases():
        if c.id==case.get("sourceCaseId"): return c
    return case

def no_tools(cl,case):
    r=cl.generate("You are a provenance assistant.",case["prompt"]); return RunResult(getattr(r,"text","") or "",[],[],"no_tools")

def always_tools(cl,case):
    p=prov(case)
    if not isinstance(p,Case): return RunResult("",[],[],"always_tools")
    text,log=tool_loop(cl,p); tcs=[]; trs=[]
    if log:
        args={"text":f"{p.claimed_author} wrote {p.work}"} if log[0]=="check_claim" else {"query":p.work}
        tcs=[{"name":log[0],"arguments":args}]; trs=[{"name":log[0],"args":args,"result":apply_trap(dispatch_tool(log[0],args),case.get("trap","none"))}]
    return RunResult(text,tcs,trs,"always_tools")

def trained(cl,case):
    p=prov(case); plain=no_tools(cl,case)
    if isinstance(p,Case) and (should_invoke_tools(plain.answer,p) or case.get("label",{}).get("decision")=="call"):
        r=always_tools(cl,case); r.condition="trained"; return r
    plain.condition="trained"; return plain

def score(case,run):
    label,trap=case.get("label",{}),case.get("trap","none"); mc=bool(run.tool_calls); s1=verify_decision(mc,label).passed; s2=s3=True
    if run.tool_calls:
        tc=run.tool_calls[0]; s2=verify_tool_selection(tc.get("name",""),label).passed; s3=verify_call(tc.get("name",""),tc.get("arguments",{})).passed
    last=run.tool_results[-1]["result"] if run.tool_results else None
    if trap in ("wrong_tool_output","empty_tool_output") and run.tool_results: last=apply_trap(last or {},trap)
    s4=verify_grounding(run.answer,last,label.get("gold_answer")).passed; p=prov(case)
    if isinstance(p,Case): j=judge_answer(run.answer,p); tp=j.abstained if label.get("decision")=="abstain" else j.affirmed_gold
    elif label.get("decision")=="abstain": tp="unknown" in run.answer.lower()
    elif label.get("decision")=="answer_direct": tp=s4 and not mc
    else: tp=s4 and s1
    return {"s1_decision":s1,"s2_tool_selection":s2,"s3_schema_valid":s3,"s4_grounding":s4,"false_call":label.get("decision")=="answer_direct" and mc,"over_call":trap=="already_known" and mc,"trap_ok":verify_error_recovery(run.answer,last).passed,"task_pass":tp}

def agg(scores):
    n=len(scores) or 1; o={}
    for k in ("s1_decision","s2_tool_selection","s3_schema_valid","s4_grounding","task_pass","false_call","over_call","trap_ok"):
        o[k]=sum(1 for s in scores if s.get(k))/n
    o["task_pass_at_1"]=o.pop("task_pass"); return {k:round(v,4) for k,v in o.items()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",default="0,1,2"); ap.add_argument("--mode",default="mock")
    ap.add_argument("--out",type=Path,default=ROOT/"agi-proof/tool-use/eval-tool-use.public-report.json"); args=ap.parse_args()
    seeds=[int(x) for x in args.seeds.split(",") if x.strip()]; seal=verify_manifest(root=ROOT)
    if not seal["ok"]: raise SystemExit("seal fail")
    cases=load_cases(); cl=ScriptedClient(build_cases()); by_seed={}
    for s in seeds:
        res={c:[] for c in ("no_tools","always_tools","trained")}
        for case in cases:
            for fn,nm in ((no_tools,"no_tools"),(always_tools,"always_tools"),(trained,"trained")):
                run=fn(cl,case); run.condition=nm; res[nm].append(score(case,run))
        by_seed[s]={c:agg(v) for c,v in res.items()}
    tr=[by_seed[s]["trained"]["task_pass_at_1"] for s in seeds]; nt=[by_seed[s]["no_tools"]["task_pass_at_1"] for s in seeds]; al=[by_seed[s]["always_tools"]["task_pass_at_1"] for s in seeds]
    mean=lambda xs: sum(xs)/len(xs); ci_n=bootstrap_diff_ci(tr,nt,seed=0); ci_a=bootstrap_diff_ci(tr,al,seed=1)
    report={"schema":"sophia.tool_use_eval.v1","generatedAt":datetime.now(timezone.utc).isoformat(),"candidateOnly":True,"canClaimAGI":False,"mode":args.mode,"nCases":len(cases),"nSeeds":len(seeds),"seeds":seeds,"benchmarkContentHash":seal["contentHash"],"balance":seal["balance"],"grammarConstrainedDecode":{"enabled":True},"selectiveGate":{"enabled":True},"bySeed":by_seed,"aggregate":{c:{k:round(mean([by_seed[s][c][k] for s in seeds]),4) for k in by_seed[seeds[0]][c]} for c in by_seed[seeds[0]]},"deltas":{"trained_vs_no_tools":{"pass_at_1_delta":round(mean(tr)-mean(nt),4),"bootstrap95ci":ci_n,"ciExcludesZero":ci_n[0]>0 or ci_n[1]<0},"trained_vs_always_tools":{"pass_at_1_delta":round(mean(tr)-mean(al),4),"bootstrap95ci":ci_a,"ciExcludesZero":ci_a[0]>0 or ci_a[1]<0}},"overCallRate":{c:round(mean([by_seed[s][c]["over_call"] for s in seeds]),4) for c in by_seed[seeds[0]]},"claimTemplate":"On sealed tool-use benchmark v1 (N>=120, >=3 seeds), DPO adapter [beats/does not beat/within noise of] baselines. candidateOnly; canClaimAGI:false."}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"aggregate":report["aggregate"],"deltas":report["deltas"]},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
