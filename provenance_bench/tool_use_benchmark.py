import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def load_cases(path=None):
    p = path or ROOT/"data/tool_use_benchmark/heldout_v1.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
def content_hash(cases):
    payload = sorted([{"id":c["id"],"prompt":c["prompt"]} for c in cases], key=lambda x:x["id"])
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
def balance_counts(cases):
    o={"should_call":0,"shouldnt_call":0,"traps":0,"total":len(cases)}
    for c in cases:
        t,d=c.get("trap","none"),c.get("label",{}).get("decision","")
        if t in ("wrong_tool_output","empty_tool_output","unanswerable"): o["traps"]+=1
        elif t=="already_known" or d=="answer_direct": o["shouldnt_call"]+=1
        elif d=="call": o["should_call"]+=1
        else: o["traps"]+=1
    return o
def verify_manifest(*, root=ROOT):
    m=json.loads((root/"data/tool_use_benchmark/manifest.json").read_text(encoding="utf-8"))
    cases=load_cases(root/"data/tool_use_benchmark/heldout_v1.jsonl")
    h=content_hash(cases); b=balance_counts(cases)
    return {"ok":m.get("contentHash")==h and b["total"]==m.get("nCases",0),"contentHash":h,"balance":b,"nCases":len(cases)}
