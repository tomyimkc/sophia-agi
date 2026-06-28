#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Launch the θ_search QLoRA validation run on RunPod — SSH-free, GraphQL-driven.

Why this exists alongside ``tools/runpod_rlvr.py``: that orchestrator drives the pod over
**SSH from the launcher**, which is blocked in sandboxed/web sessions (HTTPS-only egress).
This launcher needs only outbound HTTPS:

  * provisions a cheap GPU pod via the RunPod **GraphQL** API (``?api_key=`` query-string
    auth — the REST ``Bearer`` header gets mangled by some egress proxies);
  * the pod self-contains the whole job in its start command (clone branch → QLoRA-train
    → inference smoke → repo offline-invariants → serve ``RESULT.json`` on :8000 → self-
    terminate), so no inbound SSH is required;
  * the launcher polls ``https://{podId}-8000.proxy.runpod.net/RESULT.json`` over HTTPS,
    prints the metrics, and terminates the pod (belt-and-suspenders with the pod's own
    self-terminate + an outer ``timeout``).

This is the **validation** run from docs/11-Platform/Swarm-Variants-V3-V4-Spec.md (V3): it
proves θ_search trains and integrates on real GPU end to end (loss-decrease + adapter
manifest + integration invariants). A graded search-recall delta + the full dual-use gate
are the GPU follow-ups. A recorded outcome is in ``theta_search_run_result.json``.

Security: the API key is read from ``$RUNPOD_API_KEY`` ONLY — never hard-code it, never
commit it. It is passed to the pod env solely so the pod can self-terminate.

    RUNPOD_API_KEY=rpa_... python training/swarm_router/runpod_theta_search.py --yes
    python training/swarm_router/runpod_theta_search.py --dry-run   # print plan, no spend
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.request

GQL = "https://api.runpod.io/graphql"
DEFAULT_BRANCH = "claude/swarm-agent-model-design-a4h5te"
DEFAULT_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
DEFAULT_GPU = "NVIDIA RTX A5000"  # cheapest 24GB; ample for a 3B QLoRA

# Remote job. {BRANCH}/{MODEL} are filled in by the launcher; the smoke-gen uses the
# correct two-step tokenize (the earlier one-shot apply_chat_template(return_tensors=...)
# returns a BatchEncoding whose .shape attribute errors under transformers>=4.44).
REMOTE = r"""
set +e
mkdir -p /workspace && cd /workspace
RESULT=/workspace/RESULT.json; echo '{}' > "$RESULT"
put() { python3 - "$1" "$2" <<'PY'
import json,sys
try: d=json.load(open("/workspace/RESULT.json"))
except Exception: d={}
k,v=sys.argv[1],sys.argv[2]
try: v=json.loads(v)
except Exception: pass
d[k]=v; json.dump(d,open("/workspace/RESULT.json","w"),indent=2)
PY
}
put model "__MODEL__"
put gpu "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
git clone --depth 1 --branch __BRANCH__ https://github.com/tomyimkc/sophia-agi.git repo 2>&1 | tail -2
cd /workspace/repo || put phase clone_failed
pip install -q -U "transformers>=4.44" "peft>=0.12" "datasets>=2.20" accelerate bitsandbytes numpy 2>&1 | tail -2
put phase deps_installed
python3 - <<'PY' 2>/workspace/train.err
import json, glob, os, hashlib, torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
                          Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
MODEL="__MODEL__"; OUT="/workspace/theta_search"
def put(k,v):
    try: d=json.load(open("/workspace/RESULT.json"))
    except: d={}
    d[k]=v; json.dump(d,open("/workspace/RESULT.json","w"),indent=2)
rows=[json.loads(l) for l in open("training/council/traces.jsonl") if l.strip()]
put("n_train_rows",len(rows))
tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
texts=[tok.apply_chat_template(r["messages"],tokenize=False) for r in rows if r.get("messages")]
ds=Dataset.from_dict({"text":texts}).map(lambda b: tok(b["text"],truncation=True,max_length=1024),
                                         batched=True, remove_columns=["text"])
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16)
model=AutoModelForCausalLM.from_pretrained(MODEL,quantization_config=bnb,device_map="auto")
model=prepare_model_for_kbit_training(model)
model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,task_type="CAUSAL_LM",
   target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
nb=model.get_nb_trainable_parameters(); put("trainable_params",int(nb[0])); put("total_params",int(nb[1]))
args=TrainingArguments(output_dir=OUT,per_device_train_batch_size=1,gradient_accumulation_steps=8,
   num_train_epochs=3,learning_rate=2e-4,logging_steps=1,bf16=True,report_to=[],save_strategy="no",
   warmup_ratio=0.03,lr_scheduler_type="cosine")
tr=Trainer(model=model,args=args,train_dataset=ds,
   data_collator=DataCollatorForLanguageModeling(tok,mlm=False))
tr.train(); model.save_pretrained(OUT); tok.save_pretrained(OUT)
L=[h["loss"] for h in tr.state.log_history if "loss" in h]
put("loss_first",round(L[0],4) if L else None); put("loss_last",round(L[-1],4) if L else None)
put("loss_curve",[round(x,4) for x in L]); put("train_steps",len(L))
files={}
for f in glob.glob(OUT+"/*"):
    if os.path.isfile(f):
        b=open(f,"rb").read(); files[os.path.basename(f)]={"bytes":len(b),"sha256":hashlib.sha256(b).hexdigest()[:16]}
put("adapter_files",files)
# --- graded search-recall A/B in FP16 (NOT through the 4-bit+LoRA stack, which decodes
# to garbage). Free the quantized trainer, reload base fp16 for the 'before' arm, then
# attach the adapter for 'after'. Sequential single-model loads fit a 24GB card. ---
import sys, gc; sys.path.insert(0,"/workspace/repo")
SYS="You are a source-disciplined search agent. Cite sources; abstain if you cannot ground a claim."
def make_gen(mdl):
    def gen(q):
        text=tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":q}],
                                     tokenize=False,add_generation_prompt=True)
        enc=tok(text,return_tensors="pt").to(mdl.device)
        out=mdl.generate(**enc,max_new_tokens=110,do_sample=False,
                         pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][enc["input_ids"].shape[1]:],skip_special_tokens=True)
    return gen
try:
    from provenance_bench.search_recall import load_pack, source_discipline_ok
    from peft import PeftModel
    traps=[t for t in load_pack() if t.trap]   # harder pack_v2 (25 traps) for power
    del model; gc.collect(); torch.cuda.empty_cache()
    base=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,device_map="auto").eval()
    gb=make_gen(base)
    before=[1.0 if source_discipline_ok(gb(t.query)) else 0.0 for t in traps]
    ada=PeftModel.from_pretrained(base,OUT).eval()
    ga=make_gen(ada)
    after=[1.0 if source_discipline_ok(ga(t.query)) else 0.0 for t in traps]
    def gen(q): return ga(q)
    br=sum(before)/len(before); ar=sum(after)/len(after)
    put("graded_suite","source_discipline_rate"); put("graded_n_traps",len(traps))
    put("graded_before",round(br,3)); put("graded_after",round(ar,3)); put("graded_delta",round(ar-br,3))
    put("smoke_generation",gen(traps[0].query)[:600])
    try:
        from agent.dual_use_adapter import DualUseAdapter
        ad=DualUseAdapter(id="theta-search-v1",team_name="search",gain=1.0)
        dec=ad.gate(target_suite="source_discipline_rate",before=br,after=ar,
                    verifier_artifacts=("recall_eval.json","decontam.json"))
        put("graded_gate_verdict",dec.verdict); put("graded_gate_reasons",list(dec.reasons))
    except Exception as e:
        put("graded_gate_error",repr(e)[:300])
except Exception as e:
    put("graded_eval_error",repr(e)[:300])
put("phase","trained_and_validated")
PY
cd /workspace/repo
INV=""
for m in agent/swarm_router.py provenance_bench/swarm_rl.py agent/dual_use_adapter.py provenance_bench/swarm_benchmark.py; do
  PYTHONPATH=/workspace/repo python3 "$m" >/dev/null 2>&1 && INV="$INV $m:PASS" || INV="$INV $m:FAIL"
done
put offline_invariants "$INV"; put phase done
cd /workspace; echo SOPHIA_RESULT_READY; cat "$RESULT"
( python3 -m http.server 8000 >/workspace/http.log 2>&1 & )
sleep 1500
if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  curl -sS --request POST --url "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
    --header 'content-type: application/json' \
    --data "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}" || true
fi
"""


def _curl(args: "list[str]") -> tuple[int, str]:
    """Shell out to curl. Some egress proxies (e.g. sandboxed/web sessions) return 403 to
    Python-urllib but allow curl, and curl honours HTTPS_PROXY + the CA bundle transparently."""
    import subprocess
    p = subprocess.run(["curl", "-sS", *args], capture_output=True, text=True, timeout=120)
    return p.returncode, p.stdout


def _gql(api_key: str, query: str, variables: dict | None = None) -> dict:
    body = json.dumps({"query": query, "variables": variables or {}})
    url = f"{GQL}?api_key={api_key}"
    # Prefer urllib; on ANY failure (e.g. proxy 403 in web sessions) fall back to curl.
    try:
        req = urllib.request.Request(url, data=body.encode(), headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        rc, out = _curl(["-X", "POST", "--url", url, "-H", "content-type: application/json", "--data", body])
        if rc != 0 or not out:
            raise RuntimeError(f"curl GraphQL fallback failed (rc={rc})")
        return json.loads(out)


def _http_get(url: str, timeout: int = 20) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        # Proven pattern: body → temp file, %{http_code} → stdout (clean separation; the
        # earlier -o /dev/stdout form mixed body+code and broke parsing in proxy sessions).
        import os
        import tempfile
        fd, tmp = tempfile.mkstemp()
        os.close(fd)
        try:
            rc, code_str = _curl(["--max-time", str(timeout), "-o", tmp, "-w", "%{http_code}", url])
            body = open(tmp, encoding="utf-8", errors="replace").read()
        finally:
            os.unlink(tmp)
        if rc != 0:
            return 0, ""
        try:
            return int((code_str or "0").strip()), body
        except ValueError:
            return 0, body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="actually create the pod (incurs cost)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; do not spend")
    ap.add_argument("--branch", default=DEFAULT_BRANCH)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--gpu", default=DEFAULT_GPU)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    ap.add_argument("--hard-timeout", type=int, default=3000, help="pod-side outer timeout (s)")
    ap.add_argument("--poll-deadline", type=int, default=1700, help="launcher fetch deadline (s)")
    args = ap.parse_args()

    remote = (REMOTE.replace("__BRANCH__", args.branch).replace("__MODEL__", args.model))
    b64 = base64.b64encode(remote.encode()).decode()
    docker_args = f'bash -c "echo {b64} | base64 -d > /tmp/r.sh && timeout {args.hard_timeout} bash /tmp/r.sh"'

    if args.dry_run or not args.yes:
        print(f"PLAN: {args.gpu} ({args.cloud}) · {args.model} · branch {args.branch}\n"
              f"  image {args.image} · pod hard-timeout {args.hard_timeout}s · remote script {len(remote)}B\n"
              "  Re-run with --yes and RUNPOD_API_KEY set to launch.")
        return 0

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("RUNPOD_API_KEY not set in env. Refusing to launch.", file=sys.stderr)
        return 2

    var = {"in": {"cloudType": args.cloud, "gpuCount": 1, "gpuTypeId": args.gpu,
                  "name": "sophia-theta-search-lora", "imageName": args.image,
                  "dockerArgs": docker_args, "ports": "8000/http", "containerDiskInGb": 60,
                  "volumeInGb": 0, "minMemoryInGb": 24, "minVcpuCount": 4,
                  "startSsh": False, "supportPublicIp": True,
                  "env": [{"key": "RUNPOD_API_KEY", "value": api_key}]}}
    resp = _gql(api_key, "mutation($in: PodFindAndDeployOnDemandInput!){ "
                "podFindAndDeployOnDemand(input:$in){ id desiredStatus } }", var)
    pod = (resp.get("data") or {}).get("podFindAndDeployOnDemand") or {}
    pod_id = pod.get("id")
    if not pod_id:
        print("Launch failed:", json.dumps(resp), file=sys.stderr)
        return 1
    print(f"Launched pod {pod_id} ({pod.get('desiredStatus')}). Polling for RESULT.json…", flush=True)

    url = f"https://{pod_id}-8000.proxy.runpod.net/RESULT.json"
    deadline = time.time() + args.poll_deadline
    result = None
    while time.time() < deadline:
        code, body = _http_get(url)
        if code == 200:
            try:
                result = json.loads(body)
            except Exception:
                result = None
            if result and result.get("phase") == "done":
                break
        time.sleep(40)

    # Always terminate from here (the pod self-terminates too; this is the backstop).
    _gql(api_key, f'mutation{{podTerminate(input:{{podId:"{pod_id}"}})}}')
    print(f"Terminated pod {pod_id}.")
    if result:
        print(json.dumps(result, indent=2))
        return 0 if result.get("phase") == "done" else 1
    print("No completed RESULT.json fetched before the deadline (pod terminated to stop billing).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
