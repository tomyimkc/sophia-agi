#!/usr/bin/env bash
# Multi-seed θ_search: train the LoRA across 3 seeds, eval source-discipline (base vs each
# adapter) over the harder 25-trap pack, and report mean delta + a paired bootstrap CI on
# the pooled per-(seed,trap) differences — the headline-grade test (modulo a third-party
# pack). Single pod (one model download). FP16 eval; train and eval phases are separated so
# only one 7B is resident at a time (fits a 24GB card). Self-terminates after serving.
set +e
mkdir -p /workspace && cd /workspace
RESULT=/workspace/RESULT.json; echo '{}' > "$RESULT"
BRANCH=claude/swarm-agent-model-design-a4h5te; MODEL=${SOPHIA_MODEL:-Qwen/Qwen2.5-7B-Instruct}
put(){ python3 - "$1" "$2" <<'PY'
import json,sys
try: d=json.load(open("/workspace/RESULT.json"))
except Exception: d={}
k,v=sys.argv[1],sys.argv[2]
try: v=json.loads(v)
except Exception: pass
d[k]=v; json.dump(d,open("/workspace/RESULT.json","w"),indent=2)
PY
}
put model "$MODEL"; put gpu "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null|head -1)"
git clone --depth 1 --branch "$BRANCH" https://github.com/tomyimkc/sophia-agi.git repo 2>&1|tail -1
cd /workspace/repo
pip install -q -U "transformers>=4.44" "peft>=0.12" "datasets>=2.20" accelerate bitsandbytes numpy 2>&1|tail -1
put phase deps_installed
python3 - <<'PY' 2>/workspace/train.err
import json, os, torch, sys
sys.path.insert(0,"/workspace/repo")
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling, BitsAndBytesConfig, set_seed)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from provenance_bench.search_recall import load_pack, PACK_V2_PATH, SCORER_FAMILIES, cohens_kappa
from provenance_bench.swarm_benchmark import _paired_bootstrap_ci
PACK_PATH=os.environ.get("SOPHIA_PACK") or str(PACK_V2_PATH)
MODEL=os.environ.get("SOPHIA_MODEL","Qwen/Qwen2.5-7B-Instruct"); SEEDS=[0,1,2]
def put(k,v):
    d=json.load(open("/workspace/RESULT.json")); d[k]=v
    json.dump(d,open("/workspace/RESULT.json","w"),indent=2)
tok=AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token=tok.eos_token
rows=[json.loads(l) for l in open("training/council/traces.jsonl") if l.strip()]
texts=[tok.apply_chat_template(r["messages"],tokenize=False) for r in rows if r.get("messages")]
ds=Dataset.from_dict({"text":texts}).map(lambda b: tok(b["text"],truncation=True,max_length=1024),
                                         batched=True, remove_columns=["text"])
def train_one(seed):
    set_seed(seed)
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16)
    m=AutoModelForCausalLM.from_pretrained(MODEL,quantization_config=bnb,device_map="auto")
    m=prepare_model_for_kbit_training(m)
    m=get_peft_model(m,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    out=f"/workspace/seed{seed}"
    args=TrainingArguments(output_dir=out,per_device_train_batch_size=1,gradient_accumulation_steps=8,
        num_train_epochs=3,learning_rate=2e-4,logging_steps=5,bf16=True,report_to=[],save_strategy="no",
        warmup_ratio=0.03,lr_scheduler_type="cosine",seed=seed,data_seed=seed)
    tr=Trainer(model=m,args=args,train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok,mlm=False))
    tr.train(); m.save_pretrained(out)
    L=[h["loss"] for h in tr.state.log_history if "loss" in h]
    del m, tr; import gc; gc.collect(); torch.cuda.empty_cache()
    return out, (round(L[-1],4) if L else None)
# Phase 1: train all seeds (only a 4-bit model resident at a time).
adapters={}; last_losses={}
for s in SEEDS:
    adapters[s], last_losses[s] = train_one(s)
put("seed_last_loss", last_losses)
# Phase 2: FP16 eval — CAPTURE raw generations, then score with 2 independent families + κ.
SYS="You are a source-disciplined search agent. Cite sources; abstain if you cannot ground a claim."
traps=[t for t in load_pack(PACK_PATH) if t.trap]
def make_gen(mdl):
    def gen(q):
        text=tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":q}],
                                     tokenize=False,add_generation_prompt=True)
        enc=tok(text,return_tensors="pt").to(mdl.device)
        o=mdl.generate(**enc,max_new_tokens=90,do_sample=False,pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][enc["input_ids"].shape[1]:],skip_special_tokens=True)
    return gen
base=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,device_map="auto").eval()
gb=make_gen(base)
base_gens=[gb(t.query) for t in traps]
m=PeftModel.from_pretrained(base, adapters[SEEDS[0]], adapter_name=f"s{SEEDS[0]}").eval()
for s in SEEDS[1:]:
    m.load_adapter(adapters[s], adapter_name=f"s{s}")
gm=make_gen(m)
seed_gens={}
for s in SEEDS:
    m.set_adapter(f"s{s}")
    seed_gens[s]=[gm(t.query) for t in traps]
# Score each independent family over the SAME generations (scoring decoupled from GPU).
fam_results={}
for fam,scorer in SCORER_FAMILIES.items():
    base_hits=[1 if scorer(g) else 0 for g in base_gens]
    per_seed={}; pb=[]; pa=[]; deltas=[]
    for s in SEEDS:
        after=[1 if scorer(g) else 0 for g in seed_gens[s]]
        br=sum(base_hits)/len(base_hits); ar=sum(after)/len(after)
        per_seed[s]={"before":round(br,3),"after":round(ar,3),"delta":round(ar-br,3)}
        deltas.append(ar-br); pb+=base_hits; pa+=after
    lo,hi=_paired_bootstrap_ci(pb,pa,iters=4000,seed=0)
    fam_results[fam]={"base_rate":round(sum(base_hits)/len(base_hits),3),"per_seed":per_seed,
                      "mean_delta":round(sum(deltas)/len(deltas),4),"ci95":[round(lo,4),round(hi,4)],
                      "ci_excludes_zero":bool(lo>0)}
# Cohen's κ between the two families over ALL judgments (base + every seed).
fams=list(SCORER_FAMILIES)
all_gens=base_gens+[g for s in SEEDS for g in seed_gens[s]]
labA=[1 if SCORER_FAMILIES[fams[0]](g) else 0 for g in all_gens]
labB=[1 if SCORER_FAMILIES[fams[1]](g) else 0 for g in all_gens]
put("graded_suite","source_discipline_rate_multiseed_2family")
put("n_traps",len(traps)); put("seeds",SEEDS); put("pack",os.path.basename(PACK_PATH))
put("families",fam_results)
put("kappa_families",fams); put("kappa_between_families",cohens_kappa(labA,labB))
put("both_families_exclude_zero", all(fam_results[f]["ci_excludes_zero"] for f in fams))
put("raw_generations",{"base":base_gens, **{f"seed{s}":seed_gens[s] for s in SEEDS}})
put("smoke_generation", seed_gens[SEEDS[0]][0][:400])
put("phase","trained_and_validated")
PY
cd /workspace/repo
INV=""
for mod in agent/swarm_router.py provenance_bench/search_recall.py agent/dual_use_adapter.py; do
  PYTHONPATH=/workspace/repo python3 "$mod" >/dev/null 2>&1 && INV="$INV $mod:PASS" || INV="$INV $mod:FAIL"
done
put offline_invariants "$INV"; put phase done
cd /workspace; echo SOPHIA_RESULT_READY; cat "$RESULT"
( python3 -m http.server 8000 >/workspace/http.log 2>&1 & )
sleep 1800
if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  curl -sS -X POST --url "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
    -H 'content-type: application/json' \
    --data "{\"query\":\"mutation{podTerminate(input:{podId:\\\"${RUNPOD_POD_ID}\\\"})}\"}" || true
fi
