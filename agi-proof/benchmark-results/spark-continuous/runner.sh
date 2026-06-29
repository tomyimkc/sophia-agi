#!/bin/bash
# Spark-Warm continuous benchmark loop.
# Runs LOCAL-ONLY benchmarks (ollama qwen2.5:7b + Lean kernel + CPU) in rotation,
# stamps each result sparkIteration:true/registeredResult:false/canClaimAGI:false,
# and commits+pushes to the dedicated bench/spark-continuous branch (NEVER the shared branch).
# Stop gracefully: `touch $WT/STOP_SPARK_LOOP`  (or it stops after MAX_ITERS).
set -uo pipefail
export PATH="$HOME/.elan/bin:$PATH"
REPO=/home/tomyimkc/sophia-agi
PY=$REPO/.venv/bin/python
WT=/tmp/claude-1000/-home-tomyimkc-sophia-agi/f9d06274-991d-4862-a182-8dded28a65f7/scratchpad/wt-bench
RESDIR=$WT/agi-proof/benchmark-results/spark-continuous
TMP=/tmp/claude-1000/-home-tomyimkc-sophia-agi/f9d06274-991d-4862-a182-8dded28a65f7/scratchpad/bench-tmp
LOG=/tmp/claude-1000/-home-tomyimkc-sophia-agi/f9d06274-991d-4862-a182-8dded28a65f7/scratchpad/spark_bench_loop.log
GC="git -c filter.git-crypt.smudge=cat -c filter.git-crypt.clean=cat -c filter.git-crypt.required=false"
PROV='ollama:qwen2.5:7b-instruct'
MAX_ITERS=${MAX_ITERS:-300}
SLEEP_BETWEEN=${SLEEP_BETWEEN:-20}
mkdir -p "$RESDIR" "$TMP"

# Benchmark rotation: name | needs_gpu(0/1) | command (writes JSON to $OUT)
BENCH_NAMES=(roofline-fp4 nvfp4-quant-invariants gsm8k-exactmatch seib100 lean-expiter lean-passk)

run_one() {
  local name="$1" out="$2"
  case "$name" in
    roofline-fp4)
      cd "$REPO" && OUT="$out" $PY -c "import json,os; from kernels.bench.roofline import resolve_device, gemm_flops, gemm_bytes, analyze; dev=resolve_device('NVIDIA DGX Spark GB10'); f=gemm_flops(4096,4096,4096); b=gemm_bytes(4096,4096,4096,dtype_bytes=2); t=f/(dev.peak_flops('fp4')*0.45); r=analyze(flops=f,bytes_moved=b,times_s=[t,t*1.02,t*0.99],device=dev,dtype='fp4'); json.dump(r.to_dict(), open(os.environ['OUT'],'w'), indent=2)" ;;
    nvfp4-quant-invariants)
      cd "$REPO" && OUT="$out" $PY -c "import json,os; from moe.quant import offline_invariants; ok,d=offline_invariants(); json.dump({'pass':bool(ok),**d}, open(os.environ['OUT'],'w'), indent=2, default=lambda o: bool(o))" ;;
    gsm8k-exactmatch)
      cd "$REPO" && OLLAMA_API_KEY=ollama $PY "$WT/tools/run_external_eval.py" --dataset "$WT/eval/external/gsm8k-test.jsonl" --model "$PROV" --limit 20 --scorer numeric --out "$out" ;;
    seib100)
      cd "$REPO" && $PY tools/run_seib.py --real-model --model "$PROV" --limit 5 --out "$out" ;;
    lean-expiter)
      cd "$REPO" && SOPHIA_MODEL_PROVIDER="$PROV" SOPHIA_TIMEOUT_SEC=120 $PY tools/run_lean_expert_iteration.py --rounds 2 --out "$out" ;;
    lean-passk)
      cd "$REPO" && SOPHIA_MODEL_PROVIDER="$PROV" SOPHIA_TIMEOUT_SEC=120 $PY tools/run_lean_passk.py --spec "$PROV" --bench "$REPO/formal_proofs/eval/core-lean-passk.jsonl" --out "$out" --timeout 60 ;;
  esac
}

gpu_free_mb() { # returns an integer MB free, or 999999 if unknown (GB10 unified mem often reports N/A) -> proceed
  local v; v=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
  if [ -n "$v" ]; then echo "$v"; else echo 999999; fi; }

stamp() { # $1=raw json  $2=name  -> stamped final path on stdout
  local raw="$1" name="$2"
  local utc; utc=$(date -u +%Y%m%dT%H%M%SZ)
  local mainhead; mainhead=$(cd "$REPO" && git rev-parse --short origin/main 2>/dev/null || echo unknown)
  local final="$RESDIR/${name}-${utc}.json"
  OUT="$raw" FINAL="$final" NAME="$name" UTC="$utc" HEAD="$mainhead" $PY - <<'PYEOF'
import json,os
raw,final,name,utc,head=os.environ['OUT'],os.environ['FINAL'],os.environ['NAME'],os.environ['UTC'],os.environ['HEAD']
try:
    d=json.load(open(raw))
except Exception as e:
    d={"_unparseable_raw":True,"error":str(e)}
if not isinstance(d,dict): d={"result":d}
d.update({"sparkIteration":True,"registeredResult":False,"canClaimAGI":False,
          "benchName":name,"utc":utc,"host":"dgx-spark-gb10","subject":"ollama:qwen2.5:7b-instruct",
          "lane":"spark-continuous","mainHead":head})
json.dump(d,open(final,'w'),indent=2,default=str)
print(final)
PYEOF
}

echo "=== spark_bench_loop START $(date -u) max_iters=$MAX_ITERS ===" >> "$LOG"
i=0
while [ "$i" -lt "$MAX_ITERS" ]; do
  [ -f "$WT/STOP_SPARK_LOOP" ] && { echo "STOP file found, exiting at iter $i $(date -u)" >> "$LOG"; break; }
  name="${BENCH_NAMES[$((i % ${#BENCH_NAMES[@]}))]}"
  raw="$TMP/${name}.raw.json"; rm -f "$raw"
  # GPU contention guard: GPU benchmarks need ollama; if <6GB free skip to CPU-only ones
  free=$(gpu_free_mb)
  if { [ "$name" = "gsm8k-exactmatch" ] || [ "$name" = "seib100" ] || [ "$name" = "lean-expiter" ] || [ "$name" = "lean-passk" ]; } && [ "${free:-0}" -lt 6000 ]; then
    echo "iter $i: GPU low (${free}MB free) -> skip $name (run systems instead)" >> "$LOG"
    name="roofline-fp4"; raw="$TMP/${name}.raw.json"; rm -f "$raw"
  fi
  echo "--- iter $i: $name START $(date -u) (gpu_free=${free}MB) ---" >> "$LOG"
  # Gate on OUTPUT, not exit code: some benchmarks (e.g. seib) exit non-zero when their
  # internal verdict is "no improvement" — that's still a valid, committable result.
  run_one "$name" "$raw" >> "$LOG" 2>&1
  if [ -s "$raw" ]; then
    final=$(stamp "$raw" "$name")
    short=$(basename "$final")
    echo "{\"utc\":\"$(date -u +%Y%m%dT%H%M%SZ)\",\"bench\":\"$name\",\"file\":\"$short\"}" >> "$RESDIR/index.jsonl"
    ( cd "$WT" && $GC add "agi-proof/benchmark-results/spark-continuous/$short" "agi-proof/benchmark-results/spark-continuous/index.jsonl" \
        && git -c filter.git-crypt.required=false commit -q -m "bench(spark): $name @ $(date -u +%H:%MZ) (sparkIteration, not registered)" \
        && git push -q origin bench/spark-continuous ) >> "$LOG" 2>&1 \
      && echo "iter $i: committed+pushed $short" >> "$LOG" \
      || echo "iter $i: git commit/push FAILED for $short" >> "$LOG"
  else
    echo "iter $i: $name FAILED to produce output (see above)" >> "$LOG"
  fi
  i=$((i+1))
  sleep "$SLEEP_BETWEEN"
done
echo "=== spark_bench_loop END iter=$i $(date -u) ===" >> "$LOG"
