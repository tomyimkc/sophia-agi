#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
#
# Governed Speculative Sparsity on the DGX Spark (GB10) — offline validation + the
# Tier-2 gather-on-read-set NVFP4 GEMM roofline against the Spark's 273 GB/s.
#
# Usage:   bash scripts/gss_spark_run.sh
# Env:     M N K TILE RHO ITERS  (override the GEMM shape / read-set / timing)
#
# The fp4 roofline denominator only exists on Blackwell — this is exactly the box for it.
# Reports % of the Spark's 273 GB/s; the dense vs gather A/B shows the bandwidth lever.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

DEV="${DEV:-NVIDIA DGX Spark GB10}"
M="${M:-1}"; N="${N:-8192}"; K="${K:-8192}"; TILE="${TILE:-256}"; RHO="${RHO:-0.10}"; ITERS="${ITERS:-50}"

echo "############################################################"
echo "# GSS on the Spark — device: $DEV"
echo "############################################################"

echo; echo "== 1. Offline invariants (no GPU) =="
python serving/gss_feasibility.py
python serving/gss.py

echo; echo "== 2. CI test suites (no GPU) =="
python -m pytest tests/test_gss.py tests/test_gss_feasibility.py \
                 tests/test_gss_gather_gemm.py tests/test_gss_probe.py -q

echo; echo "== 3. Tier-2 gather-GEMM roofline on the Spark (fp4, 273 GB/s) =="
echo "-- dense baseline (ρ=1.0, reads every tile) --"
python kernels/src/gss_gather_gemm.py --m "$M" --n "$N" --k "$K" --tile "$TILE" \
    --rho 1.0 --iters "$ITERS" --device "$DEV"
echo "-- gather read-set (ρ=$RHO) --"
python kernels/src/gss_gather_gemm.py --m "$M" --n "$N" --k "$K" --tile "$TILE" \
    --rho "$RHO" --iters "$ITERS" --device "$DEV"

echo; echo "== 4. (optional) real-checkpoint probe on the Spark =="
echo "   aarch64 note: use --draft fakequant (bitsandbytes is x86-fragile on the Spark)."
echo "   python tools/gss_probe.py --backend hf --model allenai/OLMoE-1B-7B-0924 \\"
echo "       --draft fakequant --campaign 5 --out agi-proof/benchmark-results/gss-spark.json"

echo; echo "Compare 'bandwidth (% of 273 GB/s)' and the 'traffic ratio' line across the two"
echo "runs above: ρ=$RHO should move ~$(python -c "print(round(${RHO}+0.01,2))")× the bytes of dense."
