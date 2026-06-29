# Pre-baked CUDA image for the Sophia-Wisdom-4B GPU pilot/ORPO runs.
#
# WHY: the SSH-free pod runs its whole job in dockerStartCmd. When that startup did a
# multi-minute `pip install`, RunPod kept killing the container ~60-90s in, causing a
# restart loop that wasted GPU credit (M4 ORPO, 2026-06-26). Baking the deps into the
# image removes the long pip step entirely, so the pod reaches the actual job immediately.
#
# Build+push via .github/workflows/build-wisdom-image.yml (GH runner has Docker; the dev
# exec box does not). After the first push, set the GHCR package visibility to PUBLIC so
# RunPod can pull it without registry credentials.
FROM runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Tools the self-report pod job needs at startup (clone + push).
RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# The exact deps tools/pilot_gemma3_run.py + tools/pilot_gemma3_orpo.py import. Versions are
# UPPER-CAPPED to match the launcher's fallback pip line so behavior is identical, just
# pre-installed. The caps matter: unbounded floors resolved to transformers 5.x + trl 1.7.0,
# and trl 1.x REMOVED top-level ORPOConfig/ORPOTrainer (the API pilot_gemma3_orpo.py uses) —
# "ImportError: cannot import name 'ORPOConfig' from 'trl'" killed M4 (diagnosed 2026-06-26).
# trl 0.12-0.14 has ORPOConfig/ORPOTrainer + the processing_class arg; transformers <5 keeps
# the gemma-3 load path the SFT pilot was validated against.
RUN python -m pip install --upgrade pip \
 && python -m pip install \
      "transformers>=4.52,<5" \
      "peft>=0.13,<1" \
      "trl>=0.12,<0.15" \
      "accelerate<2" \
      datasets \
      sentencepiece \
      protobuf

# Fail the BUILD (not a wasted GPU pod) if any dep is unimportable. Importing ORPOConfig/
# ORPOTrainer specifically guards the trl-1.x regression that removed them: a bad resolution
# fails here (free) instead of on a rented GPU.
RUN python -c "import transformers, peft, trl, accelerate, datasets, sentencepiece; \
from trl import ORPOConfig, ORPOTrainer; from peft import LoraConfig; \
print('wisdom-pilot image OK:', transformers.__version__, peft.__version__, trl.__version__)"
