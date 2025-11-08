#!/bin/bash
set -e

model_path="$1"

# Check if model path is provided
if [ -z "$model_path" ]; then
    echo "Error: Please provide model path as the first argument"
    echo "Usage: $0 /path/to/model"
    exit 1
fi

export APHRODITE_ENABLE_T2I_PIPELINE="1"

############################################
# 3. Start Aphrodite
############################################
# Uncomment the next line for nsys profiling
# nsys launch --trace-fork-before-exec true --session test -t cuda,cublas,cudnn,nvtx --cuda-graph-trace=node \
aphrodite run "$model_path" \
    --trust-remote-code \
    --served-model-name hunyuan_image3 \
    --max-model-len 10000 \
    --gpu-memory-utilization 0.6 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --max-num-batched-tokens 10000 \
    --max-num-seqs 1 \
    --enforce-eager \
    --trust-request-chat-template \
    -tp 8
