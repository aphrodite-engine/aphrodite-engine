#!/bin/bash
# This file demonstrates the example usage of disaggregated prefilling
# We will launch 2 aphrodite instances (1 for prefill and 1 for decode),
# and then transfer the KV cache between them.

set -xe

echo "🚧🚧 Warning: The usage of disaggregated prefill is experimental and subject to change 🚧🚧"
sleep 1

# NousResearch/Meta-Llama-3.1-8B-Instruct or deepseek-ai/DeepSeek-V2-Lite
MODEL_NAME=${HF_MODEL_NAME:-NousResearch/Meta-Llama-3.1-8B-Instruct}

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'cleanup' INT

# Cleanup function
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Cleanup commands
    pgrep python | xargs kill -9
    pkill -f python
    echo "Cleanup complete. Exiting."
    exit 0
}

export APHRODITE_HOST_IP=$(hostname -I | awk '{print $1}')

# install quart first -- required for disagg prefill proxy run
if python3 -c "import quart" &> /dev/null; then
    echo "Quart is already installed."
else
    echo "Quart is not installed. Installing..."
    python3 -m pip install quart
fi 

# a function that waits Aphrodite server to start
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}


# You can also adjust --kv-ip and --kv-port for distributed inference.

# prefilling instance, which is the KV producer
CUDA_VISIBLE_DEVICES=0 aphrodite run $MODEL_NAME \
    --port 8100 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_ip":"'"$APHRODITE_HOST_IP"'"}' &

# decoding instance, which is the KV consumer
CUDA_VISIBLE_DEVICES=1 aphrodite run $MODEL_NAME \
    --port 8200 \
    --max-model-len 100 \
    --gpu-memory-utilization 0.8 \
    --trust-remote-code \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_ip":"'"$APHRODITE_HOST_IP"'"}' &

# wait until prefill and decode instances are ready
wait_for_server 8100
wait_for_server 8200

# launch a proxy server that opens the service at port 8000
# the workflow of this proxy:
# - send the request to prefill Aphrodite instance (port 8100), change max_tokens 
#   to 1
# - after the prefill Aphrodite finishes prefill, send the request to decode Aphrodite 
#   instance
# NOTE: the usage of this API is subject to change --- in the future we will 
# introduce "Aphrodite connect" to connect between prefill and decode instances
python3 ../benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py \
    --prefill-url http://localhost:8100/v1/completions \
    --decode-url http://localhost:8200/v1/completions \
    --prefill-zmq "$APHRODITE_HOST_IP:14579" \
    --decode-zmq "$APHRODITE_HOST_IP:14580" &
sleep 1

# run two example requests
# NOTE: Prompts should be sufficiently long (at least 2 tokens after tokenization)
# to ensure num_external_tokens > 0 for KV cache transfer
output1=$(curl -X POST -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"$MODEL_NAME"'",
"prompt": "San Francisco is a beautiful city located in",
"max_tokens": 10,
"temperature": 0
}')

output2=$(curl -X POST -s http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "'"$MODEL_NAME"'",
"prompt": "Santa Clara is a county in the state of",
"max_tokens": 10,
"temperature": 0
}')


# Cleanup commands
pgrep python | xargs kill -9
pkill -f python

echo ""

sleep 1

# Print the outputs of the curl requests
echo ""
echo "Output of first request: $output1"
echo "Output of second request: $output2"

echo "🎉🎉 Successfully finished 2 test requests! 🎉🎉"
echo ""