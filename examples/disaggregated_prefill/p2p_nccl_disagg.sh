#!/bin/bash
set -e

export APHRODITE_HOST_IP="127.0.0.1"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
PREFILL_PORT=8100
DECODE_PORT=8200
PROXY_PORT=9000

PREFILL_ZMQ_PORT=14579
DECODE_ZMQ_PORT=14580

echo "============================================"
echo "P2P NCCL Disaggregated Serving Setup"
echo "============================================"
echo "Using host IP: $APHRODITE_HOST_IP"
echo "Model: $MODEL"
echo ""

echo "Cleaning up any existing instances..."
pkill -f "aphrodite.*--port ($PREFILL_PORT|$DECODE_PORT)" 2>/dev/null || true
pkill -f "disagg_prefill_proxy_server.py" 2>/dev/null || true
sleep 2

rm -f /tmp/prefill.log /tmp/decode.log /tmp/proxy.log

echo "Launching prefill instance on port $PREFILL_PORT..."
CUDA_VISIBLE_DEVICES=0 \
python -m aphrodite.endpoints.openai.api_server \
    --model $MODEL \
    --port $PREFILL_PORT \
    --max-model-len 1024 \
    --kv-transfer-config '{
        "kv_connector": "P2pNcclConnector",
        "kv_role": "kv_producer",
        "kv_rank": 0,
        "kv_parallel_size": 2,
        "kv_ip": "'"$APHRODITE_HOST_IP"'",
        "kv_port": '"$PREFILL_ZMQ_PORT"'
    }' > /tmp/prefill.log 2>&1 &

PREFILL_PID=$!
echo "Prefill instance PID: $PREFILL_PID"

echo "Waiting for prefill instance to start..."
for i in {1..60}; do
    if curl -s http://localhost:$PREFILL_PORT/health > /dev/null 2>&1; then
        echo "✓ Prefill instance is ready"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "✗ Prefill instance failed to start within 60 seconds"
        echo "Last 50 lines of prefill log:"
        tail -50 /tmp/prefill.log
        exit 1
    fi
    sleep 1
done

echo ""
echo "Launching decode instance on port $DECODE_PORT..."
CUDA_VISIBLE_DEVICES=1 \
python -m aphrodite.endpoints.openai.api_server \
    --model $MODEL \
    --port $DECODE_PORT \
    --max-model-len 1024 \
    --kv-transfer-config '{
        "kv_connector": "P2pNcclConnector",
        "kv_role": "kv_consumer",
        "kv_rank": 1,
        "kv_parallel_size": 2,
        "kv_ip": "'"$APHRODITE_HOST_IP"'",
        "kv_port": '"$PREFILL_ZMQ_PORT"'
    }' > /tmp/decode.log 2>&1 &

DECODE_PID=$!
echo "Decode instance PID: $DECODE_PID"

echo "Waiting for decode instance to start..."
for i in {1..60}; do
    if curl -s http://localhost:$DECODE_PORT/health > /dev/null 2>&1; then
        echo "✓ Decode instance is ready"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "✗ Decode instance failed to start within 60 seconds"
        echo "Last 50 lines of decode log:"
        tail -50 /tmp/decode.log
        exit 1
    fi
    sleep 1
done

echo ""
echo "Launching proxy server on port $PROXY_PORT..."
python3 ../../benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py \
    --prefill-url http://localhost:$PREFILL_PORT/v1/completions \
    --decode-url http://localhost:$DECODE_PORT/v1/completions \
    --prefill-zmq "$APHRODITE_HOST_IP:$PREFILL_ZMQ_PORT" \
    --decode-zmq "$APHRODITE_HOST_IP:$DECODE_ZMQ_PORT" \
    --port $PROXY_PORT > /tmp/proxy.log 2>&1 &

PROXY_PID=$!
echo "Proxy server PID: $PROXY_PID"

echo "Waiting for proxy server to start..."
for i in {1..30}; do
    if curl -s http://localhost:$PROXY_PORT/health > /dev/null 2>&1; then
        echo "✓ Proxy server is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "✗ Proxy server failed to start within 30 seconds"
        echo "Last 50 lines of proxy log:"
        tail -50 /tmp/proxy.log
        exit 1
    fi
    sleep 1
done

echo ""
echo "============================================"
echo "All services started successfully!"
echo "============================================"
echo "Prefill instance: http://localhost:$PREFILL_PORT (PID: $PREFILL_PID)"
echo "Decode instance:  http://localhost:$DECODE_PORT (PID: $DECODE_PID)"
echo "Proxy server:     http://localhost:$PROXY_PORT (PID: $PROXY_PID)"
echo ""
echo "ZMQ addresses:"
echo "  Prefill: $APHRODITE_HOST_IP:$PREFILL_ZMQ_PORT"
echo "  Decode:  $APHRODITE_HOST_IP:$DECODE_ZMQ_PORT"
echo ""
echo "Log files:"
echo "  Prefill: /tmp/prefill.log"
echo "  Decode:  /tmp/decode.log"
echo "  Proxy:   /tmp/proxy.log"
echo ""
echo "============================================"

show_logs() {
    echo ""
    echo "Showing live logs (Ctrl+C to stop monitoring)..."
    echo "Legend: 🟦 Proxy | 🟩 Prefill | 🟨 Decode"
    echo "============================================"

    tail -f /tmp/proxy.log 2>/dev/null | while IFS= read -r line; do
        echo "🟦 PROXY   | $line"
    done &
    TAIL_PROXY=$!

    tail -f /tmp/prefill.log 2>/dev/null | while IFS= read -r line; do
        echo "🟩 PREFILL | $line"
    done &
    TAIL_PREFILL=$!

    tail -f /tmp/decode.log 2>/dev/null | while IFS= read -r line; do
        echo "🟨 DECODE  | $line"
    done &
    TAIL_DECODE=$!

    trap "kill $TAIL_PROXY $TAIL_PREFILL $TAIL_DECODE 2>/dev/null; exit 0" INT TERM
    wait
}

echo "Testing with a sample request..."
echo ""

curl -s http://localhost:$PROXY_PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL"'",
        "prompt": "Once upon a time in a land far away, there lived a wise old wizard who",
        "max_tokens": 50,
        "temperature": 0.7
    }' | jq -r '.choices[0].text' 2>/dev/null || echo "Request sent (jq not available for parsing)"

echo ""
echo "============================================"
echo "Sample request completed!"
echo "============================================"
echo ""
echo "To run benchmarks:"
echo "  aphrodite bench run --port $PROXY_PORT --model $MODEL \\"
echo "    --random-input-len 512 --random-output-len 128 \\"
echo "    --num-prompts 100 --request-rate 10"
echo ""
echo "To stop all services:"
echo "  pkill -f 'aphrodite.*--port' && pkill -f 'disagg_prefill_proxy_server.py'"
echo ""
echo "To view logs in real-time, run:"
echo "  bash $0 --show-logs"
echo "============================================"

if [[ "$1" == "--show-logs" ]]; then
    show_logs
fi

