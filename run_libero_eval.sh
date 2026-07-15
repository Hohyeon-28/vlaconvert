#!/bin/bash
# Script to run Libero evaluation
# Usage: ./run_libero_eval.sh [task_suite_name] [extra args...]
# task_suite_name: libero_spatial (default), libero_goal, libero_object, libero_90, libero_10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK=${1:-libero_10}
shift || true
EXTRA_ARGS=("$@")

HEADLESS_FLAG="no"
HAS_PORT="no"
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "--headless" ]]; then
        HEADLESS_FLAG="yes"
    fi
    if [[ "$arg" == "--port" || "$arg" == --port=* ]]; then
        HAS_PORT="yes"
    fi
done
PORT_VALUE="${PORT:-5556}"

# Reuse the user's active venv. Only activate conda when it exists.
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "${LIBERO_CONDA_ENV:-libero_test}" || true
fi

# Add this GR00T checkout and optional LIBERO checkout to Python path.
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
if [[ -n "${LIBERO_ROOT:-}" ]]; then
    export PYTHONPATH="$LIBERO_ROOT:$PYTHONPATH"
elif [[ -d "$HOME/private/LIBERO" ]]; then
    export PYTHONPATH="$HOME/private/LIBERO:$PYTHONPATH"
fi

echo "=========================================="
echo "Running Libero evaluation for $TASK"
echo "Headless mode: $HEADLESS_FLAG"
if [[ "$HAS_PORT" == "yes" ]]; then
    echo "Port: provided by extra args"
else
    echo "Port: $PORT_VALUE"
fi
echo "=========================================="
echo ""
echo "Make sure the inference server is running in another terminal!"
echo "Run: bash run_quantvla_converted_server.sh real $TASK <converted_checkpoint> $PORT_VALUE"
echo ""
echo "Results will be saved to:"
echo "  - Log: /tmp/logs/libero_eval_${TASK}.log"
echo "  - Latency JSONL: /tmp/logs/libero_eval_${TASK}_latency_steps.jsonl"
echo "  - Latency CSV: /tmp/logs/libero_eval_${TASK}_latency_steps.csv"
echo "  - Latency summary: /tmp/logs/libero_eval_${TASK}_latency_summary.json"
echo "  - Videos: /tmp/logs/rollout_*.mp4"
echo "=========================================="
echo ""

cd "$SCRIPT_DIR/examples/Libero/eval"

CMD=(python run_libero_eval.py --task_suite_name "$TASK")
if [[ "$HAS_PORT" == "no" ]]; then
    CMD+=(--port "$PORT_VALUE")
fi
CMD+=("${EXTRA_ARGS[@]}")
"${CMD[@]}"
