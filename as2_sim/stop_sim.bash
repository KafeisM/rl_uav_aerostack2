#!/bin/bash

# Stop AS2 simulator tmux sessions

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Default drone namespace
drone_namespace="${1:-drone0}"

echo "=== Stopping AS2 simulator sessions ==="

# Kill the drone tmux session
if tmux has-session -t "${drone_namespace}" 2>/dev/null; then
    tmux kill-session -t "${drone_namespace}"
    echo "  Killed session: ${drone_namespace}"
else
    echo "  Session '${drone_namespace}' not found"
fi

echo "=== Done ==="
