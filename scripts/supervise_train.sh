#!/usr/bin/env bash
# Supervise a PPO training run against the AS2 simulator stack.
#
# Restarts the simulator, waits for the reset service to be available,
# resumes from the newest matching checkpoint when one exists, and retries
# the whole cycle on failure up to max_attempts times.
#
# Usage: ./scripts/supervise_train.sh <config.yaml> [max_attempts]
# The supervisor itself runs in the foreground; nohup the supervisor, not
# the training process.
set -u

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.yaml> [max_attempts]" >&2
    exit 2
fi

CONFIG="$1"
MAX_ATTEMPTS="${2:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SIM_DIR=/home/jordi/TFM/project_as2_multirotor_simulator
PY=/home/jordi/miniconda3/envs/rl_uav/bin/python

CONFIG_STEM="$(basename "$CONFIG" .yaml)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_DIR/runs/ppo"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/supervise_${CONFIG_STEM}_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

restart_sim_stack() {
    (cd "$SIM_DIR" && ./stop.bash) >/dev/null 2>&1 || true
    tmux kill-session -t drone0 2>/dev/null || true
    tmux kill-session -t ground_station 2>/dev/null || true
    pkill -f 'as2_' 2>/dev/null || true
    pkill -f 'ros2 launch' 2>/dev/null || true
    sleep 3
    (cd "$SIM_DIR" && ./launch_as2.bash -n drone0 </dev/null >/dev/null 2>&1) || true
    if [ -n "${DISPLAY:-}" ]; then
        (cd "$SIM_DIR" && DISPLAY="${DISPLAY:-:0}" ./launch_ground_station.bash </dev/null >/dev/null 2>&1) || true
    fi
}

wait_for_sim_health() {
    # ROS setup scripts reference unset variables; relax nounset while sourcing.
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    # shellcheck disable=SC1091
    source /home/jordi/as2_rl_ws/install/setup.bash
    set -u
    local waited=0
    while [ "$waited" -lt 60 ]; do
        if timeout 10 ros2 service list 2>/dev/null | grep -q reset_simulator_state; then
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    return 1
}

resolve_resume_checkpoint() {
    (cd "$REPO_DIR" && "$PY" - "$CONFIG" <<'EOF'
import sys

from rl_uav.training import load_training_config, resolve_resume_checkpoint

config = load_training_config(sys.argv[1])
try:
    checkpoint = resolve_resume_checkpoint(
        'latest', config, output_root=config['experiment']['output_root']
    )
except FileNotFoundError:
    print('')
else:
    print(checkpoint)
EOF
    ) 2>/dev/null || true
}

LAST_EXIT=1
ATTEMPTS_USED=0

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    ATTEMPTS_USED=$attempt
    ATTEMPT_LOG="$LOG_DIR/supervise_${CONFIG_STEM}_attempt${attempt}_$(date +%Y%m%d_%H%M%S).log"
    log "=== Attempt $attempt/$MAX_ATTEMPTS (log: $ATTEMPT_LOG) ==="

    log "Restarting simulator stack ..."
    restart_sim_stack

    log "Waiting for reset_simulator_state service ..."
    if ! wait_for_sim_health; then
        log "Simulator never became healthy within 60s; retrying."
        continue
    fi
    log "Simulator healthy."

    RESUME="$(resolve_resume_checkpoint)"
    if [ -n "$RESUME" ]; then
        log "Resuming from checkpoint: $RESUME"
    else
        log "No matching checkpoint found; starting fresh."
    fi

    log "Launching training ..."
    (cd "$REPO_DIR" && PYTHONUNBUFFERED=1 "$PY" scripts/train_ppo.py \
        --config "$CONFIG" ${RESUME:+--resume-from "$RESUME"} \
        >> "$ATTEMPT_LOG" 2>&1)
    LAST_EXIT=$?

    if [ "$LAST_EXIT" -eq 0 ]; then
        log "Training completed successfully on attempt $attempt."
        break
    fi

    log "Training exited with code $LAST_EXIT; retrying in 10s."
    sleep 10
done

log "=== Supervisor done: attempts used $ATTEMPTS_USED, last exit code $LAST_EXIT ==="
exit "$LAST_EXIT"
