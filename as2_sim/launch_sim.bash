#!/bin/bash

# Launch AS2 multirotor simulator for RL training
# Based on: https://github.com/aerostack2/project_as2_multirotor_simulator
# Multi-drone pattern: ONE shared simulator + N per-namespace stacks
# (first drone owns base_launch=true; extra drones attach with base_launch=false).

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

usage() {
    echo "Usage: $0 [options]"
    echo "  options:"
    echo "      -n: number of drones. Default: 1"
    echo "          1   -> single-drone mode (config/world.yaml, namespace drone0)"
    echo "          >1  -> multi-drone mode (config/world_multi.yaml, namespaces drone0..droneN-1)"
    echo "      -c: motion controller plugin, choices: [pid, df]. Default: pid"
    echo "      -h: show this help"
}

# Default values
num_drones=1
motion_controller_plugin="pid"

# Parse arguments
while getopts "n:c:h" opt; do
  case ${opt} in
    n )
      num_drones="${OPTARG}"
      ;;
    c )
      motion_controller_plugin="${OPTARG}"
      ;;
    h )
      usage
      exit 0
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
  esac
done

if ! [[ "${num_drones}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid -n value: '${num_drones}' (must be a positive integer)" >&2
    exit 1
fi

# Resolve controller plugin name
case ${motion_controller_plugin} in
  pid )
    motion_controller_plugin="pid_speed_controller"
    ;;
  df )
    motion_controller_plugin="differential_flatness_controller"
    ;;
  * )
    echo "Invalid motion controller plugin: ${motion_controller_plugin}" >&2
    usage
    exit 1
    ;;
esac

# Resolve world config + drone namespaces based on -n
if [[ "${num_drones}" -eq 1 ]]; then
    simulation_config="config/world.yaml"
    namespaces=("drone0")
else
    simulation_config="config/world_multi.yaml"
    # Extract namespaces declared in world_multi.yaml via pyyaml (already in project deps).
    mapfile -t available_ns < <(python3 -c "
import sys, yaml
with open('${simulation_config}') as f:
    data = yaml.safe_load(f)
for k in data.keys():
    if k.startswith('drone'):
        print(k)
" )
    if [[ "${num_drones}" -gt "${#available_ns[@]}" ]]; then
        echo "Error: requested ${num_drones} drones but ${simulation_config} only declares ${#available_ns[@]} (${available_ns[*]})." >&2
        exit 1
    fi
    namespaces=("${available_ns[@]:0:${num_drones}}")
fi

echo "=== Launching AS2 Multirotor Simulator ==="
echo "  Drones (${num_drones}): ${namespaces[*]}"
echo "  Controller: ${motion_controller_plugin}"
echo "  World config: ${simulation_config}"

for i in "${!namespaces[@]}"; do
    ns="${namespaces[$i]}"
    if [[ "${i}" -eq 0 ]]; then
        base_launch="true"
    else
        base_launch="false"
    fi

    echo "--- Starting ${ns} (base_launch=${base_launch}) ---"
    tmuxinator start -n "${ns}" -p tmuxinator/aerostack2.yaml \
        drone_namespace="${ns}" \
        simulation_config_file="${simulation_config}" \
        motion_controller_plugin="${motion_controller_plugin}" \
        base_launch="${base_launch}"

    # Give the base simulator time to spawn physics/world before attaching
    # extra namespaced platform nodes; otherwise they race the shared sim bring-up.
    if [[ "${i}" -eq 0 && "${num_drones}" -gt 1 ]]; then
        sleep 2
    fi
done

echo "=== Simulator launched ==="
echo "  Tmux sessions: ${namespaces[*]}"
echo "  Attach with: tmux attach-session -t <drone_namespace>"
