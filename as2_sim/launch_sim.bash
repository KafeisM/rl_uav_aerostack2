#!/bin/bash

# Launch AS2 multirotor simulator for RL training
# Based on: https://github.com/aerostack2/project_as2_multirotor_simulator

set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

usage() {
    echo "Usage: $0 [options]"
    echo "  options:"
    echo "      -n: drone namespace. Default: drone0"
    echo "      -c: motion controller plugin, choices: [pid, df]. Default: pid"
}

# Default values
drone_namespace="drone0"
motion_controller_plugin="pid"
simulation_config="config/world.yaml"

# Parse arguments
while getopts "n:c:h" opt; do
  case ${opt} in
    n )
      drone_namespace="${OPTARG}"
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

echo "=== Launching AS2 Multirotor Simulator ==="
echo "  Drone namespace: ${drone_namespace}"
echo "  Controller: ${motion_controller_plugin}"
echo "  World config: ${simulation_config}"

# Launch tmuxinator session
tmuxinator start -n ${drone_namespace} -p tmuxinator/aerostack2.yaml \
    drone_namespace=${drone_namespace} \
    simulation_config_file=${simulation_config} \
    motion_controller_plugin=${motion_controller_plugin} \
    base_launch=true

echo "=== Simulator launched in tmux session '${drone_namespace}' ==="
echo "  Attach with: tmux attach-session -t ${drone_namespace}"
