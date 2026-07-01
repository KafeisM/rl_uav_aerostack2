#!/usr/bin/env python3
"""Focused live AS2 reset+motion validation for deterministic reset.

This script assumes AS2 Multirotor Simulator is already running and the caller
has sourced ROS Humble plus the AS2 overlay. It verifies behavior, not only API
acceptance: after a service-backed fixed reset, repeated positive-x actions must
reduce measured distance to the target.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from rl_uav.envs.as2_test_env import AS2TestEnv


def _position_delta_norm(positions: list[list[float]]) -> float:
    if len(positions) < 2:
        return 0.0
    first = positions[0]
    return max(
        math.dist(first[:3], current[:3])
        for current in positions[1:]
        if len(current) >= 3 and len(first) >= 3
    )


def evaluate_actionability_result(
    *,
    reset_info: dict[str, object],
    distances: list[float],
    positions: list[list[float]],
    command_acceptance: list[bool],
    min_distance_reduction: float,
    min_position_delta: float,
) -> dict[str, object]:
    """Evaluate service-backed post-reset motion without velocity-only reset fallback."""
    if not bool(reset_info.get('reset_success', False)):
        return {'success': False, 'reason': 'reset_not_command_ready'}

    service_status = str(reset_info.get('reset_service_status', ''))
    if (
        reset_info.get('reset_method') != 'simulator_service'
        or not bool(reset_info.get('reset_service_attempted', False))
        or not service_status.startswith('service_success')
    ):
        return {
            'success': False,
            'reason': 'reset_not_service_backed',
            'reset_method': reset_info.get('reset_method'),
            'reset_service_status': service_status,
        }

    accepted_steps = sum(1 for accepted in command_acceptance if accepted)
    if accepted_steps == 0:
        return {'success': False, 'reason': 'command_path_not_accepted'}

    initial_distance = distances[0] if distances else float('inf')
    final_distance = distances[-1] if distances else float('inf')
    distance_reduction = initial_distance - final_distance
    position_delta = _position_delta_norm(positions)
    success = (
        distance_reduction >= float(min_distance_reduction)
        and position_delta >= float(min_position_delta)
    )
    return {
        'success': success,
        'reason': 'distance_reduced_after_valid_action' if success else 'insufficient_physical_motion',
        'initial_distance': initial_distance,
        'final_distance': final_distance,
        'distance_reduction': distance_reduction,
        'max_position_delta': position_delta,
        'accepted_command_steps': accepted_steps,
        'command_steps': len(command_acceptance),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Validate live reset then physical motion')
    parser.add_argument('--namespace', default='drone0')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--step-duration', type=float, default=0.3)
    parser.add_argument('--action-vx', type=float, default=0.8)
    parser.add_argument('--min-distance-reduction', type=float, default=0.25)
    parser.add_argument('--min-position-delta', type=float, default=0.10)
    parser.add_argument('--reset-service-timeout', type=float, default=3.0)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    env = AS2TestEnv(
        drone_namespace=args.namespace,
        verbose=False,
        fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
        target_pose=[2.0, 0.0, 1.0, 0.0],
        step_duration=args.step_duration,
        reset_service_timeout=args.reset_service_timeout,
        hover_settle_time=0.0,
        publish_target_marker=False,
    )
    result: dict[str, object] = {
        'namespace': args.namespace,
        'steps': args.steps,
        'action_vx': args.action_vx,
        'process_id': os.getpid(),
        'single_process_no_restart': True,
    }

    try:
        _, reset_info = env.reset()
        result['reset_info'] = reset_info
        result['reset_diagnostics'] = dict(env._last_reset_diagnostics)
        if not reset_info.get('reset_success', False):
            result['success'] = False
            result['reason'] = 'reset_not_command_ready'
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2

        distances: list[float] = []
        positions: list[list[float]] = []
        command_acceptance: list[bool] = []
        action = np.array([args.action_vx, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(args.steps):
            _, _, terminated, truncated, info = env.step(action)
            distances.append(float(info['distance']))
            positions.append([float(value) for value in info['position']])
            command_acceptance.append(bool(info.get('motion_command_accepted', False)))
            if terminated or truncated:
                break
            time.sleep(0.05)

        evaluation = evaluate_actionability_result(
            reset_info=reset_info,
            distances=distances,
            positions=positions,
            command_acceptance=command_acceptance,
            min_distance_reduction=args.min_distance_reduction,
            min_position_delta=args.min_position_delta,
        )
        result.update(evaluation)
        result.update({
            'distances': distances,
            'positions': positions,
            'command_acceptance': command_acceptance,
            'min_distance_reduction': args.min_distance_reduction,
            'min_position_delta': args.min_position_delta,
        })
        if not result['success']:
            print(json.dumps(result, indent=2, sort_keys=True))
            return 2 if str(result.get('reason', '')).startswith('reset_') else 3

        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == '__main__':
    raise SystemExit(main())
