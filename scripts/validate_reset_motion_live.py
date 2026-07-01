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
import sys
import time

import numpy as np

from rl_uav.envs.as2_test_env import AS2TestEnv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Validate live reset then physical motion')
    parser.add_argument('--namespace', default='drone0')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--step-duration', type=float, default=0.3)
    parser.add_argument('--action-vx', type=float, default=0.8)
    parser.add_argument('--min-distance-reduction', type=float, default=0.25)
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
        action = np.array([args.action_vx, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(args.steps):
            _, _, terminated, truncated, info = env.step(action)
            distances.append(float(info['distance']))
            positions.append([float(value) for value in info['position']])
            if terminated or truncated:
                break
            time.sleep(0.05)

        initial_distance = distances[0] if distances else float('inf')
        final_distance = distances[-1] if distances else float('inf')
        reduction = initial_distance - final_distance
        result.update({
            'success': reduction >= args.min_distance_reduction,
            'initial_distance': initial_distance,
            'final_distance': final_distance,
            'distance_reduction': reduction,
            'distances': distances,
            'positions': positions,
            'min_distance_reduction': args.min_distance_reduction,
        })
        if not result['success']:
            result['reason'] = 'distance_did_not_decrease_enough'
            print(json.dumps(result, indent=2, sort_keys=True))
            return 3

        result['reason'] = 'distance_reduced_after_positive_x_actions'
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    finally:
        env.close()


if __name__ == '__main__':
    raise SystemExit(main())
