#!/usr/bin/env python3
"""Real one-drone AS2 smoke for randomized hover-start reset."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from pathlib import Path

import numpy as np

from as2_runtime import ensure_as2_simulator
from rl_uav.envs import AS2TestEnv


def run_with_timeout(func, timeout_s: float, operation: str):
    box: dict[str, object] = {}

    def _runner():
        try:
            box['result'] = func()
        except Exception as exc:  # noqa: BLE001
            box['error'] = exc

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise TimeoutError(f'{operation} exceeded {timeout_s:.1f}s')
    if 'error' in box:
        raise box['error']  # type: ignore[misc]
    return box.get('result')


def fail(kind: str, reason: str) -> int:
    print(json.dumps({'status': 'failure', 'failure_class': kind, 'reason': reason}, indent=2))
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate real randomized_hover_start reset')
    parser.add_argument('--namespace', default='drone0')
    parser.add_argument('--launch-wait', type=float, default=8.0)
    parser.add_argument('--reset-timeout', type=float, default=80.0)
    parser.add_argument('--step-timeout', type=float, default=10.0)
    parser.add_argument('--close-timeout', type=float, default=8.0)
    parser.add_argument('--min-start-target-distance', type=float, default=1.0)
    parser.add_argument('--hover-speed-threshold', type=float, default=0.08)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    env = None
    try:
        ensure_result = ensure_as2_simulator(
            num_drones=1,
            launch_script=project_root / 'as2_sim' / 'launch_sim.bash',
            launch_wait_s=args.launch_wait,
            readiness_timeout_s=30.0,
        )
        env = AS2TestEnv(
            drone_namespace=args.namespace,
            verbose=False,
            randomize_hover_start=True,
            scene_bounds_xy=5.0,
            height_bounds=(0.1, 2.0),
            min_start_target_distance=args.min_start_target_distance,
            hover_speed_threshold=args.hover_speed_threshold,
            hover_settle_time=0.5,
            hover_timeout=10.0,
            max_reset_sample_attempts=100,
            step_duration=0.1,
        )

        obs, info = run_with_timeout(env.reset, args.reset_timeout, 'env.reset()')
        if not info.get('reset_success'):
            return fail('reset_failure', f'reset_success is false: {info}')
        if info.get('reset_mode') != 'randomized_hover_start':
            return fail('reset_failure', f"unexpected reset_mode={info.get('reset_mode')}")
        if not bool(info.get('hover_settled')):
            return fail('hover_failure', f'hover did not settle before/after randomized reset: {info}')
        if not np.all(obs >= -1.0) or not np.all(obs <= 1.0):
            return fail('observation_range_failure', f'obs outside [-1,1]: {obs.tolist()}')

        start_pose = info['start_pose']
        target_pose = info['target_pose']
        distance = math.dist(start_pose[:3], target_pose[:3])
        threshold = max(env.distance_threshold, args.min_start_target_distance)
        if distance <= threshold:
            return fail('randomization_failure', f'start-target distance {distance:.3f} <= threshold {threshold:.3f}')
        if not (0.1 <= start_pose[2] <= 2.0 and 0.1 <= target_pose[2] <= 2.0):
            return fail('randomization_failure', f'height out of bounds: start={start_pose} target={target_pose}')

        speed = info.get('speed', [float('inf'), float('inf'), float('inf')])
        speed_norm = math.sqrt(sum(float(v) ** 2 for v in speed[:3]))
        if speed_norm > max(0.20, args.hover_speed_threshold * 3.0):
            return fail('hover_failure', f'post-reset speed too high: speed={speed} norm={speed_norm:.3f}')

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        step_obs, reward, terminated, truncated, step_info = run_with_timeout(
            lambda: env.step(action),
            args.step_timeout,
            'env.step(hold)',
        )
        if not np.all(step_obs >= -1.0) or not np.all(step_obs <= 1.0):
            return fail('observation_range_failure', f'step obs outside [-1,1]: {step_obs.tolist()}')

        print(json.dumps({
            'status': 'success',
            'namespace': args.namespace,
            'as2_ensure_launched': ensure_result.launched,
            'reset_mode': info.get('reset_mode'),
            'start_pose': start_pose,
            'target_pose': target_pose,
            'start_target_distance': distance,
            'threshold': threshold,
            'sample_attempts': info.get('sample_attempts'),
            'hover_settled': info.get('hover_settled'),
            'post_reset_speed_norm': speed_norm,
            'step_reward': reward,
            'step_terminated': bool(terminated),
            'step_truncated': bool(truncated),
            'step_distance': step_info.get('distance'),
            'observation_range': '[-1, 1]',
            'simulation_acceleration': 'not_validated',
        }, indent=2))
        return 0
    except TimeoutError as exc:
        return fail('readiness_timeout', str(exc))
    except Exception as exc:  # noqa: BLE001
        return fail('unexpected_failure', str(exc))
    finally:
        if env is not None:
            try:
                if hasattr(env, '_is_flying'):
                    env._is_flying = False
                run_with_timeout(env.close, args.close_timeout, 'env.close()')
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({'status': 'cleanup_warning', 'reason': str(exc)}, indent=2))


if __name__ == '__main__':
    sys.exit(main())
