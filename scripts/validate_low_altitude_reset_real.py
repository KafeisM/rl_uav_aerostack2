#!/usr/bin/env python3
"""Real one-drone AS2 smoke for low-altitude fixed-start reset recovery."""

from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description='Validate low-altitude fixed-start reset recovery')
    parser.add_argument('--namespace', default='drone0')
    parser.add_argument('--launch-wait', type=float, default=8.0)
    parser.add_argument('--initial-reset-timeout', type=float, default=90.0)
    parser.add_argument('--recovery-reset-timeout', type=float, default=90.0)
    parser.add_argument('--close-timeout', type=float, default=20.0)
    parser.add_argument('--max-descent-steps', type=int, default=80)
    parser.add_argument('--descent-speed', type=float, default=-0.3)
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
            fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
            target_pose=[2.0, 0.0, 1.0, 0.0],
            max_vel=1.0,
            max_yaw_vel=1.5707963267948966,
            step_duration=0.1,
            fixed_start_timeout=25.0,
            reset_ground_recovery_height=0.35,
            hover_settle_time=0.5,
            hover_timeout=8.0,
            close_operation_timeout=5.0,
        )

        obs, info = run_with_timeout(env.reset, args.initial_reset_timeout, 'initial env.reset()')
        if not info.get('reset_success'):
            return fail('initial_reset_failure', f'reset_success is false: {info}')
        if not np.all(obs >= -1.0) or not np.all(obs <= 1.0):
            return fail('observation_range_failure', f'initial obs outside [-1,1]: {obs.tolist()}')

        low_altitude_pose = list(env._drone.position)
        descent_steps = 0
        descent_done = False
        descent_action = np.array([0.0, 0.0, args.descent_speed, 0.0], dtype=np.float32)
        for descent_steps in range(1, args.max_descent_steps + 1):
            _, _, terminated, truncated, descent_info = env.step(descent_action)
            low_altitude_pose = list(env._drone.position)
            descent_done = bool(terminated or truncated)
            if descent_done or float(low_altitude_pose[2]) <= env.reset_ground_recovery_height:
                break
        if float(low_altitude_pose[2]) > env.reset_ground_recovery_height:
            return fail(
                'descent_failure',
                f'could not reach low altitude <= {env.reset_ground_recovery_height}: pose={low_altitude_pose}',
            )

        obs_after, info_after = run_with_timeout(
            env.reset,
            args.recovery_reset_timeout,
            'low-altitude recovery env.reset()',
        )
        if not info_after.get('reset_success'):
            return fail('recovery_reset_failure', f'reset_success is false: {info_after}')
        if info_after.get('reset_mode') != 'fixed_start_pose':
            return fail('recovery_reset_failure', f"unexpected reset_mode={info_after.get('reset_mode')}")
        if not bool(info_after.get('hover_settled')):
            return fail('hover_failure', f'hover did not settle after recovery reset: {info_after}')
        if not np.all(obs_after >= -1.0) or not np.all(obs_after <= 1.0):
            return fail('observation_range_failure', f'recovery obs outside [-1,1]: {obs_after.tolist()}')

        pose_after = list(env._drone.position)
        print(json.dumps({
            'status': 'success',
            'namespace': args.namespace,
            'as2_ensure_launched': ensure_result.launched,
            'low_altitude_pose_before_reset': low_altitude_pose,
            'descent_steps': descent_steps,
            'descent_done': descent_done,
            'descent_terminal_reason': descent_info.get('terminal_reason') if descent_done else None,
            'reset_mode': info_after.get('reset_mode'),
            'reset_success': info_after.get('reset_success'),
            'hover_settled': info_after.get('hover_settled'),
            'pose_after_reset': pose_after,
            'observation_range': '[-1, 1]',
            'diagnostics': env._last_reset_diagnostics,
        }, indent=2))
        return 0
    except TimeoutError as exc:
        return fail('readiness_timeout', str(exc))
    except Exception as exc:  # noqa: BLE001
        return fail('unexpected_failure', str(exc))
    finally:
        if env is not None:
            try:
                run_with_timeout(env.close, args.close_timeout, 'env.close()')
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({'status': 'cleanup_warning', 'reason': str(exc)}, indent=2))


if __name__ == '__main__':
    sys.exit(main())
