#!/usr/bin/env python3
"""
Real AS2 validation for 4 vectorized environments in one Python process.

Usage:
    conda run -n rl_uav python3 scripts/validate_real_vectorized_sim.py

Failure classes:
    - launch_failure
    - readiness_timeout
    - reset_failure
    - observation_range_failure
    - step_failure
    - cleanup_failure
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path

import gymnasium
import numpy as np

import rl_uav  # noqa: F401  # triggers env registration

from as2_runtime import ensure_as2_simulator

VALIDATOR_UNSAFE_LOW_ALTITUDE_THRESHOLD = 0.30
VALIDATOR_HEIGHT_BOUNDS = (0.1, 2.0)
VALIDATOR_FIXED_START_POSE = [0.0, 0.0, 1.0, 0.0]


def run_with_timeout(func, timeout_s: float, *args, **kwargs):
    box: dict[str, object] = {}

    def _runner():
        try:
            box['result'] = func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            box['error'] = exc

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)

    if worker.is_alive():
        raise TimeoutError(f'operation exceeded {timeout_s}s')
    if 'error' in box:
        raise box['error']  # type: ignore[misc]
    return box.get('result')


def classify_failure(kind: str, reason: str, namespaces: list[str] | None = None) -> int:
    payload = {
        'status': 'failure',
        'failure_class': kind,
        'reason': reason,
        'namespaces': namespaces or [],
    }
    print(json.dumps(payload, indent=2))
    return 1


def stop_simulator(stop_script: Path, namespaces: list[str]) -> None:
    if not stop_script.exists():
        raise FileNotFoundError(f"Stop script not found: {stop_script}")
    for namespace in namespaces:
        result = subprocess.run(
            ['bash', str(stop_script), namespace],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Stop script failed for {namespace} with code {result.returncode}"
            )


def namespaces_with_out_of_range(obs: np.ndarray, namespaces: list[str]) -> list[str]:
    mask = np.logical_or(obs < -1.0, obs > 1.0).any(axis=1)
    return [ns for ns, bad in zip(namespaces, mask.tolist()) if bad]


def namespaces_with_done(
    terminated: np.ndarray,
    truncated: np.ndarray,
    namespaces: list[str],
) -> list[str]:
    done_mask = np.logical_or(terminated, truncated)
    return [ns for ns, done in zip(namespaces, done_mask.tolist()) if done]


def build_hold_actions(action_space: gymnasium.spaces.Box, num_envs: int) -> np.ndarray:
    action_dim = int(action_space.shape[0])
    return np.zeros((num_envs, action_dim), dtype=np.float32)


def build_vectorized_env_kwargs(namespace: str, target_z: float, steps: int) -> dict[str, object]:
    """Build live vectorized validator env kwargs with safe randomized starts."""
    return {
        'drone_namespace': namespace,
        'target_pose': [0.0, 0.0, target_z, 0.0],
        'distance_threshold': 0.1,
        'pos_limit': 20.0,
        'fixed_start_pose': list(VALIDATOR_FIXED_START_POSE),
        'randomize_hover_start': False,
        'scene_bounds_xy': 10.0,
        'height_bounds': VALIDATOR_HEIGHT_BOUNDS,
        'unsafe_low_altitude_threshold': VALIDATOR_UNSAFE_LOW_ALTITUDE_THRESHOLD,
        'min_start_target_distance': 1.0,
        'hover_speed_threshold': 0.08,
        'hover_settle_time': 0.5,
        'hover_timeout': 10.0,
        'max_steps': max(500, steps + 50),
    }


def close_vector_env(vec_env: gymnasium.vector.SyncVectorEnv, timeout_s: float) -> None:
    for env in vec_env.envs:
        inner = env.unwrapped
        if hasattr(inner, '_is_flying'):
            # Avoid blocking landing actions during acceptance cleanup.
            inner._is_flying = False
        run_with_timeout(env.close, timeout_s)


def main() -> int:
    parser = argparse.ArgumentParser(description='Validate real 4-drone vectorized AS2 execution')
    parser.add_argument('--num-drones', type=int, default=4)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--launch-wait', type=float, default=8.0)
    parser.add_argument('--reset-timeout', type=float, default=35.0)
    parser.add_argument('--reset-timeout-per-drone', type=float, default=8.0)
    parser.add_argument('--reset-retries', type=int, default=1)
    parser.add_argument('--step-timeout', type=float, default=15.0)
    parser.add_argument('--close-timeout', type=float, default=5.0)
    parser.add_argument('--target-z', type=float, default=5.0)
    parser.add_argument('--launch-script', default='as2_sim/launch_sim.bash')
    parser.add_argument('--stop-script', default='as2_sim/stop_sim.bash')
    parser.add_argument(
        '--stop-after',
        action='store_true',
        default=False,
        help='Stop AS2 tmux sessions after validation (default: leave running for later AS2 tests)',
    )
    args = parser.parse_args()

    if args.num_drones != 4:
        print('Only num-drones=4 is supported by this acceptance validator.')
        return 1
    if args.reset_retries < 0:
        print('--reset-retries must be >= 0')
        return 1

    project_root = Path(__file__).resolve().parents[1]
    launch_script = project_root / args.launch_script
    stop_script = project_root / args.stop_script

    namespaces = [f'drone{i}' for i in range(args.num_drones)]
    vec_env = None
    cleanup_errors: list[str] = []
    ensure_result = None

    try:
        ensure_result = ensure_as2_simulator(
            num_drones=args.num_drones,
            launch_script=launch_script,
            launch_wait_s=args.launch_wait,
            readiness_timeout_s=30.0,
        )

        vec_env = gymnasium.vector.SyncVectorEnv([
            (
                lambda ns=ns: gymnasium.make(
                    'AS2TestEnv-v0',
                    **build_vectorized_env_kwargs(ns, args.target_z, args.steps),
                )
            )
            for ns in namespaces
        ])

        configured_namespaces = [env.unwrapped.drone_namespace for env in vec_env.envs]
        if len(set(configured_namespaces)) != args.num_drones:
            return classify_failure(
                'reset_failure',
                f'Namespace collision detected: {configured_namespaces}',
                configured_namespaces,
            )

        effective_reset_timeout = args.reset_timeout + (args.reset_timeout_per_drone * args.num_drones)
        obs = None
        reset_infos = None
        last_reset_error: Exception | None = None
        total_attempts = args.reset_retries + 1
        for attempt_idx in range(total_attempts):
            try:
                obs, reset_infos = run_with_timeout(vec_env.reset, effective_reset_timeout)
                break
            except TimeoutError as exc:
                last_reset_error = exc
                print(
                    f'[warn] reset attempt {attempt_idx + 1}/{total_attempts} '
                    f'timed out after {effective_reset_timeout:.1f}s',
                    file=sys.stderr,
                )
            except Exception as exc:
                return classify_failure('reset_failure', str(exc), namespaces)

        if obs is None or reset_infos is None:
            reason = (
                f'reset() timed out after {total_attempts} attempt(s); '
                f'effective_timeout={effective_reset_timeout:.1f}s '
                f'(base={args.reset_timeout:.1f}s + per_drone={args.reset_timeout_per_drone:.1f}s*{args.num_drones})'
            )
            if last_reset_error is not None:
                reason = f'{reason}; last_error={last_reset_error}'
            return classify_failure('readiness_timeout', reason, namespaces)

        bad_namespaces = namespaces_with_out_of_range(obs, namespaces)
        if bad_namespaces:
            return classify_failure(
                'observation_range_failure',
                'Reset observation outside [-1, 1]',
                bad_namespaces,
            )

        reset_modes = reset_infos.get('reset_mode', []) if isinstance(reset_infos, dict) else []
        if len(reset_modes) and any(mode != 'fixed_start_pose' for mode in reset_modes):
            return classify_failure(
                'reset_failure',
                f'Expected fixed_start_pose reset mode, got {reset_modes}',
                namespaces,
            )

        for step_idx in range(args.steps):
            actions = build_hold_actions(vec_env.single_action_space, args.num_drones)
            try:
                obs, _, terminated, truncated, _ = run_with_timeout(
                    vec_env.step,
                    args.step_timeout,
                    actions,
                )
            except TimeoutError:
                return classify_failure(
                    'step_failure',
                    f'step() timeout at index {step_idx} after {args.step_timeout}s',
                    namespaces,
                )
            except Exception as exc:
                return classify_failure(
                    'step_failure',
                    f'step() raised at index {step_idx}: {exc}',
                    namespaces,
                )

            done_namespaces = namespaces_with_done(terminated, truncated, namespaces)
            if done_namespaces:
                return classify_failure(
                    'step_failure',
                    f'Unexpected terminated/truncated env(s) at step {step_idx}',
                    done_namespaces,
                )

            bad_namespaces = namespaces_with_out_of_range(obs, namespaces)
            if bad_namespaces:
                return classify_failure(
                    'observation_range_failure',
                    f'Step {step_idx} observation outside [-1, 1]',
                    bad_namespaces,
                )

        print(json.dumps({
            'status': 'success',
            'num_drones': args.num_drones,
            'namespaces': namespaces,
            'steps': args.steps,
            'reset_timeout_effective_s': effective_reset_timeout,
            'reset_attempts_allowed': total_attempts,
            'observation_range': '[-1, 1]',
            'as2_ensure_launched': ensure_result.launched if ensure_result else None,
            'as2_left_running': not args.stop_after,
            'simulation_acceleration': 'not_validated',
        }, indent=2))
        return 0

    except FileNotFoundError as exc:
        return classify_failure('launch_failure', str(exc), namespaces)
    except Exception as exc:
        return classify_failure('launch_failure', str(exc), namespaces)
    finally:
        if vec_env is not None:
            try:
                close_vector_env(vec_env, args.close_timeout)
            except TimeoutError:
                cleanup_errors.append(
                    f'vec_env.close timed out after {args.close_timeout}s per env'
                )
            except Exception as exc:
                cleanup_errors.append(f'vec_env.close failed: {exc}')

        if args.stop_after:
            try:
                stop_simulator(stop_script, namespaces)
            except Exception as exc:
                cleanup_errors.append(str(exc))

        if cleanup_errors:
            print(json.dumps({
                'status': 'failure',
                'failure_class': 'cleanup_failure',
                'reason': ' | '.join(cleanup_errors),
                'namespaces': namespaces,
            }, indent=2))


if __name__ == '__main__':
    sys.exit(main())
