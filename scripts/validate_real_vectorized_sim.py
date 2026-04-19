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
import time
from pathlib import Path

import gymnasium
import numpy as np

import rl_uav  # noqa: F401  # triggers env registration


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


def launch_simulator(launch_script: Path, num_drones: int, launch_wait_s: float):
    if not launch_script.exists():
        raise FileNotFoundError(f"Launch script not found: {launch_script}")

    proc = subprocess.Popen(
        ['bash', str(launch_script), '-n', str(num_drones)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(launch_wait_s)
    if proc.poll() is not None and proc.returncode != 0:
        raise RuntimeError(
            f"Simulator launcher exited early with code {proc.returncode}"
        )

    return proc


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
    parser.add_argument('--step-timeout', type=float, default=15.0)
    parser.add_argument('--close-timeout', type=float, default=5.0)
    parser.add_argument('--target-z', type=float, default=5.0)
    parser.add_argument('--launch-script', default='as2_sim/launch_sim.bash')
    parser.add_argument('--stop-script', default='as2_sim/stop_sim.bash')
    args = parser.parse_args()

    if args.num_drones != 4:
        print('Only num-drones=4 is supported by this acceptance validator.')
        return 1

    project_root = Path(__file__).resolve().parents[1]
    launch_script = project_root / args.launch_script
    stop_script = project_root / args.stop_script

    namespaces = [f'drone{i}' for i in range(args.num_drones)]
    launch_proc = None
    vec_env = None
    cleanup_errors: list[str] = []

    try:
        launch_proc = launch_simulator(launch_script, args.num_drones, args.launch_wait)

        vec_env = gymnasium.vector.SyncVectorEnv([
            (
                lambda ns=ns: gymnasium.make(
                    'AS2TestEnv-v0',
                    drone_namespace=ns,
                    target_pose=[0.0, 0.0, args.target_z, 0.0],
                    distance_threshold=0.1,
                    pos_limit=20.0,
                    max_steps=max(500, args.steps + 50),
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

        try:
            obs, _ = run_with_timeout(vec_env.reset, args.reset_timeout)
        except TimeoutError:
            return classify_failure(
                'readiness_timeout',
                f'reset() exceeded timeout of {args.reset_timeout}s',
                namespaces,
            )
        except Exception as exc:
            return classify_failure('reset_failure', str(exc), namespaces)

        bad_namespaces = namespaces_with_out_of_range(obs, namespaces)
        if bad_namespaces:
            return classify_failure(
                'observation_range_failure',
                'Reset observation outside [-1, 1]',
                bad_namespaces,
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
            'observation_range': '[-1, 1]',
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

        try:
            stop_simulator(stop_script, namespaces)
        except Exception as exc:
            cleanup_errors.append(str(exc))

        if launch_proc is not None and launch_proc.poll() is None:
            launch_proc.terminate()
            try:
                launch_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                launch_proc.kill()

        if cleanup_errors:
            print(json.dumps({
                'status': 'failure',
                'failure_class': 'cleanup_failure',
                'reason': ' | '.join(cleanup_errors),
                'namespaces': namespaces,
            }, indent=2))


if __name__ == '__main__':
    sys.exit(main())
