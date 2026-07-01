#!/usr/bin/env python3
"""Deterministic Exp008 smoke through the PPO training env builder.

This diagnostic isolates the AS2 command path from PPO policy quality. It uses
the same configuration and `build_vec_env()` path as `scripts/train_ppo.py`, but
drives a deterministic positive-x action so per-step physical motion can be
measured across explicit episode resets.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from rl_uav.training import build_vec_env, load_training_config


def build_constant_action_batch(num_envs: int, action: list[float]) -> np.ndarray:
    """Build an SB3 VecEnv action batch from one 4D AS2 action."""
    if num_envs < 1:
        raise ValueError('num_envs must be >= 1')
    action_array = np.asarray(action, dtype=np.float32)
    if action_array.shape != (4,):
        raise ValueError(f'action must have shape (4,), got {action_array.shape}')
    return np.repeat(action_array.reshape(1, 4), repeats=num_envs, axis=0)


def position_delta(before: list[float], after: list[float]) -> list[float]:
    """Return element-wise XYZ position delta."""
    if len(before) < 3 or len(after) < 3:
        raise ValueError('before and after positions must contain at least XYZ')
    return [float(after[i]) - float(before[i]) for i in range(3)]


def _finite_float(value: Any, default: float = float('nan')) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if math.isfinite(converted) else default


def _norm(values: list[float]) -> float:
    return float(math.sqrt(sum(float(value) * float(value) for value in values)))


def summarize_episode_motion(
    episode_index: int,
    samples: list[dict[str, Any]],
    min_distance_reduction: float,
    min_position_delta: float,
) -> dict[str, Any]:
    """Summarize whether accepted commands produced measured physical motion."""
    if not samples:
        return {
            'episode': episode_index,
            'success': False,
            'reason': 'no_step_samples',
            'steps': 0,
        }

    first = samples[0]
    last = samples[-1]
    initial_distance = _finite_float(first.get('distance_before'), default=float('nan'))
    if not math.isfinite(initial_distance):
        initial_distance = _finite_float(first.get('distance_after'), default=float('nan'))
    final_distance = _finite_float(last.get('distance_after'), default=float('nan'))
    distance_reduction = initial_distance - final_distance

    step_delta_norms = [_norm(list(sample.get('position_delta', [0.0, 0.0, 0.0]))) for sample in samples]
    total_position_delta_norm = float(sum(step_delta_norms))
    accepted_command_steps = sum(1 for sample in samples if bool(sample.get('motion_command_accepted', False)))
    moved_enough = (
        math.isfinite(distance_reduction)
        and distance_reduction >= float(min_distance_reduction)
        and total_position_delta_norm >= float(min_position_delta)
        and accepted_command_steps > 0
    )

    return {
        'episode': episode_index,
        'success': moved_enough,
        'reason': 'distance_reduced_after_positive_x_actions' if moved_enough else 'insufficient_physical_motion',
        'steps': len(samples),
        'initial_distance': initial_distance,
        'final_distance': final_distance,
        'distance_reduction': distance_reduction,
        'total_position_delta_norm': total_position_delta_norm,
        'max_step_position_delta_norm': max(step_delta_norms) if step_delta_norms else 0.0,
        'accepted_command_steps': accepted_command_steps,
        'terminated_or_truncated': bool(last.get('done', False)),
        'terminal_reason': last.get('terminal_reason'),
    }


def summarize_reset_diagnostics(
    use_simulator_reset_service: bool,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Return a compact reset-path summary for smoke logs."""
    reason = str(diagnostics.get('reason', 'unknown'))
    service_attempted = bool(use_simulator_reset_service) and (
        (reason.startswith('service_') and reason != 'service_disabled')
        or reason in {'post_reset_arm_failed', 'post_reset_offboard_failed'}
    )
    return {
        'use_simulator_reset_service': bool(use_simulator_reset_service),
        'service_path': reason if use_simulator_reset_service else 'disabled',
        'service_attempted': service_attempted,
        'reset_service_success': (not service_attempted) or reason == 'service_success',
        'diagnostics': dict(diagnostics),
    }


def _single_inner_env(vec_env: Any) -> Any:
    current = vec_env
    while hasattr(current, 'venv'):
        current = current.venv
    envs = getattr(current, 'envs', None)
    if not envs or len(envs) != 1:
        raise RuntimeError('Exp008 diagnostic expects exactly one VecEnv environment')
    env = envs[0]
    return getattr(env, 'unwrapped', env)


def _safe_env_info(env: Any) -> dict[str, Any]:
    try:
        return dict(env._get_info())
    except Exception as exc:  # pragma: no cover - diagnostic resilience
        return {'info_error': str(exc)}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run deterministic Exp008 training-path smoke')
    parser.add_argument('--config', default='configs/train_ppo_phase1_exp008.yaml')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--action', nargs=4, type=float, default=[0.8, 0.0, 0.0, 0.0])
    parser.add_argument('--min-distance-reduction', type=float, default=0.25)
    parser.add_argument('--min-position-delta', type=float, default=0.10)
    parser.add_argument('--output-root', default='runs/diagnostics')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--sleep-between-steps', type=float, default=0.0)
    parser.add_argument('--sleep-between-episodes', type=float, default=0.5)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = load_training_config(args.config)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(args.output_root) / f'exp008_training_smoke_{timestamp}'
    monitor_dir = run_dir / 'monitor'
    run_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        'config': str(Path(args.config).resolve()),
        'run_dir': str(run_dir.resolve()),
        'monitor_dir': str(monitor_dir.resolve()),
        'episodes_requested': args.episodes,
        'steps_requested': args.steps,
        'action': [float(value) for value in args.action],
        'min_distance_reduction': args.min_distance_reduction,
        'min_position_delta': args.min_position_delta,
        'episodes': [],
    }

    vec_env = None
    try:
        vec_env, namespaces = build_vec_env(
            config=config,
            monitor_dir=monitor_dir,
            num_envs_override=args.num_envs,
        )
        if vec_env.num_envs != 1:
            raise RuntimeError(f'Expected one env for diagnostic, got {vec_env.num_envs}')
        inner_env = _single_inner_env(vec_env)
        action_batch = build_constant_action_batch(vec_env.num_envs, [float(value) for value in args.action])
        result['namespaces'] = namespaces
        result['vec_env_num_envs'] = vec_env.num_envs

        for episode_index in range(int(args.episodes)):
            vec_env.reset()
            reset_info = _safe_env_info(inner_env)
            reset_summary = summarize_reset_diagnostics(
                use_simulator_reset_service=getattr(inner_env, 'use_simulator_reset_service', False),
                diagnostics=dict(getattr(inner_env, '_last_reset_diagnostics', {})),
            )
            samples: list[dict[str, Any]] = []

            for step_index in range(int(args.steps)):
                before_info = _safe_env_info(inner_env)
                before_position = list(before_info.get('position', getattr(inner_env._drone, 'position', [0.0, 0.0, 0.0])))
                _, rewards, dones, infos = vec_env.step(action_batch)
                info = dict(infos[0])
                after_position = list(info.get('position', getattr(inner_env._drone, 'position', before_position)))
                delta = position_delta(before_position, after_position)
                done = bool(dones[0])
                sample = {
                    'episode': episode_index,
                    'step': step_index,
                    'action': action_batch[0].tolist(),
                    'position_before': [float(value) for value in before_position[:3]],
                    'position_after': [float(value) for value in after_position[:3]],
                    'position_delta': delta,
                    'distance_before': _finite_float(before_info.get('distance')),
                    'distance_after': _finite_float(info.get('distance')),
                    'reward': float(rewards[0]),
                    'done': done,
                    'terminal_reason': info.get('terminal_reason'),
                    'motion_command_accepted': bool(info.get('motion_command_accepted', False)),
                    'motion_command_recovered': bool(info.get('motion_command_recovered', False)),
                    'reset_success': bool(info.get('reset_success', True)),
                }
                samples.append(sample)
                if done:
                    break
                if args.sleep_between_steps > 0.0:
                    time.sleep(float(args.sleep_between_steps))

            episode_summary = summarize_episode_motion(
                episode_index=episode_index,
                samples=samples,
                min_distance_reduction=float(args.min_distance_reduction),
                min_position_delta=float(args.min_position_delta),
            )
            result['episodes'].append({
                'episode': episode_index,
                'reset_info_after_reset': reset_info,
                'reset_summary': reset_summary,
                'summary': episode_summary,
                'samples': samples,
            })
            if episode_index + 1 < int(args.episodes) and args.sleep_between_episodes > 0.0:
                time.sleep(float(args.sleep_between_episodes))

        summaries = [episode['summary'] for episode in result['episodes']]
        reset_summaries = [episode['reset_summary'] for episode in result['episodes']]
        success = bool(summaries) and all(summary['success'] for summary in summaries)
        service_reset_success = all(
            (not summary['use_simulator_reset_service']) or summary['reset_service_success']
            for summary in reset_summaries
        )
        result['success'] = success and service_reset_success
        result['reason'] = (
            'physical_motion_confirmed_across_training_resets'
            if result['success']
            else 'training_path_motion_not_confirmed'
        )
        result['episode_summaries'] = summaries
        result['reset_summaries'] = reset_summaries
        output_path = run_dir / 'diagnostic.json'
        output_path.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True), encoding='utf-8')
        result['output_path'] = str(output_path.resolve())
        print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
        return 0 if result['success'] else 3
    except Exception as exc:
        result['success'] = False
        result['reason'] = 'diagnostic_error'
        result['error'] = str(exc)
        output_path = run_dir / 'diagnostic.json'
        output_path.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True), encoding='utf-8')
        result['output_path'] = str(output_path.resolve())
        print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
        return 2
    finally:
        if vec_env is not None:
            vec_env.close()


if __name__ == '__main__':
    raise SystemExit(main())
