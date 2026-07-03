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


def evaluate_stationary_command_window(
    motion_command_accepted: bool,
    position_delta_norm: float,
    consecutive_accepted_without_motion: int,
    stationary_command_window: int,
    stationary_command_epsilon: float,
) -> dict[str, Any]:
    """Evaluate fail-closed early stop for accepted commands without motion."""
    if motion_command_accepted and position_delta_norm <= float(stationary_command_epsilon):
        next_consecutive = int(consecutive_accepted_without_motion) + 1
    else:
        next_consecutive = 0

    early_stop_reason = None
    if next_consecutive >= int(stationary_command_window):
        early_stop_reason = 'accepted_commands_without_physical_motion'

    return {
        'consecutive_accepted_without_motion': next_consecutive,
        'early_stop_reason': early_stop_reason,
    }


def summarize_episode_motion(
    episode_index: int,
    samples: list[dict[str, Any]],
    min_distance_reduction: float,
    min_position_delta: float,
    require_target_progress: bool = False,
    stationary_command_epsilon: float = 1e-4,
) -> dict[str, Any]:
    """Summarize physical actionability separately from target progress."""
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
    altitude_candidates: list[float] = []
    for sample in samples:
        for key in ('min_altitude_step', 'altitude_before', 'altitude_after'):
            value = _finite_float(sample.get(key), default=float('nan'))
            if math.isfinite(value):
                altitude_candidates.append(value)
    total_position_delta_norm = float(sum(step_delta_norms))
    accepted_command_steps = sum(1 for sample in samples if bool(sample.get('motion_command_accepted', False)))
    accepted_without_motion_steps = sum(
        1
        for sample, delta_norm in zip(samples, step_delta_norms)
        if bool(sample.get('motion_command_accepted', False)) and delta_norm <= float(stationary_command_epsilon)
    )
    early_stop_reasons = [
        str(sample.get('early_stop_reason'))
        for sample in samples
        if sample.get('early_stop_reason')
    ]
    safety_terminal_reasons = [
        str(sample.get('terminal_reason'))
        for sample in samples
        if sample.get('terminal_reason') in {'unsafe_low_altitude', 'out_of_bounds'}
        or bool(sample.get('is_unsafe_low_altitude', False))
        or bool(sample.get('is_out_of_bounds', False))
    ]
    physical_actionability_confirmed = (
        total_position_delta_norm >= float(min_position_delta)
        and accepted_command_steps > 0
    )
    target_progress_confirmed = (
        math.isfinite(distance_reduction)
        and distance_reduction >= float(min_distance_reduction)
    )
    success = physical_actionability_confirmed and (
        target_progress_confirmed or not bool(require_target_progress)
    )
    if success:
        reason = (
            'physical_motion_and_target_progress_confirmed'
            if require_target_progress
            else 'physical_actionability_confirmed'
        )
    elif not physical_actionability_confirmed:
        reason = 'insufficient_physical_motion'
    else:
        reason = 'insufficient_target_progress'
    no_motion_early_stop_reasons = [
        reason
        for reason in early_stop_reasons
        if reason == 'accepted_commands_without_physical_motion'
    ]
    if safety_terminal_reasons:
        success = False
        reason = safety_terminal_reasons[-1]
    elif no_motion_early_stop_reasons:
        success = False
        reason = no_motion_early_stop_reasons[-1]
    if early_stop_reasons and not success:
        reason = (
            reason
            if safety_terminal_reasons or no_motion_early_stop_reasons
            else early_stop_reasons[-1]
        )

    return {
        'episode': episode_index,
        'success': success,
        'reason': reason,
        'steps': len(samples),
        'require_target_progress': bool(require_target_progress),
        'physical_actionability_confirmed': physical_actionability_confirmed,
        'target_progress_confirmed': target_progress_confirmed,
        'initial_distance': initial_distance,
        'final_distance': final_distance,
        'distance_reduction': distance_reduction,
        'total_position_delta_norm': total_position_delta_norm,
        'max_step_position_delta_norm': max(step_delta_norms) if step_delta_norms else 0.0,
        'accepted_command_steps': accepted_command_steps,
        'accepted_without_motion_steps': accepted_without_motion_steps,
        'terminated_or_truncated': bool(last.get('done', False)),
        'terminal_reason': last.get('terminal_reason'),
        'terminal_safety_reason': safety_terminal_reasons[-1] if safety_terminal_reasons else None,
        'min_altitude': min(altitude_candidates) if altitude_candidates else float('nan'),
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


def build_step_sample(
    episode_index: int,
    step_index: int,
    requested_action: list[float],
    before_info: dict[str, Any],
    info: dict[str, Any],
    reward: float,
    done: bool,
    inner_env: Any | None = None,
) -> dict[str, Any]:
    """Build one diagnostic step sample with command-path details."""
    fallback_before = getattr(getattr(inner_env, '_drone', None), 'position', [0.0, 0.0, 0.0])
    before_position = list(before_info.get('position', fallback_before))
    after_position = list(info.get('position', before_position))
    altitude_before = _finite_float(
        info.get(
            'step_altitude_before',
            before_info.get('altitude', before_position[2] if len(before_position) >= 3 else None),
        )
    )
    altitude_after = _finite_float(
        info.get('step_altitude_after', info.get('altitude', after_position[2] if len(after_position) >= 3 else None))
    )
    min_altitude_step = _finite_float(
        info.get('step_min_altitude'),
        default=min(altitude_before, altitude_after),
    )
    speed_before = before_info.get('speed')
    speed_after = info.get('speed')
    step_speed_z_before = _finite_float(
        info.get(
            'step_speed_z_before',
            speed_before[2] if isinstance(speed_before, (list, tuple)) and len(speed_before) >= 3 else None,
        )
    )
    step_speed_z_after = _finite_float(
        info.get(
            'step_speed_z_after',
            speed_after[2] if isinstance(speed_after, (list, tuple)) and len(speed_after) >= 3 else None,
        )
    )
    sample = {
        'episode': episode_index,
        'step': step_index,
        'action': [float(value) for value in requested_action],
        'action_requested': [float(value) for value in requested_action],
        'action_sent': info.get('action_sent'),
        'raw_action': info.get('raw_action'),
        'position_before': [float(value) for value in before_position[:3]],
        'position_after': [float(value) for value in after_position[:3]],
        'position_delta': position_delta(before_position, after_position),
        'altitude_before': altitude_before,
        'altitude_after': altitude_after,
        'min_altitude_step': min_altitude_step,
        'height_bounds': info.get('height_bounds', before_info.get('height_bounds')),
        'unsafe_low_altitude_threshold': info.get(
            'unsafe_low_altitude_threshold',
            before_info.get('unsafe_low_altitude_threshold'),
        ),
        'low_altitude_guard_height': info.get('low_altitude_guard_height', before_info.get('low_altitude_guard_height')),
        'speed_before': speed_before,
        'speed_after': speed_after,
        'speed_z_before': step_speed_z_before,
        'speed_z_after': step_speed_z_after,
        'distance_before': _finite_float(before_info.get('distance')),
        'distance_after': _finite_float(info.get('distance')),
        'reward': float(reward),
        'done': done,
        'terminal_reason': info.get('terminal_reason'),
        'is_unsafe_low_altitude': bool(info.get('is_unsafe_low_altitude', False)),
        'is_out_of_bounds': bool(info.get('is_out_of_bounds', False)),
        'step_safety_terminal': bool(info.get('step_safety_terminal', False)),
        'step_safety_checked_after_publication': bool(
            info.get('step_safety_checked_after_publication', False)
        ),
        'motion_command_accepted': bool(info.get('motion_command_accepted', False)),
        'motion_command_recovered': bool(info.get('motion_command_recovered', False)),
        'motion_command_publication_count': int(info.get('motion_command_publication_count', 0)),
        'motion_command_accepted_publication_count': int(
            info.get('motion_command_accepted_publication_count', 0)
        ),
        'reset_success': bool(info.get('reset_success', True)),
        'low_altitude_guard_active': bool(info.get('low_altitude_guard_active', False)),
        'vertical_action_requested': float(requested_action[2]),
        'vertical_action_sent': (
            float(info['action_sent'][2])
            if isinstance(info.get('action_sent'), (list, tuple)) and len(info['action_sent']) >= 3
            else None
        ),
    }
    if inner_env is not None:
        sample.update({
            'env_step_duration': float(getattr(inner_env, 'step_duration', float('nan'))),
            'env_max_vel': float(getattr(inner_env, 'max_vel', float('nan'))),
            'speed_command_reference_frame': getattr(inner_env, '_last_speed_command_reference_frame', 'unknown'),
        })
    return sample


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run deterministic Exp008 training-path smoke')
    parser.add_argument('--config', default='configs/train_ppo_phase1_exp008.yaml')
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--action', nargs=4, type=float, default=[0.8, 0.0, 0.0, 0.0])
    parser.add_argument('--min-distance-reduction', type=float, default=0.25)
    parser.add_argument('--min-position-delta', type=float, default=0.10)
    parser.add_argument(
        '--require-target-progress',
        action='store_true',
        help=(
            'Require distance-to-target reduction in addition to physical actionability. '
            'Keep disabled for explicit fixed-action command-path smoke unless the action is target-aligned.'
        ),
    )
    parser.add_argument('--output-root', default='runs/diagnostics')
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument('--sleep-between-steps', type=float, default=0.0)
    parser.add_argument('--sleep-between-episodes', type=float, default=0.5)
    parser.add_argument('--stationary-command-window', type=int, default=4)
    parser.add_argument('--stationary-command-epsilon', type=float, default=1e-4)
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
        'require_target_progress': bool(args.require_target_progress),
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
            consecutive_accepted_without_motion = 0

            for step_index in range(int(args.steps)):
                before_info = _safe_env_info(inner_env)
                _, rewards, dones, infos = vec_env.step(action_batch)
                info = dict(infos[0])
                done = bool(dones[0])
                sample = build_step_sample(
                    episode_index=episode_index,
                    step_index=step_index,
                    requested_action=action_batch[0].tolist(),
                    before_info=before_info,
                    info=info,
                    reward=float(rewards[0]),
                    done=done,
                    inner_env=inner_env,
                )
                delta = sample['position_delta']
                delta_norm = _norm(delta)
                stationary_command_decision = evaluate_stationary_command_window(
                    motion_command_accepted=sample['motion_command_accepted'],
                    position_delta_norm=delta_norm,
                    consecutive_accepted_without_motion=consecutive_accepted_without_motion,
                    stationary_command_window=int(args.stationary_command_window),
                    stationary_command_epsilon=float(args.stationary_command_epsilon),
                )
                consecutive_accepted_without_motion = int(
                    stationary_command_decision['consecutive_accepted_without_motion']
                )
                sample['position_delta_norm'] = delta_norm
                sample['consecutive_accepted_without_motion'] = consecutive_accepted_without_motion
                samples.append(sample)
                if stationary_command_decision['early_stop_reason']:
                    sample['early_stop_reason'] = stationary_command_decision['early_stop_reason']
                    break
                if done:
                    break
                if args.sleep_between_steps > 0.0:
                    time.sleep(float(args.sleep_between_steps))

            episode_summary = summarize_episode_motion(
                episode_index=episode_index,
                samples=samples,
                min_distance_reduction=float(args.min_distance_reduction),
                min_position_delta=float(args.min_position_delta),
                require_target_progress=bool(args.require_target_progress),
                stationary_command_epsilon=float(args.stationary_command_epsilon),
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
        physical_actionability_confirmed = bool(summaries) and all(
            summary.get('physical_actionability_confirmed') for summary in summaries
        )
        target_progress_confirmed = bool(summaries) and all(
            summary.get('target_progress_confirmed') for summary in summaries
        )
        service_reset_success = all(
            (not summary['use_simulator_reset_service']) or summary['reset_service_success']
            for summary in reset_summaries
        )
        result['success'] = success and service_reset_success
        result['physical_actionability_confirmed'] = physical_actionability_confirmed
        result['target_progress_confirmed'] = target_progress_confirmed
        safety_reasons = [summary.get('terminal_safety_reason') for summary in summaries if summary.get('terminal_safety_reason')]
        summary_failure_reasons = [
            summary.get('reason')
            for summary in summaries
            if not summary.get('success') and summary.get('reason')
        ]
        if result['success']:
            result['reason'] = (
                'physical_motion_and_target_progress_confirmed_across_training_resets'
                if bool(args.require_target_progress)
                else 'physical_actionability_confirmed_across_training_resets'
            )
        elif safety_reasons:
            result['reason'] = safety_reasons[-1]
        elif not service_reset_success:
            result['reason'] = 'reset_service_failed'
        elif summary_failure_reasons:
            result['reason'] = summary_failure_reasons[-1]
        else:
            result['reason'] = 'training_path_motion_not_confirmed'
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
