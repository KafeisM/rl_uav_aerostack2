#!/usr/bin/env python3
"""Evaluate a trained policy on seeded, reproducible episodes.

The same protocol (seeds, metrics, outputs) will evaluate the classical PID
baseline, so results are directly comparable controller to controller.

Usage:
    conda run -n rl_uav python scripts/evaluate_policy.py \
        --config configs/train_ppo_phase1_exp016.yaml \
        --model runs/ppo/<run>/model_final.zip \
        --episodes 100 --seed 1000
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import gymnasium as gym

import rl_uav  # noqa: F401  # env registration
from rl_uav.evaluation import (
    PIDController,
    PPOController,
    run_evaluation,
    summarize,
    write_outputs,
)
from rl_uav.training import build_env_kwargs, load_training_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Seeded policy evaluation for AS2TestEnv')
    parser.add_argument('--config', type=str, required=True, help='Env YAML config path')
    parser.add_argument(
        '--controller',
        type=str,
        choices=['ppo', 'pid'],
        default='ppo',
        help='Controller under evaluation (default: ppo)',
    )
    parser.add_argument('--model', type=str, default=None, help='SB3 PPO checkpoint .zip (required for --controller ppo)')
    parser.add_argument('--pid-kp-xy', type=float, default=0.8, help='PID proportional gain for x/y (default: 0.8)')
    parser.add_argument('--pid-kp-z', type=float, default=0.8, help='PID proportional gain for z (default: 0.8)')
    parser.add_argument('--pid-kp-yaw', type=float, default=0.8, help='PID proportional gain for yaw (default: 0.8)')
    parser.add_argument('--pid-kd-xy', type=float, default=0.0, help='PID derivative gain for x/y (default: 0.0)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes (default: 100)')
    parser.add_argument('--seed', type=int, default=1000, help='Base seed; episode i uses seed+i (default: 1000)')
    parser.add_argument('--namespace', type=str, default='drone0', help='Drone namespace (default: drone0)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: runs/eval/<auto>)')
    parser.add_argument(
        '--stochastic',
        action='store_true',
        help='Sample actions from the policy distribution instead of the deterministic mean',
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    logger = logging.getLogger('evaluate_policy')

    if args.controller == 'ppo' and args.model is None:
        raise SystemExit('--model is required with --controller ppo')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    controller_stem = Path(args.model).stem if args.controller == 'ppo' else 'pid'
    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else f'runs/eval/{controller_stem}_{Path(args.config).stem}_{timestamp}'
    )

    config = load_training_config(args.config)
    env_cfg = config['environment']
    env_kwargs = build_env_kwargs(env_cfg)
    env = gym.make('AS2TestEnv-v0', drone_namespace=args.namespace, **env_kwargs)
    if args.controller == 'pid':
        controller = PIDController(
            pos_limit=float(env_cfg['pos_limit']),
            max_vel=float(env_cfg['max_vel']),
            max_yaw_vel=float(env_cfg['max_yaw_vel']),
            dt=float(env_cfg['step_duration']),
            kp_xy=args.pid_kp_xy,
            kp_z=args.pid_kp_z,
            kp_yaw=args.pid_kp_yaw,
            kd_xy=args.pid_kd_xy,
        )
    else:
        controller = PPOController.from_checkpoint(
            args.model,
            deterministic=not args.stochastic,
        )
    logger.info('Controller: %s', controller.name)
    logger.info('Episodes: %d (base seed %d)', args.episodes, args.seed)
    logger.info('Output: %s', output_dir)

    def _log_episode(record) -> None:
        logger.info(
            'episode %d/%d: %s steps=%d final_dist=%.2f (start-target %.2f m)',
            record.episode_index + 1,
            args.episodes,
            record.terminal_reason,
            record.steps,
            record.final_distance,
            record.start_target_distance,
        )

    try:
        records = run_evaluation(
            env,
            controller,
            episodes=args.episodes,
            base_seed=args.seed,
            on_episode=_log_episode,
        )
    finally:
        env.close()

    summary = summarize(records)
    write_outputs(
        records,
        summary,
        output_dir,
        metadata={
            'model': str(Path(args.model).resolve()) if args.model else None,
            'config': str(Path(args.config).resolve()),
            'controller': controller.name,
            'episodes': args.episodes,
            'base_seed': args.seed,
            'namespace': args.namespace,
            'timestamp': timestamp,
        },
    )

    logger.info('=== Evaluation summary ===')
    logger.info('success_rate: %s', summary['success_rate'])
    logger.info('terminal_reasons: %s', summary['terminal_reasons'])
    logger.info('final_distance: %s', summary['final_distance'])
    logger.info('path_efficiency (successes): %s', summary['path_efficiency_success'])
    for bin_name, bin_stats in summary['distance_bins'].items():
        logger.info('bin %s m: %s', bin_name, bin_stats)
    logger.info('Full results in %s', output_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
