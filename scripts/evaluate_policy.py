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
from rl_uav.evaluation import PPOController, run_evaluation, summarize, write_outputs
from rl_uav.training import build_env_kwargs, load_training_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Seeded policy evaluation for AS2TestEnv')
    parser.add_argument('--config', type=str, required=True, help='Env YAML config path')
    parser.add_argument('--model', type=str, required=True, help='SB3 PPO checkpoint .zip')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes (default: 100)')
    parser.add_argument('--seed', type=int, default=1000, help='Base seed; episode i uses seed+i (default: 1000)')
    parser.add_argument('--namespace', type=str, default='drone0', help='Drone namespace (default: drone0)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: runs/eval/<auto>)')
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    logger = logging.getLogger('evaluate_policy')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(
        args.output_dir
        if args.output_dir is not None
        else f'runs/eval/{Path(args.model).stem}_{Path(args.config).stem}_{timestamp}'
    )

    config = load_training_config(args.config)
    env_kwargs = build_env_kwargs(config['environment'])
    env = gym.make('AS2TestEnv-v0', drone_namespace=args.namespace, **env_kwargs)
    controller = PPOController.from_checkpoint(args.model)
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
            'model': str(Path(args.model).resolve()),
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
