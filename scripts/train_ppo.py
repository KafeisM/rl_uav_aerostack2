#!/usr/bin/env python3
"""Train PPO on AS2TestEnv with declarative configuration."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from rl_uav.training import (
    build_ppo_model,
    build_vec_env,
    load_training_config,
    prepare_run_paths,
    resolve_resume_checkpoint,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='PPO training entry point for AS2TestEnv')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_ppo.yaml',
        help='Path to YAML config file (default: configs/train_ppo.yaml)',
    )
    parser.add_argument(
        '--num-envs',
        type=int,
        default=None,
        help='Override number of environments declared in config',
    )
    parser.add_argument(
        '--total-timesteps',
        type=int,
        default=None,
        help='Override total training timesteps declared in config',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Build config/env/model and print summary without calling learn()',
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help=(
            "Checkpoint .zip path to resume from, or 'latest' to pick the "
            "newest checkpoint matching the config's checkpoint_prefix"
        ),
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )
    logger = logging.getLogger('train_ppo')

    config = load_training_config(args.config)
    resume_checkpoint = None
    if args.resume_from is not None:
        resume_checkpoint = resolve_resume_checkpoint(
            args.resume_from,
            config,
            output_root=config['experiment']['output_root'],
        )
    paths = prepare_run_paths(config)
    vec_env = None

    try:
        vec_env, namespaces = build_vec_env(
            config=config,
            monitor_dir=paths.monitor_dir,
            num_envs_override=args.num_envs,
        )
        if resume_checkpoint is not None:
            model = PPO.load(
                str(resume_checkpoint),
                env=vec_env,
                device=config['training'].get('device', 'cpu'),
            )
            model.tensorboard_log = str(paths.tensorboard_dir)
            logger.info(
                'Resuming from %s at num_timesteps=%d',
                resume_checkpoint,
                model.num_timesteps,
            )
        else:
            model = build_ppo_model(config=config, vec_env=vec_env, tensorboard_dir=paths.tensorboard_dir)

        checkpoint_cb = CheckpointCallback(
            save_freq=config['training']['checkpoint_freq'],
            save_path=str(paths.checkpoints_dir),
            name_prefix=config['training']['checkpoint_prefix'],
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        logger.info('=== PPO infrastructure ready ===')
        logger.info('Config: %s', Path(args.config).resolve())
        logger.info('Run dir: %s', paths.run_dir.resolve())
        logger.info('Tensorboard dir: %s', paths.tensorboard_dir.resolve())
        logger.info('Monitor dir: %s', paths.monitor_dir.resolve())
        logger.info('Checkpoints dir: %s', paths.checkpoints_dir.resolve())
        logger.info('Num envs: %d', vec_env.num_envs)
        logger.info('Namespaces: %s', namespaces)
        logger.info('Policy kwargs: %s', config['ppo']['policy_kwargs'])
        logger.info('Checkpoint callback: every %s steps', config['training']['checkpoint_freq'])

        total_timesteps = (
            args.total_timesteps
            if args.total_timesteps is not None
            else config['training']['total_timesteps']
        )
        logger.info('Total timesteps: %d', total_timesteps)

        if args.dry_run:
            if resume_checkpoint is not None:
                logger.info(
                    'Dry-run resume summary: checkpoint=%s num_timesteps=%d',
                    resume_checkpoint,
                    model.num_timesteps,
                )
            logger.info('Dry-run enabled: model/env built successfully, skipping learn().')
            return 0

        logger.info('Starting PPO learn() ...')
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_cb,
            progress_bar=False,
            reset_num_timesteps=resume_checkpoint is None,
        )
        model.save(str(paths.run_dir / 'model_final'))
        logger.info('Training completed. Final model saved at: %s', paths.run_dir / 'model_final.zip')
        return 0
    finally:
        if vec_env is not None:
            vec_env.close()


if __name__ == '__main__':
    raise SystemExit(main())
