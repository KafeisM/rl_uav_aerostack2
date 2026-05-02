#!/usr/bin/env python3
"""Validate PPO config propagation into SB3 model construction.

Usage:
    conda run -n rl_uav python3 scripts/test_ppo_config_propagation.py
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_uav.training import build_ppo_model, load_training_config


def _linear_out_features(module: nn.Module) -> list[int]:
    return [m.out_features for m in module.modules() if isinstance(m, nn.Linear)]


def main() -> int:
    config = load_training_config(Path('configs/train_ppo.yaml'))

    vec_env = DummyVecEnv([lambda: gym.make('Pendulum-v1')])
    try:
        model = build_ppo_model(config=config, vec_env=vec_env, tensorboard_dir=Path('/tmp'))

        ppo_cfg = config['ppo']

        assert model.n_steps == ppo_cfg['n_steps']
        assert model.batch_size == ppo_cfg['batch_size']
        assert model.n_epochs == ppo_cfg['n_epochs']
        assert model.normalize_advantage is ppo_cfg['normalize_advantage']
        assert model.use_sde is ppo_cfg['use_sde']
        assert model.sde_sample_freq == ppo_cfg['sde_sample_freq']

        # SB3 accepts constant LR as a float and exposes a schedule callable.
        assert float(model.learning_rate) == float(ppo_cfg['learning_rate'])
        assert float(model.lr_schedule(1.0)) == float(ppo_cfg['learning_rate'])

        assert model.policy.activation_fn is nn.ReLU

        pi_layers = _linear_out_features(model.policy.mlp_extractor.policy_net)
        vf_layers = _linear_out_features(model.policy.mlp_extractor.value_net)
        assert pi_layers == ppo_cfg['policy_kwargs']['net_arch']['pi']
        assert vf_layers == ppo_cfg['policy_kwargs']['net_arch']['vf']

        print('✓ PASS: PPO config fields are propagated to the SB3 model')
        return 0
    finally:
        vec_env.close()


if __name__ == '__main__':
    raise SystemExit(main())
