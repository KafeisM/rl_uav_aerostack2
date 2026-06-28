#!/usr/bin/env python3
"""Validate PPO config propagation into SB3 model construction.

Usage:
    conda run -n rl_uav python3 scripts/test_ppo_config_propagation.py
"""

from __future__ import annotations

from pathlib import Path
import warnings

import gymnasium as gym
import torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from rl_uav.training import build_ppo_model, build_vec_env, load_training_config


def _linear_out_features(module: nn.Module) -> list[int]:
    return [m.out_features for m in module.modules() if isinstance(m, nn.Linear)]


def main() -> int:
    config = load_training_config(Path('configs/train_ppo.yaml'))

    # New randomized-hover keys must be present in config defaults.
    env_cfg = config['environment']
    for key in [
        'randomize_hover_start',
        'scene_bounds_xy',
        'height_bounds',
        'min_start_target_distance',
        'progress_reward_weight',
        'fixed_start_pose',
        'fixed_start_tolerance',
        'fixed_start_timeout',
        'reset_min_speed',
        'reset_ground_recovery_height',
        'publish_target_marker',
        'target_marker_topic',
        'target_marker_frame_id',
        'target_marker_scale',
        'close_operation_timeout',
        'monitor_info_keywords',
        'hover_speed_threshold',
        'hover_settle_time',
        'hover_timeout',
        'max_reset_sample_attempts',
    ]:
        assert key in env_cfg, f"Missing environment key: {key}"

    vec_env = DummyVecEnv([lambda: gym.make('Pendulum-v1')])
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = build_ppo_model(config=config, vec_env=vec_env, tensorboard_dir=Path('/tmp'))

        warning_text = '\n'.join(str(item.message) for item in caught)
        assert 'You are trying to run PPO on the GPU' not in warning_text

        ppo_cfg = config['ppo']

        assert model.n_steps == ppo_cfg['n_steps']
        assert model.batch_size == ppo_cfg['batch_size']
        assert model.n_epochs == ppo_cfg['n_epochs']
        assert model.normalize_advantage is ppo_cfg['normalize_advantage']
        assert model.use_sde is ppo_cfg['use_sde']
        assert model.sde_sample_freq == ppo_cfg['sde_sample_freq']
        assert model.device.type == 'cpu'

        # SB3 accepts constant LR as a float and exposes a schedule callable.
        assert float(model.learning_rate) == float(ppo_cfg['learning_rate'])
        assert float(model.lr_schedule(1.0)) == float(ppo_cfg['learning_rate'])

        assert model.policy.activation_fn is nn.ReLU

        pi_layers = _linear_out_features(model.policy.mlp_extractor.policy_net)
        vf_layers = _linear_out_features(model.policy.mlp_extractor.value_net)
        assert pi_layers == ppo_cfg['policy_kwargs']['net_arch']['pi']
        assert vf_layers == ppo_cfg['policy_kwargs']['net_arch']['vf']

        print('✓ PASS: PPO config fields are propagated to the SB3 model')
    finally:
        vec_env.close()

    vec_env2, namespaces = build_vec_env(config=config, monitor_dir=Path('/tmp'), num_envs_override=1)
    try:
        assert namespaces == ['drone0']
        # VecMonitor -> DummyVecEnv -> AS2TestEnv. Individual envs are not
        # Monitor-wrapped because VecMonitor owns episode metrics/logging.
        assert type(vec_env2).__name__ == 'VecMonitor'
        assert not isinstance(vec_env2.venv.envs[0], Monitor)
        inner = vec_env2.venv.envs[0].unwrapped
        assert inner.randomize_hover_start == env_cfg['randomize_hover_start']
        assert inner.progress_reward_weight == env_cfg['progress_reward_weight']
        assert inner.scene_bounds_xy == env_cfg['scene_bounds_xy']
        assert tuple(inner.height_bounds) == tuple(env_cfg['height_bounds'])
        assert inner.min_start_target_distance == env_cfg['min_start_target_distance']
        assert inner.fixed_start_pose == env_cfg['fixed_start_pose']
        assert inner.fixed_start_tolerance == env_cfg['fixed_start_tolerance']
        assert inner.fixed_start_timeout == env_cfg['fixed_start_timeout']
        assert inner.reset_min_speed == env_cfg['reset_min_speed']
        assert inner.reset_ground_recovery_height == env_cfg['reset_ground_recovery_height']
        assert inner.publish_target_marker == env_cfg['publish_target_marker']
        assert inner.target_marker_topic == env_cfg['target_marker_topic']
        assert inner.target_marker_frame_id == env_cfg['target_marker_frame_id']
        assert inner.target_marker_scale == env_cfg['target_marker_scale']
        assert inner.close_operation_timeout == env_cfg['close_operation_timeout']
        assert inner.hover_speed_threshold == env_cfg['hover_speed_threshold']
        assert inner.hover_settle_time == env_cfg['hover_settle_time']
        assert inner.hover_timeout == env_cfg['hover_timeout']
        assert inner.max_reset_sample_attempts == env_cfg['max_reset_sample_attempts']
        print('✓ PASS: Randomized hover env config propagates to AS2TestEnv')
    finally:
        vec_env2.close()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
