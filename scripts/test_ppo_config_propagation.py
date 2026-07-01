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
        'reset_max_vel',
        'reset_xy_kp',
        'reset_z_kp',
        'fixed_start_pose',
        'fixed_start_tolerance',
        'fixed_start_timeout',
        'reset_yaw_tolerance',
        'reset_yaw_required',
        'reset_min_speed',
        'reset_ground_recovery_height',
        'unsafe_low_altitude_threshold',
        'low_altitude_guard_margin',
        'low_altitude_guard_climb_speed',
        'vertical_safety_band',
        'vertical_safety_penalty_weight',
        'vertical_descent_penalty_weight',
        'publish_target_marker',
        'target_marker_topic',
        'target_marker_frame_id',
        'target_marker_scale',
        'close_operation_timeout',
        'use_service_reset_after_velocity_timeout',
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
        assert inner.reset_max_vel == env_cfg['max_vel']
        assert inner.reset_xy_kp == env_cfg['reset_xy_kp']
        assert inner.reset_z_kp == env_cfg['reset_z_kp']
        assert inner.scene_bounds_xy == env_cfg['scene_bounds_xy']
        assert tuple(inner.height_bounds) == tuple(env_cfg['height_bounds'])
        assert inner.min_start_target_distance == env_cfg['min_start_target_distance']
        assert inner.fixed_start_pose == env_cfg['fixed_start_pose']
        assert inner.fixed_start_tolerance == env_cfg['fixed_start_tolerance']
        assert inner.fixed_start_timeout == env_cfg['fixed_start_timeout']
        assert inner.reset_yaw_tolerance == env_cfg['reset_yaw_tolerance']
        assert inner.reset_yaw_required == env_cfg['reset_yaw_required']
        assert inner.reset_min_speed == env_cfg['reset_min_speed']
        assert inner.reset_ground_recovery_height == env_cfg['reset_ground_recovery_height']
        assert inner.unsafe_low_altitude_threshold == env_cfg['unsafe_low_altitude_threshold']
        assert inner.low_altitude_guard_margin == env_cfg['low_altitude_guard_margin']
        assert inner.low_altitude_guard_climb_speed == env_cfg['low_altitude_guard_climb_speed']
        assert inner.vertical_safety_band == env_cfg['vertical_safety_band']
        assert inner.vertical_safety_penalty_weight == env_cfg['vertical_safety_penalty_weight']
        assert inner.vertical_descent_penalty_weight == env_cfg['vertical_descent_penalty_weight']
        assert inner.publish_target_marker == env_cfg['publish_target_marker']
        assert inner.target_marker_topic == env_cfg['target_marker_topic']
        assert inner.target_marker_frame_id == env_cfg['target_marker_frame_id']
        assert inner.target_marker_scale == env_cfg['target_marker_scale']
        assert inner.close_operation_timeout == env_cfg['close_operation_timeout']
        assert inner.use_service_reset_after_velocity_timeout == env_cfg['use_service_reset_after_velocity_timeout']
        assert inner.hover_speed_threshold == env_cfg['hover_speed_threshold']
        assert inner.hover_settle_time == env_cfg['hover_settle_time']
        assert inner.hover_timeout == env_cfg['hover_timeout']
        assert inner.max_reset_sample_attempts == env_cfg['max_reset_sample_attempts']
        monitor_keywords = set(env_cfg['monitor_info_keywords'])
        for keyword in [
            'physical_displacement',
            'path_length',
            'max_physical_displacement',
            'min_altitude',
            'motion_command_steps',
            'motion_command_accepted_steps',
            'motion_command_acceptance_rate',
            'reset_method',
            'reset_path',
            'reset_service_attempted',
            'reset_failure_class',
            'reset_position_error',
            'reset_yaw_error',
            'reset_service_status',
            'vertical_safety_penalty',
            'low_altitude_guard_active',
        ]:
            assert keyword in monitor_keywords, f'Missing PPO monitor diagnostic keyword: {keyword}'
        print('✓ PASS: Randomized hover env config propagates to AS2TestEnv')
    finally:
        vec_env2.close()

    exp008_config = load_training_config(Path('configs/train_ppo_phase1_exp008.yaml'))
    exp008_env_cfg = exp008_config['environment']
    exp008_ppo_cfg = exp008_config['ppo']
    assert exp008_env_cfg['fixed_start_tolerance'] >= 0.22
    assert exp008_env_cfg['fixed_start_timeout'] >= 30.0
    print('✓ PASS: Exp008 reset tolerance covers live post-controller settle slack')

    exp008a_config = load_training_config(Path('configs/train_ppo_phase1_exp008a.yaml'))
    exp008a_env_cfg = exp008a_config['environment']
    exp008a_training_cfg = exp008a_config['training']
    exp008a_ppo_cfg = exp008a_config['ppo']
    assert exp008a_config['experiment']['name'].startswith('ppo_phase1_exp008a')
    assert exp008a_training_cfg['total_timesteps'] <= exp008_config['training']['total_timesteps']
    assert exp008a_training_cfg['checkpoint_prefix'] == 'ppo_phase1_exp008a'
    assert exp008a_env_cfg['fixed_start_pose'][2] == exp008_env_cfg['fixed_start_pose'][2]
    assert exp008a_env_cfg['target_pose'][2] == exp008_env_cfg['target_pose'][2]
    assert exp008a_env_cfg['target_pose'][0] < exp008_env_cfg['target_pose'][0]
    assert exp008a_env_cfg['target_pose'][0] <= 0.6
    assert exp008a_env_cfg['distance_threshold'] >= exp008_env_cfg['distance_threshold']
    assert exp008a_env_cfg['distance_threshold'] >= 0.4
    assert 0.0 < exp008a_env_cfg['path_facing_weight'] < exp008_env_cfg['path_facing_weight']
    assert exp008_env_cfg['reset_yaw_required'] is True
    assert exp008a_env_cfg['reset_yaw_required'] is False
    assert exp008a_env_cfg['reset_yaw_tolerance'] > exp008_env_cfg['reset_yaw_tolerance']
    assert exp008a_env_cfg['reset_yaw_tolerance'] >= 1.0
    assert exp008a_env_cfg['fixed_start_timeout'] > exp008_env_cfg['fixed_start_timeout']
    assert exp008a_env_cfg['hover_timeout'] > exp008_env_cfg['hover_timeout']
    assert exp008a_env_cfg['max_steps'] < exp008_env_cfg['max_steps']
    assert exp008a_env_cfg['max_steps'] <= 80
    assert exp008a_env_cfg['max_vel'] <= exp008_env_cfg['max_vel']
    assert exp008a_env_cfg['max_vel'] <= 0.5
    assert exp008a_env_cfg['reset_max_vel'] > exp008a_env_cfg['max_vel']
    assert exp008a_env_cfg['reset_max_vel'] >= 1.0
    assert exp008a_env_cfg['unsafe_low_altitude_threshold'] >= exp008_env_cfg['unsafe_low_altitude_threshold']
    assert exp008a_env_cfg['unsafe_low_altitude_threshold'] >= 1.3
    assert exp008a_env_cfg['unsafe_low_altitude_threshold'] < exp008a_env_cfg['reset_ground_recovery_height']
    assert exp008a_env_cfg['reset_ground_recovery_height'] < exp008a_env_cfg['fixed_start_pose'][2]
    assert exp008a_env_cfg['low_altitude_guard_margin'] >= exp008_env_cfg['low_altitude_guard_margin']
    assert exp008a_env_cfg['vertical_safety_penalty_weight'] >= exp008_env_cfg['vertical_safety_penalty_weight']
    assert exp008a_env_cfg['vertical_descent_penalty_weight'] >= exp008_env_cfg['vertical_descent_penalty_weight']
    assert exp008a_env_cfg['randomize_hover_start'] is False
    assert exp008a_env_cfg['use_service_reset_after_velocity_timeout'] is True
    assert exp008a_ppo_cfg['n_steps'] <= exp008_ppo_cfg['n_steps']
    print('✓ PASS: Exp008a easier curriculum config is bounded and propagated')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
