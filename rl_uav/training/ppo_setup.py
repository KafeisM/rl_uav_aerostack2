"""Minimal PPO training infrastructure for AS2TestEnv."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch as th
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

import rl_uav  # noqa: F401  # ensure env registration


_ACTIVATIONS: dict[str, type[th.nn.Module]] = {
    'relu': th.nn.ReLU,
    'tanh': th.nn.Tanh,
    'elu': th.nn.ELU,
    'leakyrelu': th.nn.LeakyReLU,
}


@dataclass(frozen=True)
class RunPaths:
    """Output paths for a single training run."""

    run_dir: Path
    tensorboard_dir: Path
    monitor_dir: Path
    checkpoints_dir: Path


def load_training_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML training config and validate required sections."""
    path = Path(config_path)
    with path.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    required_sections = ['experiment', 'environment', 'training', 'ppo']
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ValueError(f'Missing required config sections: {missing}')

    return config


def prepare_run_paths(config: dict[str, Any]) -> RunPaths:
    """Create output directories for logs and checkpoints."""
    experiment = config['experiment']
    training = config['training']

    root = Path(experiment['output_root'])
    run_name = f"{experiment['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = root / run_name
    tensorboard_dir = run_dir / training['tensorboard_log_subdir']
    monitor_dir = run_dir / training['monitor_subdir']
    checkpoints_dir = run_dir / training['checkpoints_subdir']

    for directory in (run_dir, tensorboard_dir, monitor_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_dir=run_dir,
        tensorboard_dir=tensorboard_dir,
        monitor_dir=monitor_dir,
        checkpoints_dir=checkpoints_dir,
    )


def build_env_kwargs(env_cfg: dict[str, Any]) -> dict[str, Any]:
    """Map the YAML environment section to AS2TestEnv constructor kwargs.

    Shared by ``build_vec_env`` and ROS-free diagnostics so every consumer
    derives env parameters from the same config keys and defaults.
    """
    return {
        'use_sim_time': env_cfg['use_sim_time'],
        'verbose': env_cfg['verbose'],
        'max_vel': env_cfg['max_vel'],
        'reset_max_vel': env_cfg.get('reset_max_vel'),
        'reset_xy_kp': env_cfg.get('reset_xy_kp', 1.0),
        'reset_z_kp': env_cfg.get('reset_z_kp', 1.0),
        'max_yaw_vel': env_cfg['max_yaw_vel'],
        'pos_limit': env_cfg['pos_limit'],
        'step_duration': env_cfg['step_duration'],
        'command_publication_interval': env_cfg.get('command_publication_interval', 0.05),
        'min_motion_command_publications': env_cfg.get('min_motion_command_publications', 1),
        'interface_spin_rate': env_cfg.get('interface_spin_rate', 300.0),
        'reset_platform_state_timeout': env_cfg.get('reset_platform_state_timeout', 2.0),
        'target_pose': env_cfg['target_pose'],
        'distance_threshold': env_cfg['distance_threshold'],
        'max_steps': env_cfg['max_steps'],
        'success_reward': env_cfg['success_reward'],
        'oob_penalty': env_cfg['oob_penalty'],
        'path_facing_weight': env_cfg['path_facing_weight'],
        'progress_reward_weight': env_cfg.get('progress_reward_weight', 0.0),
        'terminal_yaw_penalty_weight': env_cfg.get('terminal_yaw_penalty_weight', 0.0),
        'speed_deadband': env_cfg['speed_deadband'],
        'fixed_start_pose': env_cfg.get('fixed_start_pose'),
        'fixed_start_tolerance': env_cfg.get('fixed_start_tolerance', 0.15),
        'fixed_start_timeout': env_cfg.get('fixed_start_timeout', 20.0),
        'reset_yaw_tolerance': env_cfg.get('reset_yaw_tolerance'),
        'reset_yaw_required': env_cfg.get('reset_yaw_required', True),
        'reset_min_speed': env_cfg.get('reset_min_speed', 0.15),
        'reset_ground_recovery_height': env_cfg.get('reset_ground_recovery_height', 0.35),
        'unsafe_low_altitude_threshold': env_cfg.get('unsafe_low_altitude_threshold', 0.30),
        'low_altitude_guard_margin': env_cfg.get('low_altitude_guard_margin', 0.20),
        'low_altitude_guard_climb_speed': env_cfg.get('low_altitude_guard_climb_speed', 0.15),
        'vertical_safety_band': env_cfg.get('vertical_safety_band', 0.0),
        'vertical_safety_penalty_weight': env_cfg.get('vertical_safety_penalty_weight', 0.0),
        'vertical_descent_penalty_weight': env_cfg.get('vertical_descent_penalty_weight', 0.0),
        'publish_target_marker': env_cfg.get('publish_target_marker', False),
        'target_marker_topic': env_cfg.get('target_marker_topic', 'visualization_marker'),
        'target_marker_frame_id': env_cfg.get('target_marker_frame_id', 'earth'),
        'target_marker_scale': env_cfg.get('target_marker_scale', 0.35),
        'close_operation_timeout': env_cfg.get('close_operation_timeout', 10.0),
        'use_service_reset_after_velocity_timeout': env_cfg.get('use_service_reset_after_velocity_timeout', False),
        'randomize_hover_start': env_cfg.get('randomize_hover_start', False),
        'scene_bounds_xy': env_cfg.get('scene_bounds_xy', 5.0),
        'height_bounds': env_cfg.get('height_bounds', [0.1, 2.0]),
        'min_start_target_distance': env_cfg.get('min_start_target_distance'),
        'hover_speed_threshold': env_cfg.get('hover_speed_threshold', 0.05),
        'hover_settle_time': env_cfg.get('hover_settle_time', 1.0),
        'hover_timeout': env_cfg.get('hover_timeout', 10.0),
        'max_reset_sample_attempts': env_cfg.get('max_reset_sample_attempts', 100),
        'randomize_yaw': env_cfg.get('randomize_yaw', True),
        'randomization_bounds_margin': env_cfg.get('randomization_bounds_margin', 0.0),
    }


def build_vec_env(
    config: dict[str, Any],
    monitor_dir: Path,
    num_envs_override: int | None = None,
) -> tuple[VecEnv, list[str]]:
    """Build SB3 vectorized env with unique drone namespaces."""
    env_cfg = config['environment']
    num_envs = int(num_envs_override or env_cfg['num_envs'])
    if num_envs < 1:
        raise ValueError('num_envs must be >= 1')

    namespaces = [f"{env_cfg['namespace_prefix']}{i}" for i in range(num_envs)]
    env_kwargs = build_env_kwargs(env_cfg)

    def _make_env(ns: str, rank: int):
        def _thunk():
            return gym.make(env_cfg['env_id'], drone_namespace=ns, **env_kwargs)

        return _thunk

    env_fns = [_make_env(ns=ns, rank=i) for i, ns in enumerate(namespaces)]
    vec_env_type = str(env_cfg['vec_env']).strip().lower()

    if vec_env_type == 'dummy':
        vec_env: VecEnv = DummyVecEnv(env_fns)
    elif vec_env_type == 'subproc':
        vec_env = SubprocVecEnv(env_fns)
    else:
        raise ValueError("environment.vec_env must be 'dummy' or 'subproc'")

    info_keywords = tuple(env_cfg.get('monitor_info_keywords', []))
    return VecMonitor(
        vec_env,
        filename=str(monitor_dir / 'vec_monitor.csv'),
        info_keywords=info_keywords,
    ), namespaces


def _resolve_policy_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    raw = dict(config['ppo'].get('policy_kwargs', {}))
    activation_name = str(raw.get('activation_fn', 'relu')).strip().lower()
    if activation_name not in _ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation_fn '{raw.get('activation_fn')}'. "
            f"Supported: {list(_ACTIVATIONS.keys())}"
        )

    raw['activation_fn'] = _ACTIVATIONS[activation_name]
    return raw


def build_ppo_model(config: dict[str, Any], vec_env: VecEnv, tensorboard_dir: Path) -> PPO:
    """Build PPO model from declarative config."""
    training_cfg = config['training']
    ppo_cfg = config['ppo']
    policy_kwargs = _resolve_policy_kwargs(config)

    return PPO(
        policy=ppo_cfg['policy'],
        env=vec_env,
        learning_rate=ppo_cfg['learning_rate'],
        n_steps=ppo_cfg['n_steps'],
        batch_size=ppo_cfg['batch_size'],
        n_epochs=ppo_cfg['n_epochs'],
        gamma=ppo_cfg['gamma'],
        gae_lambda=ppo_cfg['gae_lambda'],
        clip_range=ppo_cfg['clip_range'],
        ent_coef=ppo_cfg['ent_coef'],
        vf_coef=ppo_cfg['vf_coef'],
        max_grad_norm=ppo_cfg['max_grad_norm'],
        normalize_advantage=ppo_cfg['normalize_advantage'],
        use_sde=ppo_cfg['use_sde'],
        sde_sample_freq=ppo_cfg['sde_sample_freq'],
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tensorboard_dir),
        verbose=1,
        seed=config['experiment']['seed'],
        device=training_cfg['device'],
    )
