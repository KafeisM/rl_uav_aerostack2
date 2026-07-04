#!/usr/bin/env python3
"""ROS-free checks for the Exp012 bounds-guard training contract.

Exp012 keeps the exp011 tutor-spec reward and safe-area sampling, and adds
the bounds ACTION guard so boundary terminals (and their slow service
resets) ~never fire. The vertical corridor is widened: flyable band
0.8-2.2 soft-guarded, terminal band below 0.5.

Covers:
    - Config declares the guard with the specified geometry; env constructs.
    - Guard lines strictly enclose the randomized sampling volume.
    - `bounds_guard_active` is monitored.
    - Sampled episodes always start strictly inside every guard line.
    - Reward and PPO hyperparameters are identical to exp011 (tutor spec).

Usage:
    conda run -n rl_uav python scripts/test_exp012_training_contract.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from gymnasium.utils import seeding

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_uav.envs.as2_test_env import AS2TestEnv
from rl_uav.training.ppo_setup import build_env_kwargs

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP012_CONFIG = REPO_ROOT / 'configs' / 'train_ppo_phase1_exp012.yaml'

SAMPLE_ROUNDS = 200


class SamplerTestEnv(AS2TestEnv):
    def _send_speed_command(self, twist, yaw_rate):  # type: ignore[no-untyped-def]
        return True

    def _publish_target_marker(self) -> None:
        return None


def load_exp012_config() -> dict:
    with EXP012_CONFIG.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def make_exp012_env(**overrides) -> SamplerTestEnv:
    kwargs = build_env_kwargs(load_exp012_config()['environment'])
    kwargs['step_duration'] = 0.0
    kwargs['verbose'] = False
    kwargs.update(overrides)
    env = SamplerTestEnv(**kwargs)
    env._np_random, _ = seeding.np_random(7)
    return env


def test_config_declares_guard_geometry_and_env_constructs() -> None:
    env_cfg = load_exp012_config()['environment']
    assert env_cfg['bounds_action_guard'] is True
    assert env_cfg['bounds_guard_margin_xy'] == 0.5
    assert env_cfg['bounds_guard_margin_ceiling'] == 0.3
    assert env_cfg['bounds_guard_push_speed'] == 0.2
    assert env_cfg['height_bounds'] == [0.3, 2.5]
    assert env_cfg['unsafe_low_altitude_threshold'] == 0.5
    assert env_cfg['low_altitude_guard_margin'] == 0.30
    assert env_cfg['low_altitude_guard_climb_speed'] == 0.5
    assert env_cfg['randomization_bounds_margin_z'] == 0.4
    assert env_cfg['reset_ground_recovery_height'] == 1.4

    env = make_exp012_env()  # must not raise ValueError
    assert env.bounds_action_guard is True


def test_guard_lines_strictly_enclose_sampling_volume() -> None:
    # Use the config-declared step_duration so the floor guard height matches
    # the live training geometry (dynamic margin uses step_duration).
    env = make_exp012_env(step_duration=load_exp012_config()['environment']['step_duration'])

    xy_guard_line = env.scene_bounds_xy - env.bounds_guard_margin_xy
    ceiling_guard_line = env.height_bounds[1] - env.bounds_guard_margin_ceiling
    floor_guard_height = env._low_altitude_guard_height()

    sampling_xy_max = env.scene_bounds_xy - env.randomization_bounds_margin
    z_margin = float(env.randomization_bounds_margin_z)
    effective_floor = max(
        float(env.height_bounds[0]),
        float(env.unsafe_low_altitude_threshold),
    )
    sampled_z_low = effective_floor + z_margin
    sampled_z_high = env.height_bounds[1] - z_margin

    assert xy_guard_line == 4.5 and sampling_xy_max == 3.5
    assert xy_guard_line > sampling_xy_max
    assert ceiling_guard_line == 2.2 and sampled_z_high == 2.1
    assert ceiling_guard_line > sampled_z_high
    assert floor_guard_height == 0.8 and sampled_z_low == 0.9
    assert floor_guard_height < sampled_z_low
    assert env.unsafe_low_altitude_threshold == 0.5
    assert env.unsafe_low_altitude_threshold < floor_guard_height


def test_bounds_guard_active_is_monitored() -> None:
    keywords = load_exp012_config()['environment']['monitor_info_keywords']
    assert 'bounds_guard_active' in keywords
    assert 'low_altitude_guard_active' in keywords


def test_sampled_episodes_start_strictly_inside_guard_lines() -> None:
    env = make_exp012_env(step_duration=load_exp012_config()['environment']['step_duration'])
    xy_guard_line = env.scene_bounds_xy - env.bounds_guard_margin_xy
    ceiling_guard_line = env.height_bounds[1] - env.bounds_guard_margin_ceiling
    floor_guard_height = env._low_altitude_guard_height()

    for _ in range(SAMPLE_ROUNDS):
        start_pose, target_pose, _ = env._sample_randomized_episode()
        for pose in (start_pose, target_pose):
            x, y, z = pose[:3]
            assert abs(x) < xy_guard_line, f'x {x} not inside guard line'
            assert abs(y) < xy_guard_line, f'y {y} not inside guard line'
            assert z < ceiling_guard_line, f'z {z} not below ceiling guard'
            assert z > floor_guard_height, f'z {z} not above floor guard'


def test_reward_and_ppo_hyperparameters_match_exp011_tutor_spec() -> None:
    config = load_exp012_config()
    env_cfg = config['environment']
    assert env_cfg['success_reward'] == 20.0
    assert env_cfg['oob_penalty'] == 15.0
    assert env_cfg['terminal_yaw_penalty_weight'] == 10.0
    assert env_cfg['path_facing_weight'] == 0.0
    assert env_cfg['progress_reward_weight'] == 0.0

    ppo_cfg = config['ppo']
    assert ppo_cfg['learning_rate'] == 3.0e-05
    assert ppo_cfg['n_steps'] == 512
    assert ppo_cfg['batch_size'] == 32

    assert config['experiment']['name'] == 'ppo_phase1_exp012_bounds_guard_50k'
    assert config['training']['checkpoint_prefix'] == 'ppo_phase1_exp012'


def main() -> int:
    test_config_declares_guard_geometry_and_env_constructs()
    test_guard_lines_strictly_enclose_sampling_volume()
    test_bounds_guard_active_is_monitored()
    test_sampled_episodes_start_strictly_inside_guard_lines()
    test_reward_and_ppo_hyperparameters_match_exp011_tutor_spec()
    print('✓ PASS: Exp012 bounds-guard training contract holds')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
