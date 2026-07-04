#!/usr/bin/env python3
"""ROS-free checks for the Exp011 safe-area randomization contract.

Exp011 keeps the exp010 tutor-spec reward but fixes the two episode-sampling
problems observed in the exp010 run:
    - Observation saturation: start-target separation could exceed pos_limit,
      clipping obs components at +/-1 and destroying the distance gradient.
      Fixed by capping start-target distance below pos_limit.
    - Boundary-adjacent spawns: a 0.3 m margin put untrained policies ~3 steps
      from an out-of-bounds terminal (93% OOB episodes, one ~22 s service
      teleport per episode). Fixed by a wide XY safe area and a separate,
      feasible Z margin.

Usage:
    conda run -n rl_uav python scripts/test_exp011_randomization_contract.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import yaml
from gymnasium.utils import seeding

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_uav.envs.as2_test_env import AS2TestEnv
from rl_uav.training.ppo_setup import build_env_kwargs

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP011_CONFIG = REPO_ROOT / 'configs' / 'train_ppo_phase1_exp011.yaml'

SAMPLE_ROUNDS = 300


class DummyDrone:
    def __init__(self, position: list[float], yaw: float = 0.0):
        self.position = list(position)
        self.speed = [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, float(yaw)]


class SamplerTestEnv(AS2TestEnv):
    def _send_speed_command(self, twist, yaw_rate):  # type: ignore[no-untyped-def]
        return True

    def _publish_target_marker(self) -> None:
        return None


def load_exp011_env_cfg() -> dict:
    with EXP011_CONFIG.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)['environment']


def make_exp011_env(**overrides) -> SamplerTestEnv:
    kwargs = build_env_kwargs(load_exp011_env_cfg())
    kwargs['step_duration'] = 0.0
    kwargs['verbose'] = False
    kwargs.update(overrides)
    env = SamplerTestEnv(**kwargs)
    env._np_random, _ = seeding.np_random(7)
    return env


def sample_many(env: AS2TestEnv, rounds: int = SAMPLE_ROUNDS):
    for _ in range(rounds):
        yield env._sample_randomized_episode()


def test_config_declares_safe_area_geometry() -> None:
    env_cfg = load_exp011_env_cfg()
    assert env_cfg['randomization_bounds_margin'] >= 1.5
    assert env_cfg['randomization_bounds_margin_z'] is not None
    assert env_cfg['max_start_target_distance'] <= env_cfg['pos_limit']
    assert env_cfg['reset_service_max_attempts'] >= 3


def test_sampler_enforces_max_start_target_distance() -> None:
    env = make_exp011_env()
    max_distance = float(env.max_start_target_distance)
    min_distance = float(env.min_start_target_distance)
    for start_pose, target_pose, _ in sample_many(env):
        distance = math.dist(start_pose[:3], target_pose[:3])
        assert min_distance < distance < max_distance, (
            f'distance {distance} outside ({min_distance}, {max_distance})'
        )


def test_sampler_keeps_xy_inside_safe_area() -> None:
    env = make_exp011_env()
    xy_limit = env.scene_bounds_xy - env.randomization_bounds_margin
    for start_pose, target_pose, _ in sample_many(env):
        for pose in (start_pose, target_pose):
            assert abs(pose[0]) <= xy_limit
            assert abs(pose[1]) <= xy_limit


def test_sampler_applies_z_margin_above_effective_floor() -> None:
    env = make_exp011_env()
    z_margin = float(env.randomization_bounds_margin_z)
    effective_floor = max(
        float(env.height_bounds[0]),
        float(env.unsafe_low_altitude_threshold),
    )
    z_low = effective_floor + z_margin
    z_high = float(env.height_bounds[1]) - z_margin
    assert z_low < z_high, 'exp011 z margin must leave a sampling range'
    for start_pose, target_pose, _ in sample_many(env):
        for pose in (start_pose, target_pose):
            assert z_low <= pose[2] <= z_high


def test_initial_observation_never_saturates() -> None:
    env = make_exp011_env()
    for start_pose, target_pose, _ in sample_many(env):
        env._drone = DummyDrone(start_pose[:3], yaw=start_pose[3])
        env._target_pose = list(target_pose)
        obs = env._get_obs()
        assert np.all(np.abs(obs[:3]) < 1.0), (
            f'saturated obs {obs} for start {start_pose} target {target_pose}'
        )


def test_max_distance_must_exceed_min_distance() -> None:
    try:
        make_exp011_env(max_start_target_distance=0.5, min_start_target_distance=1.0)
    except ValueError as exc:
        assert 'max_start_target_distance' in str(exc)
    else:
        raise AssertionError('max <= min start-target distance must raise')


def test_z_margin_validation_rejects_empty_range() -> None:
    try:
        make_exp011_env(randomization_bounds_margin_z=2.0)
    except ValueError as exc:
        assert 'randomization_bounds_margin_z' in str(exc)
    else:
        raise AssertionError('infeasible z margin must raise ValueError')


def test_z_margin_falls_back_to_xy_margin_when_unset() -> None:
    env = make_exp011_env(
        randomization_bounds_margin=0.3,
        randomization_bounds_margin_z=None,
    )
    effective_floor = max(
        float(env.height_bounds[0]),
        float(env.unsafe_low_altitude_threshold),
    )
    for start_pose, target_pose, _ in sample_many(env, rounds=50):
        for pose in (start_pose, target_pose):
            assert effective_floor + 0.3 <= pose[2] <= env.height_bounds[1] - 0.3


def test_sampling_is_feasible_within_attempt_budget() -> None:
    env = make_exp011_env()
    attempts = [attempt for _, _, attempt in sample_many(env)]
    assert max(attempts) < env.max_reset_sample_attempts, (
        f'sampler too close to attempt budget: max {max(attempts)}'
    )


def main() -> int:
    test_config_declares_safe_area_geometry()
    test_sampler_enforces_max_start_target_distance()
    test_sampler_keeps_xy_inside_safe_area()
    test_sampler_applies_z_margin_above_effective_floor()
    test_initial_observation_never_saturates()
    test_max_distance_must_exceed_min_distance()
    test_z_margin_validation_rejects_empty_range()
    test_z_margin_falls_back_to_xy_margin_when_unset()
    test_sampling_is_feasible_within_attempt_budget()
    print('✓ PASS: Exp011 safe-area randomization contract holds')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
