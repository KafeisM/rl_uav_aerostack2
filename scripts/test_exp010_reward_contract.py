#!/usr/bin/env python3
"""ROS-free checks for the Exp010 tutor-spec reward and randomization contract.

Covers:
    - Continuous reward is exactly -d_norm with exp010 weights (no hidden terms).
    - Success terminal reward is -d_norm + success_reward - w * yaw_err_norm.
    - Terminal yaw keys are emitted on every episode-ending path (SB3 Monitor
      reads every monitor_info_keywords entry at episode end).
    - Randomized sampler: target yaw randomization, bounds margin, min-distance.

Usage:
    conda run -n rl_uav python scripts/test_exp010_reward_contract.py
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
EXP010_CONFIG = REPO_ROOT / 'configs' / 'train_ppo_phase1_exp010.yaml'


class DummyDrone:
    def __init__(self, position: list[float], speed=None, yaw: float = 0.0):
        self.position = list(position)
        self.speed = list(speed) if speed is not None else [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, float(yaw)]


class RewardTestEnv(AS2TestEnv):
    def _send_speed_command(self, twist, yaw_rate):  # type: ignore[no-untyped-def]
        self.last_command = [float(value) for value in twist] + [float(yaw_rate)]
        return True

    def _send_terminal_stop_command(self) -> bool:
        return True

    def _publish_target_marker(self) -> None:
        return None


def load_exp010_config() -> dict:
    with EXP010_CONFIG.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def make_exp010_env(position: list[float], speed=None, yaw: float = 0.0) -> RewardTestEnv:
    """Instantiate the env exactly as training does: from the exp010 config."""
    env_cfg = load_exp010_config()['environment']
    kwargs = build_env_kwargs(env_cfg)
    # Keep the mock step instantaneous; reward terms do not depend on it.
    kwargs['step_duration'] = 0.0
    kwargs['verbose'] = False
    env = RewardTestEnv(**kwargs)
    env._drone = DummyDrone(position, speed=speed, yaw=yaw)
    return env


def expected_d_norm(env: AS2TestEnv, position: list[float]) -> float:
    # Replicates the env's distance formula bit-for-bit so strict equality
    # assertions catch any hidden reward term.
    tx, ty, tz, _ = env._target_pose
    d = math.sqrt(
        (position[0] - tx) ** 2
        + (position[1] - ty) ** 2
        + (position[2] - tz) ** 2
    )
    return min(d / (env.pos_limit * math.sqrt(3.0)), 1.0)


def test_per_step_reward_is_exactly_negative_d_norm() -> None:
    # Nonzero horizontal speed and yaw so a hidden path-facing term would show.
    env = make_exp010_env([3.0, 0.5, 1.6], speed=[0.4, 0.1, 0.0], yaw=0.7)

    _, reward, terminated, truncated, info = env.step(
        np.array([0.3, 0.0, 0.0, 0.2], dtype=np.float32)
    )

    d_norm = expected_d_norm(env, [3.0, 0.5, 1.6])
    assert terminated is False and truncated is False
    assert reward == -d_norm, f'reward {reward} != -d_norm {-d_norm}'
    assert info['reward_distance'] == -d_norm
    assert info['path_facing_reward'] == 0.0
    assert info['progress_reward'] == 0.0
    assert info['vertical_safety_penalty'] == 0.0
    assert info['terminal_yaw_penalty'] == 0.0

    # Second step after physical progress: a hidden progress term would show.
    env._drone.position = [2.0, 0.5, 1.6]
    env._drone.speed = [0.5, 0.0, 0.0]
    _, reward2, _, _, info2 = env.step(
        np.array([0.3, 0.0, 0.0, 0.0], dtype=np.float32)
    )
    d_norm2 = expected_d_norm(env, [2.0, 0.5, 1.6])
    assert reward2 == -d_norm2, f'reward {reward2} != -d_norm {-d_norm2}'
    assert info2['progress_reward'] == 0.0


def test_success_terminal_reward_includes_yaw_reduction() -> None:
    env = make_exp010_env([1.2, 0.0, 1.6], yaw=0.7)
    target_yaw = float(env._target_pose[3])
    yaw_err_norm = abs(math.atan2(math.sin(0.7 - target_yaw), math.cos(0.7 - target_yaw))) / math.pi

    _, reward, terminated, _, info = env.step(
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )

    d_norm = expected_d_norm(env, [1.2, 0.0, 1.6])
    expected = -d_norm + env.success_reward - env.terminal_yaw_penalty_weight * yaw_err_norm
    assert terminated is True
    assert info['terminal_reason'] == 'success'
    assert math.isclose(reward, expected, rel_tol=0.0, abs_tol=1e-9), (
        f'success reward {reward} != {expected}'
    )
    assert math.isclose(info['terminal_yaw_error'], yaw_err_norm, abs_tol=1e-9)
    assert math.isclose(
        info['terminal_yaw_penalty'],
        env.terminal_yaw_penalty_weight * yaw_err_norm,
        abs_tol=1e-9,
    )


def test_worst_case_arrival_still_nets_positive_terminal() -> None:
    # yaw error = pi -> yaw_err_norm = 1 -> terminal bonus = 20 - 10 = +10 > 0.
    env = make_exp010_env([1.2, 0.0, 1.6], yaw=math.pi)
    assert float(env._target_pose[3]) == 0.0

    _, reward, terminated, _, info = env.step(
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )

    assert terminated is True and info['terminal_reason'] == 'success'
    assert env.success_reward - env.terminal_yaw_penalty_weight > 0.0
    assert reward > 0.0, f'worst-case arrival must stay positive, got {reward}'


def test_success_terminal_reward_with_zero_weight() -> None:
    env = RewardTestEnv(
        verbose=False,
        step_duration=0.0,
        target_pose=[0.0, 0.0, 1.5, 0.0],
        distance_threshold=0.4,
        success_reward=20.0,
        path_facing_weight=0.0,
        progress_reward_weight=0.0,
        terminal_yaw_penalty_weight=0.0,
        height_bounds=(0.1, 2.0),
        unsafe_low_altitude_threshold=0.5,
    )
    env._drone = DummyDrone([0.2, 0.0, 1.5], yaw=1.0)

    _, reward, terminated, _, info = env.step(
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )

    d_norm = expected_d_norm(env, [0.2, 0.0, 1.5])
    assert terminated is True and info['terminal_reason'] == 'success'
    assert math.isclose(reward, -d_norm + env.success_reward, abs_tol=1e-9)
    assert info['terminal_yaw_penalty'] == 0.0
    assert info['terminal_yaw_error'] > 0.0


def test_pre_step_terminal_emits_every_monitor_keyword() -> None:
    # Exp009 lesson: pre-step terminals must emit every monitor keyword too.
    env = make_exp010_env([0.0, 0.0, 0.9], yaw=0.3)
    env._recover_low_altitude_hover_before_velocity_reset = lambda height: True  # type: ignore[method-assign]

    _, reward, terminated, truncated, info = env.step(
        np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    )

    assert terminated is True and truncated is False
    assert reward == -env.oob_penalty
    assert info['terminal_reason'] == 'unsafe_low_altitude'
    assert info['terminal_yaw_penalty'] == 0.0
    assert math.isclose(info['terminal_yaw_error'], 0.3 / math.pi, abs_tol=1e-6)

    monitor_keywords = load_exp010_config()['environment']['monitor_info_keywords']
    assert 'terminal_yaw_error' in monitor_keywords
    assert 'terminal_yaw_penalty' in monitor_keywords
    missing = [key for key in monitor_keywords if key not in info]
    assert not missing, f'pre-step terminal info missing monitor keywords: {missing}'


def test_success_terminal_emits_every_monitor_keyword() -> None:
    env = make_exp010_env([1.2, 0.0, 1.6], yaw=0.0)

    _, _, terminated, _, info = env.step(
        np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )

    assert terminated is True and info['terminal_reason'] == 'success'
    monitor_keywords = load_exp010_config()['environment']['monitor_info_keywords']
    missing = [key for key in monitor_keywords if key not in info]
    assert not missing, f'success terminal info missing monitor keywords: {missing}'


def make_sampler_env(**overrides) -> RewardTestEnv:
    kwargs = dict(
        verbose=False,
        step_duration=0.0,
        randomize_hover_start=True,
        scene_bounds_xy=5.0,
        height_bounds=(0.8, 2.5),
        unsafe_low_altitude_threshold=1.0,
        min_start_target_distance=1.0,
        distance_threshold=0.4,
        randomize_yaw=True,
        randomization_bounds_margin=0.3,
        max_reset_sample_attempts=100,
    )
    kwargs.update(overrides)
    env = RewardTestEnv(**kwargs)
    env._np_random, _ = seeding.np_random(42)
    return env


def test_sampler_randomizes_target_yaw() -> None:
    env = make_sampler_env()
    target_yaws = []
    start_yaws = []
    for _ in range(200):
        start_pose, target_pose, _ = env._sample_randomized_episode()
        start_yaws.append(start_pose[3])
        target_yaws.append(target_pose[3])

    assert all(-math.pi <= yaw <= math.pi for yaw in target_yaws)
    assert np.std(target_yaws) > 0.5, 'target yaw must be randomized'
    assert np.std(start_yaws) > 0.5, 'start yaw must be randomized'
    assert not np.allclose(start_yaws, target_yaws), (
        'start and target yaw must be sampled independently'
    )


def test_sampler_fixed_yaw_when_randomize_yaw_disabled() -> None:
    env = make_sampler_env(randomize_yaw=False)
    for _ in range(50):
        start_pose, target_pose, _ = env._sample_randomized_episode()
        assert start_pose[3] == 0.0
        assert target_pose[3] == 0.0


def test_sampler_respects_bounds_margin() -> None:
    env = make_sampler_env()
    margin = env.randomization_bounds_margin
    xy_limit = env.scene_bounds_xy - margin
    z_low = env.height_bounds[0] + margin
    z_high = env.height_bounds[1] - margin

    for _ in range(500):
        start_pose, target_pose, _ = env._sample_randomized_episode()
        for label, pose in (('start', start_pose), ('target', target_pose)):
            x, y, z = pose[:3]
            assert abs(x) <= xy_limit, f'{label} x {x} violates margin {margin}'
            assert abs(y) <= xy_limit, f'{label} y {y} violates margin {margin}'
            assert z_low <= z <= z_high, f'{label} z {z} violates margin {margin}'


def test_sampler_enforces_min_start_target_distance() -> None:
    env = make_sampler_env()
    assert env.distance_threshold < env.min_start_target_distance
    for _ in range(500):
        start_pose, target_pose, _ = env._sample_randomized_episode()
        distance = math.dist(start_pose[:3], target_pose[:3])
        assert distance > env.min_start_target_distance, (
            f'sampled pair distance {distance} below minimum'
        )


def test_sampler_raises_when_min_distance_is_unsatisfiable() -> None:
    env = make_sampler_env(min_start_target_distance=100.0, max_reset_sample_attempts=5)
    try:
        env._sample_randomized_episode()
    except RuntimeError:
        return
    raise AssertionError('expected RuntimeError for unsatisfiable min distance')


def test_margin_validation_rejects_bad_values() -> None:
    for bad_kwargs in (
        {'randomization_bounds_margin': -0.1},
        {'randomization_bounds_margin': 5.0, 'scene_bounds_xy': 5.0},
        {'randomization_bounds_margin': 1.0, 'height_bounds': (0.8, 2.5)},
    ):
        try:
            make_sampler_env(**bad_kwargs)
        except ValueError:
            continue
        raise AssertionError(f'expected ValueError for {bad_kwargs}')


def main() -> int:
    test_per_step_reward_is_exactly_negative_d_norm()
    test_success_terminal_reward_includes_yaw_reduction()
    test_worst_case_arrival_still_nets_positive_terminal()
    test_success_terminal_reward_with_zero_weight()
    test_pre_step_terminal_emits_every_monitor_keyword()
    test_success_terminal_emits_every_monitor_keyword()
    test_sampler_randomizes_target_yaw()
    test_sampler_fixed_yaw_when_randomize_yaw_disabled()
    test_sampler_respects_bounds_margin()
    test_sampler_enforces_min_start_target_distance()
    test_sampler_raises_when_min_distance_is_unsatisfiable()
    test_margin_validation_rejects_bad_values()
    print('✓ PASS: Exp010 reward contract and randomization behave per tutor spec')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
