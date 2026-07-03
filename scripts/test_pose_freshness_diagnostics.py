#!/usr/bin/env python3
"""ROS-free checks for per-step pose freshness diagnostics.

Usage:
    conda run -n rl_uav python scripts/test_pose_freshness_diagnostics.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_uav.envs.as2_test_env import AS2TestEnv
from scripts.validate_exp008_training_smoke import build_step_sample


class DummyDrone:
    def __init__(self, position: list[float]):
        self.position = position
        self.speed = [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, 0.0]


class FreshnessTestEnv(AS2TestEnv):
    """Env stub that advances (or freezes) the drone pose per accepted command."""

    move_per_step = [0.05, 0.0, 0.0]
    frozen = False

    def _send_speed_command(self, twist, yaw_rate):  # type: ignore[no-untyped-def]
        if not self.frozen:
            self._drone.position = [
                float(value) + float(delta)
                for value, delta in zip(self._drone.position, self.move_per_step)
            ]
        return True

    def _send_terminal_stop_command(self) -> bool:
        return True

    def _publish_target_marker(self) -> None:
        return None


def make_env(z: float = 1.5) -> FreshnessTestEnv:
    env = FreshnessTestEnv(
        verbose=False,
        step_duration=0.0,
        target_pose=[2.0, 0.0, 1.5, 0.0],
        height_bounds=(0.1, 2.0),
        unsafe_low_altitude_threshold=0.5,
    )
    env._drone = DummyDrone([0.0, 0.0, z])
    env._reset_pose_freshness(env._current_position_xyz())
    return env


ACTION = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)


def test_moving_pose_reports_fresh_reads() -> None:
    env = make_env()
    for _ in range(3):
        _, _, terminated, truncated, info = env.step(ACTION)
        assert terminated is False and truncated is False
        assert info['pose_freshness_changed'] is True
        assert info['pose_freshness_identical_reads'] == 0
        assert info['pose_freshness_age_seconds'] == 0.0 or (
            math.isfinite(info['pose_freshness_age_seconds'])
            and info['pose_freshness_age_seconds'] < 0.1
        )


def test_frozen_pose_accumulates_identical_reads() -> None:
    env = make_env()
    env.frozen = True
    counts = []
    for _ in range(3):
        _, _, _, _, info = env.step(ACTION)
        counts.append(info['pose_freshness_identical_reads'])
        assert info['pose_freshness_changed'] is False
        assert math.isfinite(info['pose_freshness_age_seconds'])
        assert info['pose_freshness_age_seconds'] >= 0.0
    assert counts == [1, 2, 3]


def test_pose_change_after_frozen_streak_resets_counter() -> None:
    env = make_env()
    env.frozen = True
    for _ in range(2):
        _, _, _, _, info = env.step(ACTION)
    assert info['pose_freshness_identical_reads'] == 2

    env.frozen = False
    _, _, _, _, info = env.step(ACTION)
    assert info['pose_freshness_changed'] is True
    assert info['pose_freshness_identical_reads'] == 0


def test_reset_pose_freshness_restarts_tracking() -> None:
    env = make_env()
    env.frozen = True
    for _ in range(3):
        env.step(ACTION)
    assert env._pose_freshness_consecutive_identical_reads == 3

    env._reset_pose_freshness(env._current_position_xyz())
    assert env._pose_freshness_consecutive_identical_reads == 0

    # The first post-reset step with a frozen pose counts from 1 again.
    _, _, _, _, info = env.step(ACTION)
    assert info['pose_freshness_identical_reads'] == 1


def test_unreadable_pose_counts_as_stale_sample() -> None:
    env = make_env()
    diagnostics = env._update_pose_freshness(None)
    assert diagnostics['pose_freshness_changed'] is False
    assert diagnostics['pose_freshness_identical_reads'] == 1


def test_pre_step_terminal_branch_reports_freshness() -> None:
    env = make_env(z=0.4)  # below unsafe_low_altitude_threshold
    _, _, terminated, _, info = env.step(ACTION)
    assert terminated is True
    assert 'pose_freshness_identical_reads' in info
    assert 'pose_freshness_changed' in info
    assert 'pose_freshness_age_seconds' in info


def test_interface_spin_rate_default_and_validation() -> None:
    env = make_env()
    assert env.interface_spin_rate == 300.0

    try:
        FreshnessTestEnv(verbose=False, interface_spin_rate=0.0)
    except ValueError as exc:
        assert 'interface_spin_rate' in str(exc)
    else:
        raise AssertionError('interface_spin_rate=0.0 must raise ValueError')

    try:
        FreshnessTestEnv(verbose=False, interface_spin_rate=float('nan'))
    except ValueError as exc:
        assert 'interface_spin_rate' in str(exc)
    else:
        raise AssertionError('interface_spin_rate=nan must raise ValueError')


def test_step_sample_carries_pose_freshness_fields() -> None:
    sample = build_step_sample(
        episode_index=0,
        step_index=0,
        requested_action=[0.5, 0.0, 0.0, 0.0],
        before_info={
            'position': [0.0, 0.0, 1.5],
            'altitude': 1.5,
            'distance': 2.0,
        },
        info={
            'position': [0.0, 0.0, 1.5],
            'altitude': 1.5,
            'distance': 2.0,
            'pose_freshness_changed': False,
            'pose_freshness_identical_reads': 2,
            'pose_freshness_age_seconds': 0.41,
        },
        reward=-1.0,
        done=False,
    )

    assert sample['pose_freshness_changed'] is False
    assert sample['pose_freshness_identical_reads'] == 2
    assert sample['pose_freshness_age_seconds'] == 0.41


def main() -> int:
    test_moving_pose_reports_fresh_reads()
    test_frozen_pose_accumulates_identical_reads()
    test_pose_change_after_frozen_streak_resets_counter()
    test_reset_pose_freshness_restarts_tracking()
    test_unreadable_pose_counts_as_stale_sample()
    test_pre_step_terminal_branch_reports_freshness()
    test_interface_spin_rate_default_and_validation()
    test_step_sample_carries_pose_freshness_fields()
    print('✓ PASS: pose freshness diagnostics detect stale telemetry reads')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
