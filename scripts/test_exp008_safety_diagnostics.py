#!/usr/bin/env python3
"""ROS-free checks for Exp008 terminal safety diagnostics."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_uav.envs.as2_test_env import AS2TestEnv
from scripts.validate_exp008_training_smoke import build_step_sample, summarize_episode_motion


class DummyDrone:
    def __init__(self, position: list[float]):
        self.position = position
        self.speed = [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, 0.0]


class SafetyTestEnv(AS2TestEnv):
    def _send_speed_command(self, twist, yaw_rate):  # type: ignore[no-untyped-def]
        self.last_command = [float(value) for value in twist] + [float(yaw_rate)]
        return True

    def _send_terminal_stop_command(self) -> bool:
        self.stop_command_sent = True
        return True

    def _publish_target_marker(self) -> None:
        return None


class DroppingAfterPublicationEnv(SafetyTestEnv):
    def _send_speed_command(self, twist, yaw_rate):  # type: ignore[no-untyped-def]
        accepted = super()._send_speed_command(twist, yaw_rate)
        self._drone.position = [0.0, 0.0, 0.49]
        self._drone.speed = [0.0, 0.0, -2.0]
        return accepted


def make_env(z: float) -> SafetyTestEnv:
    env = SafetyTestEnv(
        verbose=False,
        step_duration=0.0,
        target_pose=[2.0, 0.0, 1.5, 0.0],
        height_bounds=(0.1, 2.0),
        unsafe_low_altitude_threshold=0.5,
        low_altitude_guard_margin=0.2,
        low_altitude_guard_climb_speed=0.15,
    )
    env._drone = DummyDrone([0.0, 0.0, z])
    env.stop_command_sent = False
    return env


def test_unsafe_low_altitude_pre_step_terminal() -> None:
    env = make_env(0.49)
    _, reward, terminated, truncated, info = env.step(np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert reward == -env.oob_penalty
    assert info['terminal_reason'] == 'unsafe_low_altitude'
    assert info['is_unsafe_low_altitude'] is True
    assert info['motion_command_publication_count'] == 0
    assert info['motion_command_accepted_publication_count'] == 0
    assert info['terminal_stop_command_accepted'] is True
    assert env._terminal_reset_requires_service is True


def test_unsafe_low_altitude_pre_step_terminal_survives_recovery_failure() -> None:
    env = make_env(0.49)
    env.fixed_start_pose = [0.0, 0.0, 1.0, 0.0]
    recovery_calls: list[float] = []

    def fail_recovery(hover_height: float) -> bool:
        recovery_calls.append(float(hover_height))
        return False

    env._recover_low_altitude_hover_before_velocity_reset = fail_recovery  # type: ignore[method-assign]

    _, reward, terminated, truncated, info = env.step(np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32))

    assert recovery_calls == [env._reset_recovery_hover_height()]
    assert terminated is True
    assert truncated is False
    assert reward == -env.oob_penalty
    assert info['terminal_reason'] == 'unsafe_low_altitude'
    assert info['is_unsafe_low_altitude'] is True
    assert info['terminal_stop_command_accepted'] is True
    assert env.stop_command_sent is True
    assert env._terminal_reset_requires_service is True


def test_unsafe_low_altitude_pre_step_terminal_survives_recovery_exception() -> None:
    env = make_env(0.49)
    env.fixed_start_pose = [0.0, 0.0, 1.0, 0.0]
    recovery_calls: list[float] = []

    def raise_recovery(hover_height: float) -> bool:
        recovery_calls.append(float(hover_height))
        raise RuntimeError('synthetic recovery failure')

    env._recover_low_altitude_hover_before_velocity_reset = raise_recovery  # type: ignore[method-assign]

    _, reward, terminated, truncated, info = env.step(np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32))

    assert recovery_calls == [env._reset_recovery_hover_height()]
    assert terminated is True
    assert truncated is False
    assert reward == -env.oob_penalty
    assert info['terminal_reason'] == 'unsafe_low_altitude'
    assert info['is_unsafe_low_altitude'] is True
    assert info['terminal_stop_command_accepted'] is True
    assert env.stop_command_sent is True
    assert env._terminal_reset_requires_service is True


def test_low_altitude_guard_climbs_on_zero_vertical_action() -> None:
    env = make_env(0.65)
    _, _, terminated, _, info = env.step(np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32))

    assert terminated is False
    assert info['low_altitude_guard_active'] is True
    assert info['action_sent'][2] > 0.0
    assert info['raw_action'][2] == 0.0


def test_post_publication_drop_is_terminal_safety_before_no_motion() -> None:
    env = DroppingAfterPublicationEnv(
        verbose=False,
        step_duration=0.0,
        target_pose=[2.0, 0.0, 1.5, 0.0],
        height_bounds=(0.1, 2.0),
        unsafe_low_altitude_threshold=0.5,
        low_altitude_guard_margin=0.2,
        low_altitude_guard_climb_speed=0.15,
    )
    env._drone = DummyDrone([0.0, 0.0, 0.8])
    env.stop_command_sent = False

    _, reward, terminated, truncated, info = env.step(np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert reward == -env.oob_penalty
    assert info['terminal_reason'] == 'unsafe_low_altitude'
    assert info['is_unsafe_low_altitude'] is True
    assert info['step_safety_terminal'] is True
    assert info['step_safety_checked_after_publication'] is True
    assert math.isclose(info['step_altitude_before'], 0.8)
    assert math.isclose(info['step_altitude_after'], 0.49)
    assert math.isclose(info['step_min_altitude'], 0.49)
    assert math.isclose(info['step_speed_z_after'], -2.0)
    assert info['terminal_stop_command_accepted'] is True


def test_in_step_unsafe_low_altitude_terminal_survives_recovery_exception() -> None:
    env = DroppingAfterPublicationEnv(
        verbose=False,
        step_duration=0.0,
        target_pose=[2.0, 0.0, 1.5, 0.0],
        height_bounds=(0.1, 2.0),
        unsafe_low_altitude_threshold=0.5,
        low_altitude_guard_margin=0.2,
        low_altitude_guard_climb_speed=0.15,
    )
    env._drone = DummyDrone([0.0, 0.0, 0.8])
    env.stop_command_sent = False
    env.fixed_start_pose = [0.0, 0.0, 1.0, 0.0]
    recovery_calls: list[float] = []

    def raise_recovery(hover_height: float) -> bool:
        recovery_calls.append(float(hover_height))
        raise RuntimeError('synthetic in-step recovery failure')

    env._recover_low_altitude_hover_before_velocity_reset = raise_recovery  # type: ignore[method-assign]

    _, reward, terminated, truncated, info = env.step(np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32))

    assert recovery_calls == [env._reset_recovery_hover_height()]
    assert terminated is True
    assert truncated is False
    assert reward == -env.oob_penalty
    assert info['terminal_reason'] == 'unsafe_low_altitude'
    assert info['is_unsafe_low_altitude'] is True
    assert info['terminal_stop_command_accepted'] is True
    assert env.stop_command_sent is True
    assert env._terminal_reset_requires_service is True


def test_height_upper_bound_is_terminal_box_boundary() -> None:
    env = make_env(2.51)
    _, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))

    assert terminated is True
    assert truncated is False
    assert reward == -env.oob_penalty
    assert info['terminal_reason'] == 'out_of_bounds'
    assert info['is_out_of_bounds'] is True
    assert env._terminal_reset_requires_service is True


def test_smoke_summary_prefers_terminal_safety_over_no_motion() -> None:
    samples = [
        {
            'distance_before': 2.0,
            'distance_after': 2.0,
            'position_delta': [0.0, 0.0, 0.0],
            'motion_command_accepted': True,
            'early_stop_reason': 'accepted_commands_without_physical_motion',
            'done': True,
            'terminal_reason': 'unsafe_low_altitude',
            'is_unsafe_low_altitude': True,
            'altitude_after': 0.49,
        }
    ]

    summary = summarize_episode_motion(
        episode_index=0,
        samples=samples,
        min_distance_reduction=0.25,
        min_position_delta=0.10,
    )

    assert summary['success'] is False
    assert summary['reason'] == 'unsafe_low_altitude'
    assert summary['terminal_safety_reason'] == 'unsafe_low_altitude'
    assert math.isclose(summary['min_altitude'], 0.49)


def test_fixed_action_movement_away_from_target_confirms_actionability_only() -> None:
    samples = [
        {
            'distance_before': 1.0,
            'distance_after': 1.0,
            'position_delta': [0.08, 0.0, 0.0],
            'motion_command_accepted': True,
        },
        {
            'distance_before': 1.0,
            'distance_after': 1.2,
            'position_delta': [0.07, 0.0, 0.0],
            'motion_command_accepted': True,
        },
    ]

    summary = summarize_episode_motion(
        episode_index=0,
        samples=samples,
        min_distance_reduction=0.25,
        min_position_delta=0.10,
        require_target_progress=False,
    )

    assert summary['success'] is True
    assert summary['reason'] == 'physical_actionability_confirmed'
    assert summary['physical_actionability_confirmed'] is True
    assert summary['target_progress_confirmed'] is False
    assert math.isclose(summary['distance_reduction'], -0.2)


def test_accepted_without_motion_early_stop_fails_after_initial_movement() -> None:
    samples = [
        {
            'distance_before': 1.0,
            'distance_after': 0.9,
            'position_delta': [0.12, 0.0, 0.0],
            'motion_command_accepted': True,
        },
        {
            'distance_before': 0.9,
            'distance_after': 0.9,
            'position_delta': [0.0, 0.0, 0.0],
            'motion_command_accepted': True,
        },
        {
            'distance_before': 0.9,
            'distance_after': 0.9,
            'position_delta': [0.0, 0.0, 0.0],
            'motion_command_accepted': True,
            'early_stop_reason': 'accepted_commands_without_physical_motion',
            'done': True,
        },
    ]

    summary = summarize_episode_motion(
        episode_index=0,
        samples=samples,
        min_distance_reduction=0.05,
        min_position_delta=0.10,
        require_target_progress=False,
    )

    assert summary['success'] is False
    assert summary['reason'] == 'accepted_commands_without_physical_motion'
    assert summary['physical_actionability_confirmed'] is True
    assert summary['accepted_without_motion_steps'] == 2


def test_policy_smoke_can_require_target_progress() -> None:
    samples = [
        {
            'distance_before': 1.0,
            'distance_after': 1.0,
            'position_delta': [0.08, 0.0, 0.0],
            'motion_command_accepted': True,
        },
        {
            'distance_before': 1.0,
            'distance_after': 1.2,
            'position_delta': [0.07, 0.0, 0.0],
            'motion_command_accepted': True,
        },
    ]

    summary = summarize_episode_motion(
        episode_index=0,
        samples=samples,
        min_distance_reduction=0.25,
        min_position_delta=0.10,
        require_target_progress=True,
    )

    assert summary['success'] is False
    assert summary['reason'] == 'insufficient_target_progress'
    assert summary['physical_actionability_confirmed'] is True
    assert summary['target_progress_confirmed'] is False


def test_no_motion_still_fails_actionability() -> None:
    summary = summarize_episode_motion(
        episode_index=0,
        samples=[
            {
                'distance_before': 1.0,
                'distance_after': 0.6,
                'position_delta': [0.0, 0.0, 0.0],
                'motion_command_accepted': True,
            }
        ],
        min_distance_reduction=0.25,
        min_position_delta=0.10,
        require_target_progress=False,
    )

    assert summary['success'] is False
    assert summary['reason'] == 'insufficient_physical_motion'
    assert summary['physical_actionability_confirmed'] is False
    assert summary['target_progress_confirmed'] is True


def test_step_sample_uses_env_step_altitude_before_and_publication_counters() -> None:
    sample = build_step_sample(
        episode_index=0,
        step_index=0,
        requested_action=[0.5, 0.0, 0.0, 0.0],
        before_info={
            'position': [0.0, 0.0, 1.0],
            'altitude': 1.2,
            'distance': 2.0,
        },
        info={
            'position': [0.1, 0.0, 0.9],
            'altitude': 0.9,
            'step_altitude_before': 1.0,
            'step_altitude_after': 0.9,
            'step_min_altitude': 0.85,
            'step_speed_z_before': -0.1,
            'step_speed_z_after': -0.2,
            'distance': 1.9,
            'action_sent': [0.5, 0.0, 0.15, 0.0],
            'motion_command_publication_count': 3,
            'motion_command_accepted_publication_count': 3,
            'unsafe_low_altitude_threshold': 0.5,
            'height_bounds': [0.1, 2.0],
            'step_safety_checked_after_publication': True,
        },
        reward=-1.0,
        done=False,
    )

    assert sample['altitude_before'] == 1.0
    assert sample['altitude_after'] == 0.9
    assert sample['min_altitude_step'] == 0.85
    assert sample['speed_z_before'] == -0.1
    assert sample['speed_z_after'] == -0.2
    assert sample['vertical_action_requested'] == 0.0
    assert sample['vertical_action_sent'] == 0.15
    assert sample['motion_command_publication_count'] == 3
    assert sample['motion_command_accepted_publication_count'] == 3
    assert sample['step_safety_checked_after_publication'] is True


def test_step_sample_altitude_before_falls_back_when_env_snapshot_is_absent() -> None:
    sample = build_step_sample(
        episode_index=0,
        step_index=0,
        requested_action=[0.0, 0.0, 0.0, 0.0],
        before_info={
            'position': [0.0, 0.0, 1.1],
            'altitude': 1.1,
            'distance': 2.0,
        },
        info={
            'position': [0.0, 0.0, 1.0],
            'altitude': 1.0,
            'distance': 1.9,
        },
        reward=-1.0,
        done=False,
    )

    assert sample['altitude_before'] == 1.1


def main() -> int:
    test_unsafe_low_altitude_pre_step_terminal()
    test_unsafe_low_altitude_pre_step_terminal_survives_recovery_failure()
    test_unsafe_low_altitude_pre_step_terminal_survives_recovery_exception()
    test_low_altitude_guard_climbs_on_zero_vertical_action()
    test_post_publication_drop_is_terminal_safety_before_no_motion()
    test_height_upper_bound_is_terminal_box_boundary()
    test_smoke_summary_prefers_terminal_safety_over_no_motion()
    test_fixed_action_movement_away_from_target_confirms_actionability_only()
    test_accepted_without_motion_early_stop_fails_after_initial_movement()
    test_policy_smoke_can_require_target_progress()
    test_no_motion_still_fails_actionability()
    test_step_sample_uses_env_step_altitude_before_and_publication_counters()
    test_step_sample_altitude_before_falls_back_when_env_snapshot_is_absent()
    print('✓ PASS: Exp008 safety diagnostics are classified before no-motion failures')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
