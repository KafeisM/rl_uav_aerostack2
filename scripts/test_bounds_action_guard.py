#!/usr/bin/env python3
"""ROS-free checks for the bounds ACTION guard in AS2TestEnv.

The guard pushes the drone back toward the interior before it can reach
the out-of-bounds / ceiling terminals, so boundary terminals (which force
slow simulator service resets) should ~never fire. Terminal conditions
themselves stay unchanged as a fail-closed safety net.

Covers:
    - Guard disabled by default: actions pass through unchanged.
    - XY guard lines (both signs, x and y) clamp outward commands only.
    - Ceiling guard line clamps upward commands only.
    - Inside the safe volume the command is untouched.
    - `bounds_guard_active` info key on normal and pre-step terminal paths.
    - Constructor validation of the guard geometry parameters.

Usage:
    conda run -n rl_uav python scripts/test_bounds_action_guard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_uav.envs.as2_test_env import AS2TestEnv


class DummyDrone:
    def __init__(self, position: list[float], speed=None, yaw: float = 0.0):
        self.position = list(position)
        self.speed = list(speed) if speed is not None else [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, float(yaw)]


class GuardTestEnv(AS2TestEnv):
    def _send_speed_command(self, twist, yaw_rate):  # type: ignore[no-untyped-def]
        self.last_command = [float(value) for value in twist] + [float(yaw_rate)]
        return True

    def _send_terminal_stop_command(self) -> bool:
        return True

    def _publish_target_marker(self) -> None:
        return None


BASE_KWARGS = dict(
    verbose=False,
    step_duration=0.0,
    max_vel=0.5,
    pos_limit=5.0,
    scene_bounds_xy=5.0,
    height_bounds=(0.3, 2.5),
    unsafe_low_altitude_threshold=0.5,
    low_altitude_guard_margin=0.3,
    target_pose=[2.0, 0.0, 1.5, 0.0],
)

GUARD_KWARGS = dict(
    bounds_action_guard=True,
    bounds_guard_margin_xy=0.5,
    bounds_guard_margin_ceiling=0.3,
    bounds_guard_push_speed=0.2,
)

# Derived geometry with BASE_KWARGS + GUARD_KWARGS:
#   xy guard line     = 5.0 - 0.5 = 4.5
#   ceiling guard line = 2.5 - 0.3 = 2.2
#   push speed        = min(max_vel 0.5, 0.2) = 0.2
XY_LINE = 4.5
CEILING_LINE = 2.2
PUSH = 0.2


def make_env(position: list[float], **overrides) -> GuardTestEnv:
    kwargs = dict(BASE_KWARGS)
    kwargs.update(overrides)
    env = GuardTestEnv(**kwargs)
    env._drone = DummyDrone(position)
    return env


def make_guard_env(position: list[float], **overrides) -> GuardTestEnv:
    kwargs = dict(GUARD_KWARGS)
    kwargs.update(overrides)
    return make_env(position, **kwargs)


def step_env(env: GuardTestEnv, vx: float, vy: float, vz: float, vyaw: float = 0.0):
    return env.step(np.array([vx, vy, vz, vyaw], dtype=np.float32))


def test_guard_disabled_by_default_passes_action_through() -> None:
    env = make_env([XY_LINE + 0.1, 0.0, 1.5])
    assert env.bounds_action_guard is False

    _, _, terminated, truncated, info = step_env(env, 0.4, 0.1, 0.1)

    assert terminated is False and truncated is False
    assert env.last_command == [
        np.float32(0.4), np.float32(0.1), np.float32(0.1), np.float32(0.0)
    ]
    assert info['bounds_guard_active'] is False


def test_xy_guard_clamps_outward_positive_x() -> None:
    env = make_guard_env([XY_LINE + 0.1, 0.0, 1.5])

    _, _, terminated, _, info = step_env(env, 0.4, 0.0, 0.0)

    assert terminated is False
    assert env.last_command[0] == -PUSH, f'vx {env.last_command[0]} != {-PUSH}'
    assert info['bounds_guard_active'] is True

    # Inward command passes through untouched.
    env2 = make_guard_env([XY_LINE + 0.1, 0.0, 1.5])
    _, _, _, _, info2 = step_env(env2, -0.3, 0.0, 0.0)
    assert env2.last_command[0] == np.float32(-0.3)
    assert info2['bounds_guard_active'] is False


def test_xy_guard_clamps_outward_negative_x_and_positive_y() -> None:
    env = make_guard_env([-(XY_LINE + 0.1), 0.0, 1.5])
    _, _, _, _, info = step_env(env, -0.4, 0.0, 0.0)
    assert env.last_command[0] == PUSH, f'vx {env.last_command[0]} != {PUSH}'
    assert info['bounds_guard_active'] is True

    env_y = make_guard_env([0.0, XY_LINE + 0.1, 1.5])
    _, _, _, _, info_y = step_env(env_y, 0.0, 0.3, 0.0)
    assert env_y.last_command[1] == -PUSH, f'vy {env_y.last_command[1]} != {-PUSH}'
    assert info_y['bounds_guard_active'] is True


def test_ceiling_guard_clamps_upward_command() -> None:
    env = make_guard_env([0.0, 0.0, CEILING_LINE + 0.1])
    _, _, terminated, _, info = step_env(env, 0.0, 0.0, 0.3)
    assert terminated is False
    assert env.last_command[2] == -PUSH, f'vz {env.last_command[2]} != {-PUSH}'
    assert info['bounds_guard_active'] is True

    # Descending command near the ceiling passes through untouched.
    env2 = make_guard_env([0.0, 0.0, CEILING_LINE + 0.1])
    _, _, _, _, info2 = step_env(env2, 0.0, 0.0, -0.2)
    assert env2.last_command[2] == np.float32(-0.2)
    assert info2['bounds_guard_active'] is False


def test_inside_safe_volume_command_untouched() -> None:
    env = make_guard_env([1.0, -1.0, 1.5])
    _, _, _, _, info = step_env(env, 0.4, -0.4, 0.2, 0.5)
    assert env.last_command == [
        np.float32(0.4), np.float32(-0.4), np.float32(0.2), np.float32(0.5)
    ]
    assert info['bounds_guard_active'] is False
    assert env._last_bounds_guard_active is False


def test_pre_step_terminal_still_fires_and_emits_guard_key() -> None:
    # Fail-closed safety net: with the guard on, an already out-of-bounds
    # drone still terminates before any action is processed.
    env = make_guard_env([5.5, 0.0, 1.5])

    _, reward, terminated, truncated, info = step_env(env, 0.0, 0.0, 0.0)

    assert terminated is True and truncated is False
    assert reward == -env.oob_penalty
    assert info['terminal_reason'] == 'out_of_bounds'
    assert 'bounds_guard_active' in info
    assert info['bounds_guard_active'] is False
    assert 'low_altitude_guard_active' in info


def test_validation_rejects_bad_guard_parameters() -> None:
    # Guard margin must stay strictly inside the randomized spawn margin.
    try:
        make_guard_env(
            [0.0, 0.0, 1.5],
            randomize_hover_start=True,
            randomization_bounds_margin=0.4,
            bounds_guard_margin_xy=0.5,
        )
    except ValueError as exc:
        assert 'bounds_guard_margin_xy' in str(exc)
    else:
        raise AssertionError('guard margin >= randomization margin must raise')

    # Push speed must be positive when the guard is enabled.
    try:
        make_guard_env([0.0, 0.0, 1.5], bounds_guard_push_speed=0.0)
    except ValueError as exc:
        assert 'bounds_guard_push_speed' in str(exc)
    else:
        raise AssertionError('zero push speed with guard enabled must raise')


def main() -> int:
    test_guard_disabled_by_default_passes_action_through()
    test_xy_guard_clamps_outward_positive_x()
    test_xy_guard_clamps_outward_negative_x_and_positive_y()
    test_ceiling_guard_clamps_upward_command()
    test_inside_safe_volume_command_untouched()
    test_pre_step_terminal_still_fires_and_emits_guard_key()
    test_validation_rejects_bad_guard_parameters()
    print('✓ PASS: bounds action guard behaves per spec')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
