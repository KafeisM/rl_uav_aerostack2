#!/usr/bin/env python3
"""ROS-free checks for the PREDICTIVE bounds / low-altitude action guards.

Exp012 attempt-1 live data showed the position-reactive guards being
defeated by vertical momentum: 285 ceiling deaths (z > 2.5 despite the
2.2 guard line) and 119 floor deaths (min_altitude 0.00 despite the 0.8
guard). The platform reverses vertical velocity slowly, so by the time
the clamped command takes effect the drone has coasted past the terminal
bound. The fix triggers both guards on the PREDICTED coordinate
(position + measured velocity * lookahead) and gives the vertical push
full authority via a dedicated speed parameter.

Covers:
    - Lookahead 0 (default): behavior identical to the position-only
      guard (backward compatibility).
    - Ceiling predictive trigger replaces upward commands with the
      full-authority vertical push.
    - Floor predictive trigger forces the climb on downward commands,
      and also on small positive commands when the measured fall speed
      exceeds the climb authority.
    - XY predictive trigger clamps outward commands before the line.
    - Unreadable speed degrades gracefully to the position-only guard.
    - Constructor validation of the new parameters.

Usage:
    conda run -n rl_uav python scripts/test_predictive_guard.py
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


class RaisingSpeedDrone(DummyDrone):
    @property
    def speed(self):  # type: ignore[override]
        raise RuntimeError('speed unreadable')

    @speed.setter
    def speed(self, value):  # constructor assigns; ignore it
        return None


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
    low_altitude_guard_margin=0.45,
    low_altitude_guard_climb_speed=0.5,
    target_pose=[2.0, 0.0, 1.5, 0.0],
)

GUARD_KWARGS = dict(
    bounds_action_guard=True,
    bounds_guard_margin_xy=0.5,
    bounds_guard_margin_ceiling=0.3,
    bounds_guard_push_speed=0.2,
)

PREDICTIVE_KWARGS = dict(
    bounds_guard_lookahead_s=0.6,
    bounds_guard_vertical_push_speed=0.5,
    low_altitude_guard_lookahead_s=0.6,
)

# Derived geometry with BASE_KWARGS + GUARD_KWARGS:
#   xy guard line       = 5.0 - 0.5 = 4.5
#   ceiling guard line  = 2.5 - 0.3 = 2.2
#   floor guard height  = 0.5 + max(0.45, dynamic 0.1) = 0.95
#   xy push speed       = min(max_vel 0.5, 0.2) = 0.2
#   vertical push speed = min(max_vel 0.5, 0.5) = 0.5
#   climb speed         = min(max_vel 0.5, max(0.5, reset_min_speed)) = 0.5
XY_LINE = 4.5
CEILING_LINE = 2.2
FLOOR_GUARD = 0.95
PUSH = 0.2
VERTICAL_PUSH = 0.5
CLIMB = 0.5


def make_env(position: list[float], speed=None, **overrides) -> GuardTestEnv:
    kwargs = dict(BASE_KWARGS)
    kwargs.update(GUARD_KWARGS)
    kwargs.update(overrides)
    env = GuardTestEnv(**kwargs)
    env._drone = DummyDrone(position, speed=speed)
    return env


def make_predictive_env(position: list[float], speed=None, **overrides) -> GuardTestEnv:
    kwargs = dict(PREDICTIVE_KWARGS)
    kwargs.update(overrides)
    return make_env(position, speed=speed, **kwargs)


def step_env(env: GuardTestEnv, vx: float, vy: float, vz: float, vyaw: float = 0.0):
    return env.step(np.array([vx, vy, vz, vyaw], dtype=np.float32))


def test_default_lookahead_zero_matches_position_only_guard() -> None:
    # Defaults must reproduce the exp012 position-reactive behavior exactly:
    # a drone strictly inside every guard line is never clamped, no matter
    # how fast it is moving outward.
    env = make_env([4.0, 0.0, 1.5], speed=[1.0, 0.0, 0.0])
    assert env.bounds_guard_lookahead_s == 0.0
    assert env.low_altitude_guard_lookahead_s == 0.0
    assert env.bounds_guard_vertical_push_speed is None

    _, _, terminated, _, info = step_env(env, 0.4, 0.0, 0.0)
    assert terminated is False
    assert env.last_command[0] == np.float32(0.4)
    assert info['bounds_guard_active'] is False

    # Fast ascent below the ceiling line: untouched.
    env_up = make_env([0.0, 0.0, 2.0], speed=[0.0, 0.0, 0.5])
    _, _, _, _, info_up = step_env(env_up, 0.0, 0.0, 0.3)
    assert env_up.last_command[2] == np.float32(0.3)
    assert info_up['bounds_guard_active'] is False

    # Fast descent above the floor guard: untouched.
    env_down = make_env([0.0, 0.0, 1.0], speed=[0.0, 0.0, -0.5])
    _, _, _, _, info_down = step_env(env_down, 0.0, 0.0, -0.2)
    assert env_down.last_command[2] == np.float32(-0.2)
    assert info_down['low_altitude_guard_active'] is False


def test_ceiling_predictive_trigger_uses_full_authority_vertical_push() -> None:
    # z = 2.0 (inside the 2.2 line), measured vz = +0.5, lookahead 0.6
    # -> predicted 2.3 >= 2.2 -> upward command replaced with -vertical_push.
    env = make_predictive_env([0.0, 0.0, 2.0], speed=[0.0, 0.0, 0.5])

    _, _, terminated, _, info = step_env(env, 0.0, 0.0, 0.3)

    assert terminated is False
    assert env.last_command[2] == -VERTICAL_PUSH, (
        f'vz {env.last_command[2]} != {-VERTICAL_PUSH}'
    )
    assert info['bounds_guard_active'] is True

    # Descending command with the same prediction passes through: only
    # outward (upward) commands are replaced.
    env2 = make_predictive_env([0.0, 0.0, 2.0], speed=[0.0, 0.0, 0.5])
    _, _, _, _, info2 = step_env(env2, 0.0, 0.0, -0.2)
    assert env2.last_command[2] == np.float32(-0.2)
    assert info2['bounds_guard_active'] is False


def test_ceiling_vertical_push_falls_back_to_push_speed_when_unset() -> None:
    # Without the dedicated vertical push speed, the ceiling clamp keeps
    # using bounds_guard_push_speed (position trigger past the line).
    env = make_predictive_env(
        [0.0, 0.0, CEILING_LINE + 0.1],
        speed=[0.0, 0.0, 0.0],
        bounds_guard_vertical_push_speed=None,
    )
    _, _, _, _, info = step_env(env, 0.0, 0.0, 0.3)
    assert env.last_command[2] == -PUSH, f'vz {env.last_command[2]} != {-PUSH}'
    assert info['bounds_guard_active'] is True


def test_floor_predictive_trigger_forces_climb() -> None:
    # z = 1.0 (above the 0.95 guard), measured vz = -0.5, lookahead 0.6
    # -> predicted 0.7 <= 0.95 -> downward command replaced with the climb.
    env = make_predictive_env([0.0, 0.0, 1.0], speed=[0.0, 0.0, -0.5])

    _, _, terminated, _, info = step_env(env, 0.0, 0.0, -0.2)

    assert terminated is False
    assert env.last_command[2] == CLIMB, f'vz {env.last_command[2]} != {CLIMB}'
    assert info['low_altitude_guard_active'] is True

    # Falling faster than the climb authority (measured -0.6 < -0.5): even a
    # small positive command cannot arrest the fall in time -> forced climb.
    env2 = make_predictive_env([0.0, 0.0, 1.0], speed=[0.0, 0.0, -0.6])
    _, _, _, _, info2 = step_env(env2, 0.0, 0.0, 0.1)
    assert env2.last_command[2] == CLIMB, f'vz {env2.last_command[2]} != {CLIMB}'
    assert info2['low_altitude_guard_active'] is True

    # Falling slower than the climb authority with a positive command:
    # the command already points inward, pass it through.
    env3 = make_predictive_env([0.0, 0.0, 1.0], speed=[0.0, 0.0, -0.5])
    _, _, _, _, info3 = step_env(env3, 0.0, 0.0, 0.1)
    assert env3.last_command[2] == np.float32(0.1)
    assert info3['low_altitude_guard_active'] is False


def test_xy_predictive_trigger_clamps_outward_command() -> None:
    # x = 4.0 (inside the 4.5 line), measured vx = +1.0, lookahead 0.6
    # -> predicted 4.6 >= 4.5 -> outward command replaced with -push.
    env = make_predictive_env([4.0, 0.0, 1.5], speed=[1.0, 0.0, 0.0])

    _, _, terminated, _, info = step_env(env, 0.4, 0.0, 0.0)

    assert terminated is False
    assert env.last_command[0] == -PUSH, f'vx {env.last_command[0]} != {-PUSH}'
    assert info['bounds_guard_active'] is True

    # Inward command with the same prediction passes through.
    env2 = make_predictive_env([4.0, 0.0, 1.5], speed=[1.0, 0.0, 0.0])
    _, _, _, _, info2 = step_env(env2, -0.3, 0.0, 0.0)
    assert env2.last_command[0] == np.float32(-0.3)
    assert info2['bounds_guard_active'] is False

    # Negative side, y axis: y = -4.0, vy = -1.0 -> predicted -4.6 -> +push.
    env3 = make_predictive_env([0.0, -4.0, 1.5], speed=[0.0, -1.0, 0.0])
    _, _, _, _, info3 = step_env(env3, 0.0, -0.4, 0.0)
    assert env3.last_command[1] == PUSH, f'vy {env3.last_command[1]} != {PUSH}'
    assert info3['bounds_guard_active'] is True


def test_unreadable_speed_falls_back_to_position_only() -> None:
    # speed = None: prediction unavailable -> position-only behavior, no crash.
    env = make_predictive_env([4.0, 0.0, 2.0])
    env._drone.speed = None
    _, _, _, _, info = step_env(env, 0.4, 0.0, 0.3)
    assert env.last_command[0] == np.float32(0.4)
    assert env.last_command[2] == np.float32(0.3)
    assert info['bounds_guard_active'] is False
    assert info['low_altitude_guard_active'] is False

    # speed property that raises: same graceful degradation.
    env2 = make_predictive_env([4.0, 0.0, 1.0])
    env2._drone = RaisingSpeedDrone([4.0, 0.0, 1.0])
    _, _, _, _, info2 = step_env(env2, 0.4, 0.0, -0.2)
    assert env2.last_command[0] == np.float32(0.4)
    assert env2.last_command[2] == np.float32(-0.2)
    assert info2['bounds_guard_active'] is False
    assert info2['low_altitude_guard_active'] is False

    # Position-only triggers still work with unreadable speed.
    env3 = make_predictive_env([XY_LINE + 0.1, 0.0, 1.5])
    env3._drone.speed = None
    _, _, _, _, info3 = step_env(env3, 0.4, 0.0, 0.0)
    assert env3.last_command[0] == -PUSH
    assert info3['bounds_guard_active'] is True


def test_validation_rejects_bad_predictive_parameters() -> None:
    cases = [
        (dict(bounds_guard_lookahead_s=-0.1), 'bounds_guard_lookahead_s'),
        (dict(bounds_guard_lookahead_s=5.1), 'bounds_guard_lookahead_s'),
        (dict(low_altitude_guard_lookahead_s=-0.1), 'low_altitude_guard_lookahead_s'),
        (dict(low_altitude_guard_lookahead_s=5.1), 'low_altitude_guard_lookahead_s'),
        (dict(bounds_guard_vertical_push_speed=0.0), 'bounds_guard_vertical_push_speed'),
        (dict(bounds_guard_vertical_push_speed=-0.2), 'bounds_guard_vertical_push_speed'),
    ]
    for overrides, expected_token in cases:
        try:
            make_predictive_env([0.0, 0.0, 1.5], **overrides)
        except ValueError as exc:
            assert expected_token in str(exc), f'{overrides}: {exc}'
        else:
            raise AssertionError(f'{overrides} must raise ValueError')

    # Lookaheads are validated even with the bounds guard disabled: they are
    # plain floats with safe defaults, so validation is unconditional.
    try:
        make_env(
            [0.0, 0.0, 1.5],
            bounds_action_guard=False,
            low_altitude_guard_lookahead_s=-1.0,
        )
    except ValueError as exc:
        assert 'low_altitude_guard_lookahead_s' in str(exc)
    else:
        raise AssertionError('negative lookahead must raise even with guard off')


def main() -> int:
    test_default_lookahead_zero_matches_position_only_guard()
    test_ceiling_predictive_trigger_uses_full_authority_vertical_push()
    test_ceiling_vertical_push_falls_back_to_push_speed_when_unset()
    test_floor_predictive_trigger_forces_climb()
    test_xy_predictive_trigger_clamps_outward_command()
    test_unreadable_speed_falls_back_to_position_only()
    test_validation_rejects_bad_predictive_parameters()
    print('✓ PASS: predictive guard behaves per spec')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
