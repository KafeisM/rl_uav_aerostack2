#!/usr/bin/env python3
"""ROS-free checks for the classical PID baseline controller.

The PID consumes the SAME normalized relative observation the policy sees
([dx, dy, dz, dyaw] in [-1, 1]) and emits actions in the env's physical
action space ([vx, vy, vz, vyaw]), so it plugs into the shared evaluation
protocol unchanged.

Usage:
    conda run -n rl_uav python3 scripts/test_pid_controller.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_uav.evaluation import PIDController

POS_LIMIT = 5.0
MAX_VEL = 0.5
MAX_YAW_VEL = math.pi / 2
DT = 0.2


def make_pid(**overrides) -> PIDController:
    kwargs = {
        'pos_limit': POS_LIMIT,
        'max_vel': MAX_VEL,
        'max_yaw_vel': MAX_YAW_VEL,
        'dt': DT,
        'kp_xy': 0.8,
        'kp_z': 0.8,
        'kp_yaw': 0.8,
    }
    kwargs.update(overrides)
    return PIDController(**kwargs)


def obs_for(drone_minus_target_xyz, dyaw_rad=0.0) -> np.ndarray:
    dx, dy, dz = drone_minus_target_xyz
    return np.array(
        [dx / POS_LIMIT, dy / POS_LIMIT, dz / POS_LIMIT, dyaw_rad / math.pi],
        dtype=np.float32,
    )


def test_proportional_action_points_toward_target() -> None:
    pid = make_pid()
    pid.reset_episode()
    # Drone is +2 m in x from the target: must command negative vx.
    action = pid.act(obs_for([2.0, -1.0, 0.5], dyaw_rad=0.5))
    assert action[0] < 0.0
    assert action[1] > 0.0
    assert action[2] < 0.0
    assert action[3] < 0.0
    assert action.shape == (4,)


def test_saturation_at_action_bounds() -> None:
    pid = make_pid(kp_xy=10.0, kp_z=10.0, kp_yaw=10.0)
    pid.reset_episode()
    action = pid.act(obs_for([4.0, -4.0, 2.0], dyaw_rad=3.0))
    assert abs(action[0]) <= MAX_VEL + 1e-9
    assert abs(action[1]) <= MAX_VEL + 1e-9
    assert abs(action[2]) <= MAX_VEL + 1e-9
    assert abs(action[3]) <= MAX_YAW_VEL + 1e-9
    assert action[0] == -MAX_VEL and action[1] == MAX_VEL


def test_zero_error_zero_action() -> None:
    pid = make_pid()
    pid.reset_episode()
    action = pid.act(obs_for([0.0, 0.0, 0.0], dyaw_rad=0.0))
    assert np.allclose(action, 0.0)


def test_integral_accumulates_and_reset_clears() -> None:
    pid = make_pid(kp_xy=0.0, ki_xy=0.5)
    pid.reset_episode()
    obs = obs_for([1.0, 0.0, 0.0])
    first = pid.act(obs)
    second = pid.act(obs)
    # Constant error: the integral term must grow between calls.
    assert abs(second[0]) > abs(first[0])

    pid.reset_episode()
    after_reset = pid.act(obs)
    assert abs(after_reset[0]) == abs(first[0])


def test_derivative_damps_fast_error_change() -> None:
    pid = make_pid(kp_xy=0.0, kd_xy=0.4)
    pid.reset_episode()
    pid.act(obs_for([2.0, 0.0, 0.0]))
    # Error shrinking (drone approaching target): derivative opposes motion
    # along -x, i.e. commands positive vx (braking).
    action = pid.act(obs_for([1.0, 0.0, 0.0]))
    assert action[0] > 0.0


def test_closed_loop_kinematic_convergence() -> None:
    """P-controller on integrated kinematics reaches the success sphere."""
    pid = make_pid()
    pid.reset_episode()
    drone = np.array([-3.0, 2.0, 1.0], dtype=np.float64)
    target = np.array([1.0, -1.0, 1.8], dtype=np.float64)
    yaw, target_yaw = 2.0, -1.0

    for step in range(200):
        dyaw = math.atan2(math.sin(yaw - target_yaw), math.cos(yaw - target_yaw))
        obs = obs_for(list(drone - target), dyaw_rad=dyaw)
        action = pid.act(obs)
        drone = drone + np.asarray(action[:3], dtype=np.float64) * DT
        yaw += float(action[3]) * DT
        if (
            float(np.linalg.norm(drone - target)) < 0.4
            and abs(math.atan2(math.sin(yaw - target_yaw), math.cos(yaw - target_yaw))) < 0.3
        ):
            break
    else:
        raise AssertionError(f'PID did not converge: dist={np.linalg.norm(drone - target):.2f}')
    assert step < 150


def test_yaw_error_uses_wrapped_shortest_path() -> None:
    pid = make_pid()
    pid.reset_episode()
    # dyaw_norm near +1 (~pi): either turn direction is valid, but the
    # magnitude must be finite and bounded.
    action = pid.act(obs_for([0.0, 0.0, 0.0], dyaw_rad=0.9 * math.pi))
    assert abs(action[3]) <= MAX_YAW_VEL + 1e-9
    assert action[3] < 0.0


def main() -> int:
    test_proportional_action_points_toward_target()
    test_saturation_at_action_bounds()
    test_zero_error_zero_action()
    test_integral_accumulates_and_reset_clears()
    test_derivative_damps_fast_error_change()
    test_closed_loop_kinematic_convergence()
    test_yaw_error_uses_wrapped_shortest_path()
    print('✓ PASS: PID baseline controller behaves per spec')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
