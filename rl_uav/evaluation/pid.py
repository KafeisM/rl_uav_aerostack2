"""Classical PID baseline controller over the shared evaluation protocol.

Consumes the same normalized relative observation the trained policy sees
([dx, dy, dz, dyaw] in [-1, 1]) and emits actions in the env's physical
action space ([vx, vy, vz, vyaw] with per-axis saturation), so it runs the
exact same seeded episodes and metrics as the PPO policy.
"""

from __future__ import annotations

import math

import numpy as np

from rl_uav.evaluation.harness import Controller


class PIDController(Controller):
    """Per-axis PID on the relative position/yaw error.

    The error convention follows the observation: obs_xyz = (drone - target)
    / pos_limit, so the control error is the NEGATED denormalized obs (we
    command velocity toward the target). Yaw uses the wrapped error in
    radians. Integrators are clamped to the saturation-equivalent error and
    cleared per episode.
    """

    def __init__(
        self,
        pos_limit: float,
        max_vel: float,
        max_yaw_vel: float,
        dt: float,
        kp_xy: float = 0.8,
        ki_xy: float = 0.0,
        kd_xy: float = 0.0,
        kp_z: float = 0.8,
        ki_z: float = 0.0,
        kd_z: float = 0.0,
        kp_yaw: float = 0.8,
        ki_yaw: float = 0.0,
        kd_yaw: float = 0.0,
    ):
        if dt <= 0.0:
            raise ValueError('dt must be > 0')
        self._pos_limit = float(pos_limit)
        self._max_vel = float(max_vel)
        self._max_yaw_vel = float(max_yaw_vel)
        self._dt = float(dt)
        self._gains = {
            'x': (float(kp_xy), float(ki_xy), float(kd_xy)),
            'y': (float(kp_xy), float(ki_xy), float(kd_xy)),
            'z': (float(kp_z), float(ki_z), float(kd_z)),
            'yaw': (float(kp_yaw), float(ki_yaw), float(kd_yaw)),
        }
        self.reset_episode()

    @property
    def name(self) -> str:
        return 'pid'

    def reset_episode(self) -> None:
        self._integral = {axis: 0.0 for axis in self._gains}
        self._previous_error: dict[str, float] | None = None

    def _axis_output(self, axis: str, error: float, saturation: float) -> float:
        kp, ki, kd = self._gains[axis]

        derivative = 0.0
        if self._previous_error is not None:
            derivative = (error - self._previous_error[axis]) / self._dt

        if ki > 0.0:
            self._integral[axis] += error * self._dt
            # Anti-windup: keep the integral contribution within saturation.
            integral_cap = saturation / ki
            self._integral[axis] = float(
                np.clip(self._integral[axis], -integral_cap, integral_cap)
            )

        output = kp * error + ki * self._integral[axis] + kd * derivative
        return float(np.clip(output, -saturation, saturation))

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        # Control error: vector from drone to target, in physical units.
        errors = {
            'x': -float(obs[0]) * self._pos_limit,
            'y': -float(obs[1]) * self._pos_limit,
            'z': -float(obs[2]) * self._pos_limit,
            'yaw': -float(obs[3]) * math.pi,
        }

        action = np.array(
            [
                self._axis_output('x', errors['x'], self._max_vel),
                self._axis_output('y', errors['y'], self._max_vel),
                self._axis_output('z', errors['z'], self._max_vel),
                self._axis_output('yaw', errors['yaw'], self._max_yaw_vel),
            ],
            dtype=np.float32,
        )
        self._previous_error = errors
        return action
