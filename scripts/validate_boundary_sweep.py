#!/usr/bin/env python3
"""Systematic boundary sweep validating env states and rewards at every limit.

Tutor requirement: before training, teleport the agent to all boundary/limit
coordinates and check the environment does what it must and returns what it
must — both observation and reward.

The script has two layers:

1. A pure, ROS-free layer (sweep generation + expected-outcome computation)
   exercised by ``scripts/test_boundary_sweep_expectations.py`` against the
   real ``AS2TestEnv.step()`` with a mock drone, so the expectations are
   self-validating without the simulator.
2. A live runner (requires the AS2 multirotor simulator fork with the
   ``ResetSimulatorState`` service). For each sweep point it resets to the
   configured safe start pose, service-teleports mid-episode to the point,
   steps once with a zero action, and asserts observation, reward, and
   terminal reason against the pure expectations computed from the observed
   (post-teleport) state. Out-of-bounds points intentionally bypass the env
   reset validators: the env's pre-step safety branch is the code path under
   test.

Run live (simulator up), for example:

    conda run -n rl_uav python scripts/validate_boundary_sweep.py \
        --config configs/train_ppo_phase1_exp008a.yaml

A JSON report with per-point pass/fail (observed vs expected) is written to
``runs/diagnostics/boundary_sweep_<timestamp>/``. Exit code is non-zero when
any point fails.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_uav.envs.as2_test_env import AS2TestEnv
from rl_uav.training import build_env_kwargs, build_vec_env, load_training_config
from scripts.validate_deterministic_reset import normalized_observation_from_state

ZERO_ACTION = [0.0, 0.0, 0.0, 0.0]
TERMINAL_REASONS = {'out_of_bounds', 'unsafe_low_altitude'}


# ---------------------------------------------------------------------------
# Pure, ROS-free layer: env-derived parameters, sweep generation, expectations
# ---------------------------------------------------------------------------


def build_probe_env(config: dict[str, Any]) -> AS2TestEnv:
    """Build a ROS-free AS2TestEnv instance from the training config.

    The constructor performs no ROS work (DroneInterface is created lazily on
    first reset), so the instance can be used to read the env's own
    config-derived boundary attributes and helper formulas without a
    simulator.
    """
    env_kwargs = build_env_kwargs(config['environment'])
    return AS2TestEnv(drone_namespace='boundary_probe', **env_kwargs)


@dataclass(frozen=True)
class BoundaryParams:
    """Env-derived boundary and reward parameters used by the pure layer."""

    xy_limit: float
    z_min: float
    z_max: float
    pos_limit: float
    unsafe_low_altitude_threshold: float
    low_altitude_guard_height: float
    guard_climb_speed: float
    max_vel: float
    target_pose: tuple[float, float, float, float]
    start_pose: tuple[float, float, float, float] | None
    oob_penalty: float
    success_reward: float
    distance_threshold: float
    progress_reward_weight: float
    vertical_safety_band: float
    vertical_safety_penalty_weight: float
    vertical_descent_penalty_weight: float
    speed_deadband: float
    path_facing_weight: float


def boundary_params_from_env(env: AS2TestEnv) -> BoundaryParams:
    """Read boundary/reward parameters from the env's own attributes.

    Limits mirror ``AS2TestEnv._is_out_of_bounds`` (position box) and
    ``AS2TestEnv._is_unsafe_low_altitude`` (training safety band); the guard
    height comes from the env's own ``_low_altitude_guard_height`` formula so
    nothing is duplicated.
    """
    z_min, z_max = env.height_bounds
    return BoundaryParams(
        xy_limit=min(float(env.scene_bounds_xy), float(env.pos_limit)),
        z_min=float(z_min),
        z_max=min(float(z_max), float(env.pos_limit)),
        pos_limit=float(env.pos_limit),
        unsafe_low_altitude_threshold=float(env.unsafe_low_altitude_threshold),
        low_altitude_guard_height=float(env._low_altitude_guard_height()),
        guard_climb_speed=min(
            float(env.max_vel),
            max(float(env.low_altitude_guard_climb_speed), float(env.reset_min_speed), 0.0),
        ),
        max_vel=float(env.max_vel),
        target_pose=tuple(float(v) for v in env._target_pose),
        start_pose=(
            tuple(float(v) for v in env.fixed_start_pose)
            if env.fixed_start_pose is not None
            else None
        ),
        oob_penalty=float(env.oob_penalty),
        success_reward=float(env.success_reward),
        distance_threshold=float(env.distance_threshold),
        progress_reward_weight=float(env.progress_reward_weight),
        vertical_safety_band=float(env.vertical_safety_band),
        vertical_safety_penalty_weight=float(env.vertical_safety_penalty_weight),
        vertical_descent_penalty_weight=float(env.vertical_descent_penalty_weight),
        speed_deadband=float(env.speed_deadband),
        path_facing_weight=float(env.path_facing_weight),
    )


def classify_position(params: BoundaryParams, position: list[float] | tuple[float, ...]) -> str:
    """Classify a position exactly like the env's pre-step safety check.

    Returns 'out_of_bounds', 'unsafe_low_altitude', or 'safe'. Boundary
    semantics are strict inequalities: a point exactly ON a face is inside.
    """
    x, y, z = (float(position[0]), float(position[1]), float(position[2]))
    if not all(math.isfinite(value) for value in (x, y, z)):
        return 'out_of_bounds'
    if (
        abs(x) > params.xy_limit
        or abs(y) > params.xy_limit
        or z < params.z_min
        or z > params.z_max
        or abs(z) > params.pos_limit
    ):
        return 'out_of_bounds'
    if params.z_min <= z < params.unsafe_low_altitude_threshold:
        return 'unsafe_low_altitude'
    return 'safe'


def expected_vertical_safety_penalty(params: BoundaryParams, z: float, raw_vz: float) -> float:
    """Expected value of the env's vertical safety penalty at altitude z."""
    band = params.vertical_safety_band
    altitude_weight = params.vertical_safety_penalty_weight
    descent_weight = params.vertical_descent_penalty_weight
    if band <= 0.0 or (altitude_weight <= 0.0 and descent_weight <= 0.0):
        return 0.0
    band_top = params.unsafe_low_altitude_threshold + band
    if float(z) >= band_top:
        return 0.0
    severity = min(max((band_top - float(z)) / band, 0.0), 1.0)
    penalty = -altitude_weight * severity
    if float(raw_vz) < 0.0:
        normalized_descent = min(abs(float(raw_vz)) / max(params.max_vel, 1e-6), 1.0)
        penalty -= descent_weight * severity * normalized_descent
    return float(penalty)


def _distance_to_target(params: BoundaryParams, position: list[float] | tuple[float, ...]) -> float:
    tx, ty, tz, _ = params.target_pose
    return math.sqrt(
        (float(position[0]) - tx) ** 2
        + (float(position[1]) - ty) ** 2
        + (float(position[2]) - tz) ** 2
    )


def compute_expected_outcome(
    params: BoundaryParams,
    position: list[float] | tuple[float, ...],
    yaw: float = 0.0,
    action: list[float] | tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    previous_distance: float | None = None,
) -> dict[str, Any]:
    """Expected env step() outcome for a drone at `position` taking `action`.

    Mirrors AS2TestEnv semantics:
    - Pre-step safety check fires first: out_of_bounds / unsafe_low_altitude
      states terminate immediately with reward == -oob_penalty, before any
      motion command is sent.
    - Otherwise the low-altitude action guard replaces a non-climbing vz when
      z <= guard height, and the reward is
      -distance_norm + path_facing + progress + vertical_safety_penalty
      (+ success_reward when distance < distance_threshold).

    ``expected_reward`` is exact for terminal points. For non-terminal points
    it is the exact reward under static dynamics (drone does not move during
    the step and horizontal speed is below the deadband) — this holds for the
    mock-drone tests; the live runner instead checks the env's reward
    composition against observed telemetry.
    """
    classification = classify_position(params, position)
    expected_obs = normalized_observation_from_state(
        position=[float(v) for v in position[:3]],
        yaw=float(yaw),
        target_pose=list(params.target_pose),
        pos_limit=params.pos_limit,
    )
    outcome: dict[str, Any] = {
        'classification': classification,
        'expected_obs': expected_obs,
        'truncated': False,
    }

    if classification in TERMINAL_REASONS:
        outcome.update({
            'terminated': True,
            'terminal_reason': classification,
            'expected_reward': -params.oob_penalty,
            'reward_exact': True,
            'is_out_of_bounds': classification == 'out_of_bounds',
            'is_unsafe_low_altitude': classification == 'unsafe_low_altitude',
            'guard_active': False,
            'motion_command_publication_count': 0,
        })
        return outcome

    vx = min(max(float(action[0]), -params.max_vel), params.max_vel)
    vy = min(max(float(action[1]), -params.max_vel), params.max_vel)
    raw_vz = min(max(float(action[2]), -params.max_vel), params.max_vel)
    guard_active = raw_vz <= 0.0 and float(position[2]) <= params.low_altitude_guard_height
    expected_action_sent_vz = params.guard_climb_speed if guard_active else raw_vz

    distance = _distance_to_target(params, position)
    d_max = params.pos_limit * math.sqrt(3.0)
    distance_norm = min(distance / d_max, 1.0)
    reward_distance = -distance_norm
    path_facing_reward = 0.0  # static assumption: speed below deadband
    progress_reward = (
        params.progress_reward_weight * (float(previous_distance) - distance)
        if previous_distance is not None and math.isfinite(distance)
        else 0.0
    )
    vertical_penalty = expected_vertical_safety_penalty(params, float(position[2]), raw_vz)
    reward = reward_distance + path_facing_reward + progress_reward + vertical_penalty

    success = distance < params.distance_threshold
    if success:
        reward += params.success_reward

    outcome.update({
        'terminated': bool(success),
        'terminal_reason': 'success' if success else None,
        'expected_reward': float(reward),
        'reward_exact': False,
        'is_out_of_bounds': False,
        'is_unsafe_low_altitude': False,
        'is_success': bool(success),
        'guard_active': bool(guard_active),
        'expected_action_sent': [vx, vy, float(expected_action_sent_vz), float(action[3])],
        'expected_reward_components': {
            'reward_distance': reward_distance,
            'path_facing_reward': path_facing_reward,
            'progress_reward': progress_reward,
            'vertical_safety_penalty': vertical_penalty,
        },
        'distance': distance,
        'distance_norm': distance_norm,
    })
    return outcome


def generate_boundary_sweep(params: BoundaryParams, epsilon: float = 0.05) -> list[dict[str, Any]]:
    """Generate the boundary sweep set from env-derived parameters.

    Categories:
    - face_on:       point exactly on each of the 6 position-box faces
    - face_inside:   epsilon inside each face (must NOT be out_of_bounds)
    - face_outside:  epsilon outside each face (must be out_of_bounds)
    - corner:        the 8 corners of the position box
    - unsafe_low_altitude_pair: epsilon above (safe) / below (terminal) the
      unsafe_low_altitude_threshold
    - guard_pair:    epsilon below (guard active on vz<=0) / above (inactive)
      the low-altitude guard height
    - start_pose / target_pose sanity points

    Note: with configs where unsafe_low_altitude_threshold > height_bounds[0],
    the bottom face and bottom corners are inside the position box but within
    the unsafe band, so their expected outcome is an unsafe_low_altitude
    terminal — computed, not hardcoded.
    """
    if epsilon <= 0.0:
        raise ValueError('epsilon must be > 0')
    lo = params.xy_limit
    z_min, z_max = params.z_min, params.z_max
    guard_h = params.low_altitude_guard_height
    threshold = params.unsafe_low_altitude_threshold

    # A z well inside the safe band and above the guard height when possible.
    if guard_h < z_max:
        safe_z = guard_h + 0.5 * (z_max - guard_h)
    else:
        safe_z = threshold + 0.5 * (z_max - threshold)

    def point(name: str, category: str, position: list[float], yaw: float = 0.0) -> dict[str, Any]:
        return {
            'name': name,
            'category': category,
            'position': [float(v) for v in position],
            'yaw': float(yaw),
            'action': list(ZERO_ACTION),
        }

    points: list[dict[str, Any]] = []

    faces = [
        ('x_pos', [lo, 0.0, safe_z], [lo - epsilon, 0.0, safe_z], [lo + epsilon, 0.0, safe_z]),
        ('x_neg', [-lo, 0.0, safe_z], [-lo + epsilon, 0.0, safe_z], [-lo - epsilon, 0.0, safe_z]),
        ('y_pos', [0.0, lo, safe_z], [0.0, lo - epsilon, safe_z], [0.0, lo + epsilon, safe_z]),
        ('y_neg', [0.0, -lo, safe_z], [0.0, -lo + epsilon, safe_z], [0.0, -lo - epsilon, safe_z]),
        ('z_top', [0.0, 0.0, z_max], [0.0, 0.0, z_max - epsilon], [0.0, 0.0, z_max + epsilon]),
        ('z_bottom', [0.0, 0.0, z_min], [0.0, 0.0, z_min + epsilon], [0.0, 0.0, z_min - epsilon]),
    ]
    for face_name, on, inside, outside in faces:
        points.append(point(f'face_on_{face_name}', 'face_on', on))
        points.append(point(f'face_inside_{face_name}', 'face_inside', inside))
        points.append(point(f'face_outside_{face_name}', 'face_outside', outside))

    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for cz, cz_name in ((z_min, 'bottom'), (z_max, 'top')):
                name = f'corner_{"p" if sx > 0 else "n"}x_{"p" if sy > 0 else "n"}y_{cz_name}'
                points.append(point(name, 'corner', [sx * lo, sy * lo, cz]))

    points.append(point(
        'unsafe_threshold_above', 'unsafe_low_altitude_pair', [0.0, 0.0, threshold + epsilon],
    ))
    points.append(point(
        'unsafe_threshold_below', 'unsafe_low_altitude_pair', [0.0, 0.0, threshold - epsilon],
    ))

    guard_below_z = max(threshold + epsilon, guard_h - epsilon)
    guard_above_z = min(z_max - epsilon, guard_h + epsilon)
    points.append(point('guard_below_height', 'guard_pair', [0.0, 0.0, guard_below_z]))
    points.append(point('guard_above_height', 'guard_pair', [0.0, 0.0, guard_above_z]))

    if params.start_pose is not None:
        points.append(point(
            'start_pose', 'start_pose', list(params.start_pose[:3]), yaw=params.start_pose[3],
        ))
    points.append(point(
        'target_pose', 'target_pose', list(params.target_pose[:3]), yaw=params.target_pose[3],
    ))

    return points


# ---------------------------------------------------------------------------
# Live layer: service teleport + per-point evaluation
# ---------------------------------------------------------------------------


def _single_inner_env(vec_env: Any) -> Any:
    current = vec_env
    while hasattr(current, 'venv'):
        current = current.venv
    envs = getattr(current, 'envs', None)
    if not envs or len(envs) != 1:
        raise RuntimeError('Boundary sweep expects exactly one VecEnv environment')
    env = envs[0]
    return getattr(env, 'unwrapped', env)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def reestablish_speed_command_path_after_teleport(inner_env: Any) -> None:
    """Re-enter SPEED mode after a raw teleport, mirroring the env's post-service sequence.

    The simulator fork's ``ResetSimulatorState`` callback leaves both the
    platform-info mirror and the inner simulator in POSITION mode holding the
    teleport pose. The env's service-backed reset always re-establishes the
    SPEED command path right after the service call
    (``_try_service_backed_reset``: offboard reassert -> controller mode
    refresh -> speed controller rebuild). A raw teleport must do the same:
    otherwise later speed commands are accepted on the ROS side but overridden
    by the platform's position hold, and the next in-air velocity reset stalls
    at the teleport altitude until timeout.

    Reuses the env's own helpers in the env's order. The takeoff, start-pose
    hold, and actionability probes are intentionally skipped so out-of-bounds
    teleport poses remain reachable for the pre-step safety branch under test.
    """
    if not inner_env._ensure_offboard_after_reset():
        raise RuntimeError(
            'offboard reassert failed after raw teleport: '
            f'{inner_env._last_reset_diagnostics}'
        )
    if not inner_env._refresh_controller_mode_after_service_reset():
        raise RuntimeError(
            'controller mode refresh failed after raw teleport: '
            f'{inner_env._last_reset_diagnostics}'
        )
    if not inner_env._reset_velocity_controller():
        raise RuntimeError(
            'speed controller rebuild failed after raw teleport: '
            f'{inner_env._last_reset_diagnostics}'
        )


def teleport_via_service(
    inner_env: Any,
    pose: list[float],
    settle_tolerance: float,
    settle_timeout: float,
    poll_interval: float = 0.05,
) -> dict[str, Any]:
    """Teleport the simulator mid-episode via the raw ResetSimulatorState call.

    Deliberately bypasses the env's `_try_service_backed_reset` post-reset
    validators (start-pose hold, actionability probe): out-of-bounds sweep
    poses must be reachable so the env's own pre-step safety branch can be
    exercised on the next `step()`. The SPEED command path is still
    re-established (see ``reestablish_speed_command_path_after_teleport``)
    because the service forces the platform into a POSITION hold.
    """
    if not inner_env._ensure_reset_service_client():
        raise RuntimeError(
            f'reset service client unavailable: {inner_env._last_reset_diagnostics}'
        )
    client = inner_env._reset_service_client
    if hasattr(client, 'wait_for_service') and not client.wait_for_service(
        timeout_sec=inner_env.reset_service_timeout,
    ):
        raise RuntimeError(f'reset service not available: {inner_env._reset_service_name()}')

    request = inner_env._build_reset_service_request([float(v) for v in pose])
    future = client.call_async(request)
    if not inner_env._spin_reset_future_until_complete(future, timeout_sec=settle_timeout):
        raise RuntimeError('teleport service future timed out')
    response = future.result()
    if not bool(getattr(response, 'success', False)):
        raise RuntimeError(f'teleport rejected: {getattr(response, "message", "")}')

    reestablish_speed_command_path_after_teleport(inner_env)

    deadline = time.monotonic() + max(float(settle_timeout), 0.5)
    observed = None
    while time.monotonic() < deadline:
        observed = inner_env._current_position_xyz()
        if observed is not None:
            error = math.sqrt(sum((observed[i] - float(pose[i])) ** 2 for i in range(3)))
            if error <= settle_tolerance:
                return {
                    'position': observed,
                    'position_error': error,
                    'settled': True,
                    'command_path_reestablished': True,
                }
        time.sleep(poll_interval)
    error = (
        math.sqrt(sum((observed[i] - float(pose[i])) ** 2 for i in range(3)))
        if observed is not None else float('inf')
    )
    return {
        'position': observed,
        'position_error': error,
        'settled': False,
        'command_path_reestablished': True,
    }


def _check(name: str, passed: bool, expected: Any, observed: Any) -> dict[str, Any]:
    return {
        'check': name,
        'passed': bool(passed),
        'expected': _json_safe(expected),
        'observed': _json_safe(observed),
    }


def _obs_matches(expected_obs: list[float], obs: Any, tolerance: float) -> bool:
    values = np.asarray(obs, dtype=np.float64).reshape(-1)
    if values.shape[0] != 4:
        return False
    return bool(np.all(np.abs(values - np.asarray(expected_obs, dtype=np.float64)) <= tolerance))


def evaluate_step_against_expected(
    params: BoundaryParams,
    expected: dict[str, Any],
    step_obs: Any,
    reward: float,
    done: bool,
    info: dict[str, Any],
    previous_distance: float | None,
    obs_tolerance: float,
    reward_tolerance: float,
    component_tolerance: float,
) -> list[dict[str, Any]]:
    """Assert one live step result against a pure expected outcome."""
    checks: list[dict[str, Any]] = []
    terminal_reason = info.get('terminal_reason')

    checks.append(_check(
        'terminated', bool(done) == bool(expected['terminated']), expected['terminated'], done,
    ))
    checks.append(_check(
        'terminal_reason',
        terminal_reason == expected['terminal_reason'],
        expected['terminal_reason'],
        terminal_reason,
    ))
    checks.append(_check(
        'not_truncated',
        not bool(info.get('TimeLimit.truncated', False)),
        False,
        bool(info.get('TimeLimit.truncated', False)),
    ))

    # Observation contract: obs must encode the env-reported state per the
    # normalized [-1, 1] contract (position/yaw relative to target).
    reported_position = info.get('position')
    reported_yaw = (
        float(info['orientation'][2])
        if isinstance(info.get('orientation'), (list, tuple)) and len(info['orientation']) >= 3
        else 0.0
    )
    if reported_position is not None:
        obs_from_reported_state = normalized_observation_from_state(
            position=[float(v) for v in reported_position[:3]],
            yaw=reported_yaw,
            target_pose=list(params.target_pose),
            pos_limit=params.pos_limit,
        )
        checks.append(_check(
            'obs_matches_env_reported_state',
            _obs_matches(obs_from_reported_state, step_obs, obs_tolerance),
            obs_from_reported_state,
            step_obs,
        ))
    checks.append(_check(
        'obs_matches_teleport_state',
        _obs_matches(expected['expected_obs'], step_obs, obs_tolerance),
        expected['expected_obs'],
        step_obs,
    ))

    if expected['classification'] in TERMINAL_REASONS:
        checks.append(_check(
            'reward_exact_oob_penalty',
            abs(float(reward) - float(expected['expected_reward'])) <= reward_tolerance,
            expected['expected_reward'],
            reward,
        ))
        checks.append(_check(
            'is_out_of_bounds_flag',
            bool(info.get('is_out_of_bounds', False)) == bool(expected['is_out_of_bounds']),
            expected['is_out_of_bounds'],
            info.get('is_out_of_bounds'),
        ))
        checks.append(_check(
            'is_unsafe_low_altitude_flag',
            bool(info.get('is_unsafe_low_altitude', False)) == bool(expected['is_unsafe_low_altitude']),
            expected['is_unsafe_low_altitude'],
            info.get('is_unsafe_low_altitude'),
        ))
        checks.append(_check(
            'no_motion_command_before_terminal',
            int(info.get('motion_command_publication_count', -1)) == 0,
            0,
            info.get('motion_command_publication_count'),
        ))
        return checks

    # Non-terminal (or success): validate reward composition against the
    # env-reported components and the components against telemetry.
    components = {
        key: float(info.get(key, float('nan')))
        for key in (
            'reward_distance', 'path_facing_reward', 'progress_reward', 'vertical_safety_penalty',
        )
    }
    component_sum = sum(components.values())
    if expected.get('is_success'):
        component_sum += params.success_reward
    checks.append(_check(
        'reward_equals_component_sum',
        math.isfinite(component_sum) and abs(float(reward) - component_sum) <= component_tolerance,
        component_sum,
        reward,
    ))

    reported_position = info.get('position')
    if reported_position is not None:
        distance = _distance_to_target(params, reported_position)
        expected_reward_distance = -min(distance / (params.pos_limit * math.sqrt(3.0)), 1.0)
        checks.append(_check(
            'reward_distance_matches_position',
            abs(components['reward_distance'] - expected_reward_distance) <= component_tolerance,
            expected_reward_distance,
            components['reward_distance'],
        ))
        if previous_distance is not None:
            expected_progress = params.progress_reward_weight * (float(previous_distance) - distance)
            checks.append(_check(
                'progress_reward_matches_distances',
                abs(components['progress_reward'] - expected_progress) <= component_tolerance,
                expected_progress,
                components['progress_reward'],
            ))

    altitude_before = info.get('step_altitude_before')
    if altitude_before is not None and math.isfinite(float(altitude_before)):
        expected_vertical = expected_vertical_safety_penalty(params, float(altitude_before), 0.0)
        checks.append(_check(
            'vertical_safety_penalty_matches_altitude',
            abs(components['vertical_safety_penalty'] - expected_vertical) <= component_tolerance,
            expected_vertical,
            components['vertical_safety_penalty'],
        ))

    checks.append(_check(
        'low_altitude_guard_active',
        bool(info.get('low_altitude_guard_active', False)) == bool(expected['guard_active']),
        expected['guard_active'],
        info.get('low_altitude_guard_active'),
    ))
    if expected['guard_active']:
        action_sent = info.get('action_sent')
        sent_vz = (
            float(action_sent[2])
            if isinstance(action_sent, (list, tuple)) and len(action_sent) >= 3
            else float('nan')
        )
        checks.append(_check(
            'guard_climb_speed_applied',
            abs(sent_vz - params.guard_climb_speed) <= 1e-6,
            params.guard_climb_speed,
            sent_vz,
        ))
    return checks


def evaluate_live_point(
    params: BoundaryParams,
    point: dict[str, Any],
    pre_step_position: list[float],
    pre_step_yaw: float,
    previous_distance: float | None,
    step_obs: Any,
    reward: float,
    done: bool,
    info: dict[str, Any],
    obs_tolerance: float,
    reward_tolerance: float,
    component_tolerance: float,
) -> dict[str, Any]:
    """Evaluate one live sweep point, tolerating boundary drift during the step.

    The primary expectation is computed from the observed post-teleport
    (pre-step) state. When the drone drifts across a boundary during the step
    hold (only possible for points at/near a face), the result is re-evaluated
    against the env-reported post-step state before declaring failure.
    """
    expected = compute_expected_outcome(
        params,
        pre_step_position,
        yaw=pre_step_yaw,
        action=point['action'],
        previous_distance=previous_distance,
    )
    checks = evaluate_step_against_expected(
        params, expected, step_obs, reward, done, info,
        previous_distance, obs_tolerance, reward_tolerance, component_tolerance,
    )
    passed = all(check['passed'] for check in checks)
    notes: list[str] = []

    requested_classification = classify_position(params, point['position'])
    observed_classification = str(expected['classification'])
    if requested_classification != observed_classification:
        notes.append(
            f'teleport landed in a different region than requested '
            f'(requested={requested_classification}, observed={observed_classification})'
        )

    if not passed:
        reported_position = info.get('position')
        if reported_position is not None:
            post_classification = classify_position(params, reported_position)
            if post_classification != observed_classification:
                reported_yaw = (
                    float(info['orientation'][2])
                    if isinstance(info.get('orientation'), (list, tuple))
                    and len(info['orientation']) >= 3
                    else pre_step_yaw
                )
                drift_expected = compute_expected_outcome(
                    params,
                    reported_position,
                    yaw=reported_yaw,
                    action=point['action'],
                    previous_distance=previous_distance,
                )
                drift_checks = evaluate_step_against_expected(
                    params, drift_expected, step_obs, reward, done, info,
                    previous_distance, obs_tolerance, reward_tolerance, component_tolerance,
                )
                if all(check['passed'] for check in drift_checks):
                    passed = True
                    checks = drift_checks
                    expected = drift_expected
                    notes.append(
                        'boundary drift during step: evaluated against env-reported '
                        f'post-step state ({observed_classification} -> {post_classification})'
                    )

    return {
        'name': point['name'],
        'category': point['category'],
        'requested_position': point['position'],
        'requested_yaw': point['yaw'],
        'requested_classification': requested_classification,
        'observed_pre_step_position': list(pre_step_position),
        'observed_pre_step_yaw': float(pre_step_yaw),
        'expected': _json_safe(expected),
        'observed': {
            'obs': _json_safe(step_obs),
            'reward': float(reward),
            'done': bool(done),
            'terminal_reason': info.get('terminal_reason'),
            'position': _json_safe(info.get('position')),
            'is_out_of_bounds': bool(info.get('is_out_of_bounds', False)),
            'is_unsafe_low_altitude': bool(info.get('is_unsafe_low_altitude', False)),
            'low_altitude_guard_active': bool(info.get('low_altitude_guard_active', False)),
            'action_sent': _json_safe(info.get('action_sent')),
            'reward_components': {
                key: _json_safe(info.get(key))
                for key in (
                    'reward_distance', 'path_facing_reward',
                    'progress_reward', 'vertical_safety_penalty',
                )
            },
        },
        'checks': checks,
        'notes': notes,
        'passed': bool(passed),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Live boundary sweep validating env observations and rewards at every limit',
    )
    parser.add_argument('--config', default='configs/train_ppo_phase1_exp008a.yaml')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Just-inside/just-outside offset in meters')
    parser.add_argument('--obs-tolerance', type=float, default=0.02,
                        help='Per-component tolerance on normalized observations')
    parser.add_argument('--reward-tolerance', type=float, default=1e-6,
                        help='Tolerance for exact terminal rewards (-oob_penalty)')
    parser.add_argument('--component-tolerance', type=float, default=0.05,
                        help='Tolerance for telemetry-derived reward components')
    parser.add_argument('--teleport-settle-tolerance', type=float, default=0.10,
                        help='Position error accepted before stepping after teleport')
    parser.add_argument('--teleport-settle-timeout', type=float, default=5.0)
    parser.add_argument('--categories', nargs='*', default=None,
                        help='Optional subset of sweep categories to run')
    parser.add_argument('--output-root', default='runs/diagnostics')
    parser.add_argument('--sleep-between-points', type=float, default=0.2)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    config = load_training_config(args.config)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(args.output_root) / f'boundary_sweep_{timestamp}'
    monitor_dir = run_dir / 'monitor'
    run_dir.mkdir(parents=True, exist_ok=True)
    monitor_dir.mkdir(parents=True, exist_ok=True)

    probe_env = build_probe_env(config)
    params = boundary_params_from_env(probe_env)
    points = generate_boundary_sweep(params, epsilon=float(args.epsilon))
    if args.categories:
        wanted = set(args.categories)
        points = [pt for pt in points if pt['category'] in wanted]

    result: dict[str, Any] = {
        'config': str(Path(args.config).resolve()),
        'run_dir': str(run_dir.resolve()),
        'epsilon': float(args.epsilon),
        'obs_tolerance': float(args.obs_tolerance),
        'reward_tolerance': float(args.reward_tolerance),
        'component_tolerance': float(args.component_tolerance),
        'boundary_params': asdict(params),
        'points_requested': len(points),
        'points': [],
    }
    output_path = run_dir / 'boundary_sweep_report.json'

    vec_env = None
    try:
        vec_env, namespaces = build_vec_env(
            config=config,
            monitor_dir=monitor_dir,
            num_envs_override=1,
        )
        inner_env = _single_inner_env(vec_env)
        result['namespaces'] = namespaces
        zero_batch = np.zeros((1, 4), dtype=np.float32)

        for point in points:
            print(f"[boundary-sweep] point '{point['name']}' ({point['category']}) "
                  f"-> {point['position']}")
            vec_env.reset()
            previous_distance = getattr(inner_env, '_previous_distance', None)
            teleport = teleport_via_service(
                inner_env,
                pose=point['position'] + [point['yaw']],
                settle_tolerance=float(args.teleport_settle_tolerance),
                settle_timeout=float(args.teleport_settle_timeout),
            )
            pre_position = teleport['position']
            if pre_position is None:
                raise RuntimeError(f"teleport left no readable pose for point '{point['name']}'")
            try:
                pre_yaw = float(inner_env._drone.orientation[2])
            except Exception:
                pre_yaw = float(point['yaw'])

            obs_batch, rewards, dones, infos = vec_env.step(zero_batch)
            info = dict(infos[0])
            done = bool(dones[0])
            # DummyVecEnv auto-resets on done and moves the terminal step
            # observation into info['terminal_observation'].
            step_obs = info.get('terminal_observation') if done else obs_batch[0]
            record = evaluate_live_point(
                params=params,
                point=point,
                pre_step_position=list(pre_position),
                pre_step_yaw=pre_yaw,
                previous_distance=previous_distance,
                step_obs=step_obs,
                reward=float(rewards[0]),
                done=done,
                info=info,
                obs_tolerance=float(args.obs_tolerance),
                reward_tolerance=float(args.reward_tolerance),
                component_tolerance=float(args.component_tolerance),
            )
            record['teleport'] = _json_safe(teleport)
            result['points'].append(record)
            status = 'PASS' if record['passed'] else 'FAIL'
            print(f"[boundary-sweep]   {status}: terminal_reason={info.get('terminal_reason')} "
                  f"reward={float(rewards[0]):.4f}")
            if args.sleep_between_points > 0.0:
                time.sleep(float(args.sleep_between_points))

        failed = [record for record in result['points'] if not record['passed']]
        result['points_evaluated'] = len(result['points'])
        result['points_failed'] = len(failed)
        result['failed_point_names'] = [record['name'] for record in failed]
        result['success'] = not failed
        output_path.write_text(
            json.dumps(_json_safe(result), indent=2, sort_keys=True), encoding='utf-8',
        )
        print(f'[boundary-sweep] report: {output_path.resolve()}')
        print(f"[boundary-sweep] {'SUCCESS' if result['success'] else 'FAILURE'}: "
              f"{result['points_evaluated'] - len(failed)}/{result['points_evaluated']} points passed")
        return 0 if result['success'] else 3
    except Exception as exc:
        result['success'] = False
        result['error'] = str(exc)
        output_path.write_text(
            json.dumps(_json_safe(result), indent=2, sort_keys=True), encoding='utf-8',
        )
        print(f'[boundary-sweep] ERROR: {exc}')
        print(f'[boundary-sweep] report: {output_path.resolve()}')
        return 2
    finally:
        if vec_env is not None:
            vec_env.close()


if __name__ == '__main__':
    raise SystemExit(main())
