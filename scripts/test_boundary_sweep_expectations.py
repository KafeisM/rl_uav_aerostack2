#!/usr/bin/env python3
"""ROS-free checks for the boundary sweep generation and expected outcomes.

Validates two things without the simulator:
1. Sweep-set generation covers every face, corner, epsilon pair, the unsafe
   low-altitude threshold, the guard height, and the start/target poses.
2. The pure expected-outcome computation matches the real ``AS2TestEnv.step()``
   behavior (observation, reward, terminal flag/reason, guard) when the drone
   is placed at representative sweep points with a mock drone — so the
   expectations used by the live sweep are self-validating.

Usage:
    conda run -n rl_uav python scripts/test_boundary_sweep_expectations.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_uav.envs.as2_test_env import AS2TestEnv
from rl_uav.training import build_env_kwargs, load_training_config
from scripts.validate_boundary_sweep import (
    TERMINAL_REASONS,
    boundary_params_from_env,
    build_probe_env,
    classify_position,
    compute_expected_outcome,
    generate_boundary_sweep,
    reestablish_speed_command_path_after_teleport,
    teleport_via_service,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / 'configs' / 'train_ppo_phase1_exp008a.yaml'
EPSILON = 0.05


class DummyDrone:
    def __init__(self, position: list[float], yaw: float = 0.0):
        self.position = list(position)
        self.speed = [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, float(yaw)]


class MockBoundaryEnv(AS2TestEnv):
    """AS2TestEnv with the ROS command path mocked out (SafetyTestEnv pattern)."""

    def _send_speed_command(self, twist, yaw_rate):  # type: ignore[no-untyped-def]
        self.last_command = [float(value) for value in twist] + [float(yaw_rate)]
        return True

    def _send_terminal_stop_command(self) -> bool:
        self.stop_command_sent = True
        return True

    def _publish_target_marker(self) -> None:
        return None

    def _recover_low_altitude_hover_before_velocity_reset(self, hover_height: float) -> bool:
        self.recovery_hover_requested = float(hover_height)
        return True


def load_config() -> dict:
    return load_training_config(DEFAULT_CONFIG)


def make_params(config: dict):
    return boundary_params_from_env(build_probe_env(config))


def make_mock_env(config: dict, position: list[float], yaw: float = 0.0) -> MockBoundaryEnv:
    env = MockBoundaryEnv(drone_namespace='drone0', **build_env_kwargs(config['environment']))
    env._drone = DummyDrone(position, yaw=yaw)
    env.stop_command_sent = False
    return env


def sweep_by_name(points: list[dict]) -> dict[str, dict]:
    by_name = {point['name']: point for point in points}
    assert len(by_name) == len(points), 'sweep point names must be unique'
    return by_name


def test_sweep_counts_and_categories() -> None:
    config = load_config()
    params = make_params(config)
    points = generate_boundary_sweep(params, epsilon=EPSILON)

    counts: dict[str, int] = {}
    for point in points:
        counts[point['category']] = counts.get(point['category'], 0) + 1

    assert counts['face_on'] == 6, counts
    assert counts['face_inside'] == 6, counts
    assert counts['face_outside'] == 6, counts
    assert counts['corner'] == 8, counts
    assert counts['unsafe_low_altitude_pair'] == 2, counts
    assert counts['guard_pair'] == 2, counts
    assert counts['target_pose'] == 1, counts
    expected_total = 30 + (1 if params.start_pose is not None else 0) + 1
    if params.start_pose is not None:
        assert counts['start_pose'] == 1, counts
    assert len(points) == expected_total, (len(points), expected_total)


def test_face_pairs_epsilon_geometry_and_classification() -> None:
    config = load_config()
    params = make_params(config)
    points = sweep_by_name(generate_boundary_sweep(params, epsilon=EPSILON))

    axis_of_face = {'x_pos': 0, 'x_neg': 0, 'y_pos': 1, 'y_neg': 1, 'z_top': 2, 'z_bottom': 2}
    for face, axis in axis_of_face.items():
        on = points[f'face_on_{face}']['position']
        inside = points[f'face_inside_{face}']['position']
        outside = points[f'face_outside_{face}']['position']
        # Epsilon pairs sit exactly EPSILON from the face along its axis.
        assert math.isclose(abs(inside[axis] - on[axis]), EPSILON, abs_tol=1e-12), face
        assert math.isclose(abs(outside[axis] - on[axis]), EPSILON, abs_tol=1e-12), face
        # Points exactly ON a face are inside the box (strict inequalities).
        assert classify_position(params, on) != 'out_of_bounds', face
        assert classify_position(params, inside) != 'out_of_bounds', face
        assert classify_position(params, outside) == 'out_of_bounds', face

    # Just-inside x/y/top faces are at a safe altitude and must not terminate.
    for face in ('x_pos', 'x_neg', 'y_pos', 'y_neg', 'z_top'):
        assert classify_position(params, points[f'face_inside_{face}']['position']) == 'safe', face

    # The bottom face may fall inside the unsafe training band when the
    # threshold sits above the physical floor (config-dependent, computed).
    bottom_inside = classify_position(params, points['face_inside_z_bottom']['position'])
    if params.unsafe_low_altitude_threshold > params.z_min + EPSILON:
        assert bottom_inside == 'unsafe_low_altitude'
    else:
        assert bottom_inside == 'safe'


def test_corners_cover_all_sign_combinations() -> None:
    config = load_config()
    params = make_params(config)
    points = generate_boundary_sweep(params, epsilon=EPSILON)
    corners = {
        tuple(point['position'])
        for point in points
        if point['category'] == 'corner'
    }
    expected = {
        (sx * params.xy_limit, sy * params.xy_limit, cz)
        for sx in (1.0, -1.0)
        for sy in (1.0, -1.0)
        for cz in (params.z_min, params.z_max)
    }
    assert corners == expected, corners.symmetric_difference(expected)


def test_threshold_and_guard_pairs_use_env_derived_heights() -> None:
    config = load_config()
    params = make_params(config)
    points = sweep_by_name(generate_boundary_sweep(params, epsilon=EPSILON))

    above = points['unsafe_threshold_above']['position'][2]
    below = points['unsafe_threshold_below']['position'][2]
    assert math.isclose(above, params.unsafe_low_altitude_threshold + EPSILON, abs_tol=1e-12)
    assert math.isclose(below, params.unsafe_low_altitude_threshold - EPSILON, abs_tol=1e-12)
    assert classify_position(params, points['unsafe_threshold_above']['position']) == 'safe'
    assert classify_position(params, points['unsafe_threshold_below']['position']) == 'unsafe_low_altitude'

    guard_below = points['guard_below_height']['position'][2]
    guard_above = points['guard_above_height']['position'][2]
    assert guard_below <= params.low_altitude_guard_height
    assert guard_above > params.low_altitude_guard_height
    expected_below = compute_expected_outcome(params, points['guard_below_height']['position'])
    expected_above = compute_expected_outcome(params, points['guard_above_height']['position'])
    assert expected_below['guard_active'] is True
    assert expected_above['guard_active'] is False


def _assert_env_step_matches_expected(config: dict, params, point: dict) -> None:
    env = make_mock_env(config, point['position'], yaw=point['yaw'])
    expected = compute_expected_outcome(
        params,
        point['position'],
        yaw=point['yaw'],
        action=point['action'],
        previous_distance=None,
    )
    obs, reward, terminated, truncated, info = env.step(
        np.asarray(point['action'], dtype=np.float32)
    )

    context = point['name']
    assert truncated is False, context
    assert terminated == expected['terminated'], (context, expected, info.get('terminal_reason'))
    assert info.get('terminal_reason') == expected['terminal_reason'], (
        context, expected['terminal_reason'], info.get('terminal_reason'),
    )
    assert bool(info.get('is_out_of_bounds', False)) == bool(expected['is_out_of_bounds']), context
    assert bool(info.get('is_unsafe_low_altitude', False)) == bool(
        expected['is_unsafe_low_altitude']
    ), context

    assert math.isclose(float(reward), float(expected['expected_reward']), abs_tol=1e-9), (
        context, float(reward), float(expected['expected_reward']),
    )
    if expected['classification'] in TERMINAL_REASONS:
        assert float(reward) == -env.oob_penalty, context
        assert info['motion_command_publication_count'] == 0, context
        assert env.stop_command_sent is True, context
    else:
        assert bool(info['low_altitude_guard_active']) == bool(expected['guard_active']), context
        if expected['guard_active']:
            assert math.isclose(
                float(info['action_sent'][2]),
                float(expected['expected_action_sent'][2]),
                abs_tol=1e-9,
            ), context

    np.testing.assert_allclose(
        np.asarray(obs, dtype=np.float64),
        np.asarray(expected['expected_obs'], dtype=np.float64),
        atol=1e-6,
        err_msg=context,
    )


def test_expected_outcomes_match_env_step_for_representative_points() -> None:
    config = load_config()
    params = make_params(config)
    points = sweep_by_name(generate_boundary_sweep(params, epsilon=EPSILON))

    representative = [
        'face_inside_x_pos',        # inside-safe
        'face_on_x_pos',            # exactly on a face is still inside
        'face_outside_x_pos',       # oob-face +x
        'face_outside_y_neg',       # oob-face -y
        'face_outside_z_top',       # oob above ceiling
        'face_outside_z_bottom',    # oob below physical floor
        'face_inside_z_bottom',     # inside box but inside unsafe band
        'unsafe_threshold_above',   # safe epsilon above threshold
        'unsafe_threshold_below',   # unsafe-low terminal
        'guard_below_height',       # guard active on vz<=0
        'guard_above_height',       # guard inactive
        'corner_px_py_top',         # top corner (safe boundary point)
        'corner_nx_ny_bottom',      # bottom corner (unsafe band)
        'target_pose',              # success terminal with success_reward
    ]
    if params.start_pose is not None:
        representative.append('start_pose')  # sanity: no terminal at episode start

    for name in representative:
        _assert_env_step_matches_expected(config, params, points[name])


def test_start_pose_expectation_is_non_terminal() -> None:
    config = load_config()
    params = make_params(config)
    if params.start_pose is None:
        return
    expected = compute_expected_outcome(params, list(params.start_pose[:3]), yaw=params.start_pose[3])
    assert expected['classification'] == 'safe'
    assert expected['terminated'] is False
    assert expected['terminal_reason'] is None


class _FakeFuture:
    def done(self) -> bool:
        return True

    def result(self):
        class _Response:
            success = True
            message = ''
        return _Response()


class _FakeResetClient:
    def __init__(self, env: '_FakeTeleportEnv'):
        self._env = env

    def wait_for_service(self, timeout_sec: float) -> bool:
        return True

    def call_async(self, request):
        self._env.calls.append('service_call')
        return _FakeFuture()


class _FakeTeleportEnv:
    """Duck-typed inner env recording the raw-teleport command-path sequence.

    Mirrors only the attributes ``teleport_via_service`` touches; the helper
    names must stay in sync with AS2TestEnv (see
    ``test_env_exposes_post_teleport_command_path_helpers``).
    """

    def __init__(
        self,
        pose_after_teleport: list[float],
        offboard_ok: bool = True,
        mode_refresh_ok: bool = True,
        controller_ok: bool = True,
    ):
        self.calls: list[str] = []
        self.reset_service_timeout = 1.0
        self._last_reset_diagnostics: dict = {}
        self._reset_service_client = _FakeResetClient(self)
        self._pose_after_teleport = list(pose_after_teleport)
        self._offboard_ok = offboard_ok
        self._mode_refresh_ok = mode_refresh_ok
        self._controller_ok = controller_ok

    def _ensure_reset_service_client(self) -> bool:
        self.calls.append('ensure_client')
        return True

    def _reset_service_name(self) -> str:
        return '/drone0/platform/reset_simulator_state'

    def _build_reset_service_request(self, pose: list[float]):
        self.calls.append('build_request')
        return object()

    def _spin_reset_future_until_complete(self, future, timeout_sec=None) -> bool:
        return True

    def _ensure_offboard_after_reset(self, confirm_platform_state: bool = False) -> bool:
        self.calls.append('offboard')
        return self._offboard_ok

    def _refresh_controller_mode_after_service_reset(self) -> bool:
        self.calls.append('mode_refresh')
        return self._mode_refresh_ok

    def _reset_velocity_controller(self) -> bool:
        self.calls.append('velocity_controller')
        return self._controller_ok

    def _current_position_xyz(self) -> list[float]:
        self.calls.append('read_pose')
        return list(self._pose_after_teleport)


def test_teleport_reestablishes_speed_command_path_in_env_order() -> None:
    """A raw teleport must re-enter SPEED mode exactly like the env's
    post-service-reset sequence, before the settle poll reads the pose."""
    pose = [0.0, 0.0, 1.05, 0.0]
    env = _FakeTeleportEnv(pose_after_teleport=pose[:3])
    result = teleport_via_service(
        env, pose=pose, settle_tolerance=0.10, settle_timeout=1.0,
    )
    assert result['settled'] is True, result
    assert result['command_path_reestablished'] is True, result
    first_read = env.calls.index('read_pose')
    assert env.calls.index('service_call') < env.calls.index('offboard'), env.calls
    assert env.calls[env.calls.index('offboard'):first_read] == [
        'offboard', 'mode_refresh', 'velocity_controller',
    ], env.calls


def test_teleport_fails_fast_when_command_path_cannot_be_reestablished() -> None:
    pose = [0.0, 0.0, 1.05, 0.0]
    scenarios = [
        ({'offboard_ok': False}, 'offboard reassert failed'),
        ({'mode_refresh_ok': False}, 'controller mode refresh failed'),
        ({'controller_ok': False}, 'speed controller rebuild failed'),
    ]
    for kwargs, expected_message in scenarios:
        env = _FakeTeleportEnv(pose_after_teleport=pose[:3], **kwargs)
        try:
            teleport_via_service(
                env, pose=pose, settle_tolerance=0.10, settle_timeout=1.0,
            )
        except RuntimeError as exc:
            assert expected_message in str(exc), (kwargs, str(exc))
            assert 'read_pose' not in env.calls, (kwargs, env.calls)
        else:
            raise AssertionError(f'expected RuntimeError for {kwargs}')


def test_env_exposes_post_teleport_command_path_helpers() -> None:
    """The sweep reuses env helpers by name; renames must fail loudly here."""
    for helper in (
        '_ensure_offboard_after_reset',
        '_refresh_controller_mode_after_service_reset',
        '_reset_velocity_controller',
        '_ensure_reset_service_client',
        '_build_reset_service_request',
        '_spin_reset_future_until_complete',
        '_current_position_xyz',
        '_reset_service_name',
    ):
        assert callable(getattr(AS2TestEnv, helper, None)), helper


def test_reestablish_helper_order_matches_env_post_service_sequence() -> None:
    """Guards parity with _try_service_backed_reset: offboard -> mode refresh
    -> velocity controller (takeoff/pose-hold/probes intentionally skipped)."""
    env = _FakeTeleportEnv(pose_after_teleport=[0.0, 0.0, 1.6])
    reestablish_speed_command_path_after_teleport(env)
    assert env.calls == ['offboard', 'mode_refresh', 'velocity_controller'], env.calls


def test_target_pose_expectation_is_success_with_bonus() -> None:
    config = load_config()
    params = make_params(config)
    expected = compute_expected_outcome(params, list(params.target_pose[:3]), yaw=params.target_pose[3])
    assert expected['classification'] == 'safe'
    assert expected['terminated'] is True
    assert expected['terminal_reason'] == 'success'
    assert float(expected['expected_reward']) >= params.success_reward - 1.0


def main() -> int:
    test_sweep_counts_and_categories()
    test_face_pairs_epsilon_geometry_and_classification()
    test_corners_cover_all_sign_combinations()
    test_threshold_and_guard_pairs_use_env_derived_heights()
    test_expected_outcomes_match_env_step_for_representative_points()
    test_start_pose_expectation_is_non_terminal()
    test_teleport_reestablishes_speed_command_path_in_env_order()
    test_teleport_fails_fast_when_command_path_cannot_be_reestablished()
    test_env_exposes_post_teleport_command_path_helpers()
    test_reestablish_helper_order_matches_env_post_service_sequence()
    test_target_pose_expectation_is_success_with_bonus()
    print('✓ PASS: boundary sweep expectations match real env step() semantics')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
