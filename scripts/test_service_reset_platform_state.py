#!/usr/bin/env python3
"""ROS-free checks for post-service-reset AS2 platform state confirmation.

A service-backed reset must not be reported successful unless the platform
info confirms the AS2 command gate (connected + armed + offboard + FLYING +
settled control mode). These tests cover both the confirmed-success path and
the explicit failure path when the platform state cannot be confirmed.

Usage:
    conda run -n rl_uav python3 scripts/test_service_reset_platform_state.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_uav.envs.as2_test_env import AS2TestEnv

PLATFORM_STATUS_LANDED = 1
PLATFORM_STATUS_FLYING = 3
CONTROL_MODE_UNSET = 0
CONTROL_MODE_POSITION = 2

READY_PLATFORM_INFO = {
    'connected': True,
    'armed': True,
    'offboard': True,
    'state': PLATFORM_STATUS_FLYING,
    'control_mode': CONTROL_MODE_POSITION,
}

NOT_FLYING_PLATFORM_INFO = {
    'connected': True,
    'armed': True,
    'offboard': True,
    'state': PLATFORM_STATUS_LANDED,
    'control_mode': CONTROL_MODE_POSITION,
}


class FakeResetFuture:
    def __init__(self, response):
        self._response = response

    def done(self) -> bool:
        return True

    def result(self):
        return self._response


class FakeResetClient:
    def __init__(self, response):
        self._response = response

    def wait_for_service(self, timeout_sec=None) -> bool:
        return True

    def call_async(self, request):
        return FakeResetFuture(self._response)


class FakeResetService:
    @staticmethod
    def Request():
        return types.SimpleNamespace()


def make_success_response():
    return types.SimpleNamespace(
        success=True,
        message='reset state applied',
        position_error=0.0,
        yaw_error=0.0,
        linear_speed_norm=0.0,
        angular_speed_norm=0.0,
        platform_fsm_synced=True,
        platform_control_mode_synced=True,
    )


class PlatformInfoDrone:
    def __init__(self, info: dict, reassertion_flips_to_flying: bool = False):
        self.position = [0.0, 0.0, 1.0]
        self.speed = [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, 0.0]
        self.info = dict(info)
        self._reassertion_flips = reassertion_flips_to_flying

    def arm(self, **kw) -> bool:
        return True

    def offboard(self, **kw) -> bool:
        return True

    def takeoff(self, **kw) -> bool:
        return True

    def land(self, **kw) -> bool:
        return True

    def manual(self, **kw) -> None:
        pass

    def shutdown(self, **kw) -> None:
        pass

    def apply_fsm_reassertion(self) -> bool:
        if self._reassertion_flips:
            self.info['state'] = PLATFORM_STATUS_FLYING
            return True
        return False


class PlatformStateTestEnv(AS2TestEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fsm_reassertions = 0

    def _publish_target_marker(self) -> None:
        return None

    def _send_speed_command(self, twist, yaw_speed):  # type: ignore[no-untyped-def]
        return True

    def _set_platform_flying_after_service_reset(self) -> bool:
        self.fsm_reassertions += 1
        return self._drone.apply_fsm_reassertion()


def make_env(drone: PlatformInfoDrone, **env_kwargs) -> PlatformStateTestEnv:
    kwargs = {
        'verbose': False,
        'step_duration': 0.0,
        'target_pose': [2.0, 0.0, 1.5, 0.0],
        'reset_service_timeout': 0.05,
        'reset_platform_state_timeout': 0.15,
    }
    kwargs.update(env_kwargs)
    env = PlatformStateTestEnv(**kwargs)
    env._drone = drone
    env._reset_service_type = FakeResetService
    env._reset_service_client = FakeResetClient(make_success_response())
    env._reset_velocity_controller = lambda: True  # type: ignore[method-assign]
    env._hold_service_reset_start_pose = lambda start_pose: True  # type: ignore[method-assign]
    env._wait_until_service_reset_pose_observed = lambda start_pose: True  # type: ignore[method-assign]
    env._confirm_post_service_command_path = lambda start_pose: True  # type: ignore[method-assign]
    return env


def test_service_reset_success_records_confirmed_platform_state() -> None:
    env = make_env(PlatformInfoDrone(READY_PLATFORM_INFO))

    assert env._try_service_backed_reset([0.0, 0.0, 1.0, 0.0]) is True
    diagnostics = env._last_reset_diagnostics
    assert diagnostics['reason'] == 'service_success'
    assert diagnostics['post_reset_offboard_ready'] is True
    assert diagnostics['post_reset_platform_fsm_ready'] is True
    assert diagnostics['post_reset_platform_reassertion_attempted'] is False
    assert diagnostics['post_reset_platform_state'] == READY_PLATFORM_INFO
    assert env.fsm_reassertions == 0
    assert env._is_flying is True
    assert 'post_reset_platform_fsm_ready=True' in env._format_reset_diagnostics()


def test_unconfirmed_platform_state_fails_service_reset() -> None:
    env = make_env(PlatformInfoDrone(NOT_FLYING_PLATFORM_INFO))

    assert env._try_service_backed_reset([0.0, 0.0, 1.0, 0.0]) is False
    diagnostics = env._last_reset_diagnostics
    assert diagnostics['reason'] == 'service_post_reset_command_path_failed'
    assert diagnostics['post_reset_platform_fsm_ready'] is False
    assert diagnostics['post_reset_platform_reassertion_attempted'] is True
    assert diagnostics['post_reset_platform_state']['state'] == PLATFORM_STATUS_LANDED
    assert env.fsm_reassertions == 1
    assert env._is_flying is False


def test_fsm_reassertion_recovers_platform_state() -> None:
    env = make_env(
        PlatformInfoDrone(NOT_FLYING_PLATFORM_INFO, reassertion_flips_to_flying=True)
    )

    assert env._try_service_backed_reset([0.0, 0.0, 1.0, 0.0]) is True
    diagnostics = env._last_reset_diagnostics
    assert diagnostics['reason'] == 'service_success'
    assert diagnostics['post_reset_platform_fsm_ready'] is True
    assert diagnostics['post_reset_platform_reassertion_attempted'] is True
    assert diagnostics['post_reset_platform_state']['state'] == PLATFORM_STATUS_FLYING
    assert env.fsm_reassertions == 1


def test_unsettled_control_mode_fails_service_reset() -> None:
    unset_mode_info = dict(READY_PLATFORM_INFO, control_mode=CONTROL_MODE_UNSET)
    env = make_env(PlatformInfoDrone(unset_mode_info))

    assert env._try_service_backed_reset([0.0, 0.0, 1.0, 0.0]) is False
    diagnostics = env._last_reset_diagnostics
    assert diagnostics['post_reset_platform_fsm_ready'] is False
    assert diagnostics['post_reset_platform_state']['control_mode'] == CONTROL_MODE_UNSET


def test_unobservable_platform_info_keeps_legacy_acceptance() -> None:
    drone = PlatformInfoDrone(READY_PLATFORM_INFO)
    del drone.info
    env = make_env(drone)

    assert env._try_service_backed_reset([0.0, 0.0, 1.0, 0.0]) is True
    diagnostics = env._last_reset_diagnostics
    assert diagnostics['post_reset_platform_fsm_ready'] is None
    assert diagnostics['post_reset_platform_state'] == 'unobservable'
    assert env.fsm_reassertions == 0


def test_fresh_reset_raises_when_platform_state_unconfirmed() -> None:
    env = make_env(
        PlatformInfoDrone(NOT_FLYING_PLATFORM_INFO),
        fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    )

    try:
        env.reset()
    except RuntimeError as exc:
        message = str(exc)
        assert 'Fresh service-backed reset failed' in message
        assert 'post_reset_platform_fsm_ready=False' in message
    else:
        raise AssertionError(
            'reset() must raise when the platform state cannot be confirmed'
        )


def test_full_reset_reports_success_with_confirmed_platform_state() -> None:
    env = make_env(
        PlatformInfoDrone(READY_PLATFORM_INFO),
        fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    )

    _, info = env.reset()
    assert info['reset_success'] is True
    diagnostics = env._last_reset_diagnostics
    assert diagnostics['reason'] == 'service_success'
    assert diagnostics['post_reset_platform_fsm_ready'] is True
    assert diagnostics['post_reset_platform_state'] == READY_PLATFORM_INFO


def main() -> int:
    test_service_reset_success_records_confirmed_platform_state()
    test_unconfirmed_platform_state_fails_service_reset()
    test_fsm_reassertion_recovers_platform_state()
    test_unsettled_control_mode_fails_service_reset()
    test_unobservable_platform_info_keeps_legacy_acceptance()
    test_fresh_reset_raises_when_platform_state_unconfirmed()
    test_full_reset_reports_success_with_confirmed_platform_state()
    print('✓ PASS: service reset requires confirmed AS2 platform state')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
