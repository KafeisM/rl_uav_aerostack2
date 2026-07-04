#!/usr/bin/env python3
"""ROS-free checks for full service-reset retry before failing an episode.

A transient post-reset command-path failure (accepted-but-ignored motion
references) must not kill a long training run when a fresh full teleport
retry would recover it. These tests cover retry recovery, retry exhaustion,
the non-retryable disabled-service path, and config validation.

Usage:
    conda run -n rl_uav python3 scripts/test_service_reset_retry.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_uav.envs.as2_test_env import AS2TestEnv

PLATFORM_STATUS_FLYING = 3
CONTROL_MODE_POSITION = 2

READY_PLATFORM_INFO = {
    'connected': True,
    'armed': True,
    'offboard': True,
    'state': PLATFORM_STATUS_FLYING,
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


class ReadyDrone:
    def __init__(self):
        self.position = [0.0, 0.0, 1.0]
        self.speed = [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, 0.0]
        self.info = dict(READY_PLATFORM_INFO)

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


class RetryTestEnv(AS2TestEnv):
    def _publish_target_marker(self) -> None:
        return None

    def _send_speed_command(self, twist, yaw_speed):  # type: ignore[no-untyped-def]
        return True


def make_env(actionability_outcomes: list[bool], **env_kwargs) -> RetryTestEnv:
    """Build an env whose actionability probe follows a scripted outcome list."""
    kwargs = {
        'verbose': False,
        'step_duration': 0.0,
        'target_pose': [2.0, 0.0, 1.5, 0.0],
        'reset_service_timeout': 0.05,
        'reset_platform_state_timeout': 0.15,
        'reset_service_retry_backoff_s': 0.0,
    }
    kwargs.update(env_kwargs)
    env = RetryTestEnv(**kwargs)
    env._drone = ReadyDrone()
    env._reset_service_type = FakeResetService
    env._reset_service_client = FakeResetClient(make_success_response())
    env._reset_velocity_controller = lambda: True  # type: ignore[method-assign]
    env._hold_service_reset_start_pose = lambda start_pose: True  # type: ignore[method-assign]
    env._wait_until_service_reset_pose_observed = lambda start_pose: True  # type: ignore[method-assign]

    outcomes = list(actionability_outcomes)
    env.actionability_calls = 0

    def scripted_actionability(start_pose) -> bool:
        env.actionability_calls += 1
        if outcomes:
            return outcomes.pop(0)
        return True

    env._confirm_post_service_command_path = scripted_actionability  # type: ignore[method-assign]
    return env


def test_retry_recovers_after_transient_command_path_failure() -> None:
    env = make_env(actionability_outcomes=[False, True], reset_service_max_attempts=3)

    assert env._service_backed_reset_with_retries([0.0, 0.0, 1.0, 0.0]) is True
    assert env.actionability_calls == 2
    diagnostics = env._last_reset_diagnostics
    assert diagnostics['reason'] == 'service_success'
    assert diagnostics['reset_service_attempts'] == 2
    failures = diagnostics['reset_service_attempt_failures']
    assert len(failures) == 1
    assert failures[0]['attempt'] == 1
    assert failures[0]['reason'] == 'service_post_reset_command_path_failed'
    assert env._is_flying is True


def test_first_attempt_success_records_single_attempt() -> None:
    env = make_env(actionability_outcomes=[True], reset_service_max_attempts=3)

    assert env._service_backed_reset_with_retries([0.0, 0.0, 1.0, 0.0]) is True
    assert env.actionability_calls == 1
    diagnostics = env._last_reset_diagnostics
    assert diagnostics['reset_service_attempts'] == 1
    assert 'reset_service_attempt_failures' not in diagnostics


def test_retry_exhaustion_returns_failure_with_attempt_history() -> None:
    env = make_env(
        actionability_outcomes=[False, False, False],
        reset_service_max_attempts=3,
    )

    assert env._service_backed_reset_with_retries([0.0, 0.0, 1.0, 0.0]) is False
    assert env.actionability_calls == 3
    diagnostics = env._last_reset_diagnostics
    assert diagnostics['reason'] == 'service_post_reset_command_path_failed'
    assert diagnostics['reset_service_attempts'] == 3
    assert len(diagnostics['reset_service_attempt_failures']) == 3
    assert env._is_flying is False


def test_full_reset_raises_only_after_retry_exhaustion() -> None:
    env = make_env(
        actionability_outcomes=[False, False],
        reset_service_max_attempts=2,
        fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    )

    try:
        env.reset()
    except RuntimeError as exc:
        message = str(exc)
        assert 'Fresh service-backed reset failed' in message
        assert 'reset_service_attempts=2' in message
    else:
        raise AssertionError('reset() must raise after retry exhaustion')
    assert env.actionability_calls == 2


def test_full_reset_survives_transient_failure_via_retry() -> None:
    env = make_env(
        actionability_outcomes=[False, True],
        reset_service_max_attempts=3,
        fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    )

    _, info = env.reset()
    assert info['reset_success'] is True
    assert env.actionability_calls == 2
    assert env._last_reset_diagnostics['reset_service_attempts'] == 2


def test_service_disabled_does_not_retry() -> None:
    env = make_env(
        actionability_outcomes=[],
        reset_service_max_attempts=3,
        use_simulator_reset_service=False,
    )

    assert env._service_backed_reset_with_retries([0.0, 0.0, 1.0, 0.0]) is False
    assert env.actionability_calls == 0
    diagnostics = env._last_reset_diagnostics
    assert diagnostics['reason'] == 'service_disabled'
    assert diagnostics['reset_service_attempts'] == 1


def test_retry_config_validation() -> None:
    try:
        make_env(actionability_outcomes=[], reset_service_max_attempts=0)
    except ValueError as exc:
        assert 'reset_service_max_attempts' in str(exc)
    else:
        raise AssertionError('reset_service_max_attempts=0 must raise ValueError')

    try:
        make_env(actionability_outcomes=[], reset_service_retry_backoff_s=-1.0)
    except ValueError as exc:
        assert 'reset_service_retry_backoff_s' in str(exc)
    else:
        raise AssertionError('negative retry backoff must raise ValueError')


def test_default_retry_attempts_is_resilient() -> None:
    env = make_env(actionability_outcomes=[])
    assert env.reset_service_max_attempts == 3


def main() -> int:
    test_retry_recovers_after_transient_command_path_failure()
    test_first_attempt_success_records_single_attempt()
    test_retry_exhaustion_returns_failure_with_attempt_history()
    test_full_reset_raises_only_after_retry_exhaustion()
    test_full_reset_survives_transient_failure_via_retry()
    test_service_disabled_does_not_retry()
    test_retry_config_validation()
    test_default_retry_attempts_is_resilient()
    print('✓ PASS: full service-reset retry absorbs transient command-path failures')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
