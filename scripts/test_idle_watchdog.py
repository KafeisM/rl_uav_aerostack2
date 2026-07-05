#!/usr/bin/env python3
"""ROS-free checks for the idle-command hover watchdog in AS2TestEnv.

Live probe data showed the AS2 controller holds the LAST velocity reference
as a live setpoint indefinitely (8.05 m drift over 20 s with no commands).
In SubprocVecEnv lockstep training, a sibling drone receives no commands for
the 10-25 s of another env's certified reset and flies away on its stale
reference (root cause of the exp017 collapse). The watchdog publishes a
zero-velocity hover hold after `idle_command_watchdog_s` seconds without any
motion reference and keeps republishing it at `command_publication_interval`
cadence until real activity resumes.

Covers:
    - Default (0.0) disables the watchdog: no thread, no publishes.
    - Enabled watchdog engages after the threshold and REPEATS zero publishes.
    - Any command activity resets the idle clock.
    - The watchdog's own publishes do not count as activity (republishing
      keeps going while idle persists).
    - Not-flying vehicles never receive watchdog publishes.
    - close() stops the thread and is idempotent.
    - Constructor validation of idle_command_watchdog_s.

Usage:
    conda run -n rl_uav python scripts/test_idle_watchdog.py
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_uav.envs.as2_test_env import AS2TestEnv

# The ROS-free harness has no speed handler, so the base publish path logs an
# error on every send; silence the module logger to keep test output readable.
logging.getLogger('rl_uav.envs.as2_test_env').setLevel(logging.CRITICAL)

THRESHOLD = 0.12
STEP_ACTION = np.array([0.3, 0.0, 0.0, 0.0], dtype=np.float32)


class DummyDrone:
    def __init__(self, position: list[float], speed=None, yaw: float = 0.0):
        self.position = list(position)
        self.speed = list(speed) if speed is not None else [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, float(yaw)]

    def land(self, speed: float = 0.0) -> bool:
        return True

    def manual(self) -> bool:
        return True

    def shutdown(self) -> None:
        return None


class WatchdogTestEnv(AS2TestEnv):
    """Env with a thread-safe recording stub around the publish path."""

    def __init__(self, **kwargs):
        self._record_lock = threading.Lock()
        self._sent_commands: list[tuple[float, list[float], float]] = []
        super().__init__(**kwargs)

    def _send_speed_command(self, twist, yaw_speed):  # type: ignore[no-untyped-def]
        with self._record_lock:
            self._sent_commands.append(
                (time.monotonic(), [float(v) for v in twist], float(yaw_speed))
            )
        # Run the real base method so the idle-clock bookkeeping is exercised
        # (no speed handler exists here, so the actual publish is a no-op).
        super()._send_speed_command(twist, yaw_speed)
        return True

    def _send_terminal_stop_command(self) -> bool:
        return True

    def _publish_target_marker(self) -> None:
        return None

    def commands(self) -> list[tuple[float, list[float], float]]:
        with self._record_lock:
            return list(self._sent_commands)

    def zero_commands(self) -> list[tuple[float, list[float], float]]:
        return [
            entry for entry in self.commands()
            if entry[1] == [0.0, 0.0, 0.0] and entry[2] == 0.0
        ]


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


def make_env(**overrides) -> WatchdogTestEnv:
    kwargs = dict(BASE_KWARGS)
    kwargs.update(overrides)
    env = WatchdogTestEnv(**kwargs)
    env._drone = DummyDrone([1.0, 0.0, 1.5])
    return env


def test_default_disabled_no_thread_no_publishes() -> None:
    env = make_env()
    assert env.idle_command_watchdog_s == 0.0
    env._is_flying = True

    env.step(STEP_ACTION)
    time.sleep(0.5)

    assert env._idle_watchdog_thread is None, 'no thread must exist when disabled'
    assert env.zero_commands() == [], 'no watchdog publishes when disabled'


def test_enabled_engages_after_threshold_and_repeats() -> None:
    env = make_env(idle_command_watchdog_s=THRESHOLD)
    env._is_flying = True

    env.step(STEP_ACTION)
    last_command_time = env.commands()[-1][0]
    time.sleep(0.4)

    assert env._idle_watchdog_thread is not None, 'thread must start lazily on step'
    zeros = env.zero_commands()
    assert len(zeros) >= 2, f'expected repeated zero publishes, got {len(zeros)}'
    first_zero_delay = zeros[0][0] - last_command_time
    assert first_zero_delay >= THRESHOLD - 0.005, (
        f'watchdog fired {first_zero_delay:.3f}s after last command, '
        f'before the {THRESHOLD}s threshold'
    )
    assert first_zero_delay <= THRESHOLD + 0.2, (
        f'watchdog took {first_zero_delay:.3f}s to engage (threshold {THRESHOLD}s)'
    )
    env.close()


def test_activity_resets_idle_clock() -> None:
    env = make_env(idle_command_watchdog_s=THRESHOLD)
    env._is_flying = True

    for _ in range(6):
        env.step(STEP_ACTION)
        time.sleep(0.05)
    window_end = time.monotonic()

    zeros_in_window = [z for z in env.zero_commands() if z[0] <= window_end]
    assert zeros_in_window == [], (
        f'stepping every 0.05s must keep the watchdog quiet, got {len(zeros_in_window)} zeros'
    )
    env.close()


def test_watchdog_own_publishes_do_not_stop_republishing() -> None:
    env = make_env(idle_command_watchdog_s=THRESHOLD)
    env._is_flying = True

    env.step(STEP_ACTION)
    time.sleep(THRESHOLD + 0.3)

    zeros = env.zero_commands()
    assert len(zeros) >= 3, (
        f'watchdog must keep republishing zeros while idle persists, got {len(zeros)}'
    )
    spread = zeros[-1][0] - zeros[0][0]
    assert spread >= 0.1, (
        f'zero publishes must be spread over the idle window, spread {spread:.3f}s'
    )
    env.close()


def test_not_flying_no_publishes() -> None:
    env = make_env(idle_command_watchdog_s=THRESHOLD)
    env._is_flying = False

    env.step(STEP_ACTION)
    time.sleep(0.4)

    assert env.zero_commands() == [], 'watchdog must stay quiet while not flying'
    env.close()


def test_close_stops_thread_and_is_idempotent() -> None:
    env = make_env(idle_command_watchdog_s=THRESHOLD)
    env._is_flying = True

    env.step(STEP_ACTION)
    time.sleep(0.3)
    assert len(env.zero_commands()) >= 1, 'watchdog must be engaged before close'

    thread = env._idle_watchdog_thread
    env.close()
    assert thread is not None and not thread.is_alive(), 'close must join the thread'

    time.sleep(0.05)  # let any in-flight publish land
    count_after_close = len(env.commands())
    time.sleep(THRESHOLD + 0.2)
    assert len(env.commands()) == count_after_close, (
        'no publishes may happen after close'
    )

    env.close()  # idempotent close must not raise


def test_validation_rejects_bad_watchdog_parameters() -> None:
    try:
        make_env(idle_command_watchdog_s=-0.1)
    except ValueError as exc:
        assert 'idle_command_watchdog_s' in str(exc)
    else:
        raise AssertionError('negative idle_command_watchdog_s must raise')

    # The default harness uses step_duration=0.0, so any positive value passes
    # the '> step_duration' rule trivially; construct this case explicitly.
    try:
        make_env(step_duration=0.2, idle_command_watchdog_s=0.1)
    except ValueError as exc:
        assert 'idle_command_watchdog_s' in str(exc)
    else:
        raise AssertionError('enabled watchdog <= step_duration must raise')


def main() -> int:
    test_default_disabled_no_thread_no_publishes()
    test_enabled_engages_after_threshold_and_repeats()
    test_activity_resets_idle_clock()
    test_watchdog_own_publishes_do_not_stop_republishing()
    test_not_flying_no_publishes()
    test_close_stops_thread_and_is_idempotent()
    test_validation_rejects_bad_watchdog_parameters()
    print('✓ PASS: idle-command hover watchdog behaves per spec')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
