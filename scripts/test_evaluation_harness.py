#!/usr/bin/env python3
"""ROS-free checks for the policy evaluation harness.

The harness runs seeded, exactly-reproducible episodes through a pluggable
controller so the trained policy and the future PID baseline share one
protocol and one set of metrics.

Usage:
    conda run -n rl_uav python3 scripts/test_evaluation_harness.py
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_uav.envs.as2_test_env import AS2TestEnv
from rl_uav.evaluation import (
    Controller,
    EpisodeRecord,
    PPOController,
    run_evaluation,
    summarize,
    write_outputs,
)

logging.getLogger('rl_uav.envs.as2_test_env').setLevel(logging.CRITICAL)

TMP_DIR = Path('/home/jordi/.claude/jobs/0418d929/tmp/eval_harness_test')


class DummyDrone:
    def __init__(self):
        self.position = [0.0, 0.0, 1.5]
        self.speed = [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, 0.0]

    def arm(self, **kw) -> bool:
        return True

    def offboard(self, **kw) -> bool:
        return True

    def takeoff(self, **kw) -> bool:
        return True

    def land(self, **kw) -> bool:
        return True

    def shutdown(self, **kw) -> None:
        pass


class EvalTestEnv(AS2TestEnv):
    """ROS-free env: service resets teleport the dummy drone to start_pose."""

    def _init_ros(self) -> None:
        if self._drone is None:
            self._drone = DummyDrone()

    def _send_speed_command(self, twist, yaw_speed):  # type: ignore[no-untyped-def]
        return True

    def _send_terminal_stop_command(self) -> bool:
        return True

    def _publish_target_marker(self) -> None:
        return None

    def _try_service_backed_reset(self, start_pose):  # type: ignore[no-untyped-def]
        self._drone.position = [float(v) for v in start_pose[:3]]
        self._drone.orientation = [0.0, 0.0, float(start_pose[3])]
        self._drone.speed = [0.0, 0.0, 0.0]
        self._is_flying = True
        self._last_reset_diagnostics = {'reason': 'service_success', 'failure_class': ''}
        self._last_reset_service_status = 'service_success'
        return True

    def _prepare_in_air_velocity_reset(self) -> None:
        return None

    def _wait_for_hover_settle(self) -> bool:
        return True

    def _apply_start_pose_with_optional_service_fallback(self, start_pose) -> bool:  # type: ignore[no-untyped-def]
        return self._try_service_backed_reset(start_pose)


def make_env(**overrides) -> EvalTestEnv:
    kwargs = {
        'verbose': False,
        'step_duration': 0.0,
        'max_vel': 0.5,
        'pos_limit': 5.0,
        'scene_bounds_xy': 5.0,
        'height_bounds': (0.3, 2.5),
        'unsafe_low_altitude_threshold': 0.5,
        'low_altitude_guard_margin': 0.3,
        'target_pose': [2.0, 0.0, 1.5, 0.0],
        'distance_threshold': 0.4,
        'max_steps': 20,
        'randomize_hover_start': True,
        'randomization_bounds_margin': 1.5,
        'randomization_bounds_margin_z': 0.5,
        'min_start_target_distance': 1.0,
        'max_start_target_distance': 5.0,
    }
    kwargs.update(overrides)
    return EvalTestEnv(**kwargs)


class ScriptedController(Controller):
    """Teleports the drone per episode script: 'success', 'oob' or 'idle'."""

    def __init__(self, env: EvalTestEnv, script: list[str]):
        self._env = env
        self._script = list(script)
        self._episode = -1
        self._step = 0

    @property
    def name(self) -> str:
        return 'scripted'

    def reset_episode(self) -> None:
        self._episode += 1
        self._step = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        mode = self._script[self._episode % len(self._script)]
        self._step += 1
        if mode == 'success' and self._step >= 3:
            tx, ty, tz, _ = self._env._target_pose
            self._env._drone.position = [tx + 0.1, ty, tz]
        elif mode == 'oob' and self._step >= 2:
            self._env._drone.position = [6.5, 0.0, 1.5]
        return np.zeros(4, dtype=np.float32)


def run_scripted(base_seed: int, script: list[str]):
    env = make_env()
    controller = ScriptedController(env, script)
    records = run_evaluation(env, controller, episodes=len(script), base_seed=base_seed)
    return records


def test_records_capture_terminals_and_metrics() -> None:
    records = run_scripted(500, ['success', 'oob', 'idle'])
    assert len(records) == 3

    success, oob, idle = records
    assert success.success is True and success.terminal_reason == 'success'
    assert success.steps == 3
    assert success.final_distance < 0.4
    assert oob.success is False and oob.terminal_reason == 'out_of_bounds'
    assert idle.success is False and idle.terminal_reason == 'max_steps'
    assert idle.steps == 20

    for record in records:
        expected = math.dist(record.start_pose[:3], record.target_pose[:3])
        assert abs(record.start_target_distance - expected) < 1e-6
        assert 1.0 < record.start_target_distance < 5.0
        assert record.seed in (500, 501, 502)


def test_seeding_is_reproducible_across_runs() -> None:
    first = run_scripted(900, ['idle', 'idle'])
    second = run_scripted(900, ['idle', 'idle'])
    for a, b in zip(first, second):
        assert a.start_pose == b.start_pose
        assert a.target_pose == b.target_pose

    different = run_scripted(901, ['idle', 'idle'])
    assert any(
        a.start_pose != b.start_pose
        for a, b in zip(first, different)
    )


def test_path_efficiency_uses_straight_line_over_path_length() -> None:
    records = run_scripted(700, ['success'])
    record = records[0]
    assert record.path_length > 0
    expected = record.start_target_distance / record.path_length
    assert abs(record.path_efficiency - expected) < 1e-6

    stub = EpisodeRecord(
        episode_index=0, seed=0, success=False, terminal_reason='max_steps',
        steps=1, sim_time_s=0.2, start_pose=[0, 0, 1, 0], target_pose=[1, 0, 1, 0],
        start_target_distance=1.0, final_distance=1.0, path_length=0.0,
        path_efficiency=None, terminal_yaw_error=0.0, mean_action_norm=0.0,
        max_action_norm=0.0, motion_command_acceptance_rate=1.0, min_altitude=1.0,
    )
    assert stub.path_efficiency is None


def _stub_record(idx: int, distance: float, success: bool) -> EpisodeRecord:
    return EpisodeRecord(
        episode_index=idx, seed=idx, success=success,
        terminal_reason='success' if success else 'max_steps',
        steps=10, sim_time_s=2.0, start_pose=[0, 0, 1, 0],
        target_pose=[distance, 0, 1, 0], start_target_distance=distance,
        final_distance=0.3 if success else 2.0, path_length=distance * 1.25,
        path_efficiency=0.8, terminal_yaw_error=0.1, mean_action_norm=0.5,
        max_action_norm=0.9, motion_command_acceptance_rate=1.0, min_altitude=1.0,
    )


def test_summarize_aggregates_and_bins() -> None:
    records = [
        _stub_record(0, 1.5, True),
        _stub_record(1, 3.0, False),
        _stub_record(2, 4.0, True),
    ]
    summary = summarize(records)
    assert summary['episodes'] == 3
    # summarize rounds to 4 decimals for JSON output.
    assert abs(summary['success_rate'] - 2 / 3) < 1e-3
    assert summary['terminal_reasons'] == {'success': 2, 'max_steps': 1}
    bins = summary['distance_bins']
    assert bins['1.0-2.5']['n'] == 1 and bins['1.0-2.5']['success_rate'] == 1.0
    assert bins['2.5-3.5']['n'] == 1 and bins['2.5-3.5']['success_rate'] == 0.0
    assert bins['3.5-5.0']['n'] == 1
    json.dumps(summary)


def test_write_outputs_produces_csv_json_metadata() -> None:
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    records = [_stub_record(i, 2.0, True) for i in range(4)]
    summary = summarize(records)
    metadata = {'model': 'fake.zip', 'episodes': 4, 'timestamp': '2026-07-06T00:00:00'}
    write_outputs(records, summary, TMP_DIR, metadata)

    csv_lines = (TMP_DIR / 'episodes.csv').read_text().strip().splitlines()
    assert len(csv_lines) == 1 + len(records)
    loaded_summary = json.loads((TMP_DIR / 'summary.json').read_text())
    assert loaded_summary['episodes'] == 4
    loaded_meta = json.loads((TMP_DIR / 'metadata.json').read_text())
    assert loaded_meta == metadata


def test_ppo_controller_wraps_predict() -> None:
    class FakeModel:
        def predict(self, obs, deterministic=False):
            assert deterministic is True
            return np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), None

    controller = PPOController(FakeModel(), name='fake-ppo')
    controller.reset_episode()
    action = controller.act(np.zeros(4, dtype=np.float32))
    assert np.allclose(action, [0.1, 0.2, 0.3, 0.4])
    assert controller.name == 'fake-ppo'


def test_cli_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate_policy.py'), '--help'],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert '--episodes' in result.stdout


def main() -> int:
    test_records_capture_terminals_and_metrics()
    test_seeding_is_reproducible_across_runs()
    test_path_efficiency_uses_straight_line_over_path_length()
    test_summarize_aggregates_and_bins()
    test_write_outputs_produces_csv_json_metadata()
    test_ppo_controller_wraps_predict()
    test_cli_help_exits_zero()
    print('✓ PASS: evaluation harness protocol, metrics and outputs behave per spec')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
