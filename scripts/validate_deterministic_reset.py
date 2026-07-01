#!/usr/bin/env python3
"""Validate deterministic simulator-backed reset behavior.

This script is intentionally RED before the reset service exists: it fails
closed when the generated service type or the namespaced service endpoint is
unavailable. Once the simulator service is implemented, the same checks become
the Exp008 reset gate.
"""

__authors__ = 'Jordi'
__license__ = 'BSD-3-Clause'

import argparse
import json
import math
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_RESET_COUNT = 20
DEFAULT_POSITION_TOLERANCE = 0.15
DEFAULT_YAW_TOLERANCE = 0.2
DEFAULT_SPEED_TOLERANCE = 0.05
DEFAULT_SERVICE_TIMEOUT = 2.0
DEFAULT_STATE_TIMEOUT = 5.0
DEFAULT_POST_RESET_HOLD_TIME = 0.5
DEFAULT_POS_LIMIT = 5.0


class ResetAttemptResult:
    """One deterministic reset attempt and its observable post-reset state."""

    def __init__(
        self,
        *,
        namespace: str,
        iteration: int,
        service_success: bool,
        position_error: float = math.inf,
        yaw_error: float = math.inf,
        linear_speed_norm: float = math.inf,
        angular_speed_norm: float = math.inf,
        observation: Sequence[float] | None = None,
        other_namespace_poses: Mapping[str, Sequence[float]] | None = None,
        message: str = '',
    ) -> None:
        self.namespace = namespace
        self.iteration = int(iteration)
        self.service_success = bool(service_success)
        self.position_error = float(position_error)
        self.yaw_error = float(yaw_error)
        self.linear_speed_norm = float(linear_speed_norm)
        self.angular_speed_norm = float(angular_speed_norm)
        self.observation = list(observation) if observation is not None else []
        self.other_namespace_poses = {
            str(namespace): [float(value) for value in pose]
            for namespace, pose in (other_namespace_poses or {}).items()
        }
        self.message = str(message)

    def with_overrides(self, **overrides: Any) -> 'ResetAttemptResult':
        """Return a copy with selected fields replaced."""
        values = {
            'namespace': self.namespace,
            'iteration': self.iteration,
            'service_success': self.service_success,
            'position_error': self.position_error,
            'yaw_error': self.yaw_error,
            'linear_speed_norm': self.linear_speed_norm,
            'angular_speed_norm': self.angular_speed_norm,
            'observation': list(self.observation),
            'other_namespace_poses': dict(self.other_namespace_poses),
            'message': self.message,
        }
        values.update(overrides)
        return ResetAttemptResult(**values)


class ResetEvaluation:
    """Final reset gate result."""

    def __init__(self, success: bool, message: str) -> None:
        self.success = bool(success)
        self.message = str(message)


def reset_service_name(namespace: str) -> str:
    """Return the absolute per-namespace simulator reset service path."""
    cleaned = namespace.strip().strip('/')
    if not cleaned:
        raise ValueError('namespace must not be empty')
    return f'/{cleaned}/platform/reset_simulator_state'


def build_reset_request(
    service_type: Any,
    *,
    target_pose: Sequence[float],
    position_tolerance: float,
    yaw_tolerance: float,
    speed_tolerance: float,
) -> Any:
    """Build a ResetSimulatorState request from target pose and tolerances."""
    if len(target_pose) != 4:
        raise ValueError('target_pose must contain [x, y, z, yaw]')

    request = service_type.Request()
    request.x = float(target_pose[0])
    request.y = float(target_pose[1])
    request.z = float(target_pose[2])
    request.yaw = float(target_pose[3])
    request.position_tolerance = float(position_tolerance)
    request.yaw_tolerance = float(yaw_tolerance)
    request.linear_speed_tolerance = float(speed_tolerance)
    request.angular_speed_tolerance = float(speed_tolerance)
    return request


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def normalized_observation_from_state(
    *,
    position: Sequence[float],
    yaw: float,
    target_pose: Sequence[float],
    pos_limit: float = DEFAULT_POS_LIMIT,
) -> list[float]:
    """Build the validator observation using the env's clipped [-1, 1] contract."""
    if len(position) < 3:
        raise ValueError('position must contain at least [x, y, z]')
    if len(target_pose) != 4:
        raise ValueError('target_pose must contain [x, y, z, yaw]')
    if pos_limit <= 0.0:
        raise ValueError('pos_limit must be positive')

    values = [
        (float(position[0]) - float(target_pose[0])) / float(pos_limit),
        (float(position[1]) - float(target_pose[1])) / float(pos_limit),
        (float(position[2]) - float(target_pose[2])) / float(pos_limit),
        _wrap_angle(float(yaw) - float(target_pose[3])) / math.pi,
    ]
    return [min(1.0, max(-1.0, float(value))) for value in values]


def observation_is_normalized(observation: Sequence[float]) -> bool:
    """Return True when every observation value is finite and inside [-1, 1]."""
    if len(observation) == 0:
        return False
    return all(math.isfinite(float(value)) and -1.0 <= float(value) <= 1.0 for value in observation)


def _pose_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 3 or len(b) < 3:
        return math.inf
    return math.sqrt(sum((float(a[idx]) - float(b[idx])) ** 2 for idx in range(3)))


def _quaternion_yaw(q: Any) -> float:
    """Return yaw from a ROS geometry quaternion."""
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


def _pose_snapshot(pose_msg: Any, twist_msg: Any) -> dict[str, Any]:
    position = [
        float(pose_msg.pose.position.x),
        float(pose_msg.pose.position.y),
        float(pose_msg.pose.position.z),
    ]
    yaw = _quaternion_yaw(pose_msg.pose.orientation)
    linear_speed_norm = math.sqrt(
        float(twist_msg.twist.linear.x) ** 2
        + float(twist_msg.twist.linear.y) ** 2
        + float(twist_msg.twist.linear.z) ** 2
    )
    angular_speed_norm = math.sqrt(
        float(twist_msg.twist.angular.x) ** 2
        + float(twist_msg.twist.angular.y) ** 2
        + float(twist_msg.twist.angular.z) ** 2
    )
    return {
        'pose': [position[0], position[1], position[2], yaw],
        'position': position,
        'yaw': yaw,
        'linear_speed_norm': linear_speed_norm,
        'angular_speed_norm': angular_speed_norm,
    }


def clear_latest_namespace_state(
    latest_pose: dict[str, Any],
    latest_twist: dict[str, Any],
    namespace: str,
) -> None:
    """Discard cached state so the next sample must come from fresh ROS messages."""
    latest_pose.pop(namespace, None)
    latest_twist.pop(namespace, None)


def snapshot_matches_target(
    snapshot: Mapping[str, Any],
    target_pose: Sequence[float],
    *,
    position_tolerance: float,
    yaw_tolerance: float,
) -> bool:
    """Return True when a live snapshot is already at the requested pose/yaw."""
    position = snapshot.get('position', [])
    yaw = float(snapshot.get('yaw', math.inf))
    return (
        _pose_distance(position, target_pose[:3]) <= float(position_tolerance)
        and abs(_wrap_angle(yaw - float(target_pose[3]))) <= float(yaw_tolerance)
    )


def verify_overlay_prefix(expected_prefix: str) -> str:
    """Fail closed if ros2 resolves the simulator outside the expected overlay."""
    result = subprocess.run(
        ['ros2', 'pkg', 'prefix', 'as2_platform_multirotor_simulator'],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    actual_prefix = result.stdout.strip()
    if result.returncode != 0 or not actual_prefix.startswith(expected_prefix):
        raise RuntimeError(
            'as2_platform_multirotor_simulator resolves outside expected overlay: '
            f'expected under {expected_prefix}, actual {actual_prefix or result.stderr.strip() or "<not found>"}'
        )
    return actual_prefix


def _required_num_drones(namespaces: Sequence[str]) -> int:
    max_index = 0
    for namespace in namespaces:
        if namespace.startswith('drone') and namespace[5:].isdigit():
            max_index = max(max_index, int(namespace[5:]))
    return max(1, max_index + 1)


class LiveResetRunner:
    """Bounded ROS2 client/subscription runner for live reset validation."""

    def __init__(
        self,
        *,
        service_type: Any,
        target_namespace: str,
        other_namespaces: Sequence[str],
        target_pose: Sequence[float],
        position_tolerance: float,
        yaw_tolerance: float,
        speed_tolerance: float,
        service_timeout: float,
        state_timeout: float,
        post_reset_hold_time: float,
        pos_limit: float,
    ) -> None:
        import rclpy
        from geometry_msgs.msg import PoseStamped, TwistStamped
        from rclpy.qos import qos_profile_sensor_data

        if not rclpy.ok():
            rclpy.init(args=None)

        self._rclpy = rclpy
        self._service_type = service_type
        self._target_namespace = target_namespace
        self._other_namespaces = list(other_namespaces)
        self._namespaces = [target_namespace] + self._other_namespaces
        self._target_pose = [float(value) for value in target_pose]
        self._position_tolerance = float(position_tolerance)
        self._yaw_tolerance = float(yaw_tolerance)
        self._speed_tolerance = float(speed_tolerance)
        self._service_timeout = float(service_timeout)
        self._state_timeout = float(state_timeout)
        self._post_reset_hold_time = float(post_reset_hold_time)
        self._pos_limit = float(pos_limit)
        self._latest_pose: dict[str, Any] = {}
        self._latest_twist: dict[str, Any] = {}

        node_name = f'{target_namespace.strip().strip("/").replace("/", "_")}_deterministic_reset_validator'
        self._node = rclpy.create_node(node_name)
        self._subscriptions = []
        for namespace in self._namespaces:
            self._subscriptions.append(
                self._node.create_subscription(
                    PoseStamped,
                    f'/{namespace}/ground_truth/pose',
                    lambda msg, ns=namespace: self._latest_pose.__setitem__(ns, msg),
                    qos_profile_sensor_data,
                )
            )
            self._subscriptions.append(
                self._node.create_subscription(
                    TwistStamped,
                    f'/{namespace}/ground_truth/twist',
                    lambda msg, ns=namespace: self._latest_twist.__setitem__(ns, msg),
                    qos_profile_sensor_data,
                )
            )

        self._client = self._node.create_client(
            service_type,
            reset_service_name(target_namespace),
        )

    def close(self) -> None:
        try:
            self._node.destroy_node()
        finally:
            if self._rclpy.ok():
                self._rclpy.shutdown()

    def _spin_until(self, predicate, timeout: float) -> bool:
        deadline = time.time() + max(0.0, float(timeout))
        while time.time() <= deadline:
            if predicate():
                return True
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
        return bool(predicate())

    def _snapshot(self, namespace: str, timeout: float | None = None) -> dict[str, Any] | None:
        wait_timeout = self._state_timeout if timeout is None else float(timeout)
        ready = self._spin_until(
            lambda: namespace in self._latest_pose and namespace in self._latest_twist,
            wait_timeout,
        )
        if not ready:
            return None
        return _pose_snapshot(self._latest_pose[namespace], self._latest_twist[namespace])

    def _other_namespace_poses(self) -> dict[str, list[float]]:
        poses: dict[str, list[float]] = {}
        for namespace in self._other_namespaces:
            snapshot = self._snapshot(namespace, timeout=self._state_timeout)
            if snapshot is not None:
                poses[namespace] = list(snapshot['pose'])
        return poses

    def _wait_for_post_reset_hold(self) -> dict[str, Any] | None:
        snapshot = None
        acquisition_deadline = time.time() + self._state_timeout
        while time.time() <= acquisition_deadline:
            candidate = self._snapshot(self._target_namespace, timeout=0.05)
            if candidate is not None:
                snapshot = candidate
                if snapshot_matches_target(
                    candidate,
                    self._target_pose,
                    position_tolerance=self._position_tolerance,
                    yaw_tolerance=self._yaw_tolerance,
                ):
                    break

        hold_deadline = time.time() + max(0.0, self._post_reset_hold_time)
        while time.time() < hold_deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
            latest = self._snapshot(self._target_namespace, timeout=0.0)
            if latest is not None:
                snapshot = latest
        return snapshot

    def collect(self, reset_count: int) -> list[ResetAttemptResult]:
        attempts: list[ResetAttemptResult] = []

        if not self._client.wait_for_service(timeout_sec=self._service_timeout):
            return collect_service_absence_attempt(
                self._target_namespace,
                f'service unavailable: {reset_service_name(self._target_namespace)}',
            )

        for iteration in range(1, int(reset_count) + 1):
            request = build_reset_request(
                self._service_type,
                target_pose=self._target_pose,
                position_tolerance=self._position_tolerance,
                yaw_tolerance=self._yaw_tolerance,
                speed_tolerance=self._speed_tolerance,
            )
            future = self._client.call_async(request)
            completed = self._spin_until(lambda: future.done(), self._service_timeout)
            if not completed:
                attempts.append(
                    ResetAttemptResult(
                        namespace=self._target_namespace,
                        iteration=iteration,
                        service_success=False,
                        observation=[0.0, 0.0, 0.0, 0.0],
                        other_namespace_poses=self._other_namespace_poses(),
                        message='service future timeout',
                    )
                )
                continue

            response = future.result()
            clear_latest_namespace_state(
                self._latest_pose,
                self._latest_twist,
                self._target_namespace,
            )
            snapshot = self._wait_for_post_reset_hold()
            if snapshot is None:
                attempts.append(
                    ResetAttemptResult(
                        namespace=self._target_namespace,
                        iteration=iteration,
                        service_success=False,
                        observation=[0.0, 0.0, 0.0, 0.0],
                        other_namespace_poses=self._other_namespace_poses(),
                        message='ground truth state timeout after reset',
                    )
                )
                continue

            measured_position_error = _pose_distance(snapshot['position'], self._target_pose[:3])
            measured_yaw_error = abs(_wrap_angle(snapshot['yaw'] - self._target_pose[3]))
            position_error = max(float(getattr(response, 'position_error', math.inf)), measured_position_error)
            yaw_error = max(float(getattr(response, 'yaw_error', math.inf)), measured_yaw_error)
            linear_speed_norm = max(
                float(getattr(response, 'linear_speed_norm', math.inf)),
                float(snapshot['linear_speed_norm']),
            )
            angular_speed_norm = max(
                float(getattr(response, 'angular_speed_norm', math.inf)),
                float(snapshot['angular_speed_norm']),
            )
            attempts.append(
                ResetAttemptResult(
                    namespace=self._target_namespace,
                    iteration=iteration,
                    service_success=bool(getattr(response, 'success', False)),
                    position_error=position_error,
                    yaw_error=yaw_error,
                    linear_speed_norm=linear_speed_norm,
                    angular_speed_norm=angular_speed_norm,
                    observation=normalized_observation_from_state(
                        position=snapshot['position'],
                        yaw=snapshot['yaw'],
                        target_pose=self._target_pose,
                        pos_limit=self._pos_limit,
                    ),
                    other_namespace_poses=self._other_namespace_poses(),
                    message=str(getattr(response, 'message', '')),
                )
            )

        return attempts


def evaluate_reset_attempts(
    attempts: Sequence[ResetAttemptResult],
    *,
    expected_reset_count: int = DEFAULT_RESET_COUNT,
    namespaces: Sequence[str] | None = None,
    target_namespace: str,
    position_tolerance: float = DEFAULT_POSITION_TOLERANCE,
    yaw_tolerance: float = DEFAULT_YAW_TOLERANCE,
    linear_speed_tolerance: float = DEFAULT_SPEED_TOLERANCE,
    angular_speed_tolerance: float = DEFAULT_SPEED_TOLERANCE,
    namespace_tolerance: float = 1e-6,
) -> ResetEvaluation:
    """Evaluate deterministic reset attempts against the SDD acceptance gate."""
    if len(attempts) != expected_reset_count:
        return ResetEvaluation(False, f'expected {expected_reset_count} reset attempts, got {len(attempts)}')

    expected_other_namespaces = [
        namespace for namespace in (namespaces or [])
        if namespace != target_namespace
    ]
    first_other_poses: dict[str, list[float]] = {}

    for attempt in attempts:
        if attempt.namespace != target_namespace:
            return ResetEvaluation(False, f'attempt {attempt.iteration}: unexpected namespace {attempt.namespace}')
        if not attempt.service_success:
            detail = attempt.message or 'service failed'
            return ResetEvaluation(False, f'attempt {attempt.iteration}: {detail}')
        if attempt.position_error > position_tolerance:
            return ResetEvaluation(False, f'attempt {attempt.iteration}: position error {attempt.position_error:.4f} exceeds tolerance')
        if attempt.yaw_error > yaw_tolerance:
            return ResetEvaluation(False, f'attempt {attempt.iteration}: yaw error {attempt.yaw_error:.4f} exceeds tolerance')
        if attempt.linear_speed_norm > linear_speed_tolerance:
            return ResetEvaluation(False, f'attempt {attempt.iteration}: linear speed {attempt.linear_speed_norm:.4f} exceeds tolerance')
        if attempt.angular_speed_norm > angular_speed_tolerance:
            return ResetEvaluation(False, f'attempt {attempt.iteration}: angular speed {attempt.angular_speed_norm:.4f} exceeds tolerance')
        if not observation_is_normalized(attempt.observation):
            return ResetEvaluation(False, f'attempt {attempt.iteration}: observation outside [-1, 1]')

        for other_namespace in expected_other_namespaces:
            if other_namespace not in attempt.other_namespace_poses:
                return ResetEvaluation(False, f'attempt {attempt.iteration}: missing isolation pose for {other_namespace}')
            current_pose = list(attempt.other_namespace_poses[other_namespace])
            if other_namespace not in first_other_poses:
                first_other_poses[other_namespace] = current_pose
                continue
            if _pose_distance(first_other_poses[other_namespace], current_pose) > namespace_tolerance:
                return ResetEvaluation(
                    False,
                    f'attempt {attempt.iteration}: namespace isolation changed {other_namespace}',
                )

    return ResetEvaluation(True, f'{expected_reset_count} deterministic resets passed')


def _load_reset_service_type() -> Any:
    try:
        from as2_platform_multirotor_simulator.srv import ResetSimulatorState
    except Exception as exc:  # pragma: no cover - exercised by CLI RED gate.
        raise RuntimeError(
            'ResetSimulatorState service type is unavailable; build/source the overlay after adding the service'
        ) from exc
    return ResetSimulatorState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', default='drone0', help='Target namespace to reset')
    parser.add_argument('--other-namespace', action='append', default=[], help='Namespace that must not move')
    parser.add_argument('--reset-count', type=int, default=DEFAULT_RESET_COUNT)
    parser.add_argument('--x', type=float, default=0.0)
    parser.add_argument('--y', type=float, default=0.0)
    parser.add_argument('--z', type=float, default=1.0)
    parser.add_argument('--yaw', type=float, default=0.0)
    parser.add_argument('--position-tolerance', type=float, default=DEFAULT_POSITION_TOLERANCE)
    parser.add_argument('--yaw-tolerance', type=float, default=DEFAULT_YAW_TOLERANCE)
    parser.add_argument('--speed-tolerance', type=float, default=DEFAULT_SPEED_TOLERANCE)
    parser.add_argument('--service-timeout', type=float, default=DEFAULT_SERVICE_TIMEOUT)
    parser.add_argument('--state-timeout', type=float, default=DEFAULT_STATE_TIMEOUT)
    parser.add_argument('--post-reset-hold-time', type=float, default=DEFAULT_POST_RESET_HOLD_TIME)
    parser.add_argument('--pos-limit', type=float, default=DEFAULT_POS_LIMIT)
    parser.add_argument('--skip-ensure-as2', action='store_true', default=False)
    parser.add_argument('--force-relaunch', action='store_true', default=False)
    parser.add_argument('--launch-wait', type=float, default=8.0)
    parser.add_argument('--readiness-timeout', type=float, default=30.0)
    parser.add_argument('--expected-simulator-prefix', default='/home/jordi/as2_rl_ws')
    return parser


def collect_service_absence_attempt(namespace: str, message: str) -> list[ResetAttemptResult]:
    """Create a single failed attempt for unavailable service/type RED output."""
    return [
        ResetAttemptResult(
            namespace=namespace,
            iteration=1,
            service_success=False,
            observation=[0.0, 0.0, 0.0, 0.0],
            other_namespace_poses={},
            message=message,
        )
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    namespaces = [args.namespace] + list(args.other_namespace)

    try:
        simulator_prefix = verify_overlay_prefix(args.expected_simulator_prefix)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({'status': 'failure', 'failure_class': 'overlay_prefix', 'reason': str(exc)}, indent=2))
        return 1

    if not args.skip_ensure_as2:
        from as2_runtime import ensure_as2_simulator

        try:
            ensure_as2_simulator(
                num_drones=_required_num_drones(namespaces),
                launch_script=Path(__file__).resolve().parents[1] / 'as2_sim' / 'launch_sim.bash',
                launch_wait_s=args.launch_wait,
                readiness_timeout_s=args.readiness_timeout,
                force_relaunch=args.force_relaunch,
            )
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({'status': 'failure', 'failure_class': 'as2_readiness', 'reason': str(exc)}, indent=2))
            return 1

    try:
        service_type = _load_reset_service_type()
    except RuntimeError as exc:
        print(str(exc))
        return 1

    target_pose = [args.x, args.y, args.z, args.yaw]
    runner = LiveResetRunner(
        service_type=service_type,
        target_namespace=args.namespace,
        other_namespaces=args.other_namespace,
        target_pose=target_pose,
        position_tolerance=args.position_tolerance,
        yaw_tolerance=args.yaw_tolerance,
        speed_tolerance=args.speed_tolerance,
        service_timeout=args.service_timeout,
        state_timeout=args.state_timeout,
        post_reset_hold_time=args.post_reset_hold_time,
        pos_limit=args.pos_limit,
    )
    try:
        attempts = runner.collect(args.reset_count)
    finally:
        runner.close()

    evaluation = evaluate_reset_attempts(
        attempts,
        expected_reset_count=args.reset_count,
        namespaces=namespaces,
        target_namespace=args.namespace,
        position_tolerance=args.position_tolerance,
        yaw_tolerance=args.yaw_tolerance,
        linear_speed_tolerance=args.speed_tolerance,
        angular_speed_tolerance=args.speed_tolerance,
    )
    print(json.dumps({
        'status': 'success' if evaluation.success else 'failure',
        'message': evaluation.message,
        'simulator_prefix': simulator_prefix,
        'service_name': reset_service_name(args.namespace),
        'target_pose': target_pose,
        'reset_count': len(attempts),
        'other_namespaces': list(args.other_namespace),
        'attempts': [
            {
                'iteration': attempt.iteration,
                'service_success': attempt.service_success,
                'position_error': attempt.position_error,
                'yaw_error': attempt.yaw_error,
                'linear_speed_norm': attempt.linear_speed_norm,
                'angular_speed_norm': attempt.angular_speed_norm,
                'observation': attempt.observation,
                'message': attempt.message,
                'other_namespace_poses': attempt.other_namespace_poses,
            }
            for attempt in attempts
        ],
    }, indent=2))
    return 0 if evaluation.success else 1


if __name__ == '__main__':
    sys.exit(main())
