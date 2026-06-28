"""
AS2TestEnv — Gymnasium environment for the Aerostack2 multirotor simulator.

Observations (normalized to [-1, 1]):
    - Relative position (dx, dy, dz) — (drone - target) / pos_limit
    - Relative yaw (dyaw) — angular difference wrapped to [-π, π], then / π

Actions:
    - Linear velocity commands (vx, vy, vz) in m/s, bounded to [-max_vel, max_vel]
    - Angular yaw velocity (vyaw) in rad/s, bounded to [-max_yaw_vel, max_yaw_vel]

Reward:
    - Continuous distance penalty: reward_distance = -d_normalized (each step)
    - Continuous path-facing term: path_facing_weight * cos(yaw - path_yaw)
      based on measured horizontal velocity direction
    - Low-speed deadband neutrality: if speed_xy <= speed_deadband,
      path-facing contribution is 0.0
    - Success terminal bonus: +success_reward (default 20.0) when d < distance_threshold
    - Out-of-bounds penalty: -oob_penalty (default 10.0) when exceeding pos_limit

Termination:
    - terminated=True: success (d < threshold) or out-of-bounds
    - truncated=True: step count exceeds max_steps
"""

__authors__ = 'Jordi'
__license__ = 'BSD-3-Clause'

import logging
import math
import threading
import time
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


class AS2TestEnv(gym.Env):
    """
    Minimal Gymnasium environment for testing connectivity
    with the AS2 multirotor simulator.

    The environment interfaces with the simulator through
    as2_python_api.DroneInterface, which communicates via ROS2 topics.

    Supports vectorization via gymnasium.make_vec() or SyncVectorEnv.
    Each instance should receive a unique drone_namespace (e.g. drone0,
    drone1, ...) when used in a vectorized setup.
    """

    metadata = {'render_modes': []}

    # Class-level ROS2 init guard — rclpy.init() must be called exactly
    # once per process, even when SyncVectorEnv creates multiple instances.
    _rclpy_initialized = False
    _rclpy_lock = threading.Lock()

    def __init__(
        self,
        drone_namespace: str = 'drone0',
        use_sim_time: bool = False,
        verbose: bool = True,
        takeoff_height: float = 1.0,
        takeoff_speed: float = 0.5,
        land_speed: float = 0.5,
        max_vel: float = 2.0,
        max_yaw_vel: float = math.pi,
        pos_limit: float = 5.0,
        step_duration: float = 0.1,
        target_pose: list[float] | None = None,
        distance_threshold: float = 0.5,
        max_steps: int = 500,
        success_reward: float = 20.0,
        oob_penalty: float = 10.0,
        path_facing_weight: float = 0.25,
        progress_reward_weight: float = 0.0,
        speed_deadband: float = 0.05,
        fixed_start_pose: list[float] | None = None,
        fixed_start_tolerance: float = 0.15,
        fixed_start_timeout: float = 20.0,
        reset_min_speed: float = 0.15,
        reset_ground_recovery_height: float = 0.35,
        publish_target_marker: bool = False,
        target_marker_topic: str = 'visualization_marker',
        target_marker_frame_id: str = 'earth',
        target_marker_scale: float = 0.35,
        close_operation_timeout: float = 10.0,
        randomize_hover_start: bool = False,
        scene_bounds_xy: float = 5.0,
        height_bounds: tuple[float, float] = (0.1, 2.0),
        min_start_target_distance: float | None = None,
        hover_speed_threshold: float = 0.05,
        hover_settle_time: float = 1.0,
        hover_timeout: float = 10.0,
        max_reset_sample_attempts: int = 100,
        randomize_yaw: bool = True,
    ):
        """
        Initialize the test environment.

        Args:
            drone_namespace: ROS2 namespace of the drone.
            use_sim_time: Whether to use simulation time.
            verbose: Enable verbose output from DroneInterface.
            takeoff_height: Height for takeoff in meters.
            takeoff_speed: Speed for takeoff in m/s.
            land_speed: Speed for landing in m/s.
            max_vel: Maximum velocity command in m/s.
            max_yaw_vel: Maximum yaw angular velocity in rad/s (default: π ≈ 180°/s).
                         Defines the bounds of the yaw action dimension.
            pos_limit: Maximum absolute position in meters (scenario boundary).
                       Used to normalize relative position observations to [-1, 1].
            step_duration: Duration to wait after sending command (seconds).
            target_pose: Target pose [x, y, z, yaw] for the drone to reach.
                         Defaults to [0, 0, 1, 0] (origin, 1m height, yaw=0).
                         The target is fixed across episodes.
            distance_threshold: Distance in meters to consider target reached.
            max_steps: Maximum steps per episode before truncation.
            success_reward: Bonus reward for reaching the target.
            oob_penalty: Penalty for going out of bounds (pos_limit exceeded).
            path_facing_weight: Scale factor for continuous path-facing shaping term.
            speed_deadband: Horizontal speed threshold (m/s) below which
                            path-facing is considered undefined and neutral.
            fixed_start_pose: Optional deterministic [x, y, z, yaw] pose to apply
                              after takeoff on every reset. Useful for bounded
                              point-reaching experiments where out-of-bounds
                              episodes must not poison subsequent resets.
        """
        super().__init__()

        self.drone_namespace = drone_namespace
        self.use_sim_time = use_sim_time
        self.verbose = verbose
        self.takeoff_height = takeoff_height
        self.takeoff_speed = takeoff_speed
        self.land_speed = land_speed
        self.max_vel = max_vel
        self.max_yaw_vel = max_yaw_vel
        self.pos_limit = pos_limit
        self.step_duration = step_duration
        self.distance_threshold = distance_threshold
        self.max_steps = max_steps
        self.success_reward = success_reward
        self.oob_penalty = oob_penalty
        self.path_facing_weight = path_facing_weight
        self.progress_reward_weight = float(progress_reward_weight)
        self.speed_deadband = speed_deadband
        self.fixed_start_pose = list(fixed_start_pose) if fixed_start_pose is not None else None
        self.fixed_start_tolerance = float(fixed_start_tolerance)
        self.fixed_start_timeout = float(fixed_start_timeout)
        self.reset_min_speed = float(reset_min_speed)
        self.reset_ground_recovery_height = float(reset_ground_recovery_height)
        self.publish_target_marker = bool(publish_target_marker)
        self.target_marker_topic = target_marker_topic
        self.target_marker_frame_id = target_marker_frame_id
        self.target_marker_scale = float(target_marker_scale)
        self.close_operation_timeout = float(close_operation_timeout)
        self.randomize_hover_start = randomize_hover_start
        self.scene_bounds_xy = float(scene_bounds_xy)
        self.height_bounds = (float(height_bounds[0]), float(height_bounds[1]))
        self.min_start_target_distance = min_start_target_distance
        self.hover_speed_threshold = float(hover_speed_threshold)
        self.hover_settle_time = float(hover_settle_time)
        self.hover_timeout = float(hover_timeout)
        self.max_reset_sample_attempts = int(max_reset_sample_attempts)
        self.randomize_yaw = bool(randomize_yaw)
        self._last_sample_attempts = 0

        self._validate_bounds()
        self._validate_pose('fixed_start_pose', self.fixed_start_pose)
        if self.fixed_start_pose is not None and self.randomize_hover_start:
            raise ValueError('fixed_start_pose and randomize_hover_start are mutually exclusive')

        # Target pose [x, y, z, yaw] — goal for the drone
        self._target_pose = (
            list(target_pose) if target_pose is not None
            else [0.0, 0.0, 1.0, 0.0]
        )

        # Observation space: [dx, dy, dz, dyaw] — relative to target,
        # all normalized to [-1, 1].
        # Position diff normalized by pos_limit; yaw diff normalized by π.
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Action space: velocity commands [vx, vy, vz, vyaw]
        # Linear velocities bounded by max_vel, yaw rate bounded by max_yaw_vel
        self.action_space = spaces.Box(
            low=np.array([-max_vel, -max_vel, -max_vel, -max_yaw_vel], dtype=np.float32),
            high=np.array([max_vel, max_vel, max_vel, max_yaw_vel], dtype=np.float32),
        )

        # DroneInterface (initialized on first reset)
        self._drone = None
        self._speed_handler = None
        self._marker_node = None
        self._target_marker_pub = None
        self._is_flying = False
        self._step_count = 0
        self._previous_distance: float | None = None
        self._last_reset_diagnostics: dict[str, Any] = {}

    def _init_ros(self):
        """Initialize ROS2 (once per process) and create DroneInterface."""
        import rclpy
        from as2_python_api.drone_interface import DroneInterface

        # Thread-safe, once-per-process rclpy initialization
        with AS2TestEnv._rclpy_lock:
            if not AS2TestEnv._rclpy_initialized:
                rclpy.init()
                AS2TestEnv._rclpy_initialized = True
                logger.info("ROS2 initialized (process-wide)")

        self._drone = DroneInterface(
            drone_id=self.drone_namespace,
            use_sim_time=self.use_sim_time,
            verbose=self.verbose
        )

        # Create SpeedMotion handler directly (instead of load_module which
        # fails because MotionReferenceHandlerModule.__call__ is abstract
        # in the installed AS2 version)
        from as2_motion_reference_handlers.speed_motion import SpeedMotion
        self._speed_handler = SpeedMotion(self._drone)
        self._init_target_marker_publisher()
        logger.info(f"DroneInterface created for '{self.drone_namespace}'")

    def _init_target_marker_publisher(self) -> None:
        """Create an optional ROS2 marker publisher for the target pose."""
        if not self.publish_target_marker or self._target_marker_pub is not None:
            return

        try:
            import rclpy
            from rclpy.qos import DurabilityPolicy, QoSProfile
            from visualization_msgs.msg import Marker

            marker_qos = QoSProfile(depth=1)
            marker_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

            self._marker_node = rclpy.create_node(f'{self.drone_namespace}_target_marker')
            self._target_marker_pub = self._marker_node.create_publisher(
                Marker,
                self.target_marker_topic,
                marker_qos,
            )
            logger.info(
                "Target marker publisher created on topic '%s' frame '%s'",
                self.target_marker_topic,
                self.target_marker_frame_id,
            )
        except Exception as e:
            logger.warning(f"Unable to create target marker publisher: {e}")
            self._marker_node = None
            self._target_marker_pub = None

    def _publish_target_marker(self) -> None:
        """Publish the target as a visible RViz sphere marker."""
        if not self.publish_target_marker or self._target_marker_pub is None or self._marker_node is None:
            return

        try:
            from visualization_msgs.msg import Marker

            tx, ty, tz, _ = self._target_pose
            marker = Marker()
            marker.header.frame_id = self.target_marker_frame_id
            marker.header.stamp = self._marker_node.get_clock().now().to_msg()
            marker.ns = f'{self.drone_namespace}_rl_target'
            marker.id = 0
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(tx)
            marker.pose.position.y = float(ty)
            marker.pose.position.z = float(tz)
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.target_marker_scale
            marker.scale.y = self.target_marker_scale
            marker.scale.z = self.target_marker_scale
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            self._target_marker_pub.publish(marker)
        except Exception as e:
            logger.warning(f"Unable to publish target marker: {e}")

    def _get_obs(self) -> np.ndarray:
        """
        Compute relative state between drone and target, normalized to [-1, 1].

        Position difference is divided by pos_limit.
        Yaw difference is wrapped to [-π, π] via atan2 and divided by π.
        Values are clipped to [-1, 1] in case the drone exceeds scenario bounds.

        Returns:
            Normalized observation array [dx, dy, dz, dyaw] ∈ [-1, 1]^4
        """
        try:
            # Drone position
            pose = self._drone.position
            x, y, z = pose[0], pose[1], pose[2]

            # Drone yaw from orientation [roll, pitch, yaw]
            yaw = self._drone.orientation[2]

            # Target
            tx, ty, tz, tyaw = self._target_pose

            # Relative position (drone - target), normalized by pos_limit
            dx = (x - tx) / self.pos_limit
            dy = (y - ty) / self.pos_limit
            dz = (z - tz) / self.pos_limit

            # Relative yaw with angular wrapping to [-π, π], then / π
            dyaw_raw = yaw - tyaw
            dyaw = math.atan2(math.sin(dyaw_raw), math.cos(dyaw_raw))
            dyaw_norm = dyaw / math.pi

            obs = np.array([dx, dy, dz, dyaw_norm], dtype=np.float32)

            # Clip in case the drone exceeds the defined boundaries
            obs = np.clip(obs, -1.0, 1.0)

        except Exception as e:
            logger.warning(f"Error reading drone state: {e}")
            obs = np.zeros(4, dtype=np.float32)

        return obs

    def _compute_distance(self) -> tuple[float, float]:
        """
        Compute Euclidean distance from drone to target.

        Uses raw (unclipped) positions for accurate distance even when
        the drone exceeds the scenario boundaries.

        Returns:
            Tuple of (raw_distance_m, normalized_distance) where
            normalized_distance ∈ [0, 1], clipped by pos_limit * √3.
        """
        try:
            pose = self._drone.position
            tx, ty, tz, _ = self._target_pose

            d = math.sqrt(
                (pose[0] - tx) ** 2
                + (pose[1] - ty) ** 2
                + (pose[2] - tz) ** 2
            )

            d_max = self.pos_limit * math.sqrt(3.0)
            d_norm = min(d / d_max, 1.0)

            return d, d_norm

        except Exception as e:
            logger.warning(f"Error computing distance: {e}")
            return float('inf'), 1.0

    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-π, π]."""
        return math.atan2(math.sin(angle), math.cos(angle))

    def _compute_reward_distance(self, d_norm: float) -> float:
        """Distance contribution to reward."""
        return -d_norm

    def _compute_path_facing_term(self) -> tuple[float, float, float, float]:
        """
        Compute continuous path-facing reward from measured horizontal velocity.

        Returns:
            Tuple (path_facing_reward, speed_xy, path_yaw, path_yaw_error)
        """
        try:
            yaw = float(self._drone.orientation[2])
            speed = self._drone.speed
            vx = float(speed[0])
            vy = float(speed[1])
            speed_xy = math.hypot(vx, vy)

            if speed_xy <= self.speed_deadband:
                return 0.0, speed_xy, yaw, 0.0

            path_yaw = math.atan2(vy, vx)
            path_yaw_error = self._wrap_angle(yaw - path_yaw)
            path_facing_reward = self.path_facing_weight * math.cos(path_yaw_error)

            return path_facing_reward, speed_xy, path_yaw, path_yaw_error

        except Exception as e:
            logger.warning(f"Error computing path-facing term: {e}")
            return 0.0, 0.0, 0.0, 0.0

    def _compute_yaw_error(self) -> float:
        """Backward-compatible target yaw error helper (unused in reward)."""
        try:
            yaw = self._drone.orientation[2]
            _, _, _, tyaw = self._target_pose
            dyaw = self._wrap_angle(yaw - tyaw)
            return abs(dyaw) / math.pi
        except Exception as e:
            logger.warning(f"Error computing yaw error: {e}")
            return 0.0

    def _get_info(self) -> dict[str, Any]:
        """
        Build info dict with detailed state for debugging.

        Returns:
            Dictionary with raw drone state data.
        """
        info = {
            'step_count': self._step_count,
            'is_flying': self._is_flying,
            'drone_namespace': self.drone_namespace,
            'target_pose': list(self._target_pose),
            'is_success': False,
            'is_out_of_bounds': False,
        }

        try:
            info['position'] = list(self._drone.position)
            info['speed'] = list(self._drone.speed)
            info['orientation'] = list(self._drone.orientation)
            d, d_norm = self._compute_distance()
            info['distance'] = d
            info['final_distance'] = d
            info['distance_norm'] = d_norm
        except Exception as e:
            info['state_error'] = str(e)

        return info

    def _validate_bounds(self) -> None:
        if self.scene_bounds_xy <= 0.0:
            raise ValueError('scene_bounds_xy must be > 0')
        z_min, z_max = self.height_bounds
        if z_min > z_max:
            raise ValueError('height_bounds must satisfy z_min <= z_max')
        if self.max_reset_sample_attempts < 1:
            raise ValueError('max_reset_sample_attempts must be >= 1')
        if self.fixed_start_tolerance <= 0.0:
            raise ValueError('fixed_start_tolerance must be > 0')
        if self.fixed_start_timeout <= 0.0:
            raise ValueError('fixed_start_timeout must be > 0')
        if self.reset_min_speed < 0.0:
            raise ValueError('reset_min_speed must be >= 0')
        if self.reset_ground_recovery_height < 0.0:
            raise ValueError('reset_ground_recovery_height must be >= 0')
        if self.close_operation_timeout <= 0.0:
            raise ValueError('close_operation_timeout must be > 0')
        if self.pos_limit < self.scene_bounds_xy:
            raise ValueError(
                'pos_limit must be >= scene_bounds_xy to preserve normalization budget'
            )
        if self.pos_limit < max(abs(z_min), abs(z_max)):
            raise ValueError(
                'pos_limit must be >= max(abs(height_bounds)) to preserve normalization budget'
            )

    def _validate_pose(self, name: str, pose: list[float] | None) -> None:
        if pose is None:
            return
        if len(pose) != 4:
            raise ValueError(f'{name} must contain [x, y, z, yaw]')
        x, y, z, _ = [float(value) for value in pose]
        if abs(x) > self.pos_limit or abs(y) > self.pos_limit or abs(z) > self.pos_limit:
            raise ValueError(f'{name} must be within +/-pos_limit to preserve reset safety')

    def _sample_randomized_episode(self) -> tuple[list[float], list[float], int]:
        min_distance = max(
            self.distance_threshold,
            float(self.min_start_target_distance)
            if self.min_start_target_distance is not None
            else self.distance_threshold,
        )

        for attempt in range(1, self.max_reset_sample_attempts + 1):
            start_yaw = float(self.np_random.uniform(-math.pi, math.pi)) if self.randomize_yaw else 0.0
            start_pose = [
                float(self.np_random.uniform(-self.scene_bounds_xy, self.scene_bounds_xy)),
                float(self.np_random.uniform(-self.scene_bounds_xy, self.scene_bounds_xy)),
                float(self.np_random.uniform(self.height_bounds[0], self.height_bounds[1])),
                start_yaw,
            ]
            target_pose = [
                float(self.np_random.uniform(-self.scene_bounds_xy, self.scene_bounds_xy)),
                float(self.np_random.uniform(-self.scene_bounds_xy, self.scene_bounds_xy)),
                float(self.np_random.uniform(self.height_bounds[0], self.height_bounds[1])),
                0.0,
            ]
            distance = math.dist(start_pose[:3], target_pose[:3])
            if distance > min_distance:
                self._last_sample_attempts = attempt
                return start_pose, target_pose, attempt

        raise RuntimeError(
            'Unable to sample valid randomized start/target pair '
            f'after {self.max_reset_sample_attempts} attempts'
        )

    def _wait_for_hover_settle(self) -> bool:
        if self.hover_settle_time <= 0.0:
            return True
        stable_elapsed = 0.0
        started = time.time()
        check_dt = max(0.05, self.step_duration)
        while time.time() - started < self.hover_timeout:
            speed = getattr(self._drone, 'speed', [0.0, 0.0, 0.0])
            speed_xy = math.hypot(float(speed[0]), float(speed[1]))
            speed_z = abs(float(speed[2]))
            if speed_xy <= self.hover_speed_threshold and speed_z <= self.hover_speed_threshold:
                stable_elapsed += check_dt
                if stable_elapsed >= self.hover_settle_time:
                    return True
            else:
                stable_elapsed = 0.0
            time.sleep(check_dt)
        return False

    def _apply_start_pose(self, start_pose: list[float]) -> None:
        if hasattr(self._drone, 'set_pose'):
            self._drone.set_pose(start_pose[0], start_pose[1], start_pose[2], start_pose[3])
            return
        if self._drive_to_start_pose_with_velocity(start_pose):
            return
        if hasattr(self._drone, 'go_to'):
            diagnostics = self._format_reset_diagnostics()
            raise RuntimeError(
                'Velocity-based start pose reset timed out; refusing to fall back '
                'to blocking go_to during in-air reset. '
                f'{diagnostics}'
            )
        # Mock fallback for tests: directly mutate position/orientation if writable.
        if hasattr(self._drone, 'position') and hasattr(self._drone, 'orientation'):
            self._drone.position = [start_pose[0], start_pose[1], start_pose[2]]
            self._drone.orientation = [0.0, 0.0, start_pose[3]]
            return
        raise RuntimeError('No supported simulator API for bounded pose reset')

    def _read_reset_pose(self) -> tuple[np.ndarray, float] | None:
        try:
            position = np.array(self._drone.position[:3], dtype=np.float32)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                return None
            yaw = float(self._drone.orientation[2]) if hasattr(self._drone, 'orientation') else 0.0
            if not math.isfinite(yaw):
                yaw = 0.0
            return position, yaw
        except Exception as e:
            self._last_reset_diagnostics = {
                'reason': 'state_read_error',
                'state_error': str(e),
            }
            return None

    def _format_reset_diagnostics(self) -> str:
        if not self._last_reset_diagnostics:
            return 'No reset diagnostics available.'
        parts = []
        for key in [
            'reason',
            'elapsed',
            'target_pose',
            'final_pose',
            'position_error',
            'yaw_error',
            'last_command',
            'state_error',
        ]:
            if key in self._last_reset_diagnostics:
                parts.append(f'{key}={self._last_reset_diagnostics[key]}')
        return 'Reset diagnostics: ' + ', '.join(parts)

    def _apply_reset_min_speed(self, command_xyz: np.ndarray, error: np.ndarray) -> np.ndarray:
        if self.reset_min_speed <= 0.0:
            return command_xyz

        adjusted = command_xyz.copy()
        for idx in range(3):
            if abs(float(error[idx])) <= self.fixed_start_tolerance:
                continue
            if abs(float(adjusted[idx])) < self.reset_min_speed:
                adjusted[idx] = math.copysign(self.reset_min_speed, float(error[idx]))
        return np.clip(adjusted, -self.max_vel, self.max_vel)

    def _start_pose_error(self, start_pose: list[float]) -> tuple[float, float, list[float] | None]:
        pose = self._read_reset_pose()
        if pose is None:
            return float('inf'), float('inf'), None
        current_xyz, yaw = pose
        target_xyz = np.array(start_pose[:3], dtype=np.float32)
        distance = float(np.linalg.norm(target_xyz - current_xyz))
        yaw_error = abs(self._wrap_angle(float(start_pose[3]) - yaw))
        return distance, yaw_error, [float(current_xyz[0]), float(current_xyz[1]), float(current_xyz[2]), float(yaw)]

    def _is_at_start_pose(self, start_pose: list[float]) -> bool:
        distance, yaw_error, _ = self._start_pose_error(start_pose)
        return distance <= self.fixed_start_tolerance and yaw_error <= 0.2

    def _apply_and_confirm_start_pose(self, start_pose: list[float]) -> bool:
        """Apply a start pose and verify it still holds after controller reset."""
        hover_settled = True
        for attempt in range(1, 3):
            self._apply_start_pose(start_pose)
            self._reset_velocity_controller()
            hover_settled = self._hold_start_pose_after_controller_reset(start_pose, attempt)
            if hover_settled and self._is_at_start_pose(start_pose):
                return True

            distance, yaw_error, pose = self._start_pose_error(start_pose)
            if self._last_reset_diagnostics.get('reason') != 'post_controller_hold_timeout':
                self._last_reset_diagnostics = {
                    'reason': 'post_settle_drift',
                    'attempt': attempt,
                    'target_pose': [float(v) for v in start_pose],
                    'final_pose': pose,
                    'position_error': round(distance, 4) if math.isfinite(distance) else 'inf',
                    'yaw_error': round(yaw_error, 4) if math.isfinite(yaw_error) else 'inf',
                }
            logger.warning(
                'Start pose drifted after controller reset; retrying bounded velocity reset. %s',
                self._format_reset_diagnostics(),
            )

        raise RuntimeError(
            'Velocity-based start pose reset did not hold after controller reset. '
            f'{self._format_reset_diagnostics()}'
        )

    def _reset_recovery_hover_height(self) -> float:
        """Return the safe hover height to recover before velocity-based reset."""
        fixed_start_height = float(self.fixed_start_pose[2]) if self.fixed_start_pose is not None else 0.0
        return max(
            float(self.takeoff_height),
            fixed_start_height,
            float(self.reset_ground_recovery_height),
        )

    def _recover_low_altitude_hover_before_velocity_reset(self, hover_height: float) -> bool:
        """
        Recover with the classical takeoff controller if reset starts too low.

        Velocity commands may be ignored while the vehicle is grounded or nearly
        grounded. In that state, bounded velocity reset can keep sending upward
        commands without ever reaching a valid hover. Before recreating the
        velocity controller for an in-air reset, use the classical controller to
        re-establish a safe hover state.
        """
        pose = self._read_reset_pose()
        if pose is None:
            return True

        current_xyz, yaw = pose
        current_z = float(current_xyz[2])
        recovery_threshold = float(self.reset_ground_recovery_height)
        if current_z > recovery_threshold:
            return True

        target_height = max(float(hover_height), recovery_threshold)
        logger.warning(
            'Reset started at low altitude z=%.3f; recovering to hover %.3fm before velocity reset',
            current_z,
            target_height,
        )

        if self._speed_handler is not None:
            try:
                for _ in range(3):
                    self._speed_handler.send_speed_command_with_yaw_speed(
                        twist=[0.0, 0.0, 0.0],
                        twist_frame_id='earth',
                        yaw_speed=0.0,
                    )
                    time.sleep(max(0.05, self.step_duration))
            except Exception as e:
                logger.warning('Unable to zero stale velocity command before low-altitude recovery: %s', e)

        if hasattr(self._drone, 'land'):
            landed = self._run_close_operation(
                'low-altitude recovery landing',
                lambda: self._drone.land(speed=self.land_speed),
            )
            if not landed:
                logger.warning(
                    'Low-altitude landing handoff did not complete; recreating DroneInterface before takeoff recovery'
                )
                if hasattr(self._drone, 'shutdown'):
                    shutdown_ok = self._run_close_operation(
                        'low-altitude recovery DroneInterface shutdown',
                        self._drone.shutdown,
                    )
                    if not shutdown_ok:
                        self._last_reset_diagnostics = {
                            'reason': 'low_altitude_interface_recreation_failed',
                            'target_pose': [float(current_xyz[0]), float(current_xyz[1]), target_height, float(yaw)],
                            'final_pose': [float(current_xyz[0]), float(current_xyz[1]), current_z, float(yaw)],
                        }
                        raise RuntimeError(
                            'Low-altitude interface recreation failed before hover recovery. '
                            f'{self._format_reset_diagnostics()}'
                        )
                self._drone = None
                self._speed_handler = None
                self._is_flying = False
                self._init_ros()
            else:
                self._is_flying = False
            time.sleep(max(0.1, self.step_duration))

        if hasattr(self._drone, 'manual'):
            try:
                self._drone.manual()
                time.sleep(max(0.1, self.step_duration))
            except Exception as e:
                logger.warning('Unable to switch to manual before low-altitude recovery: %s', e)

        arm_success = True
        offboard_success = True
        takeoff_success = False
        try:
            if hasattr(self._drone, 'arm'):
                arm_success = bool(self._drone.arm())
            if hasattr(self._drone, 'offboard'):
                offboard_success = bool(self._drone.offboard())
            if hasattr(self._drone, 'takeoff'):
                takeoff_success = bool(self._drone.takeoff(
                    height=target_height,
                    speed=self.takeoff_speed,
                ))
            else:
                takeoff_success = False
        except Exception as e:
            self._last_reset_diagnostics = {
                'reason': 'low_altitude_recovery_error',
                'target_pose': [float(current_xyz[0]), float(current_xyz[1]), target_height, float(yaw)],
                'final_pose': [float(current_xyz[0]), float(current_xyz[1]), current_z, float(yaw)],
                'state_error': str(e),
            }
            raise RuntimeError(
                'Low-altitude hover recovery failed before velocity reset. '
                f'{self._format_reset_diagnostics()}'
            ) from e

        self._is_flying = bool(takeoff_success)
        hover_settled = self._wait_for_hover_settle() if self._is_flying else False
        recovered_pose = self._read_reset_pose()
        recovered_xyz = current_xyz
        recovered_yaw = yaw
        if recovered_pose is not None:
            recovered_xyz, recovered_yaw = recovered_pose

        if not (takeoff_success and hover_settled):
            self._last_reset_diagnostics = {
                'reason': 'low_altitude_recovery_failed',
                'target_pose': [float(current_xyz[0]), float(current_xyz[1]), target_height, float(yaw)],
                'final_pose': [
                    float(recovered_xyz[0]),
                    float(recovered_xyz[1]),
                    float(recovered_xyz[2]),
                    float(recovered_yaw),
                ],
                'state_error': {
                    'arm_success': arm_success,
                    'offboard_success': offboard_success,
                    'takeoff_success': takeoff_success,
                    'hover_settled': hover_settled,
                },
            }
            raise RuntimeError(
                'Low-altitude hover recovery did not reach a stable flying state before velocity reset. '
                f'{self._format_reset_diagnostics()}'
            )

        self._last_reset_diagnostics = {
            'reason': 'low_altitude_recovery',
            'target_pose': [float(current_xyz[0]), float(current_xyz[1]), target_height, float(yaw)],
            'final_pose': [
                float(recovered_xyz[0]),
                float(recovered_xyz[1]),
                float(recovered_xyz[2]),
                float(recovered_yaw),
            ],
            'position_error': round(abs(target_height - float(recovered_xyz[2])), 4),
        }
        return True

    def _hold_start_pose_after_controller_reset(self, start_pose: list[float], attempt: int) -> bool:
        """
        Keep a bounded pose hold active after recreating the speed controller.

        AS2 can briefly report the requested pose immediately after the velocity
        reset, then drift while the new controller settles. A speed-only hover
        wait can miss that handoff drift, so this loop only counts settle time
        while the vehicle remains within the fixed-start pose tolerance. If it
        drifts out, the same bounded velocity controller reacquires the pose.
        """
        if self._speed_handler is None:
            return self._wait_for_hover_settle()

        target_xyz = np.array(start_pose[:3], dtype=np.float32)
        target_yaw = float(start_pose[3])
        required_stable_time = max(0.0, float(self.hover_settle_time))
        if required_stable_time <= 0.0:
            return self._is_at_start_pose(start_pose)

        started = time.time()
        stable_elapsed = 0.0
        check_dt = max(0.05, self.step_duration)
        last_command = [0.0, 0.0, 0.0, 0.0]
        last_pose: list[float] | None = None
        last_distance = float('inf')
        last_yaw_error = float('inf')

        while time.time() - started < self.hover_timeout:
            pose = self._read_reset_pose()
            if pose is None:
                stable_elapsed = 0.0
                time.sleep(check_dt)
                continue

            current_xyz, yaw = pose
            error = target_xyz - current_xyz
            distance = float(np.linalg.norm(error))
            yaw_error = self._wrap_angle(target_yaw - yaw)
            last_distance = distance
            last_yaw_error = abs(yaw_error)
            last_pose = [float(current_xyz[0]), float(current_xyz[1]), float(current_xyz[2]), float(yaw)]

            if distance <= self.fixed_start_tolerance and abs(yaw_error) <= 0.2:
                command_xyz = np.zeros(3, dtype=np.float32)
                yaw_speed = 0.0
                stable_elapsed += check_dt
            else:
                command_xyz = np.clip(error, -self.max_vel, self.max_vel)
                command_xyz = self._apply_reset_min_speed(command_xyz, error)
                yaw_speed = float(np.clip(yaw_error, -self.max_yaw_vel, self.max_yaw_vel))
                stable_elapsed = 0.0

            last_command = [float(command_xyz[0]), float(command_xyz[1]), float(command_xyz[2]), yaw_speed]
            self._speed_handler.send_speed_command_with_yaw_speed(
                twist=last_command[:3],
                twist_frame_id='earth',
                yaw_speed=yaw_speed,
            )

            if stable_elapsed >= required_stable_time:
                time.sleep(check_dt)
                confirmed_pose = self._read_reset_pose()
                if confirmed_pose is None:
                    stable_elapsed = 0.0
                    continue
                confirmed_xyz, confirmed_yaw = confirmed_pose
                confirmed_distance = float(np.linalg.norm(target_xyz - confirmed_xyz))
                confirmed_yaw_error = abs(self._wrap_angle(target_yaw - confirmed_yaw))
                if confirmed_distance > self.fixed_start_tolerance or confirmed_yaw_error > 0.2:
                    last_pose = [
                        float(confirmed_xyz[0]),
                        float(confirmed_xyz[1]),
                        float(confirmed_xyz[2]),
                        float(confirmed_yaw),
                    ]
                    last_distance = confirmed_distance
                    last_yaw_error = confirmed_yaw_error
                    stable_elapsed = 0.0
                    continue

                self._last_reset_diagnostics = {
                    'reason': 'post_controller_hold',
                    'attempt': attempt,
                    'elapsed': round(time.time() - started, 3),
                    'target_pose': [float(v) for v in start_pose],
                    'final_pose': [
                        float(confirmed_xyz[0]),
                        float(confirmed_xyz[1]),
                        float(confirmed_xyz[2]),
                        float(confirmed_yaw),
                    ],
                    'position_error': round(confirmed_distance, 4),
                    'yaw_error': round(confirmed_yaw_error, 4),
                    'last_command': [round(v, 4) for v in last_command],
                }
                return True

            time.sleep(check_dt)

        self._last_reset_diagnostics = {
            'reason': 'post_controller_hold_timeout',
            'attempt': attempt,
            'elapsed': round(time.time() - started, 3),
            'target_pose': [float(v) for v in start_pose],
            'final_pose': last_pose,
            'position_error': round(last_distance, 4) if math.isfinite(last_distance) else 'inf',
            'yaw_error': round(last_yaw_error, 4) if math.isfinite(last_yaw_error) else 'inf',
            'last_command': [round(v, 4) for v in last_command],
        }
        logger.warning(
            'Start pose did not remain stable during post-controller hold. %s',
            self._format_reset_diagnostics(),
        )
        return False

    def _drive_to_start_pose_with_velocity(self, start_pose: list[float]) -> bool:
        """Move toward a start pose using bounded velocity commands."""
        if self._speed_handler is None or not hasattr(self._drone, 'position'):
            return False

        target_xyz = np.array(start_pose[:3], dtype=np.float32)
        target_yaw = float(start_pose[3])
        started = time.time()
        check_dt = max(0.05, self.step_duration)
        last_command = [0.0, 0.0, 0.0, 0.0]

        while time.time() - started < self.fixed_start_timeout:
            pose = self._read_reset_pose()
            if pose is None:
                time.sleep(check_dt)
                continue

            current_xyz, yaw = pose
            error = target_xyz - current_xyz
            distance = float(np.linalg.norm(error))
            yaw_error = self._wrap_angle(target_yaw - yaw)

            if distance <= self.fixed_start_tolerance and abs(yaw_error) <= 0.2:
                self._speed_handler.send_speed_command_with_yaw_speed(
                    twist=[0.0, 0.0, 0.0],
                    twist_frame_id='earth',
                    yaw_speed=0.0,
                )
                time.sleep(check_dt)
                confirmed_pose = self._read_reset_pose()
                if confirmed_pose is None:
                    continue
                confirmed_xyz, confirmed_yaw = confirmed_pose
                confirmed_distance = float(np.linalg.norm(target_xyz - confirmed_xyz))
                confirmed_yaw_error = self._wrap_angle(target_yaw - confirmed_yaw)
                if confirmed_distance > self.fixed_start_tolerance or abs(confirmed_yaw_error) > 0.2:
                    continue
                self._last_reset_diagnostics = {
                    'reason': 'reached',
                    'elapsed': round(time.time() - started, 3),
                    'target_pose': [float(v) for v in start_pose],
                    'final_pose': [float(confirmed_xyz[0]), float(confirmed_xyz[1]), float(confirmed_xyz[2]), float(confirmed_yaw)],
                    'position_error': round(confirmed_distance, 4),
                    'yaw_error': round(abs(confirmed_yaw_error), 4),
                }
                return True

            control_error = error.copy()
            recovery_height = min(float(target_xyz[2]), self.reset_ground_recovery_height)
            if current_xyz[2] < recovery_height:
                # When AS2 reports the vehicle at/near the ground, recover
                # vertically first. Lateral velocity while grounded can be
                # ineffective and was observed to leave reset stuck out of bounds.
                control_error[0] = 0.0
                control_error[1] = 0.0
                control_error[2] = max(float(target_xyz[2]), recovery_height) - current_xyz[2]

            command_xyz = np.clip(1.0 * control_error, -self.max_vel, self.max_vel)
            command_xyz = self._apply_reset_min_speed(command_xyz, control_error)
            yaw_speed = float(np.clip(1.0 * yaw_error, -self.max_yaw_vel, self.max_yaw_vel))
            last_command = [float(command_xyz[0]), float(command_xyz[1]), float(command_xyz[2]), yaw_speed]
            self._speed_handler.send_speed_command_with_yaw_speed(
                twist=last_command[:3],
                twist_frame_id='earth',
                yaw_speed=yaw_speed,
            )
            time.sleep(check_dt)

        final_pose = self._read_reset_pose()
        final_xyz = None
        final_yaw = target_yaw
        final_distance = float('inf')
        final_yaw_error = float('inf')
        if final_pose is not None:
            final_xyz, final_yaw = final_pose
            final_distance = float(np.linalg.norm(target_xyz - final_xyz))
            final_yaw_error = abs(self._wrap_angle(target_yaw - final_yaw))

        self._speed_handler.send_speed_command_with_yaw_speed(
            twist=[0.0, 0.0, 0.0],
            twist_frame_id='earth',
            yaw_speed=0.0,
        )
        self._last_reset_diagnostics = {
            'reason': 'timeout',
            'elapsed': round(time.time() - started, 3),
            'target_pose': [float(v) for v in start_pose],
            'final_pose': (
                [float(final_xyz[0]), float(final_xyz[1]), float(final_xyz[2]), float(final_yaw)]
                if final_xyz is not None else None
            ),
            'position_error': round(final_distance, 4) if math.isfinite(final_distance) else 'inf',
            'yaw_error': round(final_yaw_error, 4) if math.isfinite(final_yaw_error) else 'inf',
            'last_command': [round(v, 4) for v in last_command],
        }
        logger.warning('Timed out while driving to fixed_start_pose. %s', self._format_reset_diagnostics())
        return False

    def _reset_velocity_controller(self) -> None:
        from as2_motion_reference_handlers.speed_motion import SpeedMotion

        self._speed_handler = SpeedMotion(self._drone)
        self._speed_handler.send_speed_command_with_yaw_speed(
            twist=[0.0, 0.0, 0.0],
            twist_frame_id='earth',
            yaw_speed=0.0,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment: arm, offboard, takeoff.

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        self._step_count = 0

        # Initialize ROS2 / DroneInterface if first call
        if self._drone is None:
            self._init_ros()

        supports_in_air_reset = self.fixed_start_pose is not None or self.randomize_hover_start

        # If already flying, keep the episode reset in-air for training modes
        # that explicitly manage their own safe start pose. Landing/takeoff on
        # every episode is slow and can block AS2 land behavior indefinitely.
        if self._is_flying:
            if supports_in_air_reset:
                logger.info("Keeping drone airborne for reset...")
                self._recover_low_altitude_hover_before_velocity_reset(
                    self._reset_recovery_hover_height()
                )
                self._reset_velocity_controller()
            else:
                logger.info("Landing before reset...")
                self._run_close_operation(
                    'landing before reset',
                    lambda: self._drone.land(speed=self.land_speed),
                )
                self._is_flying = False

        if not self._is_flying:
            # Arm
            logger.info("Arming drone...")
            success = self._drone.arm()
            logger.info(f"  Arm: {'OK' if success else 'FAILED'}")

            # Offboard mode
            logger.info("Setting offboard mode...")
            success = self._drone.offboard()
            logger.info(f"  Offboard: {'OK' if success else 'FAILED'}")

            # Takeoff
            logger.info(f"Taking off to {self.takeoff_height}m...")
            success = self._drone.takeoff(
                height=self.takeoff_height,
                speed=self.takeoff_speed
            )
            logger.info(f"  Takeoff: {'OK' if success else 'FAILED'}")
            self._is_flying = success

        hover_settled = True

        reset_mode = 'fixed_takeoff'
        start_pose = list(self._drone.position) + [float(self._drone.orientation[2])]
        sample_attempts = 0
        if self.fixed_start_pose is not None:
            hover_settled = self._wait_for_hover_settle()
            hover_settled = self._apply_and_confirm_start_pose(self.fixed_start_pose)
            reset_mode = 'fixed_start_pose'
            start_pose = list(self.fixed_start_pose)

        if self.randomize_hover_start:
            hover_settled = self._wait_for_hover_settle()
            sampled_start, sampled_target, sample_attempts = self._sample_randomized_episode()
            hover_settled = self._apply_and_confirm_start_pose(sampled_start)
            self._target_pose = sampled_target
            reset_mode = 'randomized_hover_start'
            start_pose = sampled_start

        obs = self._get_obs()
        info = self._get_info()
        self._publish_target_marker()
        current_distance, _ = self._compute_distance()
        self._previous_distance = current_distance if math.isfinite(current_distance) else None
        info['reset_success'] = self._is_flying
        info['reset_mode'] = reset_mode
        info['start_pose'] = list(start_pose)
        info['target_pose'] = list(self._target_pose)
        info['sample_attempts'] = sample_attempts
        info['hover_settled'] = hover_settled

        logger.info(f"Reset complete. Initial obs: {obs}")

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step: send velocity command and read new state.

        Args:
            action: Velocity command [vx, vy, vz, vyaw] where
                    vx, vy, vz are linear velocities in m/s and
                    vyaw is the yaw angular velocity in rad/s.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        import time

        self._step_count += 1

        # Clip action to valid range (per-dimension bounds)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        vx, vy, vz = float(action[0]), float(action[1]), float(action[2])
        vyaw = float(action[3])

        # Send velocity command with yaw rate via DroneInterface
        try:
            self._speed_handler.send_speed_command_with_yaw_speed(
                twist=[vx, vy, vz],
                twist_frame_id='earth',
                yaw_speed=vyaw,
            )
        except Exception as e:
            logger.error(f"Error sending velocity command: {e}")

        # Wait for the command to take effect
        time.sleep(self.step_duration)

        # Read new state
        obs = self._get_obs()
        info = self._get_info()
        self._publish_target_marker()
        info['action_sent'] = [vx, vy, vz, vyaw]

        # Continuous reward: distance penalty + path-facing shaping
        d_raw, d_norm = self._compute_distance()
        reward_distance = self._compute_reward_distance(d_norm)
        path_facing_reward, speed_xy, path_yaw, path_yaw_error = self._compute_path_facing_term()
        progress_reward = 0.0
        if self._previous_distance is not None and math.isfinite(d_raw):
            progress_reward = self.progress_reward_weight * (self._previous_distance - d_raw)
        self._previous_distance = d_raw if math.isfinite(d_raw) else self._previous_distance
        reward = reward_distance + path_facing_reward + progress_reward

        info['reward_distance'] = reward_distance
        info['path_facing_reward'] = path_facing_reward
        info['progress_reward'] = progress_reward
        info['speed_xy'] = speed_xy
        info['path_yaw'] = path_yaw
        info['path_yaw_error'] = path_yaw_error

        terminated = False
        truncated = False

        # --- Terminal conditions ---

        # Success: drone reached the target
        if d_raw < self.distance_threshold:
            terminated = True
            reward += self.success_reward
            info['terminal_reason'] = 'success'
            info['is_success'] = True
            info['is_out_of_bounds'] = False

        # Out-of-bounds: drone exceeded scenario limits
        elif self._is_out_of_bounds():
            terminated = True
            reward = -self.oob_penalty
            info['terminal_reason'] = 'out_of_bounds'
            info['is_success'] = False
            info['is_out_of_bounds'] = True

        # Max steps: episode truncation (time limit)
        if not terminated and self._step_count >= self.max_steps:
            truncated = True
            info['terminal_reason'] = 'max_steps'
            info['is_success'] = False
            info['is_out_of_bounds'] = False

        return obs, reward, terminated, truncated, info

    def _is_out_of_bounds(self) -> bool:
        """Check if the drone has exceeded the scenario boundaries."""
        try:
            pose = self._drone.position
            terminal_min_height = float(self.height_bounds[0])
            return any(abs(p) > self.pos_limit for p in pose) or float(pose[2]) < terminal_min_height
        except Exception:
            return False

    def _run_close_operation(self, label: str, operation) -> bool:
        """Run a close operation with a bounded wait to avoid shutdown hangs."""
        result: dict[str, BaseException | None] = {'error': None}

        def _runner():
            try:
                operation()
            except BaseException as e:  # keep close best-effort
                result['error'] = e

        worker = threading.Thread(target=_runner, daemon=True)
        worker.start()
        worker.join(timeout=self.close_operation_timeout)
        if worker.is_alive():
            logger.warning(
                "%s exceeded %.1fs during close; continuing best-effort shutdown",
                label,
                self.close_operation_timeout,
            )
            return False
        if result['error'] is not None:
            logger.error("Error during %s: %s", label, result['error'])
            return False
        return True

    def close(self):
        """Land the drone and shut down ROS2."""
        logger.info("Closing environment...")

        if self._drone is not None:
            if self._is_flying:
                logger.info("Landing...")
                self._run_close_operation(
                    'landing',
                    lambda: self._drone.land(speed=self.land_speed),
                )
                self._is_flying = False

            # Disarm
            self._run_close_operation('manual mode', self._drone.manual)

            self._run_close_operation('DroneInterface shutdown', self._drone.shutdown)
            self._drone = None
            logger.info("DroneInterface shut down")

        if self._marker_node is not None:
            try:
                self._marker_node.destroy_node()
            except Exception as e:
                logger.error(f"Error destroying marker node: {e}")
            self._marker_node = None
            self._target_marker_pub = None

        # NOTE: rclpy.shutdown() is NOT called here because other
        # vectorized instances in the same process may still need it.
        # rclpy is shut down automatically when the process exits.

        logger.info("Environment closed")
