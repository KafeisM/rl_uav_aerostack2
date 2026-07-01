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
    - Safety penalty: -oob_penalty (default 10.0) when exceeding bounds
      or crossing the training low-altitude safety threshold

Termination:
    - terminated=True: success (d < threshold), out-of-bounds, or unsafe low altitude
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

    # AS2 DroneInterfaceBase owns and spins itself on a private executor/thread.
    # Reset service clients therefore use an auxiliary per-env node/executor
    # instead of adding the DroneInterface node to another executor.
    _RESET_CLIENT_OWNERSHIP_MODEL = 'auxiliary_node'
    _RESET_YAW_TOLERANCE = 0.38

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
        reset_max_vel: float | None = None,
        reset_xy_kp: float = 1.0,
        reset_z_kp: float = 1.0,
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
        reset_yaw_tolerance: float | None = None,
        reset_yaw_required: bool = True,
        reset_min_speed: float = 0.15,
        reset_ground_recovery_height: float = 0.35,
        unsafe_low_altitude_threshold: float = 0.30,
        low_altitude_guard_margin: float = 0.20,
        low_altitude_guard_climb_speed: float = 0.15,
        vertical_safety_band: float = 0.0,
        vertical_safety_penalty_weight: float = 0.0,
        vertical_descent_penalty_weight: float = 0.0,
        publish_target_marker: bool = False,
        target_marker_topic: str = 'visualization_marker',
        target_marker_frame_id: str = 'earth',
        target_marker_scale: float = 0.35,
        close_operation_timeout: float = 10.0,
        reset_service_timeout: float = 2.0,
        use_simulator_reset_service: bool = True,
        use_service_reset_after_velocity_timeout: bool = False,
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
            reset_max_vel: Maximum linear velocity used only by reset controllers.
                           Defaults to max_vel for backward compatibility.
            reset_xy_kp: Proportional gain for reset XY velocity commands.
            reset_z_kp: Proportional gain for reset Z velocity commands.
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
            unsafe_low_altitude_threshold: Training safety cutoff in meters.
                                           This is intentionally separate from
                                           height_bounds[0], which remains the
                                           physical scene lower bound.
        """
        super().__init__()

        self.drone_namespace = drone_namespace
        self.use_sim_time = use_sim_time
        self.verbose = verbose
        self.takeoff_height = takeoff_height
        self.takeoff_speed = takeoff_speed
        self.land_speed = land_speed
        self.max_vel = max_vel
        self.reset_max_vel = float(max_vel) if reset_max_vel is None else float(reset_max_vel)
        self.reset_xy_kp = float(reset_xy_kp)
        self.reset_z_kp = float(reset_z_kp)
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
        self.reset_yaw_tolerance = (
            float(reset_yaw_tolerance)
            if reset_yaw_tolerance is not None
            else float(self._RESET_YAW_TOLERANCE)
        )
        self.reset_yaw_required = bool(reset_yaw_required)
        self.reset_min_speed = float(reset_min_speed)
        self.reset_ground_recovery_height = float(reset_ground_recovery_height)
        self.unsafe_low_altitude_threshold = float(unsafe_low_altitude_threshold)
        self.low_altitude_guard_margin = float(low_altitude_guard_margin)
        self.low_altitude_guard_climb_speed = float(low_altitude_guard_climb_speed)
        self.vertical_safety_band = float(vertical_safety_band)
        self.vertical_safety_penalty_weight = float(vertical_safety_penalty_weight)
        self.vertical_descent_penalty_weight = float(vertical_descent_penalty_weight)
        self.publish_target_marker = bool(publish_target_marker)
        self.target_marker_topic = target_marker_topic
        self.target_marker_frame_id = target_marker_frame_id
        self.target_marker_scale = float(target_marker_scale)
        self.close_operation_timeout = float(close_operation_timeout)
        self.reset_service_timeout = float(reset_service_timeout)
        self.use_simulator_reset_service = bool(use_simulator_reset_service)
        self.use_service_reset_after_velocity_timeout = bool(use_service_reset_after_velocity_timeout)
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
        self._reset_service_client = None
        self._reset_service_type = None
        self._reset_aux_node = None
        self._reset_aux_executor = None
        self._is_flying = False
        self._step_count = 0
        self._previous_distance: float | None = None
        self._last_reset_diagnostics: dict[str, Any] = {}
        self._episode_start_position: list[float] | None = None
        self._episode_last_position: list[float] | None = None
        self._episode_path_length = 0.0
        self._episode_max_physical_displacement = 0.0
        self._episode_min_altitude = float('nan')
        self._episode_motion_command_steps = 0
        self._episode_motion_command_accepted_steps = 0
        self._last_reset_method = 'unknown'
        self._last_reset_path = 'unknown'
        self._last_reset_service_attempted = False
        self._last_reset_service_status = 'not_attempted'
        self._last_low_altitude_guard_active = False

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
            'is_unsafe_low_altitude': False,
        }

        try:
            info['position'] = list(self._drone.position)
            info['altitude'] = float(self._drone.position[2])
            info['speed'] = list(self._drone.speed)
            info['orientation'] = list(self._drone.orientation)
            d, d_norm = self._compute_distance()
            info['distance'] = d
            info['final_distance'] = d
            info['distance_norm'] = d_norm
            info['is_unsafe_low_altitude'] = self._is_unsafe_low_altitude()
        except Exception as e:
            info['state_error'] = str(e)

        self._add_episode_monitor_info(info)
        return info

    def _current_position_xyz(self) -> list[float] | None:
        try:
            position = list(self._drone.position[:3])
            if len(position) != 3:
                return None
            xyz = [float(value) for value in position]
            return xyz if all(math.isfinite(value) for value in xyz) else None
        except Exception:
            return None

    @staticmethod
    def _reset_service_attempted_from_reason(reason: str) -> bool:
        return (
            reason != 'service_disabled'
            and (
                reason.startswith('service_')
                or reason == 'service_success_after_velocity_timeout'
                or reason in {'post_reset_arm_failed', 'post_reset_offboard_failed'}
            )
        )

    def _classify_reset_method(self, supports_in_air_reset: bool, reason: str) -> str:
        if reason in {'service_success', 'service_success_after_velocity_timeout'}:
            return 'simulator_service'
        if supports_in_air_reset:
            return 'velocity'
        return 'takeoff'

    def _initialize_episode_monitoring(self, start_position: list[float] | None) -> None:
        self._episode_start_position = list(start_position) if start_position is not None else None
        self._episode_last_position = list(start_position) if start_position is not None else None
        self._episode_path_length = 0.0
        self._episode_max_physical_displacement = 0.0
        self._episode_min_altitude = (
            float(start_position[2]) if start_position is not None else float('nan')
        )
        self._episode_motion_command_steps = 0
        self._episode_motion_command_accepted_steps = 0

    def _update_episode_monitoring(
        self,
        current_position: list[float] | None,
        motion_command_accepted: bool,
    ) -> None:
        self._episode_motion_command_steps += 1
        if motion_command_accepted:
            self._episode_motion_command_accepted_steps += 1

        if current_position is None:
            return

        if self._episode_start_position is None:
            self._initialize_episode_monitoring(current_position)
            return

        if self._episode_last_position is not None:
            self._episode_path_length += math.dist(self._episode_last_position, current_position)
        self._episode_last_position = list(current_position)
        self._episode_max_physical_displacement = max(
            self._episode_max_physical_displacement,
            math.dist(self._episode_start_position, current_position),
        )
        altitude = float(current_position[2])
        if not math.isfinite(self._episode_min_altitude):
            self._episode_min_altitude = altitude
        else:
            self._episode_min_altitude = min(self._episode_min_altitude, altitude)

    def _add_episode_monitor_info(self, info: dict[str, Any]) -> None:
        physical_displacement = 0.0
        current_position = self._current_position_xyz()
        if self._episode_start_position is not None and current_position is not None:
            physical_displacement = math.dist(self._episode_start_position, current_position)

        command_steps = int(self._episode_motion_command_steps)
        accepted_steps = int(self._episode_motion_command_accepted_steps)
        acceptance_rate = float(accepted_steps / command_steps) if command_steps > 0 else 0.0

        info['physical_displacement'] = float(physical_displacement)
        info['path_length'] = float(self._episode_path_length)
        info['max_physical_displacement'] = float(self._episode_max_physical_displacement)
        info['min_altitude'] = float(self._episode_min_altitude)
        info['motion_command_steps'] = command_steps
        info['motion_command_accepted_steps'] = accepted_steps
        info['motion_command_acceptance_rate'] = acceptance_rate
        info['reset_method'] = self._last_reset_method
        info['reset_path'] = self._last_reset_path
        info['reset_service_attempted'] = bool(self._last_reset_service_attempted)
        info['reset_failure_class'] = str(self._last_reset_diagnostics.get('failure_class', ''))
        info['reset_position_error'] = self._reset_diagnostic_float('position_error')
        info['reset_yaw_error'] = self._reset_diagnostic_float('yaw_error')
        info['reset_service_status'] = self._last_reset_service_status

    def _reset_diagnostic_float(self, key: str) -> float:
        value = self._last_reset_diagnostics.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float('nan')

    def _validate_bounds(self) -> None:
        if self.scene_bounds_xy <= 0.0:
            raise ValueError('scene_bounds_xy must be > 0')
        z_min, z_max = self.height_bounds
        if z_min > z_max:
            raise ValueError('height_bounds must satisfy z_min <= z_max')
        if self.max_reset_sample_attempts < 1:
            raise ValueError('max_reset_sample_attempts must be >= 1')
        if self.reset_max_vel <= 0.0:
            raise ValueError('reset_max_vel must be > 0')
        if self.reset_xy_kp <= 0.0:
            raise ValueError('reset_xy_kp must be > 0')
        if self.reset_z_kp <= 0.0:
            raise ValueError('reset_z_kp must be > 0')
        if self.fixed_start_tolerance <= 0.0:
            raise ValueError('fixed_start_tolerance must be > 0')
        if self.fixed_start_timeout <= 0.0:
            raise ValueError('fixed_start_timeout must be > 0')
        if self.reset_yaw_tolerance <= 0.0 or self.reset_yaw_tolerance > math.pi:
            raise ValueError('reset_yaw_tolerance must be in (0, pi]')
        if self.reset_min_speed < 0.0:
            raise ValueError('reset_min_speed must be >= 0')
        if self.reset_ground_recovery_height < 0.0:
            raise ValueError('reset_ground_recovery_height must be >= 0')
        if self.unsafe_low_altitude_threshold <= z_min:
            raise ValueError(
                'unsafe_low_altitude_threshold must be greater than height_bounds[0] '
                'so training terminates before the physical lower bound'
            )
        if self.unsafe_low_altitude_threshold > z_max:
            raise ValueError('unsafe_low_altitude_threshold must be <= height_bounds[1]')
        if self.close_operation_timeout <= 0.0:
            raise ValueError('close_operation_timeout must be > 0')
        if self.reset_service_timeout <= 0.0:
            raise ValueError('reset_service_timeout must be > 0')
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
        if hasattr(self._drone, 'set_pose') and not self._is_flying:
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
            'failure_class',
            'elapsed',
            'recovery_timeout',
            'reacquire_timeout',
            'hold_timeout',
            'reset_phase',
            'phase',
            'settle_elapsed',
            'target_pose',
            'final_pose',
            'position_error',
            'initial_position_error',
            'best_position_error',
            'error_reduction',
            'dominant_axis',
            'yaw_error',
            'service_name',
            'message',
            'linear_speed_norm',
            'angular_speed_norm',
            'last_command',
            'control_available',
            'state_error',
        ]:
            if key in self._last_reset_diagnostics:
                parts.append(f'{key}={self._last_reset_diagnostics[key]}')
        return 'Reset diagnostics: ' + ', '.join(parts)

    def _reset_velocity_limit(self) -> float:
        return float(self.reset_max_vel)

    def _clip_reset_velocity(self, command_xyz: np.ndarray) -> np.ndarray:
        reset_limit = self._reset_velocity_limit()
        return np.clip(command_xyz, -reset_limit, reset_limit)

    def _reset_position_command(self, error: np.ndarray) -> np.ndarray:
        gains = np.array([self.reset_xy_kp, self.reset_xy_kp, self.reset_z_kp], dtype=np.float32)
        return self._clip_reset_velocity(gains * error)

    def _reset_single_axis_command(self, error: np.ndarray, axis_index: int) -> np.ndarray:
        axis_error = np.zeros(3, dtype=np.float32)
        axis_error[axis_index] = float(error[axis_index])
        command_xyz = self._reset_position_command(axis_error)
        return self._apply_reset_min_speed(command_xyz, axis_error)

    def _reset_lateral_axis_command(self, error: np.ndarray, axis_index: int) -> np.ndarray:
        axis_error_value = float(error[axis_index])
        axis_error_abs = abs(axis_error_value)
        command_xyz = np.zeros(3, dtype=np.float32)
        tolerance = self._reset_position_tolerance()
        horizontal_error = math.hypot(float(error[0]), float(error[1]))
        if axis_error_abs <= self.fixed_start_tolerance and horizontal_error <= tolerance:
            return command_xyz

        near_target_radius = max(3.0 * tolerance, tolerance + 0.01)
        phase_limit = self._reset_velocity_limit()
        raw_speed = axis_error_abs * self.reset_xy_kp

        if axis_error_abs <= near_target_radius:
            conservative_gain = min(self.reset_xy_kp, 1.0)
            raw_speed = axis_error_abs * conservative_gain
            phase_limit = min(
                phase_limit,
                self._reset_velocity_limit() * axis_error_abs / near_target_radius,
            )

        speed = min(raw_speed, phase_limit)
        if self.reset_min_speed > 0.0:
            speed = min(max(speed, self.reset_min_speed), phase_limit)

        command_xyz[axis_index] = math.copysign(speed, axis_error_value)
        return command_xyz

    def _reset_high_z_trim_command(self, error: np.ndarray) -> np.ndarray:
        z_error_value = float(error[2])
        z_error_abs = abs(z_error_value)
        command_xyz = np.zeros(3, dtype=np.float32)
        tolerance = self._reset_position_tolerance()
        near_target_radius = max(3.0 * tolerance, tolerance + 0.01)

        if z_error_abs <= 0.0:
            return command_xyz

        if z_error_abs > near_target_radius:
            return self._reset_single_axis_command(error, 2)

        phase_limit = min(
            self._reset_velocity_limit(),
            self._reset_velocity_limit() * z_error_abs / near_target_radius,
        )
        raw_speed = z_error_abs * min(self.reset_z_kp, 1.0)
        speed = min(raw_speed, phase_limit)
        if self.reset_min_speed > 0.0:
            speed = min(max(speed, self.reset_min_speed), phase_limit)

        command_xyz[2] = math.copysign(speed, z_error_value)
        return command_xyz

    def _should_prioritize_high_z_trim(
        self,
        error: np.ndarray,
        current_z: float,
        target_z: float,
        vertical_recovery_floor: float,
        vertical_floor_tolerance: float,
    ) -> bool:
        tolerance = self._reset_position_tolerance()
        z_error = float(error[2])
        high_above_target = z_error < -tolerance
        high_above_safe_band = float(current_z) > float(self.height_bounds[1]) - tolerance
        descent_floor = max(
            float(vertical_recovery_floor),
            float(self.unsafe_low_altitude_threshold),
        ) + vertical_floor_tolerance

        return (
            (high_above_target or high_above_safe_band)
            and float(current_z) > descent_floor
            and float(current_z) > float(target_z)
        )

    def _reset_near_target_vector_command(self, error: np.ndarray) -> np.ndarray:
        command_xyz = np.array(
            [
                min(self.reset_xy_kp, 1.0) * float(error[0]),
                min(self.reset_xy_kp, 1.0) * float(error[1]),
                min(self.reset_z_kp, 1.0) * float(error[2]),
            ],
            dtype=np.float32,
        )
        speed_norm = float(np.linalg.norm(command_xyz))
        if speed_norm <= 0.0:
            return command_xyz

        distance = float(np.linalg.norm(error))
        near_target_radius = max(3.0 * self._reset_position_tolerance(), self._reset_position_tolerance() + 0.01)
        phase_limit = min(
            self._reset_velocity_limit(),
            self._reset_velocity_limit() * distance / near_target_radius,
        )
        if speed_norm > phase_limit:
            command_xyz *= phase_limit / speed_norm
        return self._clip_reset_velocity(command_xyz)

    def _dominant_reset_axis(self, error: np.ndarray) -> str:
        axis_names = ('x', 'y', 'z')
        return axis_names[int(np.argmax(np.abs(error)))]

    def _apply_reset_min_speed(self, command_xyz: np.ndarray, error: np.ndarray) -> np.ndarray:
        if self.reset_min_speed <= 0.0:
            return command_xyz

        adjusted = command_xyz.copy()
        for idx in range(3):
            if abs(float(error[idx])) <= self.fixed_start_tolerance:
                continue
            if abs(float(adjusted[idx])) < self.reset_min_speed:
                adjusted[idx] = math.copysign(self.reset_min_speed, float(error[idx]))
        return self._clip_reset_velocity(adjusted)

    def _apply_low_altitude_action_guard(self, vz: float) -> float:
        """Prevent policy actions from driving the vehicle into terminal ground contact."""
        self._last_low_altitude_guard_active = False
        if vz >= 0.0:
            return vz
        try:
            current_z = float(self._drone.position[2])
        except Exception:
            return vz
        dynamic_margin = float(self.max_vel) * max(float(self.step_duration), 0.1) * 2.0
        guard_margin = max(float(self.low_altitude_guard_margin), dynamic_margin)
        guard_height = float(self.unsafe_low_altitude_threshold) + guard_margin
        if current_z <= guard_height:
            self._last_low_altitude_guard_active = True
            climb_speed = max(float(self.low_altitude_guard_climb_speed), float(self.reset_min_speed), 0.0)
            return min(float(self.max_vel), climb_speed)
        return vz

    def _compute_vertical_safety_penalty(self, raw_vz: float) -> float:
        """Penalize unsafe altitude and downward commands only near the unsafe band."""
        band = float(self.vertical_safety_band)
        altitude_weight = float(self.vertical_safety_penalty_weight)
        descent_weight = float(self.vertical_descent_penalty_weight)
        if band <= 0.0 or (altitude_weight <= 0.0 and descent_weight <= 0.0):
            return 0.0
        try:
            current_z = float(self._drone.position[2])
        except Exception:
            return 0.0
        band_top = float(self.unsafe_low_altitude_threshold) + band
        if current_z >= band_top:
            return 0.0

        severity = float(np.clip((band_top - current_z) / band, 0.0, 1.0))
        penalty = -altitude_weight * severity
        if raw_vz < 0.0:
            normalized_descent = min(abs(float(raw_vz)) / max(float(self.max_vel), 1e-6), 1.0)
            penalty -= descent_weight * severity * normalized_descent
        return float(penalty)

    def _start_pose_error(self, start_pose: list[float]) -> tuple[float, float, list[float] | None]:
        pose = self._read_reset_pose()
        if pose is None:
            return float('inf'), float('inf'), None
        current_xyz, yaw = pose
        target_xyz = np.array(start_pose[:3], dtype=np.float32)
        distance = float(np.linalg.norm(target_xyz - current_xyz))
        yaw_error = abs(self._wrap_angle(float(start_pose[3]) - yaw))
        return distance, yaw_error, [float(current_xyz[0]), float(current_xyz[1]), float(current_xyz[2]), float(yaw)]

    def _reset_yaw_tolerance(self) -> float:
        """Yaw tolerance for velocity-based reset convergence."""
        return float(self.reset_yaw_tolerance)

    def _reset_service_yaw_tolerance(self) -> float:
        """Yaw tolerance sent to the simulator reset service."""
        if not self.reset_yaw_required:
            return math.pi
        return self._reset_yaw_tolerance()

    def _reset_yaw_converged(self, yaw_error: float) -> bool:
        """Whether yaw may block reset convergence."""
        return (not self.reset_yaw_required) or yaw_error <= self._reset_yaw_tolerance()

    def _reset_yaw_command(self, yaw_error: float) -> float:
        """Yaw command for reset convergence; disabled for position-only curricula."""
        if not self.reset_yaw_required:
            return 0.0
        return float(np.clip(yaw_error, -self.max_yaw_vel, self.max_yaw_vel))

    def _reset_position_tolerance(self) -> float:
        """Position tolerance for velocity-based reset convergence."""
        return float(self.fixed_start_tolerance) + max(0.01, 0.1 * float(self.fixed_start_tolerance))

    def _is_at_start_pose(self, start_pose: list[float]) -> bool:
        distance, yaw_error, _ = self._start_pose_error(start_pose)
        return distance <= self._reset_position_tolerance() and self._reset_yaw_converged(yaw_error)

    def _reset_vertical_recovery_floor(self, target_z: float) -> float:
        """Minimum altitude considered safe while confirming a reset hold."""
        if self.vertical_safety_band > 0.0:
            return min(
                float(target_z),
                float(self.unsafe_low_altitude_threshold) + float(self.vertical_safety_band),
            )
        return min(float(target_z), float(self.reset_ground_recovery_height))

    def _phased_reset_command(
        self,
        error: np.ndarray,
        current_z: float,
        target_z: float,
        yaw_error: float,
        phase_prefix: str = '',
    ) -> tuple[np.ndarray, float, str]:
        """Build a bounded reset command using vertical, XY, Z, then yaw phases."""
        tolerance = self._reset_position_tolerance()
        vertical_recovery_floor = self._reset_vertical_recovery_floor(float(target_z))
        vertical_floor_tolerance = min(0.05, float(self.fixed_start_tolerance))

        if float(current_z) < vertical_recovery_floor - vertical_floor_tolerance:
            recovery_error = np.array([0.0, 0.0, error[2]], dtype=np.float32)
            command_xyz = self._reset_single_axis_command(recovery_error, 2)
            yaw_speed = 0.0
            phase = 'vertical_recovery'
        elif self._should_prioritize_high_z_trim(
            error,
            float(current_z),
            float(target_z),
            vertical_recovery_floor,
            vertical_floor_tolerance,
        ):
            command_xyz = self._reset_high_z_trim_command(error)
            yaw_speed = 0.0
            phase = 'z_high_trim'
        elif math.hypot(float(error[0]), float(error[1])) > tolerance:
            horizontal_axis = 0 if abs(float(error[0])) >= abs(float(error[1])) else 1
            command_xyz = self._reset_lateral_axis_command(error, horizontal_axis)
            yaw_speed = 0.0
            phase = f'xy_axis_{"x" if horizontal_axis == 0 else "y"}'
        elif abs(float(error[2])) > tolerance:
            command_xyz = self._reset_single_axis_command(error, 2)
            yaw_speed = 0.0
            phase = 'z_trim'
        elif float(np.linalg.norm(error)) > tolerance:
            command_xyz = self._reset_near_target_vector_command(error)
            yaw_speed = 0.0
            phase = '3d_trim'
        elif not self._reset_yaw_converged(abs(yaw_error)):
            command_xyz = np.zeros(3, dtype=np.float32)
            yaw_speed = self._reset_yaw_command(yaw_error)
            phase = 'yaw_trim'
        else:
            command_xyz = np.zeros(3, dtype=np.float32)
            yaw_speed = 0.0
            phase = 'hold'

        if phase_prefix:
            phase = f'{phase_prefix}_{phase}'
        return command_xyz, yaw_speed, phase

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
            if self._last_reset_diagnostics.get('reason') not in {
                'post_controller_hold_timeout',
                'post_controller_reacquire_timeout',
            }:
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

    def _velocity_timeout_service_fallback_allowed(self) -> bool:
        """Return whether the last velocity reset failure is eligible for service fallback."""
        reason = str(self._last_reset_diagnostics.get('reason', ''))
        return bool(self.use_service_reset_after_velocity_timeout) and reason in {
            'timeout',
            'post_controller_reacquire_timeout',
            'post_controller_hold_timeout',
        }

    def _apply_start_pose_with_optional_service_fallback(self, start_pose: list[float]) -> bool:
        """Use bounded velocity reset first, then optional service fallback on timeout."""
        try:
            return self._apply_and_confirm_start_pose(start_pose)
        except RuntimeError as velocity_error:
            if not self._velocity_timeout_service_fallback_allowed():
                raise

            velocity_diagnostics = dict(self._last_reset_diagnostics)
            logger.warning(
                'Bounded velocity reset timed out; attempting simulator service fallback. %s',
                self._format_reset_diagnostics(),
            )

            if not self._try_service_backed_reset(start_pose):
                service_diagnostics = dict(self._last_reset_diagnostics)
                service_diagnostics['original_velocity_reset_diagnostics'] = velocity_diagnostics
                service_diagnostics.setdefault('reset_path', 'velocity_timeout_service_fallback')
                self._last_reset_diagnostics = service_diagnostics
                raise RuntimeError(
                    'Velocity reset timed out and simulator service fallback failed; '
                    f'{self._format_reset_diagnostics()}'
                ) from velocity_error

            if not self._is_at_start_pose(start_pose):
                distance, yaw_error, pose = self._start_pose_error(start_pose)
                self._last_reset_diagnostics = {
                    'reason': 'service_fallback_start_pose_not_confirmed',
                    'failure_class': 'service_fallback_start_pose_not_confirmed',
                    'reset_path': 'velocity_timeout_service_fallback',
                    'target_pose': [float(v) for v in start_pose],
                    'final_pose': pose,
                    'position_error': round(distance, 4) if math.isfinite(distance) else 'inf',
                    'yaw_error': round(yaw_error, 4) if math.isfinite(yaw_error) else 'inf',
                    'original_velocity_reset_diagnostics': velocity_diagnostics,
                }
                self._last_reset_service_status = 'service_fallback_start_pose_not_confirmed'
                raise RuntimeError(
                    'Simulator service fallback did not confirm the requested start pose; '
                    f'{self._format_reset_diagnostics()}'
                ) from velocity_error

            if not self._send_reset_speed_command([0.0, 0.0, 0.0], 0.0):
                self._last_reset_diagnostics = {
                    'reason': 'service_fallback_command_path_failed',
                    'failure_class': 'post_reset_command_path_failed',
                    'reset_path': 'velocity_timeout_service_fallback',
                    'target_pose': [float(v) for v in start_pose],
                    'original_velocity_reset_diagnostics': velocity_diagnostics,
                }
                self._last_reset_service_status = 'service_fallback_command_path_failed'
                raise RuntimeError(
                    'Simulator service fallback could not validate the post-reset command path; '
                    f'{self._format_reset_diagnostics()}'
                ) from velocity_error

            service_diagnostics = dict(self._last_reset_diagnostics)
            self._last_reset_diagnostics = {
                'reason': 'service_success_after_velocity_timeout',
                'failure_class': '',
                'reset_path': 'velocity_timeout_service_fallback',
                'target_pose': [float(v) for v in start_pose],
                'service_reset_diagnostics': service_diagnostics,
                'original_velocity_reset_diagnostics': velocity_diagnostics,
                'position_error': service_diagnostics.get('position_error'),
                'yaw_error': service_diagnostics.get('yaw_error'),
            }
            self._last_reset_service_status = 'service_success_after_velocity_timeout'
            self._is_flying = True
            return True

    def _prepare_in_air_velocity_reset(self) -> None:
        """Prepare the bounded velocity fallback path for an already-flying reset."""
        self._recover_low_altitude_hover_before_velocity_reset(
            self._reset_recovery_hover_height()
        )
        if not self._ensure_offboard_after_reset():
            raise RuntimeError(
                'Unable to reassert offboard command path before in-air reset. '
                f'{self._format_reset_diagnostics()}'
            )
        self._reset_velocity_controller()

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
        Recover a valid low-altitude in-air reset with bounded velocity control.

        Reset must not perform an internal land/takeoff cycle here: AS2 can stay
        in LANDING after a previous failure and reject TAKE_OFF, leaving training
        stuck. If altitude is still inside the episode contract and velocity
        control is available, climb vertically to a safe hover height. If the
        vehicle is on/near the ground or control is unavailable, fail fast so the
        caller can restart AS2 instead of looping inside reset.
        """
        pose = self._read_reset_pose()
        if pose is None:
            self._last_reset_diagnostics = {
                'reason': 'hard_reset_required',
                'failure_class': 'state_unreadable',
                'control_available': self._speed_handler is not None,
            }
            raise RuntimeError(
                'Low-altitude reset recovery cannot read simulator state; hard AS2 reset required. '
                f'{self._format_reset_diagnostics()}'
            )

        current_xyz, yaw = pose
        current_z = float(current_xyz[2])
        recovery_threshold = float(self.reset_ground_recovery_height)
        target_height = max(float(hover_height), recovery_threshold)
        start_height_tolerance = self._reset_position_tolerance()
        if current_z >= target_height - start_height_tolerance:
            return True

        terminal_min_height = float(self.height_bounds[0])
        ground_epsilon = 0.05

        if current_z <= ground_epsilon or current_z < terminal_min_height:
            self._last_reset_diagnostics = {
                'reason': 'hard_reset_required',
                'failure_class': 'ground_or_terminal_altitude',
                'target_pose': [float(current_xyz[0]), float(current_xyz[1]), target_height, float(yaw)],
                'final_pose': [float(current_xyz[0]), float(current_xyz[1]), current_z, float(yaw)],
                'control_available': self._speed_handler is not None,
                'state_error': (
                    f'z={current_z:.3f} is on/near ground or below terminal lower bound '
                    f'{terminal_min_height:.3f}'
                ),
            }
            raise RuntimeError(
                'Low-altitude reset recovery found an unrecoverable ground/terminal altitude; '
                f'hard AS2 reset required. {self._format_reset_diagnostics()}'
            )

        if self._speed_handler is None or not hasattr(self._speed_handler, 'send_speed_command_with_yaw_speed'):
            self._last_reset_diagnostics = {
                'reason': 'hard_reset_required',
                'failure_class': 'velocity_control_unavailable',
                'target_pose': [float(current_xyz[0]), float(current_xyz[1]), target_height, float(yaw)],
                'final_pose': [float(current_xyz[0]), float(current_xyz[1]), current_z, float(yaw)],
                'control_available': False,
            }
            raise RuntimeError(
                'Low-altitude reset recovery requires velocity control, but no speed handler is available; '
                f'hard AS2 reset required. {self._format_reset_diagnostics()}'
            )

        logger.warning(
            'Reset started at valid low altitude z=%.3f; climbing to hover %.3fm with bounded velocity recovery',
            current_z,
            target_height,
        )

        started = time.time()
        check_dt = max(0.05, self.step_duration)
        recovery_timeout = max(self.hover_timeout, check_dt)
        last_command = [0.0, 0.0, 0.0, 0.0]
        last_pose = [float(current_xyz[0]), float(current_xyz[1]), current_z, float(yaw)]
        position_error = abs(target_height - current_z)
        initial_position_error = position_error
        best_position_error = position_error

        try:
            while time.time() - started < recovery_timeout:
                pose = self._read_reset_pose()
                if pose is None:
                    time.sleep(check_dt)
                    continue

                current_xyz, yaw = pose
                current_z = float(current_xyz[2])
                last_pose = [float(current_xyz[0]), float(current_xyz[1]), current_z, float(yaw)]
                position_error = abs(target_height - current_z)
                best_position_error = min(best_position_error, position_error)

                if current_z >= target_height - self.fixed_start_tolerance:
                    self._speed_handler.send_speed_command_with_yaw_speed(
                        twist=[0.0, 0.0, 0.0],
                        twist_frame_id='earth',
                        yaw_speed=0.0,
                    )
                    last_command = [0.0, 0.0, 0.0, 0.0]
                    hover_settled = self._wait_for_hover_settle()
                    if hover_settled:
                        recovered_pose = self._read_reset_pose()
                        if recovered_pose is not None:
                            recovered_xyz, recovered_yaw = recovered_pose
                            last_pose = [
                                float(recovered_xyz[0]),
                                float(recovered_xyz[1]),
                                float(recovered_xyz[2]),
                                float(recovered_yaw),
                            ]
                            position_error = abs(target_height - float(recovered_xyz[2]))

                        self._is_flying = True
                        self._last_reset_diagnostics = {
                            'reason': 'low_altitude_velocity_recovery',
                            'elapsed': round(time.time() - started, 3),
                            'target_pose': [float(current_xyz[0]), float(current_xyz[1]), target_height, float(yaw)],
                            'final_pose': last_pose,
                            'position_error': round(position_error, 4),
                            'last_command': last_command,
                            'control_available': True,
                        }
                        return True

                vertical_error = target_height - current_z
                vz = float(np.clip(vertical_error * self.reset_z_kp, -self.reset_max_vel, self.reset_max_vel))
                if abs(vz) < self.reset_min_speed:
                    vz = math.copysign(self.reset_min_speed, vertical_error)
                vz = float(np.clip(vz, -self.reset_max_vel, self.reset_max_vel))
                last_command = [0.0, 0.0, vz, 0.0]
                self._speed_handler.send_speed_command_with_yaw_speed(
                    twist=last_command[:3],
                    twist_frame_id='earth',
                    yaw_speed=0.0,
                )
                time.sleep(check_dt)
        except Exception as e:
            self._last_reset_diagnostics = {
                'reason': 'hard_reset_required',
                'failure_class': 'low_altitude_velocity_recovery_error',
                'elapsed': round(time.time() - started, 3),
                'target_pose': [float(current_xyz[0]), float(current_xyz[1]), target_height, float(yaw)],
                'final_pose': last_pose,
                'position_error': round(position_error, 4),
                'initial_position_error': round(initial_position_error, 4),
                'best_position_error': round(best_position_error, 4),
                'error_reduction': round(initial_position_error - best_position_error, 4),
                'dominant_axis': 'z',
                'last_command': [round(v, 4) for v in last_command],
                'control_available': True,
                'state_error': str(e),
            }
            raise RuntimeError(
                'Low-altitude velocity hover recovery failed; hard AS2 reset required. '
                f'{self._format_reset_diagnostics()}'
            ) from e

        self._last_reset_diagnostics = {
            'reason': 'hard_reset_required',
            'failure_class': 'low_altitude_velocity_recovery_timeout',
            'elapsed': round(time.time() - started, 3),
            'recovery_timeout': round(recovery_timeout, 3),
            'target_pose': [float(current_xyz[0]), float(current_xyz[1]), target_height, float(yaw)],
            'final_pose': last_pose,
            'position_error': round(position_error, 4),
            'initial_position_error': round(initial_position_error, 4),
            'best_position_error': round(best_position_error, 4),
            'error_reduction': round(initial_position_error - best_position_error, 4),
            'dominant_axis': 'z',
            'last_command': [round(v, 4) for v in last_command],
            'control_available': True,
        }
        raise RuntimeError(
            'Low-altitude velocity hover recovery timed out; hard AS2 reset required. '
            f'{self._format_reset_diagnostics()}'
        )

    def _hold_start_pose_after_controller_reset(self, start_pose: list[float], attempt: int) -> bool:
        """
        Keep a bounded pose hold active after recreating the speed controller.

        AS2 can briefly report the requested pose immediately after the velocity
        reset, then drift while the new controller settles. If that happens,
        this method first gives the bounded velocity controller up to the fixed
        start deadline to reacquire XYZ, then starts the shorter hover hold
        deadline only after the pose is inside tolerance.
        """
        if self._speed_handler is None:
            return self._wait_for_hover_settle()

        target_xyz = np.array(start_pose[:3], dtype=np.float32)
        target_yaw = float(start_pose[3])
        vertical_recovery_floor = self._reset_vertical_recovery_floor(float(target_xyz[2]))
        vertical_floor_tolerance = min(0.05, float(self.fixed_start_tolerance))
        required_stable_time = max(0.0, float(self.hover_settle_time))
        started = time.time()
        reacquire_started = started
        check_dt = max(0.05, self.step_duration)
        reacquire_timeout = max(float(self.fixed_start_timeout), check_dt)
        hold_timeout = max(float(self.hover_timeout), check_dt)
        acquired_at: float | None = None
        hold_started_at: float | None = None
        stable_elapsed = 0.0
        settle_started = time.time()
        settle_elapsed = 0.0
        last_command = [0.0, 0.0, 0.0, 0.0]
        last_pose: list[float] | None = None
        last_distance = float('inf')
        last_yaw_error = float('inf')
        initial_position_error = float('inf')
        best_position_error = float('inf')
        dominant_axis = 'unknown'
        phase = 'controller_settle'

        self._send_reset_speed_command([0.0, 0.0, 0.0], 0.0)
        time.sleep(check_dt)
        settle_elapsed = time.time() - settle_started

        while True:
            now = time.time()
            if acquired_at is None:
                if now - reacquire_started >= reacquire_timeout:
                    timeout_reason = 'post_controller_reacquire_timeout'
                    timeout_elapsed = now - reacquire_started
                    break

            pose = self._read_reset_pose()
            if pose is None:
                stable_elapsed = 0.0
                if acquired_at is not None and hold_started_at is not None and now - hold_started_at >= hold_timeout:
                    timeout_reason = 'post_controller_hold_timeout'
                    timeout_elapsed = now - hold_started_at
                    break
                time.sleep(check_dt)
                continue

            current_xyz, yaw = pose
            error = target_xyz - current_xyz
            distance = float(np.linalg.norm(error))
            yaw_error = self._wrap_angle(target_yaw - yaw)
            last_distance = distance
            last_yaw_error = abs(yaw_error)
            last_pose = [float(current_xyz[0]), float(current_xyz[1]), float(current_xyz[2]), float(yaw)]
            if not math.isfinite(initial_position_error):
                initial_position_error = distance
            best_position_error = min(best_position_error, distance)
            dominant_axis = self._dominant_reset_axis(error)
            altitude_ready = float(current_xyz[2]) >= vertical_recovery_floor - vertical_floor_tolerance

            if (
                altitude_ready
                and distance <= self._reset_position_tolerance()
                and self._reset_yaw_converged(abs(yaw_error))
            ):
                if acquired_at is None:
                    acquired_at = now
                    if hold_started_at is None:
                        hold_started_at = now
                command_xyz = np.zeros(3, dtype=np.float32)
                yaw_speed = 0.0
                stable_elapsed += check_dt
                phase = 'hold'
            else:
                if acquired_at is not None:
                    acquired_at = None
                    reacquire_started = now
                    initial_position_error = distance
                    best_position_error = distance
                command_xyz, yaw_speed, phase = self._phased_reset_command(
                    error,
                    float(current_xyz[2]),
                    float(target_xyz[2]),
                    yaw_error,
                )
                stable_elapsed = 0.0

            last_command = [float(command_xyz[0]), float(command_xyz[1]), float(command_xyz[2]), yaw_speed]
            self._send_reset_speed_command(last_command[:3], yaw_speed)

            if stable_elapsed >= required_stable_time:
                time.sleep(check_dt)
                confirmed_pose = self._read_reset_pose()
                if confirmed_pose is None:
                    stable_elapsed = 0.0
                    continue
                confirmed_xyz, confirmed_yaw = confirmed_pose
                confirmed_distance = float(np.linalg.norm(target_xyz - confirmed_xyz))
                confirmed_yaw_error = abs(self._wrap_angle(target_yaw - confirmed_yaw))
                confirmed_altitude_ready = (
                    float(confirmed_xyz[2]) >= vertical_recovery_floor - vertical_floor_tolerance
                )
                if (
                    not confirmed_altitude_ready
                    or confirmed_distance > self._reset_position_tolerance()
                    or not self._reset_yaw_converged(confirmed_yaw_error)
                ):
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
                    'reset_phase': 'hold',
                    'phase': 'hold',
                    'settle_elapsed': round(settle_elapsed, 3),
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

            if acquired_at is not None and hold_started_at is not None and now - hold_started_at >= hold_timeout:
                timeout_reason = 'post_controller_hold_timeout'
                timeout_elapsed = now - hold_started_at
                break

            time.sleep(check_dt)

        self._last_reset_diagnostics = {
            'reason': timeout_reason,
            'attempt': attempt,
            'elapsed': round(timeout_elapsed, 3),
            'reacquire_timeout': round(reacquire_timeout, 3),
            'hold_timeout': round(hold_timeout, 3),
            'reset_phase': phase,
            'phase': phase,
            'settle_elapsed': round(settle_elapsed, 3),
            'target_pose': [float(v) for v in start_pose],
            'final_pose': last_pose,
            'position_error': round(last_distance, 4) if math.isfinite(last_distance) else 'inf',
            'initial_position_error': round(initial_position_error, 4) if math.isfinite(initial_position_error) else 'inf',
            'best_position_error': round(best_position_error, 4) if math.isfinite(best_position_error) else 'inf',
            'error_reduction': (
                round(initial_position_error - best_position_error, 4)
                if math.isfinite(initial_position_error) and math.isfinite(best_position_error)
                else 'inf'
            ),
            'dominant_axis': dominant_axis,
            'yaw_error': round(last_yaw_error, 4) if math.isfinite(last_yaw_error) else 'inf',
            'last_command': [round(v, 4) for v in last_command],
        }
        if timeout_reason == 'post_controller_reacquire_timeout':
            logger.warning(
                'Start pose was not reacquired after controller reset. %s',
                self._format_reset_diagnostics(),
            )
        else:
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
        initial_position_error = float('inf')
        best_position_error = float('inf')
        dominant_axis = 'unknown'
        phase = 'drive_start'

        while time.time() - started < self.fixed_start_timeout:
            pose = self._read_reset_pose()
            if pose is None:
                time.sleep(check_dt)
                continue

            current_xyz, yaw = pose
            error = target_xyz - current_xyz
            distance = float(np.linalg.norm(error))
            yaw_error = self._wrap_angle(target_yaw - yaw)
            if not math.isfinite(initial_position_error):
                initial_position_error = distance
            best_position_error = min(best_position_error, distance)
            dominant_axis = self._dominant_reset_axis(error)

            if distance <= self._reset_position_tolerance() and self._reset_yaw_converged(abs(yaw_error)):
                self._send_reset_speed_command([0.0, 0.0, 0.0], 0.0)
                time.sleep(check_dt)
                confirmed_pose = self._read_reset_pose()
                if confirmed_pose is None:
                    continue
                confirmed_xyz, confirmed_yaw = confirmed_pose
                confirmed_distance = float(np.linalg.norm(target_xyz - confirmed_xyz))
                confirmed_yaw_error = self._wrap_angle(target_yaw - confirmed_yaw)
                if confirmed_distance > self._reset_position_tolerance() or not self._reset_yaw_converged(abs(confirmed_yaw_error)):
                    continue
                self._last_reset_diagnostics = {
                    'reason': 'reached',
                    'elapsed': round(time.time() - started, 3),
                    'reset_phase': 'drive_hold',
                    'phase': 'drive_hold',
                    'target_pose': [float(v) for v in start_pose],
                    'final_pose': [float(confirmed_xyz[0]), float(confirmed_xyz[1]), float(confirmed_xyz[2]), float(confirmed_yaw)],
                    'position_error': round(confirmed_distance, 4),
                    'yaw_error': round(abs(confirmed_yaw_error), 4),
                    'last_command': [0.0, 0.0, 0.0, 0.0],
                }
                return True

            command_xyz, yaw_speed, phase = self._phased_reset_command(
                error,
                float(current_xyz[2]),
                float(target_xyz[2]),
                yaw_error,
                phase_prefix='drive',
            )
            last_command = [float(command_xyz[0]), float(command_xyz[1]), float(command_xyz[2]), yaw_speed]
            self._send_reset_speed_command(last_command[:3], yaw_speed)
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
            best_position_error = min(best_position_error, final_distance)
            dominant_axis = self._dominant_reset_axis(target_xyz - final_xyz)

        self._send_reset_speed_command([0.0, 0.0, 0.0], 0.0)
        self._last_reset_diagnostics = {
            'reason': 'timeout',
            'elapsed': round(time.time() - started, 3),
            'reset_phase': phase,
            'phase': phase,
            'target_pose': [float(v) for v in start_pose],
            'final_pose': (
                [float(final_xyz[0]), float(final_xyz[1]), float(final_xyz[2]), float(final_yaw)]
                if final_xyz is not None else None
            ),
            'position_error': round(final_distance, 4) if math.isfinite(final_distance) else 'inf',
            'initial_position_error': round(initial_position_error, 4) if math.isfinite(initial_position_error) else 'inf',
            'best_position_error': round(best_position_error, 4) if math.isfinite(best_position_error) else 'inf',
            'error_reduction': (
                round(initial_position_error - best_position_error, 4)
                if math.isfinite(initial_position_error) and math.isfinite(best_position_error)
                else 'inf'
            ),
            'dominant_axis': dominant_axis,
            'yaw_error': round(final_yaw_error, 4) if math.isfinite(final_yaw_error) else 'inf',
            'last_command': [round(v, 4) for v in last_command],
        }
        logger.warning('Timed out while driving to fixed_start_pose. %s', self._format_reset_diagnostics())
        return False

    def _send_reset_speed_command(self, twist: list[float], yaw_speed: float) -> bool:
        """Send a reset motion reference, recovering the command path once if rejected."""
        if self._send_speed_command(twist, yaw_speed):
            return True
        if not self._recover_motion_reference_path():
            return False
        return self._send_speed_command(twist, yaw_speed)

    def _send_speed_command(self, twist: list[float], yaw_speed: float) -> bool:
        """Send one speed/yaw-rate motion reference and report API acceptance."""
        if self._speed_handler is None or not hasattr(
            self._speed_handler,
            'send_speed_command_with_yaw_speed',
        ):
            logger.error('Speed motion handler is not available; cannot publish motion reference')
            return False

        try:
            accepted = self._speed_handler.send_speed_command_with_yaw_speed(
                twist=twist,
                twist_frame_id='earth',
                yaw_speed=float(yaw_speed),
            )
            return bool(accepted) if accepted is not None else True
        except Exception as e:
            logger.error(f"Error sending velocity command: {e}")
            return False

    def _ensure_offboard_after_reset(self) -> bool:
        """Reassert AS2 arm/offboard mode after simulator-backed state reset."""
        def _info_flag(flag: str) -> bool:
            try:
                info = getattr(self._drone, 'info', {})
                return bool(info.get(flag, False)) if isinstance(info, dict) else False
            except Exception:
                return False

        arm = getattr(self._drone, 'arm', None)
        if arm is not None:
            try:
                if not bool(arm()) and not _info_flag('armed'):
                    self._last_reset_diagnostics = {
                        'reason': 'post_reset_arm_failed',
                    }
                    logger.error('Unable to reassert arm state after simulator reset')
                    return False
            except Exception as e:
                self._last_reset_diagnostics = {
                    'reason': 'post_reset_arm_failed',
                    'state_error': str(e),
                }
                logger.error('Unable to reassert arm state after simulator reset: %s', e)
                return False

        offboard = getattr(self._drone, 'offboard', None)
        if offboard is None:
            return True
        try:
            return bool(offboard()) or _info_flag('offboard')
        except Exception as e:
            self._last_reset_diagnostics = {
                'reason': 'post_reset_offboard_failed',
                'state_error': str(e),
            }
            logger.error('Unable to reassert offboard mode after simulator reset: %s', e)
            return False

    def _reset_velocity_controller(self) -> bool:
        from as2_motion_reference_handlers.speed_motion import SpeedMotion

        self._speed_handler = SpeedMotion(self._drone)
        return self._send_speed_command([0.0, 0.0, 0.0], 0.0)

    def _recover_motion_reference_path(self) -> bool:
        """Best-effort recovery when AS2 rejects a speed motion reference."""
        offboard_ready = self._ensure_offboard_after_reset()
        controller_ready = self._reset_velocity_controller()
        self._is_flying = self._is_flying or (offboard_ready and controller_ready)
        return offboard_ready and controller_ready

    def _reset_service_name(self) -> str:
        """Return the absolute per-namespace simulator reset service name."""
        namespace = self.drone_namespace.strip().strip('/')
        return f'/{namespace}/platform/reset_simulator_state'

    def _reset_client_ownership_model(self) -> str:
        """Return the reset client ROS node ownership model selected for AS2."""
        return self._RESET_CLIENT_OWNERSHIP_MODEL

    def _ensure_reset_service_client(self) -> bool:
        """Create the simulator reset service client on an auxiliary node."""
        if self._reset_service_client is not None:
            return True

        service_name = self._reset_service_name()
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from as2_platform_multirotor_simulator.srv import ResetSimulatorState

            namespace = self.drone_namespace.strip().strip('/').replace('/', '_')
            node_name = f'{namespace}_reset_client'
            self._reset_aux_node = rclpy.create_node(node_name)
            self._reset_aux_executor = SingleThreadedExecutor()
            self._reset_aux_executor.add_node(self._reset_aux_node)
            self._reset_service_type = ResetSimulatorState
            self._reset_service_client = self._reset_aux_node.create_client(
                ResetSimulatorState,
                service_name,
            )
            return True
        except Exception as e:
            self._last_reset_diagnostics = {
                'reason': 'service_client_create_failed',
                'failure_class': 'service_client_unavailable',
                'service_name': service_name,
                'state_error': str(e),
            }
            self._last_reset_service_status = 'service_client_create_failed'
            self._destroy_reset_client_resources()
            return False

    def _sync_reset_service_status_from_diagnostics(self) -> None:
        """Preserve service-attempt diagnostics when a fallback later succeeds."""
        if self._last_reset_service_status != 'not_attempted':
            return
        reason = str(self._last_reset_diagnostics.get('reason', 'unknown'))
        if self._reset_service_attempted_from_reason(reason):
            self._last_reset_service_status = reason

    def _mark_service_reset_success_if_unreported(self, start_pose: list[float]) -> None:
        """Keep reset diagnostics coherent when tests stub the service helper."""
        if self._last_reset_service_status == 'not_attempted':
            self._last_reset_service_status = 'service_success'
        if str(self._last_reset_diagnostics.get('reason', 'unknown')) == 'unknown':
            self._last_reset_diagnostics = {
                'reason': 'service_success',
                'failure_class': '',
                'target_pose': [float(v) for v in start_pose],
            }

    def _build_reset_service_request(self, start_pose: list[float]):
        """Build a ResetSimulatorState request from an env start pose."""
        if self._reset_service_type is None:
            from as2_platform_multirotor_simulator.srv import ResetSimulatorState
            self._reset_service_type = ResetSimulatorState

        request = self._reset_service_type.Request()
        request.x = float(start_pose[0])
        request.y = float(start_pose[1])
        request.z = float(start_pose[2])
        request.yaw = float(start_pose[3])
        request.position_tolerance = float(self.fixed_start_tolerance)
        request.yaw_tolerance = self._reset_service_yaw_tolerance()
        request.linear_speed_tolerance = float(self.hover_speed_threshold)
        request.angular_speed_tolerance = float(self.hover_speed_threshold)
        return request

    def _stop_motion_before_service_reset(self) -> bool:
        """Drain env-side motion references before simulator state reset.

        Do not call AS2 behavior ``stop()`` here. In live AS2, stopping idle or
        already-completed behavior modules can push the platform FSM to
        EMERGENCY, which leaves speed references accepted but physically ignored.
        The simulator reset service clears simulator-side references atomically,
        and the env recreates/zeros the speed controller after success. Sending
        a pre-service zero speed reference was observed to keep AS2 in a stale
        no-motion path after reset, so pre-service cleanup is intentionally a
        no-op at the env layer.
        """
        return True

    def _spin_reset_future_until_complete(self, future, timeout_sec: float | None = None) -> bool:
        """Wait for a reset service future with a hard timeout."""
        timeout = self.reset_service_timeout if timeout_sec is None else float(timeout_sec)
        started = time.time()
        while time.time() - started < timeout:
            if future.done():
                return True
            if self._reset_aux_executor is not None and hasattr(self._reset_aux_executor, 'spin_once'):
                self._reset_aux_executor.spin_once(timeout_sec=min(0.01, timeout))
            else:
                time.sleep(min(0.01, timeout))
        return bool(future.done())

    def _try_service_backed_reset(self, start_pose: list[float]) -> bool:
        """
        Try the simulator reset service, returning False for fallback-safe failures.

        Until the simulator service is built and validated, any unavailable or
        incomplete service path must fall back to the existing bounded velocity
        reset instead of failing the episode reset outright.
        """
        if not self.use_simulator_reset_service:
            self._last_reset_diagnostics = {
                'reason': 'service_disabled',
                'failure_class': 'service_disabled',
            }
            self._last_reset_service_status = 'service_disabled'
            return False

        self._last_reset_service_status = 'service_attempted'
        service_name = self._reset_service_name()
        if not self._ensure_reset_service_client():
            self._sync_reset_service_status_from_diagnostics()
            return False

        client = self._reset_service_client

        try:
            if not self._stop_motion_before_service_reset():
                return False

            if hasattr(client, 'wait_for_service') and not client.wait_for_service(
                timeout_sec=self.reset_service_timeout,
            ):
                self._last_reset_diagnostics = {
                    'reason': 'service_unavailable',
                    'failure_class': 'service_unavailable',
                    'service_name': service_name,
                    'target_pose': [float(v) for v in start_pose],
                }
                self._last_reset_service_status = 'service_unavailable'
                return False

            if not hasattr(client, 'call_async'):
                self._last_reset_diagnostics = {
                    'reason': 'service_call_unavailable',
                    'failure_class': 'service_call_unavailable',
                    'service_name': service_name,
                    'target_pose': [float(v) for v in start_pose],
                }
                self._last_reset_service_status = 'service_call_unavailable'
                return False

            request = self._build_reset_service_request(start_pose)
            future = client.call_async(request)
            if not self._spin_reset_future_until_complete(future):
                self._last_reset_diagnostics = {
                    'reason': 'service_future_timeout',
                    'failure_class': 'service_timeout',
                    'service_name': service_name,
                    'target_pose': [float(v) for v in start_pose],
                    'elapsed': round(float(self.reset_service_timeout), 3),
                }
                self._last_reset_service_status = 'service_future_timeout'
                return False

            response = future.result()
            self._last_reset_diagnostics = {
                'reason': 'service_success' if bool(response.success) else 'service_rejected',
                'failure_class': '' if bool(response.success) else 'service_response_failure',
                'service_name': service_name,
                'target_pose': [float(v) for v in start_pose],
                'message': getattr(response, 'message', ''),
                'position_error': getattr(response, 'position_error', None),
                'yaw_error': getattr(response, 'yaw_error', None),
                'linear_speed_norm': getattr(response, 'linear_speed_norm', None),
                'angular_speed_norm': getattr(response, 'angular_speed_norm', None),
            }
            self._last_reset_service_status = str(self._last_reset_diagnostics['reason'])
            if bool(response.success):
                offboard_ready = self._ensure_offboard_after_reset()
                controller_ready = self._reset_velocity_controller()
                self._is_flying = offboard_ready and controller_ready
                if not self._is_flying:
                    self._last_reset_diagnostics.update({
                        'reason': 'service_post_reset_command_path_failed',
                        'failure_class': 'post_reset_command_path_failed',
                        'offboard_ready': offboard_ready,
                        'controller_ready': controller_ready,
                    })
                    self._last_reset_service_status = 'service_post_reset_command_path_failed'
                    return False
            return bool(response.success)
        except Exception as e:
            self._last_reset_diagnostics = {
                'reason': 'service_error',
                'failure_class': 'service_exception',
                'service_name': service_name,
                'target_pose': [float(v) for v in start_pose],
                'state_error': str(e),
            }
            self._last_reset_service_status = 'service_error'
            return False

    def _raise_fresh_service_reset_failure(self) -> None:
        """Fail fast when a fresh service-backed reset cannot prove command readiness."""
        raise RuntimeError(
            'Fresh service-backed reset failed before command path readiness; '
            'refusing velocity fallback to avoid accepted-but-ignored actions. '
            f'{self._format_reset_diagnostics()}'
        )

    def _set_platform_flying_after_service_reset(self) -> bool:
        """Advance AS2 platform FSM to FLYING after direct simulator state reset."""
        if self._reset_aux_node is None:
            return False

        client = None
        try:
            from as2_msgs.msg import PlatformStateMachineEvent
            from as2_msgs.srv import SetPlatformStateMachineEvent

            namespace = self.drone_namespace.strip().strip('/')
            service_name = f'/{namespace}/platform/state_machine_event'
            client = self._reset_aux_node.create_client(SetPlatformStateMachineEvent, service_name)
            if hasattr(client, 'wait_for_service') and not client.wait_for_service(
                timeout_sec=self.reset_service_timeout,
            ):
                return False

            ok = True
            for event in (
                PlatformStateMachineEvent.ARM,
                PlatformStateMachineEvent.TAKE_OFF,
                PlatformStateMachineEvent.TOOK_OFF,
            ):
                request = SetPlatformStateMachineEvent.Request()
                request.event.event = event
                future = client.call_async(request)
                if not self._spin_reset_future_until_complete(future):
                    ok = False
                    continue
                result = future.result()
                ok = ok and bool(getattr(result, 'success', False))

            return ok
        except Exception as e:
            logger.warning('Unable to set AS2 platform FSM after simulator reset: %s', e)
            return False
        finally:
            if client is not None and hasattr(self._reset_aux_node, 'destroy_client'):
                try:
                    self._reset_aux_node.destroy_client(client)
                except Exception as e:
                    logger.warning('Unable to destroy AS2 platform FSM client: %s', e)

    def _destroy_reset_client_resources(self) -> None:
        """Destroy only reset-client-owned ROS resources."""
        reset_client = self._reset_service_client
        reset_node = self._reset_aux_node
        if reset_client is not None and reset_node is not None:
            try:
                if hasattr(reset_node, 'destroy_client'):
                    reset_node.destroy_client(reset_client)
            except Exception as e:
                logger.error(f"Error destroying reset service client: {e}")
        self._reset_service_client = None
        self._reset_service_type = None
        if self._reset_aux_executor is not None:
            try:
                if hasattr(self._reset_aux_executor, 'shutdown'):
                    self._reset_aux_executor.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down reset auxiliary executor: {e}")
            self._reset_aux_executor = None
        if self._reset_aux_node is not None:
            try:
                if hasattr(self._reset_aux_node, 'destroy_node'):
                    self._reset_aux_node.destroy_node()
            except Exception as e:
                logger.error(f"Error destroying reset auxiliary node: {e}")
            self._reset_aux_node = None

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
        was_flying_at_reset_start = self._is_flying
        self._last_reset_diagnostics = {}
        use_service_for_this_reset = (
            supports_in_air_reset
            and self.use_simulator_reset_service
            and not was_flying_at_reset_start
        )
        defer_takeoff_for_service_reset = use_service_for_this_reset
        self._last_reset_service_status = 'not_attempted'

        # If already flying, keep the episode reset in-air for training modes
        # that explicitly manage their own safe start pose. Landing/takeoff on
        # every episode is slow and can block AS2 land behavior indefinitely.
        if self._is_flying:
            if supports_in_air_reset:
                logger.info("Keeping drone airborne for velocity reset...")
                self._prepare_in_air_velocity_reset()
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

            if defer_takeoff_for_service_reset:
                logger.info(
                    "Deferring takeoff behavior; simulator reset service will establish flying state"
                )
            else:
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
            if use_service_for_this_reset and self._try_service_backed_reset(self.fixed_start_pose):
                self._mark_service_reset_success_if_unreported(self.fixed_start_pose)
                hover_settled = True
            elif defer_takeoff_for_service_reset and not was_flying_at_reset_start:
                self._raise_fresh_service_reset_failure()
            else:
                self._sync_reset_service_status_from_diagnostics()
                hover_settled = self._wait_for_hover_settle()
                hover_settled = self._apply_start_pose_with_optional_service_fallback(self.fixed_start_pose)
            reset_mode = 'fixed_start_pose'
            start_pose = list(self.fixed_start_pose)

        if self.randomize_hover_start:
            sampled_start, sampled_target, sample_attempts = self._sample_randomized_episode()
            if use_service_for_this_reset and self._try_service_backed_reset(sampled_start):
                self._mark_service_reset_success_if_unreported(sampled_start)
                hover_settled = True
            elif defer_takeoff_for_service_reset and not was_flying_at_reset_start:
                self._raise_fresh_service_reset_failure()
            else:
                self._sync_reset_service_status_from_diagnostics()
                hover_settled = self._wait_for_hover_settle()
                hover_settled = self._apply_start_pose_with_optional_service_fallback(sampled_start)
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
        reset_reason = str(self._last_reset_diagnostics.get('reason', 'unknown'))
        self._last_reset_path = str(self._last_reset_diagnostics.get('reset_path', reset_reason))
        self._sync_reset_service_status_from_diagnostics()
        self._last_reset_service_attempted = (
            self._last_reset_service_status not in {'not_attempted', 'service_disabled'}
            or self._reset_service_attempted_from_reason(reset_reason)
        )
        self._last_reset_method = self._classify_reset_method(supports_in_air_reset, reset_reason)
        self._initialize_episode_monitoring(self._current_position_xyz())
        self._add_episode_monitor_info(info)

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
        vx, vy, raw_vz = float(action[0]), float(action[1]), float(action[2])
        vertical_safety_penalty = self._compute_vertical_safety_penalty(raw_vz)
        vz = raw_vz
        vz = self._apply_low_altitude_action_guard(vz)
        vyaw = float(action[3])

        # Send velocity command with yaw rate via DroneInterface. If AS2 rejects
        # the command path after a simulator reset, reassert offboard and rebuild
        # the speed handler once so the first policy action is not silently lost.
        motion_command_accepted = self._send_speed_command([vx, vy, vz], vyaw)
        motion_command_recovered = False
        if not motion_command_accepted:
            motion_command_recovered = self._recover_motion_reference_path()
            if motion_command_recovered:
                motion_command_accepted = self._send_speed_command([vx, vy, vz], vyaw)

        # Keep the motion reference fresh while the simulator advances. AS2's
        # controller treats motion references as live setpoints, not durable
        # commands, so one-shot publishes can be missed around reset/mode
        # transitions and leave the controller waiting for a reference.
        elapsed = 0.0
        publish_period = min(0.05, self.step_duration) if self.step_duration > 0.0 else 0.0
        while elapsed < self.step_duration:
            sleep_dt = min(publish_period, self.step_duration - elapsed)
            if sleep_dt > 0.0:
                time.sleep(sleep_dt)
                elapsed += sleep_dt
            if elapsed < self.step_duration:
                self._send_speed_command([vx, vy, vz], vyaw)

        # Read new state
        obs = self._get_obs()
        info = self._get_info()
        self._publish_target_marker()
        info['action_sent'] = [vx, vy, vz, vyaw]
        info['raw_action'] = [vx, vy, raw_vz, vyaw]
        info['motion_command_accepted'] = motion_command_accepted
        info['motion_command_recovered'] = motion_command_recovered
        info['low_altitude_guard_active'] = self._last_low_altitude_guard_active
        self._update_episode_monitoring(
            self._current_position_xyz(),
            motion_command_accepted,
        )
        self._add_episode_monitor_info(info)

        # Continuous reward: distance penalty + path-facing shaping
        d_raw, d_norm = self._compute_distance()
        reward_distance = self._compute_reward_distance(d_norm)
        path_facing_reward, speed_xy, path_yaw, path_yaw_error = self._compute_path_facing_term()
        progress_reward = 0.0
        if self._previous_distance is not None and math.isfinite(d_raw):
            progress_reward = self.progress_reward_weight * (self._previous_distance - d_raw)
        self._previous_distance = d_raw if math.isfinite(d_raw) else self._previous_distance
        reward = reward_distance + path_facing_reward + progress_reward + vertical_safety_penalty

        info['reward_distance'] = reward_distance
        info['path_facing_reward'] = path_facing_reward
        info['progress_reward'] = progress_reward
        info['vertical_safety_penalty'] = vertical_safety_penalty
        info['speed_xy'] = speed_xy
        info['path_yaw'] = path_yaw
        info['path_yaw_error'] = path_yaw_error

        terminated = False
        truncated = False

        # --- Terminal conditions ---

        # Out-of-bounds: drone exceeded physical scenario limits.
        if self._is_out_of_bounds():
            terminated = True
            reward = -self.oob_penalty
            info['terminal_reason'] = 'out_of_bounds'
            info['is_success'] = False
            info['is_out_of_bounds'] = True
            info['is_unsafe_low_altitude'] = False

        # Unsafe low altitude: terminate training before physical ground/lower bound.
        elif self._is_unsafe_low_altitude():
            terminated = True
            reward = -self.oob_penalty
            info['terminal_reason'] = 'unsafe_low_altitude'
            info['is_success'] = False
            info['is_out_of_bounds'] = False
            info['is_unsafe_low_altitude'] = True
            if self.fixed_start_pose is not None or self.randomize_hover_start:
                self._recover_low_altitude_hover_before_velocity_reset(
                    self._reset_recovery_hover_height()
                )

        # Success: drone reached the target
        elif d_raw < self.distance_threshold:
            terminated = True
            reward += self.success_reward
            info['terminal_reason'] = 'success'
            info['is_success'] = True
            info['is_out_of_bounds'] = False
            info['is_unsafe_low_altitude'] = False

        # Max steps: episode truncation (time limit)
        if not terminated and self._step_count >= self.max_steps:
            truncated = True
            info['terminal_reason'] = 'max_steps'
            info['is_success'] = False
            info['is_out_of_bounds'] = False
            info['is_unsafe_low_altitude'] = False

        return obs, reward, terminated, truncated, info

    def _is_out_of_bounds(self) -> bool:
        """Check if the drone has exceeded the scenario boundaries."""
        try:
            pose = self._drone.position
            terminal_min_height = float(self.height_bounds[0])
            return any(abs(p) > self.pos_limit for p in pose) or float(pose[2]) < terminal_min_height
        except Exception:
            return False

    def _is_unsafe_low_altitude(self) -> bool:
        """Check if altitude is still physically valid but unsafe for training."""
        try:
            z = float(self._drone.position[2])
            return float(self.height_bounds[0]) <= z < self.unsafe_low_altitude_threshold
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
        """Land the drone and release only this environment's ROS resources."""
        logger.info("Closing environment...")

        self._destroy_reset_client_resources()

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
