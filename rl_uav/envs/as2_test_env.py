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
    - Success terminal bonus: +success_reward (default 20.0) when
      d < distance_threshold, reduced by
      terminal_yaw_penalty_weight * |wrap(yaw - target_yaw)| / pi
      (weight defaults to 0.0, i.e. disabled)
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
        command_publication_interval: float = 0.05,
        min_motion_command_publications: int = 1,
        interface_spin_rate: float = 300.0,
        target_pose: list[float] | None = None,
        distance_threshold: float = 0.5,
        max_steps: int = 500,
        success_reward: float = 20.0,
        oob_penalty: float = 10.0,
        path_facing_weight: float = 0.25,
        progress_reward_weight: float = 0.0,
        terminal_yaw_penalty_weight: float = 0.0,
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
        reset_platform_state_timeout: float = 2.0,
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
        randomization_bounds_margin: float = 0.0,
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
            step_duration: Duration to hold one policy command (seconds).
            command_publication_interval: Period used to refresh live motion
                                          references during the step hold.
            min_motion_command_publications: Minimum sends attempted per step.
            interface_spin_rate: Spin rate (Hz) for the DroneInterface executor
                                 thread. AS2's default (20 Hz) processes at most
                                 one ROS callback per spin, which is slower than
                                 the combined inbound telemetry rate and leaves
                                 `.position` stale for several env steps. Must
                                 exceed the total inbound message rate
                                 (pose + twist + info, ~130 msg/s by default).
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
            terminal_yaw_penalty_weight: Weight of the terminal yaw reduction
                                         applied to the success reward:
                                         success_reward - weight * yaw_err_norm
                                         with yaw_err_norm in [0, 1]. Default
                                         0.0 keeps prior behavior.
            randomization_bounds_margin: Minimum distance (m) kept between
                                         randomized start/target samples and
                                         every scene box face (XY faces and
                                         both height bounds). Default 0.0.
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
        self.command_publication_interval = float(command_publication_interval)
        self.min_motion_command_publications = max(1, int(min_motion_command_publications))
        self.interface_spin_rate = float(interface_spin_rate)
        if not math.isfinite(self.interface_spin_rate) or self.interface_spin_rate <= 0.0:
            raise ValueError('interface_spin_rate must be a positive finite value')
        self.distance_threshold = distance_threshold
        self.max_steps = max_steps
        self.success_reward = success_reward
        self.oob_penalty = oob_penalty
        self.path_facing_weight = path_facing_weight
        self.progress_reward_weight = float(progress_reward_weight)
        self.terminal_yaw_penalty_weight = float(terminal_yaw_penalty_weight)
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
        self.reset_platform_state_timeout = float(reset_platform_state_timeout)
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
        self.randomization_bounds_margin = float(randomization_bounds_margin)
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
        self._episode_action_steps = 0
        self._episode_action_norm_sum = 0.0
        self._episode_max_action_norm = 0.0
        self._episode_xy_action_norm_sum = 0.0
        self._episode_max_xy_action_norm = 0.0
        self._episode_last_action = [0.0, 0.0, 0.0, 0.0]
        self._last_reset_method = 'unknown'
        self._last_reset_path = 'unknown'
        self._last_reset_service_attempted = False
        self._last_reset_service_status = 'not_attempted'
        self._last_low_altitude_guard_active = False
        self._terminal_reset_requires_service = False
        self._last_speed_command_reference_frame = 'unknown'
        self._pose_freshness_last_position: list[float] | None = None
        self._pose_freshness_consecutive_identical_reads = 0
        self._pose_freshness_last_change_time: float | None = None

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

        # spin_rate must exceed the combined inbound telemetry rate: the AS2
        # interface spins a SingleThreadedExecutor with spin_once(timeout=0)
        # once per interval, processing at most one callback per spin. With
        # the default 20 Hz, self_localization/pose (~60 Hz) plus twist and
        # platform info saturate the executor and pose reads go stale for
        # several env steps.
        self._drone = DroneInterface(
            drone_id=self.drone_namespace,
            use_sim_time=self.use_sim_time,
            verbose=self.verbose,
            spin_rate=self.interface_spin_rate,
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
        """Normalized target yaw error: |wrap(yaw - target_yaw)| / pi in [0, 1].

        Used by the terminal success reward when terminal_yaw_penalty_weight > 0.
        """
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
            info['height_bounds'] = list(self.height_bounds)
            info['unsafe_low_altitude_threshold'] = float(self.unsafe_low_altitude_threshold)
            info['low_altitude_guard_height'] = self._low_altitude_guard_height()
            info['speed'] = list(self._drone.speed)
            info['orientation'] = list(self._drone.orientation)
            d, d_norm = self._compute_distance()
            info['distance'] = d
            info['final_distance'] = d
            info['distance_norm'] = d_norm
            info['is_out_of_bounds'] = self._is_out_of_bounds()
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

    def _current_speed_xyz(self) -> list[float] | None:
        try:
            speed = list(getattr(self._drone, 'speed', [0.0, 0.0, 0.0])[:3])
            if len(speed) != 3:
                return None
            xyz = [float(value) for value in speed]
            return xyz if all(math.isfinite(value) for value in xyz) else None
        except Exception:
            return None

    def _reset_pose_freshness(self, position: list[float] | None) -> None:
        """Restart pose freshness tracking at an episode boundary."""
        self._pose_freshness_last_position = list(position) if position is not None else None
        self._pose_freshness_consecutive_identical_reads = 0
        self._pose_freshness_last_change_time = (
            time.monotonic() if position is not None else None
        )

    def _update_pose_freshness(self, position: list[float] | None) -> dict[str, Any]:
        """Track consecutive identical pose samples across env steps.

        A healthy telemetry chain refreshes the observed pose faster than one
        env step (self_localization/pose publishes at ~60 Hz), so consecutive
        step-boundary samples of a moving drone should never be bit-identical.
        Repeated identical reads indicate stale telemetry (e.g. the interface
        executor cannot keep up with the inbound message rate).

        Returns:
            Dict with pose freshness diagnostics for the step info.
        """
        now = time.monotonic()
        if position is None:
            # Unreadable pose is treated as a stale sample; keep the anchor.
            self._pose_freshness_consecutive_identical_reads += 1
            changed = False
        elif self._pose_freshness_last_position is None or (
            list(position) != self._pose_freshness_last_position
        ):
            self._pose_freshness_last_position = list(position)
            self._pose_freshness_last_change_time = now
            self._pose_freshness_consecutive_identical_reads = 0
            changed = True
        else:
            self._pose_freshness_consecutive_identical_reads += 1
            changed = False

        age = (
            now - self._pose_freshness_last_change_time
            if self._pose_freshness_last_change_time is not None
            else float('nan')
        )
        return {
            'pose_freshness_changed': bool(changed),
            'pose_freshness_identical_reads': int(
                self._pose_freshness_consecutive_identical_reads
            ),
            'pose_freshness_age_seconds': float(age),
        }

    def _step_safety_snapshot(self) -> dict[str, Any]:
        """Capture raw step telemetry used to fail closed on safety transitions."""
        position = self._current_position_xyz()
        speed = self._current_speed_xyz()
        altitude = float(position[2]) if position is not None else float('nan')
        speed_z = float(speed[2]) if speed is not None else float('nan')
        out_of_bounds = self._is_out_of_bounds()
        unsafe_low_altitude = (not out_of_bounds) and self._is_unsafe_low_altitude()
        return {
            'position': position,
            'speed': speed,
            'altitude': altitude,
            'speed_z': speed_z,
            'is_out_of_bounds': bool(out_of_bounds),
            'is_unsafe_low_altitude': bool(unsafe_low_altitude),
        }

    def _update_step_safety_minimum(
        self,
        min_altitude: float,
        snapshot: dict[str, Any],
    ) -> float:
        """Track the lowest finite altitude observed during one env step."""
        altitude = snapshot.get('altitude')
        try:
            altitude_value = float(altitude)
        except (TypeError, ValueError):
            return min_altitude
        if not math.isfinite(altitude_value):
            return min_altitude
        if not math.isfinite(min_altitude):
            return altitude_value
        return min(min_altitude, altitude_value)

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
        self._episode_action_steps = 0
        self._episode_action_norm_sum = 0.0
        self._episode_max_action_norm = 0.0
        self._episode_xy_action_norm_sum = 0.0
        self._episode_max_xy_action_norm = 0.0
        self._episode_last_action = [0.0, 0.0, 0.0, 0.0]

    def _update_episode_action_monitoring(self, sent_action: list[float]) -> None:
        vx, vy, vz, vyaw = [float(value) for value in sent_action]
        action_norm = math.sqrt(vx * vx + vy * vy + vz * vz + vyaw * vyaw)
        xy_action_norm = math.hypot(vx, vy)

        self._episode_action_steps += 1
        self._episode_action_norm_sum += action_norm
        self._episode_max_action_norm = max(self._episode_max_action_norm, action_norm)
        self._episode_xy_action_norm_sum += xy_action_norm
        self._episode_max_xy_action_norm = max(self._episode_max_xy_action_norm, xy_action_norm)
        self._episode_last_action = [vx, vy, vz, vyaw]

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
        action_steps = int(self._episode_action_steps)
        mean_action_norm = (
            float(self._episode_action_norm_sum / action_steps)
            if action_steps > 0
            else 0.0
        )
        mean_xy_action_norm = (
            float(self._episode_xy_action_norm_sum / action_steps)
            if action_steps > 0
            else 0.0
        )
        last_vx, last_vy, last_vz, last_vyaw = self._episode_last_action

        info['physical_displacement'] = float(physical_displacement)
        info['path_length'] = float(self._episode_path_length)
        info['max_physical_displacement'] = float(self._episode_max_physical_displacement)
        info['min_altitude'] = float(self._episode_min_altitude)
        info['motion_command_steps'] = command_steps
        info['motion_command_accepted_steps'] = accepted_steps
        info['motion_command_acceptance_rate'] = acceptance_rate
        info['mean_action_norm'] = mean_action_norm
        info['max_action_norm'] = float(self._episode_max_action_norm)
        info['mean_xy_action_norm'] = mean_xy_action_norm
        info['max_xy_action_norm'] = float(self._episode_max_xy_action_norm)
        info['last_action_vx'] = float(last_vx)
        info['last_action_vy'] = float(last_vy)
        info['last_action_vz'] = float(last_vz)
        info['last_action_vyaw'] = float(last_vyaw)
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
        if self.terminal_yaw_penalty_weight < 0.0:
            raise ValueError('terminal_yaw_penalty_weight must be >= 0')
        if self.randomization_bounds_margin < 0.0:
            raise ValueError('randomization_bounds_margin must be >= 0')
        if self.randomization_bounds_margin >= self.scene_bounds_xy:
            raise ValueError(
                'randomization_bounds_margin must be < scene_bounds_xy '
                'so the XY sampling range stays non-empty'
            )
        if z_min + self.randomization_bounds_margin > z_max - self.randomization_bounds_margin:
            raise ValueError(
                'randomization_bounds_margin must leave a non-empty '
                'height sampling range within height_bounds'
            )
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
        if self.reset_platform_state_timeout <= 0.0:
            raise ValueError('reset_platform_state_timeout must be > 0')
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

        # Keep sampled poses away from the box faces: the reset restore
        # tolerance (~0.24 m) can push an edge-sampled start out of bounds,
        # wasting the episode on a pre-step terminal and forcing a service
        # reset. The lower-z margin is measured from the EFFECTIVE floor —
        # the unsafe-altitude threshold when it sits above the physical
        # bound — because a restored start below the unsafe threshold also
        # terminates immediately.
        margin = float(self.randomization_bounds_margin)
        xy_low = -self.scene_bounds_xy + margin
        xy_high = self.scene_bounds_xy - margin
        effective_floor = max(
            float(self.height_bounds[0]),
            float(self.unsafe_low_altitude_threshold),
        )
        z_low = effective_floor + margin
        z_high = self.height_bounds[1] - margin
        if z_low > z_high:
            raise ValueError(
                'randomization_bounds_margin leaves no height sampling range '
                f'above the effective floor {effective_floor} '
                f'(z_low {z_low} > z_high {z_high})'
            )

        for attempt in range(1, self.max_reset_sample_attempts + 1):
            start_yaw = float(self.np_random.uniform(-math.pi, math.pi)) if self.randomize_yaw else 0.0
            target_yaw = float(self.np_random.uniform(-math.pi, math.pi)) if self.randomize_yaw else 0.0
            start_pose = [
                float(self.np_random.uniform(xy_low, xy_high)),
                float(self.np_random.uniform(xy_low, xy_high)),
                float(self.np_random.uniform(z_low, z_high)),
                start_yaw,
            ]
            target_pose = [
                float(self.np_random.uniform(xy_low, xy_high)),
                float(self.np_random.uniform(xy_low, xy_high)),
                float(self.np_random.uniform(z_low, z_high)),
                target_yaw,
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
            'post_reset_platform_reassertion_attempted',
            'post_reset_platform_fsm_ready',
            'post_reset_platform_state',
            'post_reset_platform_control_mode',
            'post_reset_takeoff_ready',
            'post_reset_takeoff_skipped',
            'post_reset_takeoff_skip_reason',
            'post_reset_takeoff_commanded_height',
            'post_reset_takeoff_observed_z',
            'post_reset_start_z',
            'post_reset_offboard_ready',
            'post_reset_controller_ready',
            'post_reset_controller_hold_ready',
            'post_reset_pose_ready',
            'post_reset_pose_observed',
            'post_reset_observed_pose',
            'post_reset_observed_position_error',
            'post_reset_actionability_ready',
            'post_reset_actionability_recovered',
            'post_reset_actionability',
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

    def _low_altitude_guard_height(self) -> float:
        dynamic_margin = float(self.max_vel) * max(float(self.step_duration), 0.1) * 2.0
        guard_margin = max(float(self.low_altitude_guard_margin), dynamic_margin)
        return float(self.unsafe_low_altitude_threshold) + guard_margin

    def _apply_low_altitude_action_guard(self, vz: float) -> float:
        """Prevent policy actions from driving the vehicle into terminal ground contact."""
        self._last_low_altitude_guard_active = False
        if vz > 0.0:
            return vz
        try:
            current_z = float(self._drone.position[2])
        except Exception:
            return vz
        guard_height = self._low_altitude_guard_height()
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

    def _required_service_reset_reason(self) -> str | None:
        """Return why velocity reset must be skipped for the next episode reset."""
        if self._terminal_reset_requires_service:
            return 'terminal_out_of_bounds'
        if self._is_out_of_bounds():
            return 'current_state_out_of_bounds'
        return None

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
        # The service-reset path records platform/service proof fields before
        # this hold runs; carry them into the hold-phase diagnostics rewrite.
        preserved_diagnostics = {
            key: self._last_reset_diagnostics[key]
            for key in (
                'service_name',
                'message',
                'platform_fsm_synced',
                'platform_control_mode_synced',
                'post_reset_platform_fsm_ready',
                'post_reset_platform_state',
                'post_reset_platform_reassertion_attempted',
                'post_reset_platform_control_mode',
            )
            if isinstance(self._last_reset_diagnostics, dict)
            and key in self._last_reset_diagnostics
        }
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
                    **preserved_diagnostics,
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
            **preserved_diagnostics,
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

    def _send_terminal_stop_command(self) -> bool:
        """Best-effort zero motion reference for terminal or truncated steps."""
        return self._send_reset_speed_command([0.0, 0.0, 0.0], 0.0)

    def _post_reset_probe_command(self, start_pose: list[float]) -> tuple[list[float], float]:
        """Build a small post-service-reset velocity probe toward the active target."""
        target_xy = np.array(self._target_pose[:2], dtype=np.float32)
        start_xy = np.array(start_pose[:2], dtype=np.float32)
        direction_xy = target_xy - start_xy
        direction_norm = float(np.linalg.norm(direction_xy))
        if direction_norm <= 1e-6:
            unit_xy = np.array([1.0, 0.0], dtype=np.float32)
        else:
            unit_xy = direction_xy / direction_norm

        probe_speed = min(float(self.max_vel), 0.10)
        return [float(unit_xy[0] * probe_speed), float(unit_xy[1] * probe_speed), 0.0], probe_speed

    def _measure_post_reset_probe_effect(
        self,
        probe_twist: list[float],
        probe_speed: float,
    ) -> dict[str, Any]:
        """Publish a non-zero probe and measure whether it changes the pose."""
        before = self._current_position_xyz()
        if before is None:
            return {
                'accepted': False,
                'actionable': False,
                'reason': 'state_unreadable_before_probe',
            }

        probe_sleep = min(0.05, self.step_duration) if self.step_duration > 0.0 else 0.0
        probe_iterations = 3 if probe_sleep <= 0.0 else max(3, int(math.ceil(0.60 / probe_sleep)))
        accepted = False
        for _ in range(probe_iterations):
            accepted = self._send_speed_command(probe_twist, 0.0) or accepted
            if probe_sleep > 0.0:
                time.sleep(probe_sleep)

        after_command = self._current_position_xyz()
        stop_accepted = False
        stop_iterations = 1 if probe_sleep <= 0.0 else max(3, int(math.ceil(0.20 / probe_sleep)))
        for _ in range(stop_iterations):
            stop_accepted = self._send_reset_speed_command([0.0, 0.0, 0.0], 0.0) or stop_accepted
            if probe_sleep > 0.0:
                time.sleep(probe_sleep)
        after_stop = self._current_position_xyz()
        probe_xy = np.array(probe_twist[:2], dtype=np.float32)
        probe_xy_norm = float(np.linalg.norm(probe_xy))
        unit_xy = probe_xy / probe_xy_norm if probe_xy_norm > 1e-9 else np.array([1.0, 0.0], dtype=np.float32)
        min_forward_delta = min(0.03, max(0.005, float(probe_speed) * 0.12))

        after = after_stop if after_stop is not None else after_command
        observation_timeout = max(0.20, min(float(self.reset_service_timeout), 1.25))
        observation_started = time.time()
        observation_samples = 0
        best_after = after
        best_forward_delta = float('-inf')
        while time.time() - observation_started <= observation_timeout:
            candidate = self._current_position_xyz()
            if candidate is not None:
                observation_samples += 1
                candidate_delta = np.array(candidate, dtype=np.float32) - np.array(before, dtype=np.float32)
                candidate_forward_delta = float(np.dot(candidate_delta[:2], unit_xy))
                if candidate_forward_delta >= best_forward_delta:
                    best_forward_delta = candidate_forward_delta
                    best_after = candidate
                if bool(accepted) and candidate_forward_delta >= min_forward_delta:
                    after = candidate
                    break
            if probe_sleep <= 0.0:
                break
            time.sleep(probe_sleep)
        else:
            after = best_after

        if after is None:
            after = best_after
        if after is None:
            return {
                'accepted': accepted,
                'stop_accepted': stop_accepted,
                'actionable': False,
                'reason': 'state_unreadable_after_probe',
            }

        delta = np.array(after, dtype=np.float32) - np.array(before, dtype=np.float32)
        forward_delta = float(np.dot(delta[:2], unit_xy))
        delta_norm = float(np.linalg.norm(delta))
        actionable = bool(accepted) and forward_delta >= min_forward_delta
        return {
            'accepted': bool(accepted),
            'stop_accepted': bool(stop_accepted),
            'actionable': actionable,
            'reason': 'probe_motion_confirmed' if actionable else 'probe_motion_not_confirmed',
            'probe_twist': [float(v) for v in probe_twist],
            'twist_frame_id': self._last_speed_command_reference_frame,
            'position_before': [float(v) for v in before],
            'position_after': [float(v) for v in after],
            'position_after_command': (
                [float(v) for v in after_command] if after_command is not None else None
            ),
            'position_after_stop': (
                [float(v) for v in after_stop] if after_stop is not None else None
            ),
            'position_delta': [float(v) for v in delta.tolist()],
            'forward_delta': round(forward_delta, 5),
            'delta_norm': round(delta_norm, 5),
            'min_forward_delta': round(min_forward_delta, 5),
            'probe_iterations': probe_iterations,
            'stop_iterations': stop_iterations,
            'observation_samples': observation_samples,
            'observation_timeout': round(observation_timeout, 3),
        }

    def _post_probe_start_pose_tolerance(self) -> float:
        """Residual tolerance after the actionability probe moves."""
        return float(self.fixed_start_tolerance)

    def _post_probe_start_pose_restored(
        self,
        start_pose: list[float],
        tolerance: float,
    ) -> tuple[bool, float, float, list[float] | None, float, list[float] | None]:
        """Confirm that probe cleanup ended near the start pose and is not drifting."""
        distance, yaw_error, pose = self._start_pose_error(start_pose)
        speed = self._current_speed_xyz()
        speed_norm = (
            float(np.linalg.norm(np.array(speed, dtype=np.float32)))
            if speed is not None else float('inf')
        )
        restored = (
            distance <= tolerance
            and self._reset_yaw_converged(yaw_error)
            and speed_norm <= float(self.hover_speed_threshold)
        )
        return bool(restored), distance, yaw_error, pose, speed_norm, speed

    def _restore_start_pose_after_actionability_probe(
        self,
        start_pose: list[float],
        actionability: dict[str, Any],
    ) -> bool:
        """Restore or fail closed after a successful post-reset motion probe."""
        tolerance = self._post_probe_start_pose_tolerance()
        restored_before, distance_before, yaw_before, pose_before, speed_before_norm, speed_before = (
            self._post_probe_start_pose_restored(start_pose, tolerance)
        )
        actionability['start_pose_restore_tolerance'] = round(tolerance, 4)
        actionability['start_pose_error_before_restore'] = (
            round(distance_before, 4) if math.isfinite(distance_before) else 'inf'
        )
        actionability['start_pose_speed_before_restore'] = speed_before
        actionability['start_pose_speed_norm_before_restore'] = (
            round(speed_before_norm, 4) if math.isfinite(speed_before_norm) else 'inf'
        )

        if restored_before:
            actionability.update({
                'start_pose_restored': True,
                'start_pose_restore_method': 'already_within_tolerance',
                'start_pose_error_after_restore': (
                    round(distance_before, 4) if math.isfinite(distance_before) else 'inf'
                ),
                'start_pose_speed_after_restore': speed_before,
                'start_pose_speed_norm_after_restore': (
                    round(speed_before_norm, 4) if math.isfinite(speed_before_norm) else 'inf'
                ),
            })
            return True

        service_diagnostics = dict(self._last_reset_diagnostics)
        restore_diagnostics: dict[str, Any] = {}
        try:
            self._apply_and_confirm_start_pose(start_pose)
            restore_diagnostics = dict(self._last_reset_diagnostics)
        except RuntimeError as exc:
            restore_diagnostics = dict(self._last_reset_diagnostics)
            actionability.update({
                'start_pose_restored': False,
                'start_pose_restore_method': 'velocity_restore_failed',
                'start_pose_restore_error': str(exc),
                'start_pose_restore_diagnostics': restore_diagnostics,
            })
            self._last_reset_diagnostics = service_diagnostics
            return False

        restored, distance_after, yaw_after, pose_after, speed_after_norm, speed_after = (
            self._post_probe_start_pose_restored(start_pose, tolerance)
        )
        actionability.update({
            'start_pose_restored': bool(restored),
            'start_pose_restore_method': 'velocity_restore',
            'start_pose_before_restore': pose_before,
            'start_pose_after_restore': pose_after,
            'start_pose_error_after_restore': (
                round(distance_after, 4) if math.isfinite(distance_after) else 'inf'
            ),
            'start_pose_speed_after_restore': speed_after,
            'start_pose_speed_norm_after_restore': (
                round(speed_after_norm, 4) if math.isfinite(speed_after_norm) else 'inf'
            ),
            'start_pose_restore_diagnostics': restore_diagnostics,
        })
        self._last_reset_diagnostics = service_diagnostics
        return bool(restored)

    def _confirm_post_service_command_path(self, start_pose: list[float]) -> bool:
        """Fail closed unless a non-zero post-reset reference produces motion."""
        probe_twist, probe_speed = self._post_reset_probe_command(start_pose)
        first_probe = self._measure_post_reset_probe_effect(probe_twist, probe_speed)
        if bool(first_probe.get('actionable', False)):
            restored = self._restore_start_pose_after_actionability_probe(start_pose, first_probe)
            self._last_reset_diagnostics['post_reset_actionability'] = first_probe
            self._last_reset_diagnostics['post_reset_actionability_recovered'] = False
            return restored

        recovered = self._recover_motion_reference_path()
        second_probe = self._measure_post_reset_probe_effect(probe_twist, probe_speed) if recovered else {}
        actionability = {
            'first_probe': first_probe,
            'recovery_attempted': True,
            'recovery_ready': bool(recovered),
            'second_probe': second_probe,
        }
        self._last_reset_diagnostics['post_reset_actionability'] = actionability
        self._last_reset_diagnostics['post_reset_actionability_recovered'] = bool(
            recovered and second_probe.get('actionable', False)
        )
        if not bool(recovered and second_probe.get('actionable', False)):
            return False
        return self._restore_start_pose_after_actionability_probe(start_pose, second_probe)

    def _send_speed_command(self, twist: list[float], yaw_speed: float) -> bool:
        """Send one speed/yaw-rate motion reference and report API acceptance."""
        if self._speed_handler is None or not hasattr(
            self._speed_handler,
            'send_speed_command_with_yaw_speed',
        ):
            logger.error('Speed motion handler is not available; cannot publish motion reference')
            return False

        try:
            twist_frame_id = 'earth'
            self._configure_speed_handler_reference_frame(twist_frame_id)
            accepted = self._speed_handler.send_speed_command_with_yaw_speed(
                twist=twist,
                twist_frame_id=twist_frame_id,
                yaw_speed=float(yaw_speed),
            )
            return bool(accepted) if accepted is not None else True
        except Exception as e:
            logger.error(f"Error sending velocity command: {e}")
            return False

    def _configure_speed_handler_reference_frame(self, twist_frame_id: str) -> None:
        """Align AS2 control-mode reference frame with the twist message frame."""
        desired_mode = getattr(self._speed_handler, 'desired_control_mode_', None)
        if desired_mode is None or not hasattr(desired_mode, 'reference_frame'):
            return

        try:
            from as2_msgs.msg import ControlMode
            local_frame = ControlMode.LOCAL_ENU_FRAME
            body_frame = ControlMode.BODY_FLU_FRAME
            undefined_frame = ControlMode.UNDEFINED_FRAME
        except Exception:
            local_frame = 1
            body_frame = 2
            undefined_frame = 0

        frame = str(twist_frame_id).strip()
        if frame == 'earth':
            reference_frame = local_frame
        elif frame.endswith('/base_link') or frame == 'base_link':
            reference_frame = body_frame
        else:
            reference_frame = undefined_frame

        desired_mode.reference_frame = reference_frame
        self._last_speed_command_reference_frame = frame

    def _ensure_offboard_after_reset(self, confirm_platform_state: bool = False) -> bool:
        """Reassert AS2 arm/offboard mode after simulator-backed state reset.

        With ``confirm_platform_state=True``, the reasserted state is not
        trusted: the reported platform info must additionally confirm the AS2
        command gate (connected + armed + offboard + FLYING + settled control
        mode) before the reset is considered command-ready.
        """
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
        if offboard is not None:
            try:
                if not (bool(offboard()) or _info_flag('offboard')):
                    return False
            except Exception as e:
                self._last_reset_diagnostics = {
                    'reason': 'post_reset_offboard_failed',
                    'state_error': str(e),
                }
                logger.error('Unable to reassert offboard mode after simulator reset: %s', e)
                return False

        if not confirm_platform_state:
            return True
        return self._confirm_platform_state_after_service_reset()

    def _platform_info_snapshot(self) -> Optional[dict]:
        """Return a copy of the drone-reported platform info when observable."""
        try:
            info = getattr(self._drone, 'info', None)
        except Exception:
            return None
        return dict(info) if isinstance(info, dict) else None

    @staticmethod
    def _platform_status_flying_value() -> int:
        try:
            from as2_msgs.msg import PlatformStatus
            return int(PlatformStatus.FLYING)
        except Exception:
            return 3

    @staticmethod
    def _platform_control_mode_unset_value() -> int:
        try:
            from as2_msgs.msg import ControlMode
            return int(ControlMode.UNSET)
        except Exception:
            return 0

    def _platform_state_summary(self, snapshot: dict) -> dict[str, Any]:
        """Extract the AS2 command-gate fields from a platform info snapshot."""
        return {
            key: snapshot.get(key)
            for key in ('connected', 'armed', 'offboard', 'state', 'control_mode')
        }

    def _platform_state_command_ready(self, snapshot: dict) -> bool:
        """Mirror the AS2 platform sendCommand() gate on reported info.

        Only fields exposed by the interface are verified; the live
        DroneInterface always reports every gate field, while ROS-free stubs
        may expose a subset.
        """
        checks = {
            'connected': lambda value: bool(value),
            'armed': lambda value: bool(value),
            'offboard': lambda value: bool(value),
            'state': lambda value: value == self._platform_status_flying_value(),
            'control_mode': lambda value: value != self._platform_control_mode_unset_value(),
        }
        for key, is_ready in checks.items():
            if key not in snapshot:
                continue
            try:
                if not is_ready(snapshot[key]):
                    return False
            except Exception:
                return False
        return True

    def _confirm_platform_state_after_service_reset(self) -> bool:
        """Confirm the platform reports a command-ready state after reset."""
        snapshot = self._platform_info_snapshot()
        if snapshot is None:
            # Platform info is not observable through this drone interface;
            # keep the service-result-based acceptance for ROS-free stubs.
            self._last_reset_diagnostics.update({
                'post_reset_platform_fsm_ready': None,
                'post_reset_platform_state': 'unobservable',
            })
            return True

        timeout = max(float(self.reset_platform_state_timeout), 0.05)
        check_dt = 0.05
        started = time.time()
        reassertion_attempted = False
        while True:
            # A snapshot that turns unobservable mid-loop must not confirm
            # readiness vacuously; keep polling until timeout instead.
            snapshot = self._platform_info_snapshot()
            if snapshot is not None and self._platform_state_command_ready(snapshot):
                self._last_reset_diagnostics.update({
                    'post_reset_platform_fsm_ready': True,
                    'post_reset_platform_reassertion_attempted': reassertion_attempted,
                    'post_reset_platform_state': self._platform_state_summary(snapshot),
                })
                return True
            if not reassertion_attempted:
                reassertion_attempted = True
                self._set_platform_flying_after_service_reset()
                continue
            if time.time() - started >= timeout:
                break
            time.sleep(check_dt)

        self._last_reset_diagnostics.update({
            'post_reset_platform_fsm_ready': False,
            'post_reset_platform_reassertion_attempted': reassertion_attempted,
            'post_reset_platform_state': (
                self._platform_state_summary(snapshot)
                if snapshot is not None else 'unobservable'
            ),
        })
        logger.error(
            'Platform state is not command-ready after simulator reset: %s',
            self._last_reset_diagnostics['post_reset_platform_state'],
        )
        return False

    def _reset_velocity_controller(self) -> bool:
        from as2_motion_reference_handlers.speed_motion import SpeedMotion

        self._speed_handler = SpeedMotion(self._drone)
        return self._send_speed_command([0.0, 0.0, 0.0], 0.0)

    def _refresh_controller_mode_after_service_reset(self) -> bool:
        """Force a non-speed mode before re-entering speed mode after simulator reset."""
        if self._reset_aux_node is None:
            return True

        client = None
        try:
            from as2_msgs.msg import ControlMode
            from as2_msgs.srv import SetControlMode

            namespace = self.drone_namespace.strip().strip('/')
            service_name = f'/{namespace}/controller/set_control_mode'
            client = self._reset_aux_node.create_client(SetControlMode, service_name)
            if hasattr(client, 'wait_for_service') and not client.wait_for_service(
                timeout_sec=self.reset_service_timeout,
            ):
                self._last_reset_diagnostics.update({
                    'post_reset_mode_refresh_ready': False,
                    'post_reset_mode_refresh_reason': 'service_unavailable',
                })
                return False

            request = SetControlMode.Request()
            request.control_mode.control_mode = ControlMode.POSITION
            request.control_mode.yaw_mode = ControlMode.YAW_ANGLE
            request.control_mode.reference_frame = ControlMode.LOCAL_ENU_FRAME
            future = client.call_async(request)
            if not self._spin_reset_future_until_complete(future):
                self._last_reset_diagnostics.update({
                    'post_reset_mode_refresh_ready': False,
                    'post_reset_mode_refresh_reason': 'timeout',
                })
                return False

            result = future.result()
            success = bool(getattr(result, 'success', False))
            self._last_reset_diagnostics.update({
                'post_reset_mode_refresh_ready': success,
                'post_reset_mode_refresh_reason': 'success' if success else 'service_rejected',
            })
            if success:
                snapshot = self._platform_info_snapshot()
                if snapshot is not None and 'control_mode' in snapshot:
                    self._last_reset_diagnostics['post_reset_platform_control_mode'] = (
                        snapshot.get('control_mode')
                    )
            return success
        except Exception as e:
            self._last_reset_diagnostics.update({
                'post_reset_mode_refresh_ready': False,
                'post_reset_mode_refresh_reason': 'exception',
                'post_reset_mode_refresh_error': str(e),
            })
            logger.error('Unable to refresh controller mode after simulator reset: %s', e)
            return False
        finally:
            if client is not None and hasattr(self._reset_aux_node, 'destroy_client'):
                try:
                    self._reset_aux_node.destroy_client(client)
                except Exception as e:
                    logger.warning('Unable to destroy controller mode refresh client: %s', e)

    def _takeoff_after_service_reset(self, start_pose: list[float]) -> bool:
        """Mark a service-reset pose as flying without issuing a conflicting takeoff."""
        if self._is_flying:
            return True
        start_z = float(start_pose[2])
        pose_ready_before_takeoff = self._wait_until_service_reset_pose_observed(start_pose)
        if pose_ready_before_takeoff:
            _, _, observed_pose = self._start_pose_error(start_pose)
            observed_z = observed_pose[2] if observed_pose is not None and len(observed_pose) >= 3 else None
            self._last_reset_diagnostics.update({
                'post_reset_takeoff_skipped': True,
                'post_reset_takeoff_skip_reason': 'service_reset_pose_already_observed',
                'post_reset_start_z': round(start_z, 4),
                'post_reset_takeoff_commanded_height': None,
                'post_reset_takeoff_observed_z': (
                    round(float(observed_z), 4) if observed_z is not None else None
                ),
                'post_reset_takeoff_ready_from_pose': True,
            })
            self._is_flying = True
            return True

        takeoff = getattr(self._drone, 'takeoff', None)
        if takeoff is None or not callable(takeoff):
            self._last_reset_diagnostics.update({
                'post_reset_takeoff_skipped': True,
                'post_reset_takeoff_skip_reason': 'takeoff_api_unavailable',
                'post_reset_start_z': round(start_z, 4),
                'post_reset_takeoff_commanded_height': None,
            })
            return True
        try:
            self._last_reset_diagnostics.update({
                'post_reset_takeoff_skipped': False,
                'post_reset_start_z': round(start_z, 4),
                'post_reset_takeoff_commanded_height': round(start_z, 4),
            })
            success = bool(takeoff(height=start_z, speed=self.takeoff_speed))
            if success:
                success = self._wait_until_service_reset_pose_observed(start_pose)
            pose_ready_after_rejected_takeoff = (
                (not success) and self._wait_until_service_reset_pose_observed(start_pose)
            )
            if pose_ready_after_rejected_takeoff:
                self._last_reset_diagnostics.update({
                    'post_reset_takeoff_goal_accepted': False,
                    'post_reset_takeoff_ready_from_pose': True,
                })
                success = True
            self._is_flying = success
            return success
        except Exception as e:
            self._last_reset_diagnostics.update({
                'post_reset_takeoff_error': str(e),
            })
            logger.error('Unable to run takeoff after simulator reset: %s', e)
            return False

    def _hold_service_reset_start_pose(self, start_pose: list[float]) -> bool:
        """Actively hold the requested start pose after recreating speed control."""
        if self._speed_handler is None:
            return self._wait_until_service_reset_pose_observed(start_pose)
        return self._hold_start_pose_after_controller_reset(start_pose, 1)

    def _wait_until_service_reset_pose_observed(self, start_pose: list[float]) -> bool:
        """Wait until DroneInterface telemetry reflects a settled service reset pose."""
        started = time.time()
        timeout = max(float(self.reset_service_timeout), 0.5)
        check_dt = max(0.05, min(float(self.step_duration), 0.1))
        required_stable_time = min(0.30, max(0.15, check_dt))
        stable_elapsed = 0.0
        best_distance = float('inf')
        last_distance = float('inf')
        last_yaw_error = float('inf')
        last_pose = None
        last_speed = None
        last_speed_norm = float('inf')

        while time.time() - started < timeout:
            last_distance, last_yaw_error, last_pose = self._start_pose_error(start_pose)
            last_speed = self._current_speed_xyz()
            last_speed_norm = (
                float(np.linalg.norm(np.array(last_speed, dtype=np.float32)))
                if last_speed is not None else float('inf')
            )
            if math.isfinite(last_distance):
                best_distance = min(best_distance, last_distance)
            pose_ready = (
                last_distance <= self._reset_position_tolerance()
                and self._reset_yaw_converged(last_yaw_error)
            )
            speed_ready = last_speed_norm <= float(self.hover_speed_threshold)
            if pose_ready and speed_ready:
                stable_elapsed += check_dt
            else:
                stable_elapsed = 0.0
            if stable_elapsed >= required_stable_time:
                self._last_reset_diagnostics.update({
                    'post_reset_pose_observed': True,
                    'post_reset_pose_observe_elapsed': round(time.time() - started, 3),
                    'post_reset_pose_stable_elapsed': round(stable_elapsed, 3),
                    'post_reset_observed_pose': last_pose,
                    'post_reset_observed_position_error': round(last_distance, 4),
                    'post_reset_observed_yaw_error': round(last_yaw_error, 4),
                    'post_reset_observed_speed': last_speed,
                    'post_reset_observed_speed_norm': round(last_speed_norm, 4),
                })
                return True
            time.sleep(check_dt)

        self._last_reset_diagnostics.update({
            'post_reset_pose_observed': False,
            'post_reset_pose_observe_elapsed': round(time.time() - started, 3),
            'post_reset_pose_stable_elapsed': round(stable_elapsed, 3),
            'post_reset_observed_pose': last_pose,
            'post_reset_observed_position_error': (
                round(last_distance, 4) if math.isfinite(last_distance) else 'inf'
            ),
            'post_reset_best_observed_position_error': (
                round(best_distance, 4) if math.isfinite(best_distance) else 'inf'
            ),
            'post_reset_observed_yaw_error': (
                round(last_yaw_error, 4) if math.isfinite(last_yaw_error) else 'inf'
            ),
            'post_reset_observed_speed': last_speed,
            'post_reset_observed_speed_norm': (
                round(last_speed_norm, 4) if math.isfinite(last_speed_norm) else 'inf'
            ),
        })
        return False

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
                'platform_fsm_synced': getattr(response, 'platform_fsm_synced', None),
                'platform_control_mode_synced': getattr(
                    response, 'platform_control_mode_synced', None),
            }
            self._last_reset_service_status = str(self._last_reset_diagnostics['reason'])
            if bool(response.success):
                takeoff_ready = self._takeoff_after_service_reset(start_pose)
                offboard_ready = self._ensure_offboard_after_reset(confirm_platform_state=True)
                mode_refresh_ready = bool(offboard_ready) and self._refresh_controller_mode_after_service_reset()
                controller_ready = bool(mode_refresh_ready) and self._reset_velocity_controller()
                controller_hold_ready = bool(takeoff_ready and offboard_ready and controller_ready) and self._hold_service_reset_start_pose(start_pose)
                pose_observed = bool(takeoff_ready and offboard_ready and controller_hold_ready) and self._wait_until_service_reset_pose_observed(start_pose)
                actionability_ready = bool(pose_observed) and self._confirm_post_service_command_path(start_pose)
                self._last_reset_diagnostics.update({
                    'post_reset_takeoff_ready': bool(takeoff_ready),
                    'post_reset_offboard_ready': bool(offboard_ready),
                    'post_reset_mode_refresh_ready': bool(mode_refresh_ready),
                    'post_reset_controller_ready': bool(controller_ready),
                    'post_reset_controller_hold_ready': bool(controller_hold_ready),
                    'post_reset_pose_ready': bool(pose_observed),
                    'post_reset_actionability_ready': bool(actionability_ready),
                })
                self._is_flying = offboard_ready and controller_ready and actionability_ready
                if not self._is_flying:
                    self._last_reset_diagnostics.update({
                        'reason': 'service_post_reset_command_path_failed',
                        'failure_class': 'post_reset_command_path_failed',
                        'takeoff_ready': takeoff_ready,
                        'offboard_ready': offboard_ready,
                        'mode_refresh_ready': mode_refresh_ready,
                        'controller_ready': controller_ready,
                        'controller_hold_ready': controller_hold_ready,
                        'pose_observed': pose_observed,
                        'actionability_ready': actionability_ready,
                    })
                    self._last_reset_service_status = 'service_post_reset_command_path_failed'
                    return False
                self._last_reset_diagnostics['reason'] = 'service_success'
                self._last_reset_diagnostics['failure_class'] = ''
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

    def _raise_required_service_reset_failure(self, reason: str) -> None:
        """Fail fast when unsafe state requires simulator service reset and it fails."""
        raise RuntimeError(
            'Simulator service reset is required after unsafe/out-of-bounds state; '
            'refusing velocity reset flyback. '
            f'required_reason={reason}. {self._format_reset_diagnostics()}'
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
        required_service_reset_reason = (
            self._required_service_reset_reason()
            if supports_in_air_reset and self.use_simulator_reset_service and was_flying_at_reset_start
            else None
        )
        use_service_for_this_reset = (
            supports_in_air_reset
            and self.use_simulator_reset_service
            and (not was_flying_at_reset_start or required_service_reset_reason is not None)
        )
        self._last_reset_service_status = 'not_attempted'

        # If already flying, keep the episode reset in-air for training modes
        # that explicitly manage their own safe start pose. Landing/takeoff on
        # every episode is slow and can block AS2 land behavior indefinitely.
        if self._is_flying:
            if supports_in_air_reset:
                if use_service_for_this_reset:
                    logger.warning(
                        "Skipping velocity reset; simulator service reset required (%s)",
                        required_service_reset_reason or 'fresh_reset',
                    )
                    self._send_terminal_stop_command()
                else:
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
            elif use_service_for_this_reset and required_service_reset_reason is not None:
                self._raise_required_service_reset_failure(required_service_reset_reason)
            elif use_service_for_this_reset and not was_flying_at_reset_start:
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
            elif use_service_for_this_reset and required_service_reset_reason is not None:
                self._raise_required_service_reset_failure(required_service_reset_reason)
            elif use_service_for_this_reset and not was_flying_at_reset_start:
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
        self._terminal_reset_requires_service = False
        reset_position = self._current_position_xyz()
        self._initialize_episode_monitoring(reset_position)
        self._reset_pose_freshness(reset_position)
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
        pre_step_snapshot = self._step_safety_snapshot()
        step_min_altitude = self._update_step_safety_minimum(
            float('nan'),
            pre_step_snapshot,
        )

        pre_step_out_of_bounds = bool(pre_step_snapshot['is_out_of_bounds'])
        pre_step_unsafe_low_altitude = bool(pre_step_snapshot['is_unsafe_low_altitude'])
        if pre_step_out_of_bounds or pre_step_unsafe_low_altitude:
            obs = self._get_obs()
            info = self._get_info()
            terminal_reason = 'out_of_bounds' if pre_step_out_of_bounds else 'unsafe_low_altitude'
            info['terminal_reason'] = terminal_reason
            info['is_success'] = False
            info['is_out_of_bounds'] = bool(pre_step_out_of_bounds)
            info['is_unsafe_low_altitude'] = bool(pre_step_unsafe_low_altitude)
            info['motion_command_accepted'] = False
            info['motion_command_recovered'] = False
            info['motion_command_publication_count'] = 0
            info['motion_command_accepted_publication_count'] = 0
            info['low_altitude_guard_active'] = False
            # No action is processed on this early return; the keys must still
            # exist because SB3 Monitor reads every monitor_info_keywords entry
            # at episode end.
            info['vertical_safety_penalty'] = 0.0
            info['terminal_yaw_error'] = self._compute_yaw_error()
            info['terminal_yaw_penalty'] = 0.0
            info['step_altitude_before'] = pre_step_snapshot['altitude']
            info['step_altitude_after'] = pre_step_snapshot['altitude']
            info['step_min_altitude'] = step_min_altitude
            info['step_speed_z_before'] = pre_step_snapshot['speed_z']
            info['step_speed_z_after'] = pre_step_snapshot['speed_z']
            info['step_safety_checked_after_publication'] = False
            info['step_safety_terminal'] = True
            info.update(self._update_pose_freshness(pre_step_snapshot['position']))
            self._terminal_reset_requires_service = True
            if pre_step_unsafe_low_altitude and (self.fixed_start_pose is not None or self.randomize_hover_start):
                try:
                    self._recover_low_altitude_hover_before_velocity_reset(
                        self._reset_recovery_hover_height()
                    )
                except Exception as exc:
                    logger.warning(
                        'Low-altitude terminal recovery failed before reset; '
                        'preserving terminal stop and service reset requirement: %s',
                        exc,
                    )
            info['terminal_stop_command_accepted'] = self._send_terminal_stop_command()
            return obs, -self.oob_penalty, True, False, info

        # A non-finite action (diverged policy) would poison the simulator
        # velocity reference and silently freeze telemetry; np.clip propagates
        # NaN, so sanitize before clipping.
        action = np.asarray(action, dtype=np.float32)
        if not np.all(np.isfinite(action)):
            logger.warning(
                'Non-finite action received; replacing with zero command: %s',
                action,
            )
            action = np.where(np.isfinite(action), action, 0.0).astype(np.float32)

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
        command_twist = [vx, vy, vz]
        publication_count = 1
        accepted_publication_count = 0
        motion_command_accepted = self._send_speed_command(command_twist, vyaw)
        if motion_command_accepted:
            accepted_publication_count += 1
        motion_command_recovered = False
        if not motion_command_accepted:
            motion_command_recovered = self._recover_motion_reference_path()
            if motion_command_recovered:
                publication_count += 1
                motion_command_accepted = self._send_speed_command(command_twist, vyaw)
                if motion_command_accepted:
                    accepted_publication_count += 1

        post_publication_snapshot = self._step_safety_snapshot()
        step_min_altitude = self._update_step_safety_minimum(
            step_min_altitude,
            post_publication_snapshot,
        )
        safety_terminal_reason = None
        if post_publication_snapshot['is_out_of_bounds']:
            safety_terminal_reason = 'out_of_bounds'
        elif post_publication_snapshot['is_unsafe_low_altitude']:
            safety_terminal_reason = 'unsafe_low_altitude'

        # Keep the motion reference fresh while the simulator advances. AS2's
        # controller treats motion references as live setpoints, not durable
        # commands, so one-shot publishes can be missed around reset/mode
        # transitions and leave the controller waiting for a reference.
        elapsed = 0.0
        hold_duration = max(0.0, float(self.step_duration))
        publish_period = min(
            max(float(self.command_publication_interval), 1e-3),
            hold_duration,
        ) if hold_duration > 0.0 else 0.0
        while elapsed < hold_duration and safety_terminal_reason is None:
            sleep_dt = min(publish_period, hold_duration - elapsed)
            if sleep_dt > 0.0:
                time.sleep(sleep_dt)
                elapsed += sleep_dt
                post_publication_snapshot = self._step_safety_snapshot()
                step_min_altitude = self._update_step_safety_minimum(
                    step_min_altitude,
                    post_publication_snapshot,
                )
                if post_publication_snapshot['is_out_of_bounds']:
                    safety_terminal_reason = 'out_of_bounds'
                    break
                if post_publication_snapshot['is_unsafe_low_altitude']:
                    safety_terminal_reason = 'unsafe_low_altitude'
                    break
            if elapsed < hold_duration:
                publication_count += 1
                repeated_command_accepted = self._send_speed_command(command_twist, vyaw)
                if repeated_command_accepted:
                    accepted_publication_count += 1
                    motion_command_accepted = True
                post_publication_snapshot = self._step_safety_snapshot()
                step_min_altitude = self._update_step_safety_minimum(
                    step_min_altitude,
                    post_publication_snapshot,
                )
                if post_publication_snapshot['is_out_of_bounds']:
                    safety_terminal_reason = 'out_of_bounds'
                    break
                if post_publication_snapshot['is_unsafe_low_altitude']:
                    safety_terminal_reason = 'unsafe_low_altitude'
                    break

        while publication_count < self.min_motion_command_publications and safety_terminal_reason is None:
            publication_count += 1
            repeated_command_accepted = self._send_speed_command(command_twist, vyaw)
            if repeated_command_accepted:
                accepted_publication_count += 1
                motion_command_accepted = True
            post_publication_snapshot = self._step_safety_snapshot()
            step_min_altitude = self._update_step_safety_minimum(
                step_min_altitude,
                post_publication_snapshot,
            )
            if post_publication_snapshot['is_out_of_bounds']:
                safety_terminal_reason = 'out_of_bounds'
                break
            if post_publication_snapshot['is_unsafe_low_altitude']:
                safety_terminal_reason = 'unsafe_low_altitude'
                break

        post_step_snapshot = self._step_safety_snapshot()
        step_min_altitude = self._update_step_safety_minimum(
            step_min_altitude,
            post_step_snapshot,
        )
        if post_step_snapshot['is_out_of_bounds']:
            safety_terminal_reason = 'out_of_bounds'
        elif post_step_snapshot['is_unsafe_low_altitude'] and safety_terminal_reason != 'out_of_bounds':
            safety_terminal_reason = 'unsafe_low_altitude'

        # Read new state
        obs = self._get_obs()
        info = self._get_info()
        self._publish_target_marker()
        info['action_sent'] = [vx, vy, vz, vyaw]
        info['raw_action'] = [vx, vy, raw_vz, vyaw]
        info['motion_command_accepted'] = motion_command_accepted
        info['motion_command_recovered'] = motion_command_recovered
        info['motion_command_publication_count'] = int(publication_count)
        info['motion_command_accepted_publication_count'] = int(accepted_publication_count)
        info['low_altitude_guard_active'] = self._last_low_altitude_guard_active
        info['step_altitude_before'] = pre_step_snapshot['altitude']
        info['step_altitude_after'] = post_step_snapshot['altitude']
        info['step_min_altitude'] = step_min_altitude
        info['step_speed_z_before'] = pre_step_snapshot['speed_z']
        info['step_speed_z_after'] = post_step_snapshot['speed_z']
        info['step_safety_checked_after_publication'] = True
        info['step_safety_terminal'] = safety_terminal_reason in {'out_of_bounds', 'unsafe_low_altitude'}
        info.update(self._update_pose_freshness(post_step_snapshot['position']))
        self._update_episode_action_monitoring(info['action_sent'])
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
        # Terminal yaw diagnostics must exist on every episode-ending path
        # because SB3 Monitor reads all monitor_info_keywords at episode end.
        # The penalty is only applied (and overwritten) on the success branch.
        info['terminal_yaw_error'] = self._compute_yaw_error()
        info['terminal_yaw_penalty'] = 0.0

        terminated = False
        truncated = False

        # --- Terminal conditions ---

        # Out-of-bounds: drone exceeded physical scenario limits.
        if safety_terminal_reason == 'out_of_bounds' or self._is_out_of_bounds():
            terminated = True
            reward = -self.oob_penalty
            info['terminal_reason'] = 'out_of_bounds'
            info['is_success'] = False
            info['is_out_of_bounds'] = True
            info['is_unsafe_low_altitude'] = False
            self._terminal_reset_requires_service = True

        # Unsafe low altitude: terminate training before physical ground/lower bound.
        elif safety_terminal_reason == 'unsafe_low_altitude' or self._is_unsafe_low_altitude():
            terminated = True
            reward = -self.oob_penalty
            info['terminal_reason'] = 'unsafe_low_altitude'
            info['is_success'] = False
            info['is_out_of_bounds'] = False
            info['is_unsafe_low_altitude'] = True
            self._terminal_reset_requires_service = True
            if self.fixed_start_pose is not None or self.randomize_hover_start:
                try:
                    self._recover_low_altitude_hover_before_velocity_reset(
                        self._reset_recovery_hover_height()
                    )
                except Exception as exc:
                    logger.warning(
                        'Low-altitude terminal recovery failed before reset; '
                        'preserving terminal stop and service reset requirement: %s',
                        exc,
                    )

        # Success: drone reached the target. The terminal reward includes a
        # yaw-alignment reduction: success_reward - w * |wrap(yaw - tyaw)|/pi.
        # With w <= success_reward the worst-case arrival still nets a
        # positive terminal bonus, so arrival is never disincentivized.
        elif d_raw < self.distance_threshold:
            terminated = True
            terminal_yaw_error = self._compute_yaw_error()
            terminal_yaw_penalty = self.terminal_yaw_penalty_weight * terminal_yaw_error
            reward += self.success_reward - terminal_yaw_penalty
            info['terminal_reason'] = 'success'
            info['is_success'] = True
            info['is_out_of_bounds'] = False
            info['is_unsafe_low_altitude'] = False
            info['terminal_yaw_error'] = terminal_yaw_error
            info['terminal_yaw_penalty'] = terminal_yaw_penalty

        # Max steps: episode truncation (time limit)
        if not terminated and self._step_count >= self.max_steps:
            truncated = True
            info['terminal_reason'] = 'max_steps'
            info['is_success'] = False
            info['is_out_of_bounds'] = False
            info['is_unsafe_low_altitude'] = False

        if terminated or truncated:
            info['terminal_stop_command_accepted'] = self._send_terminal_stop_command()

        return obs, reward, terminated, truncated, info

    def _is_out_of_bounds(self) -> bool:
        """Check if the drone has exceeded the scenario boundaries."""
        try:
            pose = self._drone.position
            if len(pose) < 3:
                return True
            if not all(math.isfinite(float(p)) for p in pose[:3]):
                return True
            x, y, z = [float(value) for value in pose[:3]]
            z_min, z_max = self.height_bounds
            return (
                abs(x) > float(self.scene_bounds_xy)
                or abs(y) > float(self.scene_bounds_xy)
                or abs(x) > float(self.pos_limit)
                or abs(y) > float(self.pos_limit)
                or z < float(z_min)
                or z > float(z_max)
                or abs(z) > float(self.pos_limit)
            )
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
