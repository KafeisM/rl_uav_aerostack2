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
        speed_deadband: float = 0.05,
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
        self.speed_deadband = speed_deadband
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
        self._is_flying = False
        self._step_count = 0

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
        logger.info(f"DroneInterface created for '{self.drone_namespace}'")

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
        }

        try:
            info['position'] = list(self._drone.position)
            info['speed'] = list(self._drone.speed)
            info['orientation'] = list(self._drone.orientation)
            d, d_norm = self._compute_distance()
            info['distance'] = d
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
        if self.pos_limit < self.scene_bounds_xy:
            raise ValueError(
                'pos_limit must be >= scene_bounds_xy to preserve normalization budget'
            )
        if self.pos_limit < max(abs(z_min), abs(z_max)):
            raise ValueError(
                'pos_limit must be >= max(abs(height_bounds)) to preserve normalization budget'
            )

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
        if hasattr(self._drone, 'go_to'):  # fallback motion-based reposition
            self._drone.go_to(start_pose[0], start_pose[1], start_pose[2], speed=self.max_vel)
            return
        # Mock fallback for tests: directly mutate position/orientation if writable.
        if hasattr(self._drone, 'position') and hasattr(self._drone, 'orientation'):
            self._drone.position = [start_pose[0], start_pose[1], start_pose[2]]
            self._drone.orientation = [0.0, 0.0, start_pose[3]]
            return
        raise RuntimeError('No supported simulator API for pose reset (set_pose/go_to unavailable)')

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

        # If already flying, land first
        if self._is_flying:
            logger.info("Landing before reset...")
            self._drone.land(speed=self.land_speed)
            self._is_flying = False

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
        if self.randomize_hover_start:
            hover_settled = self._wait_for_hover_settle()
            sampled_start, sampled_target, sample_attempts = self._sample_randomized_episode()
            self._apply_start_pose(sampled_start)
            self._target_pose = sampled_target
            self._reset_velocity_controller()
            hover_settled = self._wait_for_hover_settle()
            reset_mode = 'randomized_hover_start'
            start_pose = sampled_start

        obs = self._get_obs()
        info = self._get_info()
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
        info['action_sent'] = [vx, vy, vz, vyaw]

        # Continuous reward: distance penalty + path-facing shaping
        d_raw, d_norm = self._compute_distance()
        reward_distance = self._compute_reward_distance(d_norm)
        path_facing_reward, speed_xy, path_yaw, path_yaw_error = self._compute_path_facing_term()
        reward = reward_distance + path_facing_reward

        info['reward_distance'] = reward_distance
        info['path_facing_reward'] = path_facing_reward
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

        # Out-of-bounds: drone exceeded scenario limits
        elif self._is_out_of_bounds():
            terminated = True
            reward = -self.oob_penalty
            info['terminal_reason'] = 'out_of_bounds'

        # Max steps: episode truncation (time limit)
        if not terminated and self._step_count >= self.max_steps:
            truncated = True
            info['terminal_reason'] = 'max_steps'

        return obs, reward, terminated, truncated, info

    def _is_out_of_bounds(self) -> bool:
        """Check if the drone has exceeded the scenario boundaries."""
        try:
            pose = self._drone.position
            return any(abs(p) > self.pos_limit for p in pose)
        except Exception:
            return False

    def close(self):
        """Land the drone and shut down ROS2."""
        logger.info("Closing environment...")

        if self._drone is not None:
            if self._is_flying:
                logger.info("Landing...")
                try:
                    self._drone.land(speed=self.land_speed)
                except Exception as e:
                    logger.error(f"Error during landing: {e}")
                self._is_flying = False

            # Disarm
            try:
                self._drone.manual()
            except Exception as e:
                logger.error(f"Error setting manual mode: {e}")

            self._drone.shutdown()
            self._drone = None
            logger.info("DroneInterface shut down")

        # NOTE: rclpy.shutdown() is NOT called here because other
        # vectorized instances in the same process may still need it.
        # rclpy is shut down automatically when the process exits.

        logger.info("Environment closed")
