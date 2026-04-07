"""
AS2TestEnv — Minimal Gymnasium environment for verifying connectivity
with the Aerostack2 multirotor simulator.

This environment is NOT intended for RL training. Its purpose is to:
1. Verify that the DroneInterface connects correctly to the simulator.
2. Confirm that state information (position, velocity) is received properly.
3. Validate that velocity commands are sent and executed by the drone.

Observations (normalized to [-1, 1]):
    - Position (x, y, z) — divided by pos_limit (boundary of the scenario)
    - Velocity (vx, vy, vz) — divided by max_vel

Actions:
    - Velocity commands (vx, vy, vz) in m/s, bounded to [-max_vel, max_vel]

Reward:
    - Always 0 (not meaningful, only for connectivity testing)
"""

__authors__ = 'Jordi'
__license__ = 'BSD-3-Clause'

import logging
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
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        drone_namespace: str = 'drone0',
        use_sim_time: bool = False,
        verbose: bool = True,
        takeoff_height: float = 1.0,
        takeoff_speed: float = 0.5,
        land_speed: float = 0.5,
        max_vel: float = 2.0,
        pos_limit: float = 5.0,
        step_duration: float = 0.1,
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
            max_vel: Maximum velocity command in m/s. Used to normalize velocity observations.
            pos_limit: Maximum absolute position in meters (scenario boundary).
                       Used to normalize position observations to [-1, 1].
            step_duration: Duration to wait after sending command (seconds).
        """
        super().__init__()

        self.drone_namespace = drone_namespace
        self.use_sim_time = use_sim_time
        self.verbose = verbose
        self.takeoff_height = takeoff_height
        self.takeoff_speed = takeoff_speed
        self.land_speed = land_speed
        self.max_vel = max_vel
        self.pos_limit = pos_limit
        self.step_duration = step_duration

        # Observation space: [x, y, z, vx, vy, vz] — all normalized to [-1, 1]
        # Position normalized by pos_limit; velocity normalized by max_vel.
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )

        # Action space: velocity commands [vx, vy, vz]
        self.action_space = spaces.Box(
            low=-max_vel,
            high=max_vel,
            shape=(3,),
            dtype=np.float32
        )

        # DroneInterface (initialized on first reset)
        self._drone = None
        self._rclpy_initialized = False
        self._is_flying = False
        self._step_count = 0

    def _init_ros(self):
        """Initialize ROS2 and create DroneInterface."""
        import rclpy
        from as2_python_api.drone_interface import DroneInterface

        if not self._rclpy_initialized:
            rclpy.init()
            self._rclpy_initialized = True
            logger.info("ROS2 initialized")

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
        Read current state from the DroneInterface and normalize to [-1, 1].

        Position is divided by pos_limit; velocity is divided by max_vel.
        Values are clipped to [-1, 1] in case the drone exceeds scenario bounds.

        Returns:
            Normalized observation array [x, y, z, vx, vy, vz] ∈ [-1, 1]^6
        """
        try:
            # Position from drone pose
            pose = self._drone.position
            x, y, z = pose[0], pose[1], pose[2]

            # Velocity from drone speed
            speed = self._drone.speed
            vx, vy, vz = speed[0], speed[1], speed[2]

            obs = np.array([
                x / self.pos_limit,
                y / self.pos_limit,
                z / self.pos_limit,
                vx / self.max_vel,
                vy / self.max_vel,
                vz / self.max_vel,
            ], dtype=np.float32)

            # Clip in case the drone exceeds the defined boundaries
            obs = np.clip(obs, -1.0, 1.0)

        except Exception as e:
            logger.warning(f"Error reading drone state: {e}")
            obs = np.zeros(6, dtype=np.float32)

        return obs

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
        }

        try:
            info['position'] = list(self._drone.position)
            info['speed'] = list(self._drone.speed)
            info['orientation'] = list(self._drone.orientation)
        except Exception as e:
            info['state_error'] = str(e)

        return info

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

        obs = self._get_obs()
        info = self._get_info()
        info['reset_success'] = self._is_flying

        logger.info(f"Reset complete. Initial obs: {obs}")

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step: send velocity command and read new state.

        Args:
            action: Velocity command [vx, vy, vz] in m/s

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        import time

        self._step_count += 1

        # Clip action to valid range
        action = np.clip(action, -self.max_vel, self.max_vel)
        vx, vy, vz = float(action[0]), float(action[1]), float(action[2])

        # Send velocity command via DroneInterface
        try:
            self._speed_handler.send_speed_command_with_yaw_angle(
                twist=[vx, vy, vz],
                twist_frame_id='earth',
                yaw_angle=0.0,
            )
        except Exception as e:
            logger.error(f"Error sending velocity command: {e}")

        # Wait for the command to take effect
        time.sleep(self.step_duration)

        # Read new state
        obs = self._get_obs()
        info = self._get_info()
        info['action_sent'] = [vx, vy, vz]

        # Reward = 0 (connectivity test only)
        reward = 0.0

        # Never terminate (user stops manually)
        terminated = False
        truncated = False

        return obs, reward, terminated, truncated, info

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

        if self._rclpy_initialized:
            import rclpy
            try:
                rclpy.shutdown()
            except Exception:
                pass  # Already shut down by DroneInterface.shutdown()
            self._rclpy_initialized = False
            logger.info("ROS2 shut down")

        logger.info("Environment closed")
