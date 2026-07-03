#!/usr/bin/env python3
"""
Vectorization test for AS2TestEnv.

Verifies that the environment works correctly with gymnasium.vector.SyncVectorEnv
without requiring the simulator or ROS2 (uses mocks).

Tests:
  1. gymnasium.make('AS2TestEnv-v0') works (correct registration)
  2. Custom kwargs (drone_namespace, max_vel, pos_limit)
  3. SyncVectorEnv creates N independent instances with unique namespaces
  4. Correct shapes in vectorized reset() and step()
  5. All observations are normalized to [-1, 1]
  6. Numerical normalization correctness per sub-environment
  7. Class-level _rclpy_initialized flag shared across instances
  8. Reward correctness — distance + continuous path-facing term
  9. Terminal conditions — success, out-of-bounds, max_steps
 10. Deadband determinism and neutrality at threshold

Usage:
    conda run -n rl_uav python3 scripts/test_vectorization.py
"""

__authors__ = 'Jordi'
__license__ = 'BSD-3-Clause'

import sys
import types
import logging
import math
import importlib.util
import shutil
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING)  # suppress environment logs

# ---------------------------------------------------------------------------
# Minimal stubs to avoid importing rclpy / as2_python_api
# ---------------------------------------------------------------------------
def _stub(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod

for _pkg in [
    'rclpy',
    'as2_python_api',
    'as2_python_api.drone_interface',
    'as2_motion_reference_handlers',
    'as2_motion_reference_handlers.speed_motion',
]:
    _stub(_pkg)

sys.modules['as2_python_api.drone_interface'].DroneInterface = object
sys.modules['as2_motion_reference_handlers.speed_motion'].SpeedMotion = object

sys.path.insert(0, '/home/jordi/TFM/rl_uav_aerostack2')

import importlib
import rl_uav
importlib.reload(rl_uav)  # triggers gymnasium.register(...)

import gymnasium
import numpy as np
import rl_uav.envs.as2_test_env as as2_env_module
from rl_uav.envs.as2_test_env import AS2TestEnv

# ---------------------------------------------------------------------------
# Test utilities
# ---------------------------------------------------------------------------
PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
errors = 0

def check(label, condition):
    global errors
    print(f"  {PASS if condition else FAIL}  {label}")
    if not condition:
        errors += 1


def fake_drone(pos, vel, yaw=0.0):
    """Mock drone with fixed position, velocity and yaw."""
    class _Drone:
        position = list(pos)
        speed = list(vel)
        orientation = [0.0, 0.0, yaw]
        def arm(self, **kw): return True
        def offboard(self, **kw): return True
        def takeoff(self, **kw): return True
        def land(self, **kw): return True
        def manual(self, **kw): pass
        def shutdown(self, **kw): pass
    return _Drone()


class FakeSpeedHandler:
    def send_speed_command_with_yaw_speed(self, **kw): pass


class RecordingSpeedHandler:
    def __init__(self, drone=None):
        self.drone = drone
        self.commands = []

    def send_speed_command_with_yaw_speed(self, **kw):
        twist = list(kw.get('twist', [0.0, 0.0, 0.0]))
        yaw_speed = float(kw.get('yaw_speed', 0.0))
        self.commands.append((twist, yaw_speed))
        if self.drone is not None:
            self.drone.position = [
                self.drone.position[0] + twist[0] * 0.05,
                self.drone.position[1] + twist[1] * 0.05,
                self.drone.position[2] + twist[2] * 0.05,
            ]


def inject_mocks(vec_env, positions, velocities, yaws=None):
    """Inject mock drones into each sub-environment."""
    if yaws is None:
        yaws = [0.0] * len(positions)
    for i, env in enumerate(vec_env.envs):
        inner = env.unwrapped
        inner._drone = fake_drone(positions[i], velocities[i], yaws[i])
        inner._speed_handler = FakeSpeedHandler()
        inner._is_flying = True


def build_mocked_env(
    namespace='drone0',
    *,
    pos=(1.0, 2.0, 1.0),
    vel=(0.0, 0.0, 0.0),
    yaw=0.0,
    speed_deadband=0.05,
    path_facing_weight=0.25,
):
    """Create a gym env with deterministic mocked drone state."""
    env = gymnasium.make(
        'AS2TestEnv-v0',
        drone_namespace=namespace,
        step_duration=0.0,
        speed_deadband=speed_deadband,
        path_facing_weight=path_facing_weight,
    )
    inner = env.unwrapped
    inner._drone = fake_drone(pos, vel, yaw=yaw)
    inner._speed_handler = FakeSpeedHandler()
    inner._is_flying = True
    env.reset()  # required by OrderEnforcing wrapper
    inner._drone = fake_drone(pos, vel, yaw=yaw)  # re-inject after reset
    return env, inner


def build_deterministic_inner_env(
    namespace='drone0',
    *,
    pos=(0.0, 0.0, 1.0),
    vel=(0.0, 0.0, 0.0),
    yaw=0.0,
    target_pose=None,
    pos_limit=5.0,
    max_vel=2.0,
    speed_deadband=0.05,
    path_facing_weight=0.25,
):
    """Create a ROS-free AS2TestEnv instance with deterministic mocked state."""
    env = AS2TestEnv(
        drone_namespace=namespace,
        target_pose=target_pose,
        pos_limit=pos_limit,
        max_vel=max_vel,
        step_duration=0.0,
        speed_deadband=speed_deadband,
        path_facing_weight=path_facing_weight,
    )
    env._drone = fake_drone(pos, vel, yaw=yaw)
    env._speed_handler = FakeSpeedHandler()
    env._is_flying = True
    return env


def vector_env_boundary_config(vec_env):
    """Return per-instance boundary config used by vectorized environments."""
    return [
        {
            'namespace': env.unwrapped.drone_namespace,
            'pos_limit': env.unwrapped.pos_limit,
            'max_vel': env.unwrapped.max_vel,
        }
        for env in vec_env.envs
    ]


def vector_env_config_matches(vec_env):
    """Whether pos_limit and max_vel are identical across vectorized instances."""
    configs = vector_env_boundary_config(vec_env)
    pos_limits = {cfg['pos_limit'] for cfg in configs}
    max_vels = {cfg['max_vel'] for cfg in configs}
    return len(pos_limits) == 1 and len(max_vels) == 1


def expected_reward_components(inner, pos, vel, yaw):
    """Expected reward decomposition from mocked state."""
    tx, ty, tz, _ = inner._target_pose
    d = math.sqrt((pos[0] - tx) ** 2 + (pos[1] - ty) ** 2 + (pos[2] - tz) ** 2)
    d_max = inner.pos_limit * math.sqrt(3.0)
    d_norm = min(d / d_max, 1.0)
    reward_distance = -d_norm

    speed_xy = math.hypot(vel[0], vel[1])
    if speed_xy <= inner.speed_deadband:
        path_yaw = yaw
        path_yaw_error = 0.0
        path_facing_reward = 0.0
    else:
        path_yaw = math.atan2(vel[1], vel[0])
        path_yaw_error = math.atan2(math.sin(yaw - path_yaw), math.cos(yaw - path_yaw))
        path_facing_reward = inner.path_facing_weight * math.cos(path_yaw_error)

    return {
        'distance_norm': d_norm,
        'reward_distance': reward_distance,
        'speed_xy': speed_xy,
        'path_yaw': path_yaw,
        'path_yaw_error': path_yaw_error,
        'path_facing_reward': path_facing_reward,
        'reward_total': reward_distance + path_facing_reward,
    }


def check_reward_info(info, expected):
    """Validate deterministic reward diagnostics in info."""
    check(
        "info has reward diagnostics",
        all(k in info for k in ['reward_distance', 'path_facing_reward', 'speed_xy', 'path_yaw', 'path_yaw_error'])
    )
    check(
        "info.reward_distance is correct",
        math.isclose(info['reward_distance'], expected['reward_distance'], rel_tol=1e-5),
    )
    check(
        "info.path_facing_reward is correct",
        math.isclose(info['path_facing_reward'], expected['path_facing_reward'], rel_tol=1e-5),
    )
    check(
        "info.speed_xy is correct",
        math.isclose(info['speed_xy'], expected['speed_xy'], rel_tol=1e-5),
    )


# Test data: 4 drones with different positions, velocities, and yaw angles
NUM_ENVS = 4
POSITIONS = [
    [ 1.0,  2.0,  3.0],   # normal values, within boundary
    [-1.0, -2.0,  1.0],   # negative values
    [ 0.5,  0.0, -0.5],   # small values
    [ 5.0, -5.0,  5.0],   # exactly at pos_limit boundary
]
VELOCITIES = [
    [ 0.5, -0.5,  1.0],
    [ 2.0, -2.0,  0.0],   # exactly at max_vel boundary
    [ 0.0,  0.0,  0.0],
    [ 1.0,  1.0,  1.0],
]
YAWS = [
    0.5,           # small positive yaw
    -1.0,          # negative yaw
    0.0,           # zero yaw
    3.0,           # near π — tests wrapping
]
# Default target_pose is [0, 0, 1, 0]


# ===========================================================================
# TEST 1 — gymnasium.make() registration
# ===========================================================================
print("\n[1] gymnasium.make() — correct registration")
try:
    env = gymnasium.make('AS2TestEnv-v0', drone_namespace='drone0')
    check("make('AS2TestEnv-v0') does not raise", True)
    check("returns AS2TestEnv instance", isinstance(env.unwrapped, AS2TestEnv))
    check("drone_namespace == 'drone0'", env.unwrapped.drone_namespace == 'drone0')
    env.close()
except Exception as e:
    check(f"make() failed: {e}", False)

# ===========================================================================
# TEST 2 — Custom kwargs
# ===========================================================================
print("\n[2] Custom kwargs via gymnasium.make()")
env = gymnasium.make('AS2TestEnv-v0', drone_namespace='drone9',
                     max_vel=3.0, pos_limit=10.0)
inner = env.unwrapped
check("drone_namespace == 'drone9'", inner.drone_namespace == 'drone9')
check("max_vel == 3.0",              inner.max_vel == 3.0)
check("pos_limit == 10.0",           inner.pos_limit == 10.0)
check("obs space low  == -1.0",      np.all(inner.observation_space.low  == -1.0))
check("obs space high ==  1.0",      np.all(inner.observation_space.high ==  1.0))
env.close()

# ===========================================================================
# TEST 3 — SyncVectorEnv: independent instances
# ===========================================================================
print(f"\n[3] SyncVectorEnv — {NUM_ENVS} independent instances")
vec_env = gymnasium.vector.SyncVectorEnv([
    (lambda ns=f'drone{i}': gymnasium.make('AS2TestEnv-v0', drone_namespace=ns))
    for i in range(NUM_ENVS)
])
check(f"num_envs == {NUM_ENVS}",                   vec_env.num_envs == NUM_ENVS)
check("single_obs_space.shape == (4,)",            vec_env.single_observation_space.shape == (4,))
check("single_action_space.shape == (4,)",         vec_env.single_action_space.shape == (4,))
check(f"batched obs_space.shape == ({NUM_ENVS},4)", vec_env.observation_space.shape == (NUM_ENVS, 4))
for i in range(NUM_ENVS):
    ns = vec_env.envs[i].unwrapped.drone_namespace
    check(f"  sub-env[{i}].drone_namespace == 'drone{i}'", ns == f'drone{i}')
_default_vec_config = vector_env_boundary_config(vec_env)
check("vectorized config exposes pos_limit and max_vel per instance", all('pos_limit' in cfg and 'max_vel' in cfg for cfg in _default_vec_config))
check("vectorized default config keeps pos_limit/max_vel consistent", vector_env_config_matches(vec_env))
vec_env.close()

_override_vec_env = gymnasium.vector.SyncVectorEnv([
    (lambda ns='drone_cfg0': gymnasium.make('AS2TestEnv-v0', drone_namespace=ns, pos_limit=5.0, max_vel=2.0)),
    (lambda ns='drone_cfg1': gymnasium.make('AS2TestEnv-v0', drone_namespace=ns, pos_limit=6.0, max_vel=2.0)),
])
_override_config = vector_env_boundary_config(_override_vec_env)
check("intentional vectorized pos_limit override is detectable", not vector_env_config_matches(_override_vec_env))
check(
    "intentional override keeps namespace-local config values explicit",
    _override_config == [
        {'namespace': 'drone_cfg0', 'pos_limit': 5.0, 'max_vel': 2.0},
        {'namespace': 'drone_cfg1', 'pos_limit': 6.0, 'max_vel': 2.0},
    ],
)
_override_vec_env.close()

# ===========================================================================
# TEST 4 — Shapes in vectorized reset() and step()
# ===========================================================================
print("\n[4] Shapes in vectorized reset() and step() (mocked)")
vec_env = gymnasium.vector.SyncVectorEnv([
    (lambda ns=f'drone{i}': gymnasium.make('AS2TestEnv-v0', drone_namespace=ns))
    for i in range(NUM_ENVS)
])
inject_mocks(vec_env, POSITIONS, VELOCITIES, YAWS)

obs_r, infos_r = vec_env.reset()
check(f"reset: obs.shape == ({NUM_ENVS}, 4)", obs_r.shape == (NUM_ENVS, 4))
check("reset: obs.dtype == float32",           obs_r.dtype == np.float32)
check("reset: infos is not empty",             len(infos_r) > 0)

inject_mocks(vec_env, POSITIONS, VELOCITIES, YAWS)  # re-inject after reset
actions = vec_env.action_space.sample()
obs_s, rews, terms, truncs, infos_s = vec_env.step(actions)
check(f"step: obs.shape == ({NUM_ENVS}, 4)",      obs_s.shape == (NUM_ENVS, 4))
check(f"step: rewards.shape == ({NUM_ENVS},)",    rews.shape == (NUM_ENVS,))
check(f"step: terminated.shape == ({NUM_ENVS},)", terms.shape == (NUM_ENVS,))
check(f"step: truncated.shape == ({NUM_ENVS},)",  truncs.shape == (NUM_ENVS,))
vec_env.close()

# ===========================================================================
# TEST 5 — Observations normalized to [-1, 1]
# ===========================================================================
print("\n[5] Observations normalized to [-1, 1]")
vec_env = gymnasium.vector.SyncVectorEnv([
    (lambda ns=f'drone{i}': gymnasium.make('AS2TestEnv-v0', drone_namespace=ns))
    for i in range(NUM_ENVS)
])
inject_mocks(vec_env, POSITIONS, VELOCITIES, YAWS)
obs_r, _ = vec_env.reset()
check("reset: all obs >= -1.0", np.all(obs_r >= -1.0))
check("reset: all obs <=  1.0", np.all(obs_r <=  1.0))

# Test clipping when drone exceeds boundaries (10 m with pos_limit=5)
# Target is at [0, 0, 1, 0], so relative pos = [10, -10, 9] / 5 = [2, -2, 1.8]
# All exceed ±1.0 and should be clipped
out_pos  = [[10.0, -10.0, 10.0]] * NUM_ENVS
out_vels = [[ 5.0,  -5.0,  5.0]] * NUM_ENVS
inject_mocks(vec_env, out_pos, out_vels)
obs_out, _ = vec_env.reset()
check("out-of-bounds: pos dims clipped to ±1.0",
      np.all(np.abs(obs_out[:, :3]) == 1.0))
check("out-of-bounds: all obs in [-1, 1]",
      np.all(obs_out >= -1.0) and np.all(obs_out <= 1.0))
vec_env.close()

# Deterministic clipping fixtures do not use Gym wrappers or ROS initialization.
_obs_fixtures = [
    {
        'label': 'positive x/z, negative y, positive yaw boundary',
        'position': [12.0, -13.0, 20.0],
        'yaw': math.pi,
        'expected': [1.0, -1.0, 1.0, 1.0],
    },
    {
        'label': 'negative x/z, positive y, negative yaw boundary',
        'position': [-12.0, 13.0, -20.0],
        'yaw': -math.pi,
        'expected': [-1.0, 1.0, -1.0, -1.0],
    },
    {
        'label': 'mixed in-range and clipped axes preserve ordering',
        'position': [2.5, -7.5, 1.5],
        'yaw': math.pi / 2.0,
        'expected': [0.5, -1.0, 0.1, 0.5],
    },
]
for _fixture in _obs_fixtures:
    _fixture_env = build_deterministic_inner_env(
        namespace=f"fixture_{_fixture['label'].split()[0]}",
        pos=_fixture['position'],
        yaw=_fixture['yaw'],
    )
    _fixture_obs = _fixture_env._get_obs()
    check(
        f"fixture: {_fixture['label']} clipped with sign/axis preservation",
        np.allclose(_fixture_obs, np.array(_fixture['expected'], dtype=np.float32), atol=1e-6)
        and np.all(_fixture_obs >= -1.0)
        and np.all(_fixture_obs <= 1.0),
    )
    _fixture_env.close()

# ===========================================================================
# TEST 6 — Numerical normalization correctness per sub-environment
# ===========================================================================
print("\n[6] Numerical normalization correctness per sub-environment")
vec_env = gymnasium.vector.SyncVectorEnv([
    (lambda ns=f'drone{i}': gymnasium.make('AS2TestEnv-v0', drone_namespace=ns))
    for i in range(NUM_ENVS)
])
inject_mocks(vec_env, POSITIONS, VELOCITIES, YAWS)
vec_env.reset()

# Default target_pose = [0, 0, 1, 0]
tx, ty, tz, tyaw = 0.0, 0.0, 1.0, 0.0

for i in range(NUM_ENVS):
    inner = vec_env.envs[i].unwrapped
    inner._drone = fake_drone(POSITIONS[i], VELOCITIES[i], YAWS[i])  # re-inject

    # Relative position
    dx = (POSITIONS[i][0] - tx) / inner.pos_limit
    dy = (POSITIONS[i][1] - ty) / inner.pos_limit
    dz = (POSITIONS[i][2] - tz) / inner.pos_limit

    # Relative yaw with wrapping
    dyaw_raw = YAWS[i] - tyaw
    dyaw = math.atan2(math.sin(dyaw_raw), math.cos(dyaw_raw))
    dyaw_norm = dyaw / math.pi

    expected = np.clip(np.array([dx, dy, dz, dyaw_norm], dtype=np.float32), -1.0, 1.0)
    actual = inner._get_obs()
    check(f"  sub-env[{i}] relative obs is correct", np.allclose(actual, expected))
vec_env.close()

# ===========================================================================
# TEST 7 — Class-level _rclpy_initialized guard
# ===========================================================================
print("\n[7] Class-level _rclpy_initialized guard (single rclpy.init() per process)")
check("_rclpy_initialized is a class attribute",
      '_rclpy_initialized' in AS2TestEnv.__dict__)
check("_rclpy_lock is a class attribute",
      '_rclpy_lock' in AS2TestEnv.__dict__)

e1 = AS2TestEnv(drone_namespace='drone0')
e2 = AS2TestEnv(drone_namespace='drone1')
e3 = AS2TestEnv(drone_namespace='drone2')
# All instances must share the exact same class-level flag object
check("e1, e2, e3 share _rclpy_initialized",
      (e1._rclpy_initialized is e2._rclpy_initialized
       and e2._rclpy_initialized is e3._rclpy_initialized))
e1.close(); e2.close(); e3.close()

# ===========================================================================
# TEST 8 — Reward correctness: distance + path-facing
# ===========================================================================
print("\n[8] Reward correctness — distance + continuous path-facing")

# Monotonic distance penalty: same yaw/velocity, different target distance.
_near_reward_env = build_deterministic_inner_env(
    namespace='reward_near',
    pos=(1.0, 0.0, 1.0),
    vel=(0.0, 0.0, 0.0),
)
_far_reward_env = build_deterministic_inner_env(
    namespace='reward_far',
    pos=(3.0, 0.0, 1.0),
    vel=(0.0, 0.0, 0.0),
)
_near_distance_raw, _near_distance_norm = _near_reward_env._compute_distance()
_far_distance_raw, _far_distance_norm = _far_reward_env._compute_distance()
_near_distance_reward = _near_reward_env._compute_reward_distance(_near_distance_norm)
_far_distance_reward = _far_reward_env._compute_reward_distance(_far_distance_norm)
check("distance fixture: near state has smaller raw distance", _near_distance_raw < _far_distance_raw)
check("distance reward is monotonic with position error", _near_distance_reward >= _far_distance_reward)
_near_reward_env.close(); _far_reward_env.close()

# Path-facing contribution: equal position error and speed, aligned versus opposed yaw.
_aligned_pf_env = build_deterministic_inner_env(
    namespace='path_facing_aligned',
    pos=(1.0, 0.0, 1.0),
    vel=(1.0, 0.0, 0.0),
    yaw=0.0,
)
_opposed_pf_env = build_deterministic_inner_env(
    namespace='path_facing_opposed',
    pos=(1.0, 0.0, 1.0),
    vel=(1.0, 0.0, 0.0),
    yaw=math.pi,
)
_aligned_pf, _aligned_speed_xy, _, _aligned_yaw_error = _aligned_pf_env._compute_path_facing_term()
_opposed_pf, _opposed_speed_xy, _, _opposed_yaw_error = _opposed_pf_env._compute_path_facing_term()
check("path-facing aligned yaw has zero error", math.isclose(_aligned_yaw_error, 0.0, abs_tol=1e-9))
check("path-facing opposed yaw has pi error", math.isclose(abs(_opposed_yaw_error), math.pi, rel_tol=1e-6))
check("path-facing aligned contribution exceeds opposed", _aligned_pf > _opposed_pf)
check(
    "path-facing aligned/opposed uses equal non-zero speed",
    math.isclose(_aligned_speed_xy, _opposed_speed_xy, rel_tol=1e-9) and _aligned_speed_xy > 0.0,
)
_aligned_pf_env.close(); _opposed_pf_env.close()

# Below-deadband horizontal speed must be neutral for arbitrary yaw.
_deadband_low = 0.05
_low_speed_a = build_deterministic_inner_env(
    namespace='deadband_low_a',
    pos=(1.0, 0.0, 1.0),
    vel=(_deadband_low / 2.0, 0.0, 0.0),
    yaw=-2.0,
    speed_deadband=_deadband_low,
)
_low_speed_b = build_deterministic_inner_env(
    namespace='deadband_low_b',
    pos=(1.0, 0.0, 1.0),
    vel=(0.0, _deadband_low / 2.0, 0.0),
    yaw=2.5,
    speed_deadband=_deadband_low,
)
_low_pf_a, _low_speed_xy_a, _, _low_yaw_error_a = _low_speed_a._compute_path_facing_term()
_low_pf_b, _low_speed_xy_b, _, _low_yaw_error_b = _low_speed_b._compute_path_facing_term()
check("low-speed deadband reports speed below threshold", _low_speed_xy_a < _deadband_low and _low_speed_xy_b < _deadband_low)
check("low-speed deadband keeps arbitrary yaw neutral", _low_pf_a == 0.0 and _low_pf_b == 0.0)
check("low-speed deadband zeroes yaw error contribution", _low_yaw_error_a == 0.0 and _low_yaw_error_b == 0.0)
_low_speed_a.close(); _low_speed_b.close()

# Alignment test: measured velocity along +x and yaw=0 should maximize path-facing.
_pos8 = (1.0, 2.0, 1.0)
_vel8 = (1.0, 0.0, 0.0)
_yaw8 = 0.0
_env8, _inner8 = build_mocked_env(pos=_pos8, vel=_vel8, yaw=_yaw8)
_, _rew8, _term8, _trunc8, _info8 = _env8.step(_env8.action_space.sample())
_expected8 = expected_reward_components(_inner8, _pos8, _vel8, _yaw8)

check("reward == distance + path-facing", math.isclose(_rew8, _expected8['reward_total'], rel_tol=1e-5))
check("path-facing aligned is positive", _info8['path_facing_reward'] > 0.0)
check("non-terminal: terminated=False",        not _term8)
check("non-terminal: truncated=False",         not _trunc8)
check("non-terminal: terminal_reason absent",  'terminal_reason' not in _info8)
check_reward_info(_info8, _expected8)
_env8.close()

# Opposition test: same measured velocity, yaw opposite (pi) should be lower.
_opposed_yaw8 = math.pi
_env8b, _inner8b = build_mocked_env(namespace='drone1', pos=_pos8, vel=_vel8, yaw=_opposed_yaw8)
_, _rew8b, _, _, _info8b = _env8b.step(_env8b.action_space.sample())
_expected8b = expected_reward_components(_inner8b, _pos8, _vel8, _opposed_yaw8)

check("opposed yaw lowers total reward", _rew8b < _rew8)
check("opposed yaw has negative path-facing", _info8b['path_facing_reward'] < 0.0)
check_reward_info(_info8b, _expected8b)
_env8b.close()

# Progress shaping is opt-in and rewards distance reduction.
_env8c = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone8c',
    step_duration=0.01,
    progress_reward_weight=2.0,
)
_inner8c = _env8c.unwrapped
_inner8c._drone = fake_drone([1.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner8c._speed_handler = FakeSpeedHandler()
_inner8c._is_flying = True
_env8c.reset()
_inner8c._drone = fake_drone([1.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner8c._previous_distance = 2.0
_, _rew8c, _, _, _info8c = _env8c.step(np.zeros(4, dtype=np.float32))
_d8c = 1.0
_exp_progress8c = 2.0 * (2.0 - _d8c)
_exp_rew8c = -(_d8c / (_inner8c.pos_limit * math.sqrt(3.0))) + _exp_progress8c
check("progress reward is reported", math.isclose(_info8c['progress_reward'], _exp_progress8c, rel_tol=1e-5))
check("progress reward contributes to total reward", math.isclose(_rew8c, _exp_rew8c, rel_tol=1e-5))
_env8c.close()

# ===========================================================================
# TEST 9 — Terminal conditions: success, out-of-bounds, max_steps
# ===========================================================================
print("\n[9] Terminal conditions — success / out-of-bounds / max_steps")

# 9a — Success: drone within distance_threshold of target
# target=[0,0,1,0], drone at [0.1, 0.0, 1.0] → d=0.1 < threshold=0.5
# terminal success reward should not include obsolete target-yaw penalty
_env9a = gymnasium.make('AS2TestEnv-v0', drone_namespace='drone0', step_duration=0.0)
_inner9a = _env9a.unwrapped
_inner9a._drone = fake_drone([0.1, 0.0, 1.0], [0.0, 0.0, 0.0], yaw=math.pi)
_inner9a._speed_handler = FakeSpeedHandler()
_inner9a._is_flying = True
_env9a.reset()
_inner9a._drone = fake_drone([0.1, 0.0, 1.0], [0.0, 0.0, 0.0], yaw=math.pi)  # re-inject

_, _rew9a, _term9a, _trunc9a, _info9a = _env9a.step(_env9a.action_space.sample())

_d9a = 0.1
_dmax9a = _inner9a.pos_limit * math.sqrt(3.0)
_exp_rew9a = -(_d9a / _dmax9a) + _inner9a.success_reward

check("success: terminated=True",              _term9a)
check("success: truncated=False",              not _trunc9a)
check("success: terminal_reason='success'",    _info9a.get('terminal_reason') == 'success')
check("success: reward = -d_norm + bonus (no target-yaw penalty)", math.isclose(_rew9a, _exp_rew9a, rel_tol=1e-5))
check("success: yaw_error_norm removed from info", 'yaw_error_norm' not in _info9a)
_env9a.close()

# 9b — Out-of-bounds: drone at [6.0, 0.0, 0.0] → 6.0 > pos_limit=5.0
# reward is OVERWRITTEN to -oob_penalty (not additive with -d_norm)
_env9b = gymnasium.make('AS2TestEnv-v0', drone_namespace='drone1', step_duration=0.0)
_inner9b = _env9b.unwrapped
_inner9b._drone = fake_drone([6.0, 0.0, 0.0], [0.0, 0.0, 0.0])
_inner9b._speed_handler = FakeSpeedHandler()
_inner9b._is_flying = True
_env9b.reset()
_inner9b._drone = fake_drone([6.0, 0.0, 0.0], [0.0, 0.0, 0.0])  # re-inject

_, _rew9b, _term9b, _trunc9b, _info9b = _env9b.step(_env9b.action_space.sample())

check("oob: terminated=True",                     _term9b)
check("oob: truncated=False",                     not _trunc9b)
check("oob: terminal_reason='out_of_bounds'",     _info9b.get('terminal_reason') == 'out_of_bounds')
check("oob: reward == -oob_penalty (overwritten)", math.isclose(_rew9b, -_inner9b.oob_penalty, rel_tol=1e-5))
_env9b.close()

_env9f = gymnasium.make('AS2TestEnv-v0', drone_namespace='drone9f', step_duration=0.0)
_inner9f = _env9f.unwrapped
_drone9f = fake_drone([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner9f._drone = _drone9f
_inner9f._speed_handler = RecordingSpeedHandler(_drone9f)
_inner9f._is_flying = True
_env9f.reset()
_drone9f.position = [6.0, 0.0, 1.0]
_handler9f = RecordingSpeedHandler(_drone9f)
_inner9f._speed_handler = _handler9f
_, _, _term9f, _trunc9f, _info9f = _env9f.step(np.array([0.4, -0.2, 0.1, 0.3], dtype=np.float32))
_last_twist9f, _last_yaw9f = _handler9f.commands[-1]
check("terminal out-of-bounds sends a final zero speed command", _term9f and not _trunc9f and np.allclose(_last_twist9f, [0.0, 0.0, 0.0], atol=1e-6) and math.isclose(_last_yaw9f, 0.0, abs_tol=1e-6))
check("terminal out-of-bounds reports stop command acceptance", _info9f.get('terminal_stop_command_accepted') is True)
check("terminal out-of-bounds marks next reset for service", _inner9f._terminal_reset_requires_service is True)
_env9f.close()

_env9g = gymnasium.make('AS2TestEnv-v0', drone_namespace='drone9g', step_duration=0.0)
_inner9g = _env9g.unwrapped
_drone9g = fake_drone([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner9g._drone = _drone9g
_inner9g._speed_handler = RecordingSpeedHandler(_drone9g)
_inner9g._is_flying = True
_env9g.reset()
_drone9g.position = [-110.97, 37.50, 137.28]
_handler9g = RecordingSpeedHandler(_drone9g)
_inner9g._speed_handler = _handler9g
_, _rew9g, _term9g, _trunc9g, _info9g = _env9g.step(np.zeros(4, dtype=np.float32))
check("gross runaway position terminates", _term9g and not _trunc9g)
check("gross runaway reports out-of-bounds", _info9g.get('terminal_reason') == 'out_of_bounds' and _info9g.get('is_out_of_bounds') is True)
check("gross runaway uses oob penalty", math.isclose(_rew9g, -_inner9g.oob_penalty, rel_tol=1e-5))
check("gross runaway sends terminal zero command", np.allclose(_handler9g.commands[-1][0], [0.0, 0.0, 0.0], atol=1e-6))
_env9g.close()

# 9d — Unsafe low-altitude threshold terminates before physical lower bound.
_env9d = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone9d',
    step_duration=0.0,
    height_bounds=(0.1, 2.0),
    reset_ground_recovery_height=0.35,
    unsafe_low_altitude_threshold=0.3,
)
_inner9d = _env9d.unwrapped
_inner9d._drone = fake_drone([0.0, 1.0, 1.0], [0.0, 0.0, 0.0])
_inner9d._speed_handler = FakeSpeedHandler()
_inner9d._is_flying = True
_env9d.reset()
_inner9d._drone = fake_drone([0.0, 1.0, 0.4], [0.0, 0.0, 0.0])
_, _rew9d_high, _term9d_high, _trunc9d_high, _info9d_high = _env9d.step(np.zeros(4, dtype=np.float32))
check("terminal altitude accepts z above unsafe threshold", not _term9d_high and not _trunc9d_high)
check("terminal altitude above unsafe threshold is not unsafe", not _info9d_high.get('is_unsafe_low_altitude', False))
_inner9d._drone = fake_drone([0.0, 1.0, 0.2], [0.0, 0.0, 0.0])
_, _rew9d, _term9d, _trunc9d, _info9d = _env9d.step(np.zeros(4, dtype=np.float32))
check("unsafe low altitude terminates above physical lower bound", _term9d and not _trunc9d)
check("unsafe low altitude reports explicit terminal reason", _info9d.get('terminal_reason') == 'unsafe_low_altitude')
check("unsafe low altitude is not hidden as out_of_bounds", not _info9d.get('is_out_of_bounds', False))
check("unsafe low altitude sets explicit metric", _info9d.get('is_unsafe_low_altitude') is True)
check("unsafe low altitude uses oob penalty", math.isclose(_rew9d, -_inner9d.oob_penalty, rel_tol=1e-5))
_terminal_recovery_calls9d = []


def _record_terminal_recovery9d(hover_height):
    _terminal_recovery_calls9d.append(float(hover_height))
    _inner9d._drone.position[2] = float(hover_height)
    return True


_inner9d._recover_low_altitude_hover_before_velocity_reset = _record_terminal_recovery9d
_inner9d.fixed_start_pose = [0.0, 0.0, 1.0, 0.0]
_inner9d._drone = fake_drone([0.0, 1.0, 0.2], [0.0, 0.0, 0.0])
_, _, _term9d_recovery, _, _info9d_recovery = _env9d.step(np.zeros(4, dtype=np.float32))
check("unsafe low altitude triggers terminal recovery before reset", _term9d_recovery and len(_terminal_recovery_calls9d) == 1)
check("unsafe low altitude terminal info keeps original reason", _info9d_recovery.get('terminal_reason') == 'unsafe_low_altitude')
_inner9d._drone = fake_drone([0.0, 1.0, 0.05], [0.0, 0.0, 0.0])
_, _rew9d_low, _term9d_low, _trunc9d_low, _info9d_low = _env9d.step(np.zeros(4, dtype=np.float32))
check("terminal altitude terminates below height lower bound before ground", _term9d_low and not _trunc9d_low)
check("terminal altitude below lower bound reports out_of_bounds", _info9d_low.get('terminal_reason') == 'out_of_bounds')
check("terminal altitude below lower bound uses oob penalty", math.isclose(_rew9d_low, -_inner9d.oob_penalty, rel_tol=1e-5))
_env9d.close()

_env9e = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone9e',
    step_duration=0.0,
    max_vel=1.0,
    unsafe_low_altitude_threshold=1.0,
    low_altitude_guard_margin=0.5,
    low_altitude_guard_climb_speed=0.25,
    vertical_safety_band=0.5,
    vertical_safety_penalty_weight=2.0,
    vertical_descent_penalty_weight=3.0,
)
_inner9e = _env9e.unwrapped
_inner9e._drone = fake_drone([0.0, 0.0, 1.4], [0.0, 0.0, 0.0])
_handler9e = RecordingSpeedHandler(_inner9e._drone)
_inner9e._speed_handler = _handler9e
_inner9e._is_flying = True
_env9e.reset()
_inner9e._drone = fake_drone([0.0, 0.0, 1.4], [0.0, 0.0, 0.0])
_handler9e = RecordingSpeedHandler(_inner9e._drone)
_inner9e._speed_handler = _handler9e
_, _, _, _, _info9e = _env9e.step(np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32))
check("near unsafe altitude clamps downward vertical action", math.isclose(_handler9e.commands[0][0][2], 0.25, abs_tol=1e-6))
check("clamped vertical action is reported", math.isclose(_info9e.get('action_sent')[2], 0.25, abs_tol=1e-6))
check("action diagnostics use guarded vertical action", math.isclose(_info9e.get('last_action_vz', 0.0), 0.25, abs_tol=1e-6))
check("action diagnostics use guarded action norm", math.isclose(_info9e.get('mean_action_norm', 0.0), 0.25, abs_tol=1e-6))
check("action diagnostics report zero XY norm for guarded climb", math.isclose(_info9e.get('mean_xy_action_norm', -1.0), 0.0, abs_tol=1e-6))
check("low-altitude guard activation is reported", _info9e.get('low_altitude_guard_active') is True)
check("near unsafe downward action receives vertical safety penalty", _info9e.get('vertical_safety_penalty', 0.0) < 0.0)
_inner9e._drone = fake_drone([0.0, 0.0, 1.8], [0.0, 0.0, 0.0])
_handler9e_high = RecordingSpeedHandler(_inner9e._drone)
_inner9e._speed_handler = _handler9e_high
_, _, _, _, _info9e_high = _env9e.step(np.array([0.0, 0.0, -0.5, 0.0], dtype=np.float32))
check("safe altitude preserves downward vertical action", math.isclose(_handler9e_high.commands[0][0][2], -0.5, abs_tol=1e-6))
check("safe altitude does not report guard activation", _info9e_high.get('low_altitude_guard_active') is False)
check("safe altitude does not penalize downward action", math.isclose(_info9e_high.get('vertical_safety_penalty', 0.0), 0.0, abs_tol=1e-8))
_env9e.close()

# 9c — max_steps: step_count reaches max_steps → truncated, NOT terminated
# drone at [1.0, 0.0, 1.0] — not at target, not OOB
_env9c = gymnasium.make('AS2TestEnv-v0', drone_namespace='drone2',
                        step_duration=0.0, max_steps=5)
_inner9c = _env9c.unwrapped
_inner9c._drone = fake_drone([1.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner9c._speed_handler = FakeSpeedHandler()
_inner9c._is_flying = True
_env9c.reset()
_inner9c._drone = fake_drone([1.0, 0.0, 1.0], [0.0, 0.0, 0.0])  # re-inject
_inner9c._step_count = _inner9c.max_steps - 1  # set AFTER reset (reset zeroes it)

_, _, _term9c, _trunc9c, _info9c = _env9c.step(_env9c.action_space.sample())

check("max_steps: truncated=True",               _trunc9c)
check("max_steps: terminated=False",             not _term9c)
check("max_steps: terminal_reason='max_steps'",  _info9c.get('terminal_reason') == 'max_steps')
_env9c.close()

# ===========================================================================
# TEST 10 — Deadband neutrality and deterministic threshold behavior
# ===========================================================================
print("\n[10] Deadband neutrality and deterministic threshold behavior")

_deadband = 0.05
_pos10 = (1.0, 0.0, 1.0)
_vel10 = (_deadband, 0.0, 0.0)  # exactly on threshold
_yaw10 = 1.7  # arbitrary yaw should not matter in deadband

_env10, _inner10 = build_mocked_env(
    namespace='drone3',
    pos=_pos10,
    vel=_vel10,
    yaw=_yaw10,
    speed_deadband=_deadband,
)

_, _rew10a, _, _, _info10a = _env10.step(_env10.action_space.sample())
_inner10._drone = fake_drone(_pos10, _vel10, yaw=_yaw10)  # same exact state
_, _rew10b, _, _, _info10b = _env10.step(_env10.action_space.sample())

check("deadband: path-facing is neutral at threshold", math.isclose(_info10a['path_facing_reward'], 0.0, abs_tol=1e-8))
check("deadband: repeated threshold reward is deterministic", math.isclose(_rew10a, _rew10b, rel_tol=1e-9))
check("deadband: repeated threshold path-facing deterministic", math.isclose(_info10a['path_facing_reward'], _info10b['path_facing_reward'], abs_tol=1e-9))

_expected10 = expected_reward_components(_inner10, _pos10, _vel10, _yaw10)
check_reward_info(_info10a, _expected10)
_env10.close()

# ===========================================================================
# TEST 11 — Real validator helper contracts (deterministic)
# ===========================================================================
print("\n[11] Real validator helper contracts")

_validator_path = Path(__file__).resolve().parent / 'validate_real_vectorized_sim.py'
_validator_spec = importlib.util.spec_from_file_location('validate_real_vectorized_sim', _validator_path)
_validator = importlib.util.module_from_spec(_validator_spec)
assert _validator_spec is not None and _validator_spec.loader is not None
_validator_spec.loader.exec_module(_validator)

_action_space = gymnasium.spaces.Box(
    low=np.array([-2.0, -2.0, -2.0, -math.pi], dtype=np.float32),
    high=np.array([2.0, 2.0, 2.0, math.pi], dtype=np.float32),
    dtype=np.float32,
)
_hold_actions = _validator.build_hold_actions(_action_space, num_envs=4)
check("validator: hold actions shape is (4,4)", _hold_actions.shape == (4, 4))
check("validator: hold actions are zeroed", np.allclose(_hold_actions, 0.0))

_done_namespaces = _validator.namespaces_with_done(
    np.array([False, True, False, False]),
    np.array([False, False, False, True]),
    ['drone0', 'drone1', 'drone2', 'drone3'],
)
check("validator: namespaces_with_done reports terminated/truncated envs", _done_namespaces == ['drone1', 'drone3'])

_validator_env_kwargs = _validator.build_vectorized_env_kwargs(
    namespace='drone0',
    target_z=5.0,
    steps=30,
)
check("validator: fixed reset avoids stochastic unsafe-low samples", _validator_env_kwargs['randomize_hover_start'] is False)
check("validator: fixed reset starts above unsafe low altitude", _validator_env_kwargs['fixed_start_pose'][2] > _validator_env_kwargs['unsafe_low_altitude_threshold'])
check("validator: physical lower bound remains below unsafe threshold", _validator_env_kwargs['height_bounds'][0] < _validator_env_kwargs['unsafe_low_altitude_threshold'])
check("validator: vectorized max_steps covers requested live steps", _validator_env_kwargs['max_steps'] >= 80)

# ===========================================================================
# TEST 12 — Randomized hover reset contracts
# ===========================================================================
print("\n[12] Randomized hover reset contracts")
_env12 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone12',
    step_duration=0.0,
    randomize_hover_start=True,
    scene_bounds_xy=5.0,
    height_bounds=(0.1, 2.0),
    min_start_target_distance=0.7,
)
_inner12 = _env12.unwrapped
_inner12._drone = fake_drone([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner12._speed_handler = FakeSpeedHandler()
_inner12._is_flying = True

_start12, _target12, _attempts12 = _inner12._sample_randomized_episode()
_d12 = math.dist(_start12[:3], _target12[:3])
check("randomized start x in bounds", -5.0 <= _start12[0] <= 5.0)
check("randomized target x in bounds", -5.0 <= _target12[0] <= 5.0)
check("randomized start z in bounds", 0.1 <= _start12[2] <= 2.0)
check("randomized target z in bounds", 0.1 <= _target12[2] <= 2.0)
check("randomized start-target distance respects threshold", _d12 > max(_inner12.distance_threshold, _inner12.min_start_target_distance))
check("randomization attempts >= 1", _attempts12 >= 1)
_env12.close()

# ===========================================================================
# TEST 13 — Randomized reset calls controller reset hook
# ===========================================================================
print("\n[13] Randomized reset calls controller reset hook")
_env13 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone13',
    step_duration=0.0,
    randomize_hover_start=True,
    use_simulator_reset_service=False,
)
_inner13 = _env13.unwrapped
_inner13._drone = fake_drone([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner13._is_flying = False
_calls13 = {'controller_reset': 0}


def _stub_wait_for_hover_settle():
    return True


def _stub_sample_randomized_episode():
    return [1.0, 1.0, 1.0, 0.0], [2.0, 2.0, 1.2, 0.0], 1


def _stub_apply_start_pose(start_pose):
    _inner13._drone.position = [start_pose[0], start_pose[1], start_pose[2]]
    _inner13._drone.orientation = [0.0, 0.0, start_pose[3]]


def _stub_reset_velocity_controller():
    _calls13['controller_reset'] += 1


_inner13._wait_for_hover_settle = _stub_wait_for_hover_settle
_inner13._sample_randomized_episode = _stub_sample_randomized_episode
_inner13._apply_start_pose = _stub_apply_start_pose
_inner13._reset_velocity_controller = _stub_reset_velocity_controller

_obs13, _info13 = _env13.reset()
check("controller reset called once per randomized reset", _calls13['controller_reset'] == 1)
check("reset mode is randomized_hover_start", _info13.get('reset_mode') == 'randomized_hover_start')
check("randomized reset emits normalized obs", np.all(_obs13 >= -1.0) and np.all(_obs13 <= 1.0))
_env13.close()

# ===========================================================================
# TEST 14 — Fixed start reset returns to safe pose
# ===========================================================================
print("\n[14] Fixed start reset returns to safe pose")
_fixed_start14 = [0.0, 0.0, 1.0, 0.0]
_env14 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone14',
    step_duration=0.0,
    target_pose=[2.0, 0.0, 1.0, 0.0],
    fixed_start_pose=_fixed_start14,
    use_simulator_reset_service=False,
)
_inner14 = _env14.unwrapped
_inner14._drone = fake_drone([9.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner14._is_flying = False
_calls14 = {'controller_reset': 0}


def _stub_apply_fixed_start_pose(start_pose):
    _inner14._drone.position = [start_pose[0], start_pose[1], start_pose[2]]
    _inner14._drone.orientation = [0.0, 0.0, start_pose[3]]


def _stub_reset_fixed_velocity_controller():
    _calls14['controller_reset'] += 1


_inner14._wait_for_hover_settle = _stub_wait_for_hover_settle
_inner14._apply_start_pose = _stub_apply_fixed_start_pose
_inner14._reset_velocity_controller = _stub_reset_fixed_velocity_controller

_obs14, _info14 = _env14.reset()
check("fixed reset controller reset called once", _calls14['controller_reset'] == 1)
check("reset mode is fixed_start_pose", _info14.get('reset_mode') == 'fixed_start_pose')
check("fixed reset start pose is reported", _info14.get('start_pose') == _fixed_start14)
check("fixed reset drone position is safe", _inner14._drone.position == _fixed_start14[:3])
check("fixed reset emits normalized obs", np.all(_obs14 >= -1.0) and np.all(_obs14 <= 1.0))
_env14.close()

# ===========================================================================
# TEST 15 — Close operations are bounded
# ===========================================================================
print("\n[15] Close operations are bounded")
_env15 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone15',
    step_duration=0.0,
    close_operation_timeout=0.01,
)
_inner15 = _env15.unwrapped
_inner15._is_flying = True


class _HangingCloseDrone:
    position = [0.0, 0.0, 1.0]
    speed = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0]
    def land(self, **kw): time.sleep(0.2)
    def manual(self, **kw): pass
    def shutdown(self, **kw): pass


_inner15._drone = _HangingCloseDrone()
_close15_started = time.time()
_env15.close()
_close15_elapsed = time.time() - _close15_started
check("close returns despite hanging land", _close15_elapsed < 0.15)
check("close clears drone reference", _inner15._drone is None)

# ===========================================================================
# TEST 16 — Fixed start reset stays airborne between episodes
# ===========================================================================
print("\n[16] Fixed start reset stays airborne between episodes")
_env16 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16',
    step_duration=0.0,
    target_pose=[2.0, 0.0, 1.0, 0.0],
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
)
_inner16 = _env16.unwrapped
_calls16 = {'land': 0, 'controller_reset': 0}


class _AlreadyFlyingDrone:
    position = [1.0, 0.0, 1.0]
    speed = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0]
    def arm(self, **kw): return True
    def offboard(self, **kw): return True
    def takeoff(self, **kw): return True
    def land(self, **kw):
        _calls16['land'] += 1
        return True
    def manual(self, **kw): pass
    def shutdown(self, **kw): pass


_inner16._drone = _AlreadyFlyingDrone()
_inner16._is_flying = True
_inner16._wait_for_hover_settle = _stub_wait_for_hover_settle
_inner16._apply_start_pose = lambda start_pose: setattr(_inner16._drone, 'position', start_pose[:3])
_inner16._reset_velocity_controller = lambda: _calls16.__setitem__('controller_reset', _calls16['controller_reset'] + 1)
_obs16, _info16 = _env16.reset()
check("fixed in-air reset does not land", _calls16['land'] == 0)
check("fixed in-air reset resets controller", _calls16['controller_reset'] >= 1)
check("fixed in-air reset mode is fixed_start_pose", _info16.get('reset_mode') == 'fixed_start_pose')
check("fixed in-air reset emits normalized obs", np.all(_obs16 >= -1.0) and np.all(_obs16 <= 1.0))
_env16.close()

_env16b = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16b',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
)
_inner16b = _env16b.unwrapped
_calls16b = {'go_to': 0, 'velocity_reset': 0}


class _DroneWithGoTo:
    position = [0.0, 0.0, 1.0]
    speed = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0]
    def go_to(self, *args, **kw):
        _calls16b['go_to'] += 1
    def manual(self, **kw): pass
    def shutdown(self, **kw): pass


_inner16b._drone = _DroneWithGoTo()
_inner16b._speed_handler = FakeSpeedHandler()
_inner16b._drive_to_start_pose_with_velocity = lambda start_pose: _calls16b.__setitem__('velocity_reset', 1) or True
_inner16b._apply_start_pose([0.0, 0.0, 1.0, 0.0])
check("fixed reset prefers velocity over go_to", _calls16b['velocity_reset'] == 1 and _calls16b['go_to'] == 0)
_env16b.close()

_env16b2 = AS2TestEnv(
    drone_namespace='drone16b2',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    use_simulator_reset_service=True,
)
_drone16b2 = fake_drone([-110.97, 37.50, 137.28], [0.0, 0.0, 0.0])
_handler16b2 = RecordingSpeedHandler(_drone16b2)
_inner16b2 = _env16b2
_inner16b2._drone = _drone16b2
_inner16b2._speed_handler = _handler16b2
_inner16b2._is_flying = True
_calls16b2 = {'service_reset': 0, 'velocity_prepare': 0}


def _record_required_service_reset16b2(start_pose):
    _calls16b2['service_reset'] += 1
    _drone16b2.position = list(start_pose[:3])
    _drone16b2.orientation = [0.0, 0.0, start_pose[3]]
    _inner16b2._last_reset_diagnostics = {
        'reason': 'service_success',
        'failure_class': '',
        'reset_path': 'service_success',
        'target_pose': [float(v) for v in start_pose],
        'position_error': 0.0,
        'yaw_error': 0.0,
    }
    _inner16b2._last_reset_service_status = 'service_success'
    _inner16b2._is_flying = True
    return True


def _forbidden_velocity_prepare16b2():
    _calls16b2['velocity_prepare'] += 1
    raise AssertionError('velocity reset must not fly back from far out-of-bounds')


_inner16b2._try_service_backed_reset = _record_required_service_reset16b2
_inner16b2._prepare_in_air_velocity_reset = _forbidden_velocity_prepare16b2
_obs16b2, _info16b2 = _inner16b2.reset()
check("far out-of-bounds reset uses simulator service", _calls16b2 == {'service_reset': 1, 'velocity_prepare': 0})
check("far out-of-bounds reset does not velocity-fly back", _drone16b2.position == [0.0, 0.0, 1.0])
check("far out-of-bounds reset reports simulator service method", _info16b2.get('reset_method') == 'simulator_service')
check("far out-of-bounds reset clears terminal service requirement", _inner16b2._terminal_reset_requires_service is False)
check("far out-of-bounds reset emits normalized obs", np.all(_obs16b2 >= -1.0) and np.all(_obs16b2 <= 1.0))
_inner16b2.close()

_env16c = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16c',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
)
_inner16c = _env16c.unwrapped
_calls16c = {'go_to': 0, 'velocity_reset': 0}


class _DroneWithGoToAfterVelocityTimeout:
    position = [4.0, 0.0, 1.0]
    speed = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0]
    def go_to(self, *args, **kw):
        _calls16c['go_to'] += 1
    def manual(self, **kw): pass
    def shutdown(self, **kw): pass


_inner16c._drone = _DroneWithGoToAfterVelocityTimeout()
_inner16c._speed_handler = FakeSpeedHandler()
_inner16c._drive_to_start_pose_with_velocity = lambda start_pose: _calls16c.__setitem__('velocity_reset', 1) or False
try:
    _inner16c._apply_start_pose([0.0, 0.0, 1.0, 0.0])
    _velocity_timeout_raised = False
except RuntimeError as _exc16c:
    _velocity_timeout_raised = 'refusing to fall back to blocking go_to' in str(_exc16c)
check("fixed reset velocity timeout raises bounded failure", _velocity_timeout_raised)
check("fixed reset velocity timeout does not call go_to", _calls16c['velocity_reset'] == 1 and _calls16c['go_to'] == 0)
_env16c.close()

_env16d = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16d',
    step_duration=0.05,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    fixed_start_timeout=0.01,
)
_inner16d = _env16d.unwrapped
_calls16d = {'go_to': 0}


class _StaticDroneWithGoTo:
    position = [4.0, -2.0, 0.0]
    speed = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0]
    def go_to(self, *args, **kw):
        _calls16d['go_to'] += 1
    def manual(self, **kw): pass
    def shutdown(self, **kw): pass


_inner16d._drone = _StaticDroneWithGoTo()
_inner16d._speed_handler = RecordingSpeedHandler()
try:
    _inner16d._apply_start_pose([0.0, 0.0, 1.0, 0.0])
    _diagnostic_timeout_raised = False
    _diagnostic_text = ''
except RuntimeError as _exc16d:
    _diagnostic_timeout_raised = True
    _diagnostic_text = str(_exc16d)
check("fixed reset timeout reports final pose", _diagnostic_timeout_raised and 'final_pose=' in _diagnostic_text)
check("fixed reset timeout reports position error", 'position_error=' in _diagnostic_text)
check("fixed reset timeout reports progress diagnostics", 'initial_position_error=' in _diagnostic_text and 'best_position_error=' in _diagnostic_text and 'error_reduction=' in _diagnostic_text)
check("fixed reset diagnostic timeout does not call go_to", _calls16d['go_to'] == 0)
_env16d.close()

_env16e = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16e',
    step_duration=0.05,
    fixed_start_timeout=0.12,
    reset_min_speed=0.15,
    reset_ground_recovery_height=0.35,
)
_inner16e = _env16e.unwrapped
_inner16e._drone = fake_drone([4.0, -2.0, 0.0], [0.0, 0.0, 0.0])
_handler16e = RecordingSpeedHandler(_inner16e._drone)
_inner16e._speed_handler = _handler16e
_inner16e._drive_to_start_pose_with_velocity([0.0, 0.0, 1.0, 0.0])
_first_twist16e = _handler16e.commands[0][0]
check("grounded reset recovers vertically before lateral motion", _first_twist16e[0] == 0.0 and _first_twist16e[1] == 0.0 and _first_twist16e[2] >= 0.15)
_env16e.close()

_env16e2 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16e2',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    height_bounds=(0.1, 2.0),
    reset_ground_recovery_height=0.35,
    use_simulator_reset_service=False,
)
_inner16e2 = _env16e2.unwrapped
_calls16e2 = {'go_to': 0, 'land': 0, 'takeoff': 0}
_events16e2 = []


class _LowAltitudeResetDrone:
    position = [0.3, -0.2, 0.2]
    speed = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0]
    def arm(self, **kw):
        _events16e2.append('arm')
        return True
    def offboard(self, **kw):
        _events16e2.append('offboard')
        return True
    def takeoff(self, **kw):
        _calls16e2['takeoff'] += 1
        return True
    def go_to(self, *args, **kw):
        _calls16e2['go_to'] += 1
    def land(self, **kw):
        _calls16e2['land'] += 1
        return True
    def manual(self, **kw): pass
    def shutdown(self, **kw): pass


_inner16e2._drone = _LowAltitudeResetDrone()
_handler16e2 = RecordingSpeedHandler(_inner16e2._drone)
_inner16e2._speed_handler = _handler16e2
_inner16e2._is_flying = True
_inner16e2._wait_for_hover_settle = _stub_wait_for_hover_settle
_inner16e2._reset_velocity_controller = lambda: _events16e2.append('controller_reset')


def _apply_start_after_low_altitude_recovery(start_pose):
    _events16e2.append(f'apply_start_from_z={_inner16e2._drone.position[2]:.1f}')
    _inner16e2._drone.position = list(start_pose[:3])
    _inner16e2._drone.orientation = [0.0, 0.0, start_pose[3]]


_inner16e2._apply_start_pose = _apply_start_after_low_altitude_recovery
_obs16e2, _info16e2 = _env16e2.reset()
_recovery_commands16e2 = [cmd for cmd in _handler16e2.commands if cmd[0][2] > 0.0]
check(
    "valid low-altitude in-air reset recovers and reasserts offboard before controller reset",
    len(_recovery_commands16e2) > 0 and _events16e2[:3] == ['arm', 'offboard', 'controller_reset'],
)
check("valid low-altitude recovery does not call land/takeoff", _calls16e2['land'] == 0 and _calls16e2['takeoff'] == 0)
check("low-altitude recovery starts fixed reset from hover height", 'apply_start_from_z=0.9' in _events16e2 or 'apply_start_from_z=1.0' in _events16e2)
check("low-altitude recovery preserves no go_to fallback", _calls16e2['go_to'] == 0)
check("low-altitude recovery emits normalized obs", np.all(_obs16e2 >= -1.0) and np.all(_obs16e2 <= 1.0))
check("low-altitude recovery reports fixed reset mode", _info16e2.get('reset_mode') == 'fixed_start_pose')
_env16e2.close()

_env16e2b = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16e2b',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    fixed_start_tolerance=0.15,
    hover_timeout=2.0,
    reset_ground_recovery_height=0.35,
)
_inner16e2b = _env16e2b.unwrapped
_inner16e2b._drone = fake_drone([0.0, 0.0, 0.7], [0.0, 0.0, 0.0])
_handler16e2b = RecordingSpeedHandler(_inner16e2b._drone)
_inner16e2b._speed_handler = _handler16e2b
_inner16e2b._wait_for_hover_settle = _stub_wait_for_hover_settle
_inner16e2b._recover_low_altitude_hover_before_velocity_reset(1.0)
_recovery_climb_commands16e2b = [cmd for cmd in _handler16e2b.commands if cmd[0][2] > 0.0]
check("unsafe-altitude reset preclimbs before pose drive", len(_recovery_climb_commands16e2b) > 0)
check("unsafe-altitude recovery reaches start-height tolerance", _inner16e2b._drone.position[2] >= 0.85)
_env16e2b.close()

_env16e2b1 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16e2b1',
    step_duration=0.0,
    max_vel=0.1,
    reset_max_vel=0.25,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    fixed_start_tolerance=0.15,
    hover_timeout=2.0,
    reset_ground_recovery_height=0.35,
)
_inner16e2b1 = _env16e2b1.unwrapped
_inner16e2b1._drone = fake_drone([0.0, 0.0, 0.7], [0.0, 0.0, 0.0])
_handler16e2b1 = RecordingSpeedHandler(_inner16e2b1._drone)
_inner16e2b1._speed_handler = _handler16e2b1
_inner16e2b1._wait_for_hover_settle = _stub_wait_for_hover_settle
_inner16e2b1._recover_low_altitude_hover_before_velocity_reset(1.0)
_low_altitude_reset_commands16e2b1 = [cmd for cmd in _handler16e2b1.commands if cmd[0][2] > 0.0]
_first_low_altitude_twist16e2b1 = _low_altitude_reset_commands16e2b1[0][0] if _low_altitude_reset_commands16e2b1 else [0.0, 0.0, 0.0]
check("low-altitude recovery uses reset_max_vel", math.isclose(_first_low_altitude_twist16e2b1[2], 0.25, rel_tol=1e-6))
_env16e2b1.close()

_env16e2c = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16e2c',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    fixed_start_tolerance=0.15,
    reset_ground_recovery_height=0.35,
)
_inner16e2c = _env16e2c.unwrapped
_inner16e2c._drone = fake_drone([0.0, 0.0, 0.95], [0.0, 0.0, 0.0])
_handler16e2c = RecordingSpeedHandler(_inner16e2c._drone)
_inner16e2c._speed_handler = _handler16e2c
_inner16e2c._recover_low_altitude_hover_before_velocity_reset(1.0)
check("near-start-height reset skips preclimb", len(_handler16e2c.commands) == 0)
_env16e2c.close()

_env16e3 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16e3',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    height_bounds=(0.1, 2.0),
    reset_ground_recovery_height=0.35,
    use_simulator_reset_service=False,
)
_inner16e3 = _env16e3.unwrapped
_calls16e3 = {'land': 0, 'takeoff': 0, 'go_to': 0}


class _GroundResetDrone:
    position = [0.3, -0.2, 0.0]
    speed = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0]
    def arm(self, **kw): return True
    def offboard(self, **kw): return True
    def takeoff(self, **kw):
        _calls16e3['takeoff'] += 1
        return True
    def land(self, **kw):
        _calls16e3['land'] += 1
        return True
    def go_to(self, *args, **kw):
        _calls16e3['go_to'] += 1
    def manual(self, **kw): pass
    def shutdown(self, **kw): pass


_inner16e3._drone = _GroundResetDrone()
_inner16e3._speed_handler = RecordingSpeedHandler(_inner16e3._drone)
_inner16e3._is_flying = True
try:
    _env16e3.reset()
    _ground_recovery_failed_fast = False
    _ground_recovery_text = ''
except RuntimeError as _exc16e3:
    _ground_recovery_failed_fast = True
    _ground_recovery_text = str(_exc16e3)
check("on-ground low-altitude reset fails fast with hard reset diagnostic", _ground_recovery_failed_fast and 'hard_reset_required' in _ground_recovery_text)
check("on-ground recovery does not call land/takeoff/go_to", _calls16e3 == {'land': 0, 'takeoff': 0, 'go_to': 0})
_env16e3.close()

_env16f = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16f',
    step_duration=0.05,
    fixed_start_timeout=0.06,
    fixed_start_tolerance=0.15,
    reset_min_speed=0.3,
)
_inner16f = _env16f.unwrapped
_inner16f._drone = fake_drone([0.2, 0.0, 1.0], [0.0, 0.0, 0.0])
_handler16f = RecordingSpeedHandler(_inner16f._drone)
_inner16f._speed_handler = _handler16f
_inner16f._drive_to_start_pose_with_velocity([0.0, 0.0, 1.0, 0.0])
_first_twist16f = _handler16f.commands[0][0]
check("reset applies configured minimum command above tolerance", math.isclose(_first_twist16f[0], -0.3, rel_tol=1e-6))
_env16f.close()

_env16f1 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16f1',
    step_duration=0.0,
    max_vel=0.5,
    reset_max_vel=1.0,
    fixed_start_timeout=0.01,
    fixed_start_tolerance=0.15,
    reset_min_speed=0.2,
)
_inner16f1 = _env16f1.unwrapped
_inner16f1._drone = fake_drone([2.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_handler16f1 = RecordingSpeedHandler(_inner16f1._drone)
_inner16f1._speed_handler = _handler16f1
_inner16f1._drive_to_start_pose_with_velocity([0.0, 0.0, 1.0, 0.0])
_reset_twist16f1 = _handler16f1.commands[0][0]
check("reset command may exceed training max_vel up to reset_max_vel", math.isclose(_reset_twist16f1[0], -1.0, rel_tol=1e-6))
_handler16f1.commands.clear()
_inner16f1._drone = fake_drone([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner16f1._speed_handler = _handler16f1
_, _, _, _, _info16f1 = _inner16f1.step(np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32))
check("policy action remains clipped by training max_vel", math.isclose(_info16f1['action_sent'][0], 0.5, rel_tol=1e-6))
_env16f1.close()

_env16f1b = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16f1b',
    step_duration=0.0,
    max_vel=0.5,
    fixed_start_timeout=0.01,
    fixed_start_tolerance=0.15,
    reset_min_speed=0.2,
)
_inner16f1b = _env16f1b.unwrapped
_inner16f1b._drone = fake_drone([2.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_handler16f1b = RecordingSpeedHandler(_inner16f1b._drone)
_inner16f1b._speed_handler = _handler16f1b
_inner16f1b._drive_to_start_pose_with_velocity([0.0, 0.0, 1.0, 0.0])
_default_reset_twist16f1b = _handler16f1b.commands[0][0]
check("default reset_max_vel clips reset commands to max_vel", math.isclose(_default_reset_twist16f1b[0], -0.5, rel_tol=1e-6))
_env16f1b.close()

_env16f1c = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16f1c',
    step_duration=0.0,
    max_vel=1.0,
    reset_max_vel=0.2,
    fixed_start_timeout=0.01,
    fixed_start_tolerance=0.15,
    reset_min_speed=0.5,
)
_inner16f1c = _env16f1c.unwrapped
_inner16f1c._drone = fake_drone([0.17, 0.0, 1.0], [0.0, 0.0, 0.0])
_handler16f1c = RecordingSpeedHandler(_inner16f1c._drone)
_inner16f1c._speed_handler = _handler16f1c
_inner16f1c._drive_to_start_pose_with_velocity([0.0, 0.0, 1.0, 0.0])
_min_speed_reset_twist16f1c = _handler16f1c.commands[0][0]
_near_drive_cap16f1c = _inner16f1c.reset_max_vel * 0.17 / (3.0 * _inner16f1c._reset_position_tolerance())
check("drive reset min-speed is bounded by near-target cap", abs(_min_speed_reset_twist16f1c[0]) <= _near_drive_cap16f1c + 1e-6)
_env16f1c.close()

_env16f1d = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16f1d',
    step_duration=0.0,
    max_vel=0.5,
    reset_max_vel=1.0,
    reset_xy_kp=4.0,
    reset_z_kp=4.0,
    fixed_start_timeout=0.01,
    fixed_start_tolerance=0.15,
    reset_min_speed=0.2,
)
_inner16f1d = _env16f1d.unwrapped
_inner16f1d._drone = fake_drone([0.3, -0.28, 0.8], [0.0, 0.0, 0.0], yaw=2.2)
_handler16f1d = RecordingSpeedHandler(_inner16f1d._drone)
_inner16f1d._speed_handler = _handler16f1d
_inner16f1d._drive_to_start_pose_with_velocity([0.0, 0.0, 1.0, 0.0])
_near_drive_twist16f1d, _near_drive_yaw16f1d = _handler16f1d.commands[0]
check(
    "drive reset near target uses one conservative XY axis",
    _near_drive_twist16f1d[0] < 0.0
    and abs(_near_drive_twist16f1d[0]) < _inner16f1d.reset_max_vel
    and math.isclose(_near_drive_twist16f1d[1], 0.0, abs_tol=1e-9),
)
check(
    "drive reset near target suppresses simultaneous Z/yaw correction",
    math.isclose(_near_drive_twist16f1d[2], 0.0, abs_tol=1e-9)
    and math.isclose(_near_drive_yaw16f1d, 0.0, abs_tol=1e-9),
)
_env16f1d.close()

_env16f1e = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16f1e',
    step_duration=0.0,
    max_vel=0.5,
    reset_max_vel=1.0,
    reset_xy_kp=4.0,
    reset_z_kp=4.0,
    fixed_start_timeout=0.01,
    fixed_start_tolerance=0.15,
    reset_min_speed=0.2,
)
_inner16f1e = _env16f1e.unwrapped
_inner16f1e._drone = fake_drone([2.0, -1.0, 0.8], [0.0, 0.0, 0.0], yaw=1.5)
_handler16f1e = RecordingSpeedHandler(_inner16f1e._drone)
_inner16f1e._speed_handler = _handler16f1e
_inner16f1e._drive_to_start_pose_with_velocity([0.0, 0.0, 1.0, 0.0])
_far_drive_twist16f1e, _far_drive_yaw16f1e = _handler16f1e.commands[0]
check(
    "drive reset keeps full authority for large dominant-axis error",
    math.isclose(_far_drive_twist16f1e[0], -1.0, rel_tol=1e-6),
)
check(
    "drive reset large XY phase does not command simultaneous Y/Z/yaw",
    math.isclose(_far_drive_twist16f1e[1], 0.0, abs_tol=1e-9)
    and math.isclose(_far_drive_twist16f1e[2], 0.0, abs_tol=1e-9)
    and math.isclose(_far_drive_yaw16f1e, 0.0, abs_tol=1e-9),
)
check("drive reset timeout reports drive phase", _inner16f1e._last_reset_diagnostics.get('phase') == 'drive_xy_axis_x')
_env16f1e.close()

_env16f2 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16f2',
    step_duration=0.0,
    fixed_start_timeout=0.01,
    fixed_start_tolerance=0.15,
)
_inner16f2 = _env16f2.unwrapped
_inner16f2._drone = fake_drone([0.05, 0.0, 1.0], [0.0, 0.0, 0.0], yaw=0.3092)
_handler16f2 = RecordingSpeedHandler(_inner16f2._drone)
_inner16f2._speed_handler = _handler16f2
_near_yaw_reset16f2 = _inner16f2._drive_to_start_pose_with_velocity([0.0, 0.0, 1.0, 0.0])
check("velocity reset accepts bounded near-tolerance yaw", _near_yaw_reset16f2 is True)
check("near-tolerance yaw reset reports final yaw error", math.isclose(_inner16f2._last_reset_diagnostics.get('yaw_error'), 0.3092, abs_tol=1e-3))
_last_twist16f2, _last_yaw_speed16f2 = _handler16f2.commands[-1]
check("near-tolerance yaw reset holds zero command", np.allclose(_last_twist16f2, [0.0, 0.0, 0.0], atol=1e-6) and math.isclose(_last_yaw_speed16f2, 0.0, abs_tol=1e-6))
_env16f2.close()

_env16f2b = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16f2b',
    step_duration=0.0,
    fixed_start_timeout=0.01,
    fixed_start_tolerance=0.15,
)
_inner16f2b = _env16f2b.unwrapped
_inner16f2b._drone = fake_drone([0.05, 0.0, 1.0], [0.0, 0.0, 0.0], yaw=0.368)
_inner16f2b._speed_handler = RecordingSpeedHandler(_inner16f2b._drone)
_live_yaw_reset16f2b = _inner16f2b._drive_to_start_pose_with_velocity([0.0, 0.0, 1.0, 0.0])
check("velocity reset accepts live-observed bounded yaw", _live_yaw_reset16f2b is True)
_env16f2b.close()

_env16f2c = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16f2c',
    step_duration=0.0,
    fixed_start_timeout=0.01,
    fixed_start_tolerance=0.15,
    reset_yaw_tolerance=1.2,
)
_inner16f2c = _env16f2c.unwrapped
_inner16f2c._drone = fake_drone([0.05, 0.0, 1.0], [0.0, 0.0, 0.0], yaw=1.0)
_inner16f2c._speed_handler = RecordingSpeedHandler(_inner16f2c._drone)
_relaxed_yaw_reset16f2c = _inner16f2c._drive_to_start_pose_with_velocity([0.0, 0.0, 1.0, 0.0])
check("velocity reset accepts configured relaxed early-curriculum yaw", _relaxed_yaw_reset16f2c is True)
check("relaxed yaw reset reports non-zero yaw diagnostic", math.isclose(_inner16f2c._last_reset_diagnostics.get('yaw_error'), 1.0, abs_tol=1e-3))
_env16f2c.close()

_env16f2d = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16f2d',
    step_duration=0.0,
    fixed_start_timeout=0.01,
    fixed_start_tolerance=0.15,
    reset_yaw_tolerance=1.2,
)
_inner16f2d = _env16f2d.unwrapped
_inner16f2d._drone = fake_drone([0.05, 0.0, 1.0], [0.0, 0.0, 0.0], yaw=1.3)
_inner16f2d._speed_handler = RecordingSpeedHandler(_inner16f2d._drone)
_gross_yaw_reset16f2d = _inner16f2d._drive_to_start_pose_with_velocity([0.0, 0.0, 1.0, 0.0])
check("velocity reset still rejects yaw beyond configured gross tolerance", _gross_yaw_reset16f2d is False and _inner16f2d._last_reset_diagnostics.get('reason') == 'timeout')
_env16f2d.close()

_env16f2e = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16f2e',
    step_duration=0.0,
    fixed_start_timeout=0.01,
    fixed_start_tolerance=0.15,
    reset_yaw_tolerance=0.38,
    reset_yaw_required=False,
)
_inner16f2e = _env16f2e.unwrapped
_inner16f2e._drone = fake_drone([0.05, 0.0, 1.0], [0.0, 0.0, 0.0], yaw=2.6)
_handler16f2e = RecordingSpeedHandler(_inner16f2e._drone)
_inner16f2e._speed_handler = _handler16f2e
_position_only_yaw_reset16f2e = _inner16f2e._drive_to_start_pose_with_velocity([0.0, 0.0, 1.0, 0.0])
_last_twist16f2e, _last_yaw_speed16f2e = _handler16f2e.commands[-1]
check("position-only reset accepts large yaw error within XYZ tolerance", _position_only_yaw_reset16f2e is True)
check("position-only reset records yaw error diagnostically", math.isclose(_inner16f2e._last_reset_diagnostics.get('yaw_error'), 2.6, abs_tol=1e-3))
check("position-only reset does not command yaw correction", math.isclose(_last_yaw_speed16f2e, 0.0, abs_tol=1e-9))
_env16f2e.close()

_env16f2f = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16f2f',
    step_duration=0.0,
    fixed_start_timeout=0.01,
    fixed_start_tolerance=0.15,
    reset_yaw_required=False,
)
_inner16f2f = _env16f2f.unwrapped
_inner16f2f._drone = fake_drone([0.5, 0.0, 1.0], [0.0, 0.0, 0.0], yaw=2.6)
_inner16f2f._speed_handler = RecordingSpeedHandler(_inner16f2f._drone)
_position_only_position_failure16f2f = _inner16f2f._drive_to_start_pose_with_velocity([0.0, 0.0, 1.0, 0.0])
check("position-only reset still fails when XYZ position does not converge", _position_only_position_failure16f2f is False and _inner16f2f._last_reset_diagnostics.get('reason') == 'timeout')
check("position-only reset timeout remains bounded and diagnostic", _inner16f2f._last_reset_diagnostics.get('elapsed', 999.0) < 1.0 and 'position_error' in _inner16f2f._last_reset_diagnostics)
_env16f2f.close()

_env16f3 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16f3',
    step_duration=0.0,
    fixed_start_timeout=0.01,
    fixed_start_tolerance=0.15,
)
_inner16f3 = _env16f3.unwrapped
_inner16f3._drone = fake_drone([0.05, 0.0, 1.0], [0.0, 0.0, 0.0], yaw=0.4)
_inner16f3._speed_handler = RecordingSpeedHandler(_inner16f3._drone)
_wide_yaw_reset16f3 = _inner16f3._drive_to_start_pose_with_velocity([0.0, 0.0, 1.0, 0.0])
check("velocity reset still rejects large yaw error", _wide_yaw_reset16f3 is False and _inner16f3._last_reset_diagnostics.get('reason') == 'timeout')
_env16f3.close()

_env16f4 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16f4',
    step_duration=0.0,
    fixed_start_timeout=0.01,
    fixed_start_tolerance=0.15,
)
_inner16f4 = _env16f4.unwrapped
_inner16f4._drone = fake_drone([0.152, 0.0, 1.0], [0.0, 0.0, 0.0], yaw=0.0)
_inner16f4._speed_handler = RecordingSpeedHandler(_inner16f4._drone)
_near_position_reset16f4 = _inner16f4._drive_to_start_pose_with_velocity([0.0, 0.0, 1.0, 0.0])
check("velocity reset accepts bounded near-tolerance position", _near_position_reset16f4 is True)
check("near-tolerance position reset reports final position error", math.isclose(_inner16f4._last_reset_diagnostics.get('position_error'), 0.152, abs_tol=1e-3))
_env16f4.close()

_env16f5 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16f5',
    step_duration=0.0,
    fixed_start_timeout=0.01,
    fixed_start_tolerance=0.15,
)
_inner16f5 = _env16f5.unwrapped
_inner16f5._drone = fake_drone([0.19, 0.0, 1.0], [0.0, 0.0, 0.0], yaw=0.0)
_inner16f5._speed_handler = RecordingSpeedHandler(_inner16f5._drone)
_wide_position_reset16f5 = _inner16f5._drive_to_start_pose_with_velocity([0.0, 0.0, 1.0, 0.0])
check("velocity reset still rejects large position error", _wide_position_reset16f5 is False and _inner16f5._last_reset_diagnostics.get('reason') == 'timeout')
_env16f5.close()

_env16g = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16g',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    use_simulator_reset_service=False,
)
_inner16g = _env16g.unwrapped
_calls16g = {'apply': 0, 'controller_reset': 0}
_inner16g._drone = fake_drone([2.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner16g._is_flying = True
_inner16g._wait_for_hover_settle = _stub_wait_for_hover_settle


def _apply_start_with_first_drift(start_pose):
    _calls16g['apply'] += 1
    _inner16g._drone.position = list(start_pose[:3])
    _inner16g._drone.orientation = [0.0, 0.0, start_pose[3]]


def _reset_controller_with_first_drift():
    _calls16g['controller_reset'] += 1
    if _calls16g['controller_reset'] == 2:
        _inner16g._drone.position = [0.0, 0.0, 0.55]


_inner16g._apply_start_pose = _apply_start_with_first_drift
_inner16g._reset_velocity_controller = _reset_controller_with_first_drift
_obs16g, _info16g = _env16g.reset()
check("fixed reset retries when post-settle pose drifts", _calls16g['apply'] == 2)
check("fixed reset retry finishes at requested start pose", _inner16g._drone.position == [0.0, 0.0, 1.0])
check("fixed reset retry still emits normalized obs", np.all(_obs16g >= -1.0) and np.all(_obs16g <= 1.0))
_env16g.close()

_env16h = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16h',
    step_duration=0.01,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    fixed_start_tolerance=0.15,
    hover_settle_time=0.11,
    hover_timeout=0.5,
    reset_min_speed=0.2,
    use_simulator_reset_service=False,
)
_inner16h = _env16h.unwrapped
_calls16h = {'apply': 0, 'commands': 0, 'corrections': 0}
_inner16h._drone = fake_drone([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner16h._is_flying = True


class _DriftThenCorrectHandler:
    def __init__(self, drone):
        self.drone = drone
        self.zero_commands = 0

    def send_speed_command_with_yaw_speed(self, **kw):
        twist = list(kw.get('twist', [0.0, 0.0, 0.0]))
        _calls16h['commands'] += 1
        if all(abs(v) < 1e-9 for v in twist):
            self.zero_commands += 1
            if self.zero_commands == 1:
                self.drone.position = [0.19, 0.0, 1.0]
        else:
            _calls16h['corrections'] += 1
            self.drone.position = [0.0, 0.0, 1.0]


def _apply_start_for_hold_reacquire(start_pose):
    _calls16h['apply'] += 1
    _inner16h._drone.position = list(start_pose[:3])
    _inner16h._drone.orientation = [0.0, 0.0, start_pose[3]]


_inner16h._apply_start_pose = _apply_start_for_hold_reacquire
_inner16h._reset_velocity_controller = lambda: setattr(_inner16h, '_speed_handler', _DriftThenCorrectHandler(_inner16h._drone))
_obs16h, _info16h = _env16h.reset()
check("post-controller hold reacquires drift without full reset retry", _calls16h['apply'] == 1 and _calls16h['corrections'] >= 1)
check("post-controller hold finishes at requested start pose", _inner16h._drone.position == [0.0, 0.0, 1.0])
check("post-controller hold emits normalized obs", np.all(_obs16h >= -1.0) and np.all(_obs16h <= 1.0))
_env16h.close()

_env16h1 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16h1',
    step_duration=0.0,
    max_vel=0.5,
    reset_max_vel=0.25,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    fixed_start_tolerance=0.15,
    hover_settle_time=0.0,
    hover_timeout=0.2,
    reset_min_speed=0.2,
    use_simulator_reset_service=False,
)
_inner16h1 = _env16h1.unwrapped
_inner16h1._drone = fake_drone([2.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_hold_correction_commands16h1 = []


class _RecordThenReacquireHoldHandler:
    def __init__(self, drone):
        self.drone = drone

    def send_speed_command_with_yaw_speed(self, **kw):
        twist = list(kw.get('twist', [0.0, 0.0, 0.0]))
        _hold_correction_commands16h1.append(twist)
        if any(abs(v) > 1e-9 for v in twist):
            self.drone.position = [0.0, 0.0, 1.0]
        return True


_inner16h1._speed_handler = _RecordThenReacquireHoldHandler(_inner16h1._drone)
_hold_reacquire16h1 = _inner16h1._hold_start_pose_after_controller_reset([0.0, 0.0, 1.0, 0.0], 1)
_first_hold_correction16h1 = next((cmd for cmd in _hold_correction_commands16h1 if any(abs(v) > 1e-9 for v in cmd)), [0.0, 0.0, 0.0])
check("post-controller reset sends zero command before correction", _hold_correction_commands16h1 and all(abs(v) < 1e-9 for v in _hold_correction_commands16h1[0]))
check("post-controller reacquire correction uses reset_max_vel", _hold_reacquire16h1 is True and math.isclose(_first_hold_correction16h1[0], -0.25, rel_tol=1e-6))
_env16h1.close()

_env16h1a = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16h1a',
    step_duration=0.0,
    reset_max_vel=1.0,
    reset_xy_kp=4.0,
    fixed_start_tolerance=0.15,
    reset_min_speed=0.2,
)
_inner16h1a = _env16h1a.unwrapped
_small_lateral_cmd16h1a = _inner16h1a._reset_lateral_axis_command(
    np.array([0.0, 0.314, 0.0], dtype=np.float32),
    1,
)
_larger_lateral_cmd16h1a = _inner16h1a._reset_lateral_axis_command(
    np.array([0.0, 0.45, 0.0], dtype=np.float32),
    1,
)
_near_cap16h1a = _inner16h1a.reset_max_vel * 0.314 / (3.0 * _inner16h1a._reset_position_tolerance())
check("small lateral post-controller error does not command full reset_max_vel", abs(_small_lateral_cmd16h1a[1]) < _inner16h1a.reset_max_vel)
check("lateral post-controller command is capped near tolerance", abs(_small_lateral_cmd16h1a[1]) <= _near_cap16h1a + 1e-6)
check("lateral post-controller command scales with error", abs(_larger_lateral_cmd16h1a[1]) > abs(_small_lateral_cmd16h1a[1]))
_env16h1a.close()

_env16h1a2 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16h1a2',
    step_duration=0.0,
    reset_max_vel=1.0,
    reset_xy_kp=0.1,
    fixed_start_tolerance=0.15,
    reset_min_speed=0.5,
)
_inner16h1a2 = _env16h1a2.unwrapped
_near_min_lateral_cmd16h1a2 = _inner16h1a2._reset_lateral_axis_command(
    np.array([0.0, 0.17, 0.0], dtype=np.float32),
    1,
)
_near_min_cap16h1a2 = _inner16h1a2.reset_max_vel * 0.17 / (3.0 * _inner16h1a2._reset_position_tolerance())
check("lateral min speed is bounded by near-target cap", abs(_near_min_lateral_cmd16h1a2[1]) <= _near_min_cap16h1a2 + 1e-6)
check("lateral min speed does not force overshoot near tolerance", abs(_near_min_lateral_cmd16h1a2[1]) < _inner16h1a2.reset_min_speed)
_env16h1a2.close()

_env16h1a3 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16h1a3',
    step_duration=0.0,
    reset_max_vel=1.0,
    reset_xy_kp=1.0,
    fixed_start_tolerance=0.15,
    reset_min_speed=0.5,
)
_inner16h1a3 = _env16h1a3.unwrapped
_diagonal_lateral_error16h1a3 = np.array([0.12, 0.12, 0.0], dtype=np.float32)
_diagonal_lateral_cmd16h1a3 = _inner16h1a3._reset_lateral_axis_command(
    _diagonal_lateral_error16h1a3,
    0,
)
_diagonal_near_cap16h1a3 = (
    _inner16h1a3.reset_max_vel
    * abs(float(_diagonal_lateral_error16h1a3[0]))
    / (3.0 * _inner16h1a3._reset_position_tolerance())
)
check(
    "diagonal lateral error outside reset tolerance commands correction",
    math.hypot(float(_diagonal_lateral_error16h1a3[0]), float(_diagonal_lateral_error16h1a3[1]))
    > _inner16h1a3._reset_position_tolerance()
    and abs(float(_diagonal_lateral_cmd16h1a3[0])) > 1e-9,
)
check(
    "diagonal lateral correction remains near-target capped",
    abs(float(_diagonal_lateral_cmd16h1a3[0])) <= _diagonal_near_cap16h1a3 + 1e-6,
)
_env16h1a3.close()

_env16h1a4 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16h1a4',
    step_duration=0.0,
    reset_max_vel=1.0,
    reset_xy_kp=4.0,
    reset_z_kp=4.0,
    fixed_start_tolerance=0.15,
    reset_min_speed=0.5,
    reset_yaw_required=False,
)
_inner16h1a4 = _env16h1a4.unwrapped
_near_3d_error16h1a4 = np.array([0.1, 0.1, 0.1], dtype=np.float32)
_near_3d_cmd16h1a4, _near_3d_yaw16h1a4, _near_3d_phase16h1a4 = _inner16h1a4._phased_reset_command(
    _near_3d_error16h1a4,
    current_z=0.9,
    target_z=1.0,
    yaw_error=1.5,
)
_near_3d_cap16h1a4 = (
    _inner16h1a4.reset_max_vel
    * float(np.linalg.norm(_near_3d_error16h1a4))
    / (3.0 * _inner16h1a4._reset_position_tolerance())
)
check(
    "near-threshold 3D hold error outside reset tolerance commands correction",
    float(np.linalg.norm(_near_3d_error16h1a4)) > _inner16h1a4._reset_position_tolerance()
    and math.hypot(float(_near_3d_error16h1a4[0]), float(_near_3d_error16h1a4[1])) <= _inner16h1a4._reset_position_tolerance()
    and abs(float(_near_3d_error16h1a4[2])) <= _inner16h1a4._reset_position_tolerance()
    and float(np.linalg.norm(_near_3d_cmd16h1a4)) > 1e-9
    and _near_3d_phase16h1a4 == '3d_trim',
)
check(
    "near-threshold 3D hold correction remains conservative and yaw-disabled",
    float(np.linalg.norm(_near_3d_cmd16h1a4)) <= _near_3d_cap16h1a4 + 1e-6
    and math.isclose(_near_3d_yaw16h1a4, 0.0, abs_tol=1e-9),
)
_env16h1a4.close()

_env16h1a5 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16h1a5',
    step_duration=0.0,
    reset_max_vel=1.0,
    reset_xy_kp=4.0,
    reset_z_kp=4.0,
    fixed_start_tolerance=0.15,
    reset_min_speed=0.5,
    reset_yaw_required=False,
)
_inner16h1a5 = _env16h1a5.unwrapped
_high_z_error16h1a5 = np.array([1.0, -0.7, -1.76], dtype=np.float32)
_high_z_cmd16h1a5, _high_z_yaw16h1a5, _high_z_phase16h1a5 = _inner16h1a5._phased_reset_command(
    _high_z_error16h1a5,
    current_z=3.36,
    target_z=1.6,
    yaw_error=1.5,
)
check(
    "high-altitude reset prioritizes descending before XY correction",
    _high_z_phase16h1a5 == 'z_high_trim'
    and math.isclose(float(_high_z_cmd16h1a5[0]), 0.0, abs_tol=1e-9)
    and math.isclose(float(_high_z_cmd16h1a5[1]), 0.0, abs_tol=1e-9)
    and float(_high_z_cmd16h1a5[2]) < 0.0
    and math.isclose(_high_z_yaw16h1a5, 0.0, abs_tol=1e-9),
)

_near_high_z_error16h1a5 = np.array([0.4, 0.0, -0.18], dtype=np.float32)
_near_high_z_cmd16h1a5, _, _near_high_z_phase16h1a5 = _inner16h1a5._phased_reset_command(
    _near_high_z_error16h1a5,
    current_z=1.78,
    target_z=1.6,
    yaw_error=0.0,
)
check(
    "near-target high-altitude trim remains conservative",
    _near_high_z_phase16h1a5 == 'z_high_trim'
    and float(_near_high_z_cmd16h1a5[2]) < 0.0
    and abs(float(_near_high_z_cmd16h1a5[2])) < _inner16h1a5.reset_max_vel,
)
_env16h1a5.close()

_env16h1a6 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16h1a6',
    step_duration=0.0,
    reset_max_vel=1.0,
    reset_xy_kp=4.0,
    reset_z_kp=4.0,
    fixed_start_tolerance=0.15,
    reset_min_speed=0.5,
)
_inner16h1a6 = _env16h1a6.unwrapped
_low_z_error16h1a6 = np.array([1.0, 0.0, 0.8], dtype=np.float32)
_low_z_cmd16h1a6, _low_z_yaw16h1a6, _low_z_phase16h1a6 = _inner16h1a6._phased_reset_command(
    _low_z_error16h1a6,
    current_z=0.2,
    target_z=1.0,
    yaw_error=1.5,
)
check(
    "low-altitude recovery still takes precedence over high-altitude and XY trim",
    _low_z_phase16h1a6 == 'vertical_recovery'
    and math.isclose(float(_low_z_cmd16h1a6[0]), 0.0, abs_tol=1e-9)
    and math.isclose(float(_low_z_cmd16h1a6[1]), 0.0, abs_tol=1e-9)
    and float(_low_z_cmd16h1a6[2]) > 0.0
    and math.isclose(_low_z_yaw16h1a6, 0.0, abs_tol=1e-9),
)
_env16h1a6.close()

_env16h1b = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16h1b',
    step_duration=0.01,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    fixed_start_tolerance=0.15,
    fixed_start_timeout=0.45,
    hover_settle_time=0.05,
    hover_timeout=0.12,
    reset_min_speed=0.2,
    use_simulator_reset_service=False,
)
_inner16h1b = _env16h1b.unwrapped
_inner16h1b._drone = fake_drone([0.6, 0.0, 1.0], [0.0, 0.0, 0.0])


class _SlowPostControllerReacquireHandler:
    def __init__(self, drone):
        self.drone = drone
        self.corrections = 0

    def send_speed_command_with_yaw_speed(self, **kw):
        twist = list(kw.get('twist', [0.0, 0.0, 0.0]))
        if any(abs(v) > 1e-9 for v in twist):
            self.corrections += 1
            if self.corrections >= 3:
                self.drone.position = [0.0, 0.0, 1.0]
        return True


_handler16h1b = _SlowPostControllerReacquireHandler(_inner16h1b._drone)
_inner16h1b._speed_handler = _handler16h1b
_slow_reacquire_started16h1b = time.time()
_slow_reacquire16h1b = _inner16h1b._hold_start_pose_after_controller_reset([0.0, 0.0, 1.0, 0.0], 1)
_slow_reacquire_elapsed16h1b = time.time() - _slow_reacquire_started16h1b
check("post-controller reacquire may exceed hover timeout", _slow_reacquire16h1b is True and _slow_reacquire_elapsed16h1b > _inner16h1b.hover_timeout)
check("post-controller reacquire remains bounded by fixed-start timeout", _slow_reacquire_elapsed16h1b < _inner16h1b.fixed_start_timeout)
_env16h1b.close()

_env16h1c = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16h1c',
    step_duration=0.01,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    fixed_start_tolerance=0.15,
    fixed_start_timeout=0.12,
    hover_settle_time=0.05,
    hover_timeout=0.05,
    reset_min_speed=0.2,
    use_simulator_reset_service=False,
)
_inner16h1c = _env16h1c.unwrapped
_inner16h1c._drone = fake_drone([0.8, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner16h1c._speed_handler = FakeSpeedHandler()
_far_reacquire16h1c = _inner16h1c._hold_start_pose_after_controller_reset([0.0, 0.0, 1.0, 0.0], 1)
check("post-controller far timeout fails hold", _far_reacquire16h1c is False)
check("post-controller far timeout reports reacquire diagnostic", _inner16h1c._last_reset_diagnostics.get('reason') == 'post_controller_reacquire_timeout')
check("post-controller reacquire timeout reports progress diagnostics", all(k in _inner16h1c._last_reset_diagnostics for k in ['initial_position_error', 'best_position_error', 'error_reduction', 'dominant_axis']))
_env16h1c.close()

_env16h1c2 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16h1c2',
    step_duration=0.01,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    fixed_start_tolerance=0.15,
    fixed_start_timeout=0.12,
    hover_settle_time=0.2,
    hover_timeout=0.05,
    reset_min_speed=0.2,
    use_simulator_reset_service=False,
)
_inner16h1c2 = _env16h1c2.unwrapped
_inner16h1c2._drone = fake_drone([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])


class _AcquireThenFarNoReacquireHandler:
    def __init__(self, drone):
        self.drone = drone

    def send_speed_command_with_yaw_speed(self, **kw):
        self.drone.position = [0.8, 0.0, 1.0]
        return True


_inner16h1c2._speed_handler = _AcquireThenFarNoReacquireHandler(_inner16h1c2._drone)
_lost_after_acquire16h1c2 = _inner16h1c2._hold_start_pose_after_controller_reset([0.0, 0.0, 1.0, 0.0], 1)
check("post-controller acquired-then-far timeout fails hold", _lost_after_acquire16h1c2 is False)
check("post-controller acquired-then-far timeout reports reacquire diagnostic", _inner16h1c2._last_reset_diagnostics.get('reason') == 'post_controller_reacquire_timeout')
_env16h1c2.close()

_env16h1d = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16h1d',
    step_duration=0.01,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    fixed_start_tolerance=0.15,
    fixed_start_timeout=0.4,
    hover_settle_time=0.11,
    hover_timeout=0.12,
    reset_min_speed=0.2,
    use_simulator_reset_service=False,
)
_inner16h1d = _env16h1d.unwrapped
_inner16h1d._drone = fake_drone([0.16, 0.0, 1.0], [0.0, 0.0, 0.0])


class _NearButUnstableHoldHandler:
    def __init__(self, drone):
        self.drone = drone
        self.calls = 0

    def send_speed_command_with_yaw_speed(self, **kw):
        self.calls += 1
        if self.calls % 2 == 1:
            self.drone.position = [0.18, 0.0, 1.0]
        else:
            self.drone.position = [0.16, 0.0, 1.0]
        return True


_inner16h1d._speed_handler = _NearButUnstableHoldHandler(_inner16h1d._drone)
_near_unstable16h1d = _inner16h1d._hold_start_pose_after_controller_reset([0.0, 0.0, 1.0, 0.0], 1)
check("post-controller near unstable timeout fails hold", _near_unstable16h1d is False)
check("post-controller near unstable timeout reports hold diagnostic", _inner16h1d._last_reset_diagnostics.get('reason') == 'post_controller_hold_timeout')
_env16h1d.close()

_env16h2 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16h2',
    step_duration=0.01,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    fixed_start_tolerance=0.15,
    hover_settle_time=0.11,
    hover_timeout=0.5,
    reset_min_speed=0.2,
    use_simulator_reset_service=False,
)
_inner16h2 = _env16h2.unwrapped
_calls16h2 = {'recover': 0, 'accepted_corrections': 0}
_inner16h2._drone = fake_drone([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner16h2._is_flying = True


class _RejectHoldCorrectionOnceHandler:
    def __init__(self, drone):
        self.drone = drone
        self.zero_commands = 0

    def send_speed_command_with_yaw_speed(self, **kw):
        twist = list(kw.get('twist', [0.0, 0.0, 0.0]))
        if all(abs(v) < 1e-9 for v in twist):
            self.zero_commands += 1
            if self.zero_commands == 1:
                self.drone.position = [0.19, 0.0, 1.0]
            return True
        return False


class _AcceptHoldCorrectionHandler:
    def __init__(self, drone):
        self.drone = drone

    def send_speed_command_with_yaw_speed(self, **kw):
        twist = list(kw.get('twist', [0.0, 0.0, 0.0]))
        if any(abs(v) > 1e-9 for v in twist):
            _calls16h2['accepted_corrections'] += 1
            self.drone.position = [0.0, 0.0, 1.0]
        return True


def _apply_start_for_hold_recovery(start_pose):
    _inner16h2._drone.position = list(start_pose[:3])
    _inner16h2._drone.orientation = [0.0, 0.0, start_pose[3]]


def _recover_hold_command_path16h2():
    _calls16h2['recover'] += 1
    _inner16h2._speed_handler = _AcceptHoldCorrectionHandler(_inner16h2._drone)
    return True


_inner16h2._apply_start_pose = _apply_start_for_hold_recovery
_inner16h2._reset_velocity_controller = lambda: setattr(_inner16h2, '_speed_handler', _RejectHoldCorrectionOnceHandler(_inner16h2._drone)) or True
_inner16h2._recover_motion_reference_path = _recover_hold_command_path16h2
_obs16h2, _info16h2 = _env16h2.reset()
check("post-controller hold recovers rejected correction command", _calls16h2 == {'recover': 1, 'accepted_corrections': 1})
check("post-controller hold recovery finishes at requested start pose", _inner16h2._drone.position == [0.0, 0.0, 1.0])
check("post-controller hold recovery emits normalized obs", np.all(_obs16h2 >= -1.0) and np.all(_obs16h2 <= 1.0))
_env16h2.close()

_env16h3 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16h3',
    step_duration=0.01,
    fixed_start_pose=[0.0, 0.0, 1.6, 0.0],
    fixed_start_tolerance=0.22,
    reset_yaw_tolerance=1.2,
    unsafe_low_altitude_threshold=1.3,
    vertical_safety_band=0.3,
    hover_settle_time=0.02,
    hover_timeout=0.2,
    reset_min_speed=0.2,
    use_simulator_reset_service=False,
)
_inner16h3 = _env16h3.unwrapped
_hold_commands16h3 = []
_inner16h3._drone = fake_drone([0.1, 0.0, 1.35], [0.0, 0.0, 0.0], yaw=0.5)


class _VerticalFirstHoldHandler:
    def __init__(self, drone):
        self.drone = drone

    def send_speed_command_with_yaw_speed(self, **kw):
        twist = list(kw.get('twist', [0.0, 0.0, 0.0]))
        yaw_speed = float(kw.get('yaw_speed', 999.0))
        _hold_commands16h3.append((twist, yaw_speed))
        if any(abs(v) > 1e-9 for v in twist):
            self.drone.position = [0.0, 0.0, 1.6]
        return True


_inner16h3._speed_handler = _VerticalFirstHoldHandler(_inner16h3._drone)
_hold_vertical_first16h3 = _inner16h3._hold_start_pose_after_controller_reset([0.0, 0.0, 1.6, 0.0], 1)
_first_hold_command16h3 = next((cmd for cmd in _hold_commands16h3 if any(abs(v) > 1e-9 for v in cmd[0])), ([999.0, 999.0, 999.0], 999.0))
check("post-controller hold prioritizes vertical recovery near safety floor", _hold_vertical_first16h3 is True)
check(
    "post-controller hold suppresses lateral/yaw correction during vertical recovery",
    math.isclose(_first_hold_command16h3[0][0], 0.0, abs_tol=1e-9)
    and math.isclose(_first_hold_command16h3[0][1], 0.0, abs_tol=1e-9)
    and _first_hold_command16h3[0][2] > 0.0
    and math.isclose(_first_hold_command16h3[1], 0.0, abs_tol=1e-9),
)
_env16h3.close()

_env16h4 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16h4',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    fixed_start_tolerance=0.15,
    reset_yaw_tolerance=0.38,
    reset_yaw_required=False,
    hover_settle_time=0.01,
    hover_timeout=0.2,
    use_simulator_reset_service=False,
)
_inner16h4 = _env16h4.unwrapped
_inner16h4._drone = fake_drone([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], yaw=2.6)
_inner16h4._is_flying = True
_handler16h4 = RecordingSpeedHandler(_inner16h4._drone)
_inner16h4._speed_handler = _handler16h4
_inner16h4._wait_for_hover_settle = _stub_wait_for_hover_settle
_inner16h4._reset_velocity_controller = lambda: True
_inner16h4._apply_start_pose = lambda start_pose: None
_obs16h4, _info16h4 = _env16h4.reset()
check("position-only episode reset ignores large yaw during hold", _info16h4.get('reset_success') is True and _info16h4.get('reset_method') == 'velocity')
check("position-only episode reset keeps yaw diagnostic in monitor info", math.isclose(_info16h4.get('reset_yaw_error'), 2.6, abs_tol=1e-3))
check("position-only episode reset commands zero yaw during hold", all(math.isclose(cmd[1], 0.0, abs_tol=1e-9) for cmd in _handler16h4.commands))
_env16h4.close()

_env16h5 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16h5',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    fixed_start_tolerance=0.15,
    reset_yaw_required=False,
    hover_settle_time=0.0,
    hover_timeout=0.2,
    reset_min_speed=0.2,
    use_simulator_reset_service=False,
)
_inner16h5 = _env16h5.unwrapped
_inner16h5._drone = fake_drone([0.6, 0.3, 1.05], [0.0, 0.0, 0.0], yaw=2.2)
_axis_commands16h5 = []


class _AxisPriorityPositionOnlyHandler:
    def __init__(self, drone):
        self.drone = drone

    def send_speed_command_with_yaw_speed(self, **kw):
        twist = list(kw.get('twist', [0.0, 0.0, 0.0]))
        yaw_speed = float(kw.get('yaw_speed', 999.0))
        _axis_commands16h5.append((twist, yaw_speed))
        if any(abs(v) > 1e-9 for v in twist):
            self.drone.position = [0.0, 0.0, 1.0]
        return True


_inner16h5._speed_handler = _AxisPriorityPositionOnlyHandler(_inner16h5._drone)
_axis_priority16h5 = _inner16h5._hold_start_pose_after_controller_reset([0.0, 0.0, 1.0, 0.0], 1)
_first_axis_command16h5 = next((cmd for cmd in _axis_commands16h5 if any(abs(v) > 1e-9 for v in cmd[0])), ([999.0, 999.0, 999.0], 999.0))
check("position-only XY phase uses one dominant horizontal axis", _axis_priority16h5 is True and _first_axis_command16h5[0][0] < 0.0 and math.isclose(_first_axis_command16h5[0][1], 0.0, abs_tol=1e-9))
check("position-only XY phase does not command unnecessary vertical/yaw", math.isclose(_first_axis_command16h5[0][2], 0.0, abs_tol=1e-9) and math.isclose(_first_axis_command16h5[1], 0.0, abs_tol=1e-9))
_env16h5.close()

_env16i = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16i',
    step_duration=0.01,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    fixed_start_tolerance=0.15,
    fixed_start_timeout=0.2,
    hover_settle_time=0.11,
    hover_timeout=0.16,
    reset_min_speed=0.2,
    use_simulator_reset_service=False,
)
_inner16i = _env16i.unwrapped
_inner16i._drone = fake_drone([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner16i._is_flying = True


class _PersistentDriftHandler:
    def send_speed_command_with_yaw_speed(self, **kw):
        _inner16i._drone.position = [0.19, 0.0, 1.0]


_inner16i._apply_start_pose = lambda start_pose: setattr(_inner16i._drone, 'position', list(start_pose[:3]))
_inner16i._reset_velocity_controller = lambda: setattr(_inner16i, '_speed_handler', _PersistentDriftHandler())
try:
    _env16i.reset()
    _hold_timeout_raised = False
    _hold_timeout_text = ''
except RuntimeError as _exc16i:
    _hold_timeout_raised = True
    _hold_timeout_text = str(_exc16i)
check("post-controller hold failure is bounded", _hold_timeout_raised)
check("post-controller lost hold reports reacquire diagnostics", 'post_controller_reacquire_timeout' in _hold_timeout_text and 'last_command=' in _hold_timeout_text)
_env16i.close()

_env16j = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16j',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    fixed_start_tolerance=0.15,
    use_simulator_reset_service=False,
)
_inner16j = _env16j.unwrapped
_calls16j = {'apply': 0, 'hold': 0}
_inner16j._drone = fake_drone([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner16j._is_flying = True
_inner16j._wait_for_hover_settle = _stub_wait_for_hover_settle


def _apply_start_for_timeout_at_pose(start_pose):
    _calls16j['apply'] += 1
    _inner16j._drone.position = list(start_pose[:3])
    _inner16j._drone.orientation = [0.0, 0.0, start_pose[3]]


def _hold_timeout_at_pose(start_pose, attempt):
    _calls16j['hold'] += 1
    _inner16j._drone.position = list(start_pose[:3])
    _inner16j._drone.orientation = [0.0, 0.0, start_pose[3]]
    _inner16j._last_reset_diagnostics = {
        'reason': 'post_controller_hold_timeout',
        'attempt': attempt,
        'target_pose': [float(v) for v in start_pose],
        'final_pose': [float(v) for v in start_pose],
        'position_error': 0.0,
        'yaw_error': 0.0,
        'last_command': [0.0, 0.0, 0.0, 0.0],
    }
    return False


_inner16j._apply_start_pose = _apply_start_for_timeout_at_pose
_inner16j._reset_velocity_controller = lambda: None
_inner16j._hold_start_pose_after_controller_reset = _hold_timeout_at_pose
try:
    _env16j.reset()
    _timeout_at_pose_raised = False
    _timeout_at_pose_text = ''
except RuntimeError as _exc16j:
    _timeout_at_pose_raised = True
    _timeout_at_pose_text = str(_exc16j)
check("post-controller hold timeout retries even when final pose is in tolerance", _calls16j['apply'] == 2 and _calls16j['hold'] == 2)
check("post-controller hold timeout at pose does not pass reset", _timeout_at_pose_raised)
check("post-controller hold timeout at pose reports diagnostics", 'post_controller_hold_timeout' in _timeout_at_pose_text)
_env16j.close()

_env16k = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16k',
    step_duration=0.01,
    fixed_start_timeout=0.2,
    fixed_start_tolerance=0.15,
)
_inner16k = _env16k.unwrapped
_calls16k = {'recover': 0, 'accepted': 0}
_inner16k._drone = fake_drone([0.4, 0.0, 1.0], [0.0, 0.0, 0.0], yaw=0.0)


class _RejectThenRecoverResetHandler:
    def __init__(self, drone):
        self.drone = drone
        self.calls = 0

    def send_speed_command_with_yaw_speed(self, **kw):
        self.calls += 1
        return False


class _AcceptingRecoveredResetHandler:
    def __init__(self, drone):
        self.drone = drone

    def send_speed_command_with_yaw_speed(self, **kw):
        twist = list(kw.get('twist', [0.0, 0.0, 0.0]))
        _calls16k['accepted'] += 1
        if any(abs(v) > 1e-9 for v in twist):
            self.drone.position = [0.0, 0.0, 1.0]
        return True


def _recover_rejected_reset_command16k():
    _calls16k['recover'] += 1
    _inner16k._speed_handler = _AcceptingRecoveredResetHandler(_inner16k._drone)
    return True


_inner16k._speed_handler = _RejectThenRecoverResetHandler(_inner16k._drone)
_inner16k._recover_motion_reference_path = _recover_rejected_reset_command16k
_reset_recovered16k = _inner16k._drive_to_start_pose_with_velocity([0.0, 0.0, 1.0, 0.0])
check("velocity reset recovers rejected reset command path", _reset_recovered16k is True and _calls16k['recover'] == 1)
check("velocity reset retries command after reset-path recovery", _calls16k['accepted'] >= 1)
_env16k.close()

_env16l = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone16l',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    use_simulator_reset_service=False,
)
_inner16l = _env16l.unwrapped
_calls16l = {'arm': 0, 'offboard': 0, 'reset_controller': 0, 'apply': 0}
_inner16l._drone = fake_drone([0.3, 0.0, 1.0], [0.0, 0.0, 0.0], yaw=0.0)
_inner16l._is_flying = True
_inner16l._wait_for_hover_settle = _stub_wait_for_hover_settle


def _record_arm16l(**kw):
    _calls16l['arm'] += 1
    return True


def _record_offboard16l(**kw):
    _calls16l['offboard'] += 1
    return True


def _record_reset_controller16l():
    _calls16l['reset_controller'] += 1
    _inner16l._speed_handler = RecordingSpeedHandler(_inner16l._drone)
    return True


def _record_apply_start16l(start_pose):
    _calls16l['apply'] += 1
    _inner16l._drone.position = list(start_pose[:3])
    _inner16l._drone.orientation = [0.0, 0.0, start_pose[3]]


_inner16l._drone.arm = _record_arm16l
_inner16l._drone.offboard = _record_offboard16l
_inner16l._reset_velocity_controller = _record_reset_controller16l
_inner16l._apply_start_pose = _record_apply_start16l
_obs16l, _info16l = _env16l.reset()
check("already-flying fixed reset reasserts offboard before velocity reset", _calls16l['arm'] == 1 and _calls16l['offboard'] == 1)
check("already-flying fixed reset still uses velocity reset path", _calls16l['reset_controller'] >= 1 and _calls16l['apply'] >= 1)
check("already-flying fixed reset after offboard reassert emits normalized obs", np.all(_obs16l >= -1.0) and np.all(_obs16l <= 1.0))
_env16l.close()

# ===========================================================================
# TEST 17 — Single-drone smoke helper timeout contracts
# ===========================================================================
print("\n[17] Single-drone smoke helper timeout contracts")

_connection_path = Path(__file__).resolve().parent / 'test_connection.py'
_connection_spec = importlib.util.spec_from_file_location('test_connection', _connection_path)
_connection = importlib.util.module_from_spec(_connection_spec)
assert _connection_spec is not None and _connection_spec.loader is not None
_connection_spec.loader.exec_module(_connection)

_fast_result = _connection.run_with_timeout(lambda: 'ready', 0.5, 'fast-op')
check("test_connection: bounded helper returns successful result", _fast_result == 'ready')

try:
    _connection.run_with_timeout(lambda: time.sleep(0.2), 0.01, 'slow-op')
    _timeout_raised = False
except _connection.OperationTimeout as _exc14:
    _timeout_raised = 'slow-op exceeded 0.0s' in str(_exc14) or 'slow-op exceeded 0.01s' in str(_exc14)
check("test_connection: bounded helper raises visible timeout", _timeout_raised)

try:
    _connection.run_with_timeout(lambda: (_ for _ in ()).throw(RuntimeError('boom')), 0.5, 'failing-op')
    _error_propagated = False
except RuntimeError as _exc14b:
    _error_propagated = str(_exc14b) == 'boom'
check("test_connection: bounded helper propagates operation errors", _error_propagated)

# ===========================================================================
# TEST 18 — Simulator overlay source prerequisite
# ===========================================================================
print("\n[18] Simulator overlay source prerequisite")

_overlay_src = Path('/home/jordi/as2_rl_ws/src')
_simulator_source = _overlay_src / 'as2_platform_multirotor_simulator'
_simulator_package_xml = _simulator_source / 'package.xml'
_simulator_cmake = _simulator_source / 'CMakeLists.txt'
_simulator_source_dsc = _overlay_src / 'ros-humble-as2-platform-multirotor-simulator_1.1.3-1jammy.dsc'
_simulator_pin_doc = _simulator_source / 'OVERLAY_PIN.md'

check("overlay source package directory exists", _simulator_source.is_dir())
check("overlay source package.xml exists", _simulator_package_xml.is_file())
check("overlay source CMakeLists.txt exists", _simulator_cmake.is_file())
if _simulator_package_xml.is_file():
    _package_xml_text = _simulator_package_xml.read_text(encoding='utf-8')
else:
    _package_xml_text = ''
check(
    "overlay package name is as2_platform_multirotor_simulator",
    '<name>as2_platform_multirotor_simulator</name>' in _package_xml_text,
)
check("overlay source pin .dsc exists", _simulator_source_dsc.is_file())
check("overlay package version is pinned to 1.1.3", '<version>1.1.3</version>' in _package_xml_text)
if _simulator_pin_doc.is_file():
    _simulator_pin_doc_text = _simulator_pin_doc.read_text(encoding='utf-8')
else:
    _simulator_pin_doc_text = ''
check("overlay pin document exists", _simulator_pin_doc.is_file())
check("overlay pin document records apt source package", 'ros-humble-as2-platform-multirotor-simulator' in _simulator_pin_doc_text)
check("overlay pin document records version 1.1.3-1jammy", '1.1.3-1jammy' in _simulator_pin_doc_text)
check("colcon command is available in rl_uav environment", shutil.which('colcon') is not None)
_colcon_list = subprocess.run(
    [
        'bash',
        '-lc',
        'source /opt/ros/humble/setup.bash && colcon list --base-paths /home/jordi/as2_rl_ws/src --names-only',
    ],
    check=False,
    capture_output=True,
    text=True,
)
check("colcon list command succeeds for overlay src", _colcon_list.returncode == 0)
check(
    "colcon sees as2_platform_multirotor_simulator from overlay",
    'as2_platform_multirotor_simulator' in _colcon_list.stdout.splitlines(),
)

# ===========================================================================
# TEST 19 — Deterministic reset validator contracts
# ===========================================================================
print("\n[19] Deterministic reset validator contracts")

_reset_validator_path = Path(__file__).resolve().parent / 'validate_deterministic_reset.py'
_reset_validator_text = _reset_validator_path.read_text(encoding='utf-8') if _reset_validator_path.is_file() else ''
_reset_validator_spec = importlib.util.spec_from_file_location('validate_deterministic_reset', _reset_validator_path)
_reset_validator = importlib.util.module_from_spec(_reset_validator_spec)
assert _reset_validator_spec is not None and _reset_validator_spec.loader is not None
_reset_validator_spec.loader.exec_module(_reset_validator)

check(
    "reset validator service name is absolute and namespaced",
    _reset_validator.reset_service_name('drone1') == '/drone1/platform/reset_simulator_state',
)


class _ValidatorShapeRequest:
    __slots__ = [
        '_x',
        '_y',
        '_z',
        '_yaw',
        '_position_tolerance',
        '_yaw_tolerance',
        '_linear_speed_tolerance',
        '_angular_speed_tolerance',
    ]


class _ValidatorShapeResponse:
    __slots__ = [
        '_success',
        '_message',
        '_position_error',
        '_yaw_error',
        '_linear_speed_norm',
        '_angular_speed_norm',
        '_linear_acceleration_norm',
        '_angular_acceleration_norm',
        '_force_norm',
        '_torque_norm',
        '_motor_angular_velocity_norm',
        '_motor_angular_acceleration_norm',
        '_dynamics_reset_applied',
        '_motor_state_reset_applied',
        '_controller_reset_applied',
        '_imu_reset_applied',
        '_odometry_reset_applied',
        '_references_reset_applied',
        '_dynamic_artifacts_cleared',
    ]


class _ValidatorShapeService:
    Request = _ValidatorShapeRequest
    Response = _ValidatorShapeResponse


_runtime_overlay_report19 = _reset_validator.build_runtime_overlay_report(
    simulator_prefix='/home/jordi/as2_rl_ws/install/as2_platform_multirotor_simulator',
    expected_simulator_prefix='/home/jordi/as2_rl_ws',
    service_type=_ValidatorShapeService,
    fork_path=_simulator_source,
)
check(
    "reset validator reports sourced simulator package prefix",
    _runtime_overlay_report19['simulator_prefix'] == '/home/jordi/as2_rl_ws/install/as2_platform_multirotor_simulator'
    and _runtime_overlay_report19['expected_simulator_prefix'] == '/home/jordi/as2_rl_ws',
)
check(
    "reset validator reports ResetSimulatorState request and response shape",
    'x' in _runtime_overlay_report19['service_shape']['request_fields']
    and 'angular_speed_tolerance' in _runtime_overlay_report19['service_shape']['request_fields']
    and 'dynamic_artifacts_cleared' in _runtime_overlay_report19['service_shape']['response_fields'],
)
check(
    "reset validator reports fork remote branch commit state without mutation",
    all(key in _runtime_overlay_report19['fork_git'] for key in ['is_git_repo', 'remote_url', 'branch', 'commit', 'status']),
)
try:
    _reset_validator.validate_simulator_prefix('/opt/ros/humble', '/home/jordi/as2_rl_ws')
    check("reset validator rejects base-install prefix ambiguity", False)
except RuntimeError as _exc19:
    check(
        "reset validator rejects base-install prefix ambiguity",
        'base install' in str(_exc19).lower() or 'outside expected overlay' in str(_exc19).lower(),
    )
check(
    "reset validator documents conda and AS2 Multirotor-only execution",
    'conda run -n rl_uav python3 scripts/validate_deterministic_reset.py' in _reset_validator_text
    and 'AS2 Multirotor Simulator' in _reset_validator_text
    and 'Gazebo' in _reset_validator_text
    and 'not an acceptance path' in _reset_validator_text,
)


class _ValidatorResetSimulatorStateRequest:
    pass


class _ValidatorResetSimulatorStateService:
    Request = _ValidatorResetSimulatorStateRequest

_live_request19 = _reset_validator.build_reset_request(
    _ValidatorResetSimulatorStateService,
    target_pose=[1.0, -2.0, 1.5, 0.3],
    position_tolerance=0.12,
    yaw_tolerance=0.21,
    speed_tolerance=0.04,
)
check(
    "reset validator builds live service request fields",
    [_live_request19.x, _live_request19.y, _live_request19.z, _live_request19.yaw] == [1.0, -2.0, 1.5, 0.3]
    and _live_request19.position_tolerance == 0.12
    and _live_request19.yaw_tolerance == 0.21
    and _live_request19.linear_speed_tolerance == 0.04
    and _live_request19.angular_speed_tolerance == 0.04,
)

_live_obs19 = _reset_validator.normalized_observation_from_state(
    position=[0.1, -0.2, 1.0],
    yaw=0.3,
    target_pose=[0.0, 0.0, 1.0, 0.0],
    pos_limit=5.0,
)
check(
    "reset validator builds normalized live observation",
    np.allclose(_live_obs19, [0.02, -0.04, 0.0, 0.3 / math.pi], atol=1e-6),
)
_clipped_live_obs19 = _reset_validator.normalized_observation_from_state(
    position=[20.0, -20.0, 20.0],
    yaw=4.0 * math.pi,
    target_pose=[0.0, 0.0, 0.0, 0.0],
    pos_limit=5.0,
)
check(
    "reset validator clips live observation to [-1, 1]",
    all(-1.0 <= value <= 1.0 for value in _clipped_live_obs19),
)
_pose_cache19 = {'drone0': 'stale_target_pose', 'drone1': 'other_pose'}
_twist_cache19 = {'drone0': 'stale_target_twist', 'drone1': 'other_twist'}
_reset_validator.clear_latest_namespace_state(_pose_cache19, _twist_cache19, 'drone0')
check(
    "reset validator clears stale target state before post-reset sampling",
    _pose_cache19 == {'drone1': 'other_pose'} and _twist_cache19 == {'drone1': 'other_twist'},
)
check(
    "reset validator recognizes target snapshot within tolerance",
    _reset_validator.snapshot_matches_target(
        {'position': [0.04, -0.03, 1.02], 'yaw': 0.05},
        [0.0, 0.0, 1.0, 0.0],
        position_tolerance=0.15,
        yaw_tolerance=0.2,
    ),
)
check(
    "reset validator rejects target snapshot outside tolerance",
    not _reset_validator.snapshot_matches_target(
        {'position': [0.0, 0.0, 0.0], 'yaw': 0.5},
        [0.0, 0.0, 1.0, 0.0],
        position_tolerance=0.15,
        yaw_tolerance=0.2,
    ),
)

_stable_other_pose = [9.0, 0.0, 1.0, 0.0]
_valid_reset_results = [
    _reset_validator.ResetAttemptResult(
        namespace='drone0',
        iteration=i,
        service_success=True,
        position_error=0.01,
        yaw_error=0.01,
        linear_speed_norm=0.0,
        angular_speed_norm=0.0,
        linear_acceleration_norm=0.0,
        angular_acceleration_norm=0.0,
        force_norm=0.0,
        torque_norm=0.0,
        motor_angular_velocity_norm=0.0,
        motor_angular_acceleration_norm=0.0,
        dynamic_artifacts_cleared=True,
        post_reset_max_sample_displacement=0.01,
        observation=[0.0, 0.0, 0.0, 0.0],
        other_namespace_poses={'drone1': list(_stable_other_pose)},
    )
    for i in range(20)
]

_valid_summary = _reset_validator.evaluate_reset_attempts(
    _valid_reset_results,
    expected_reset_count=20,
    namespaces=['drone0', 'drone1'],
    target_namespace='drone0',
)
check("reset validator accepts 20 clean deterministic resets", _valid_summary.success)

_missing_attempt_summary = _reset_validator.evaluate_reset_attempts(
    _valid_reset_results[:19],
    expected_reset_count=20,
    namespaces=['drone0', 'drone1'],
    target_namespace='drone0',
)
check("reset validator rejects fewer than 20 resets", not _missing_attempt_summary.success and 'expected 20 reset attempts' in _missing_attempt_summary.message)

_service_failure_results = list(_valid_reset_results)
_service_failure_results[0] = _service_failure_results[0].with_overrides(service_success=False, message='service failed')
_service_failure_summary = _reset_validator.evaluate_reset_attempts(
    _service_failure_results,
    expected_reset_count=20,
    namespaces=['drone0', 'drone1'],
    target_namespace='drone0',
)
check("reset validator rejects service failure", not _service_failure_summary.success and 'service failed' in _service_failure_summary.message)

_twist_failure_results = list(_valid_reset_results)
_twist_failure_results[3] = _twist_failure_results[3].with_overrides(linear_speed_norm=0.2)
_twist_failure_summary = _reset_validator.evaluate_reset_attempts(
    _twist_failure_results,
    expected_reset_count=20,
    namespaces=['drone0', 'drone1'],
    target_namespace='drone0',
)
check("reset validator rejects non-zero post-reset twist", not _twist_failure_summary.success and 'linear speed' in _twist_failure_summary.message)

_artifact_failure_results = list(_valid_reset_results)
_artifact_failure_results[4] = _artifact_failure_results[4].with_overrides(
    linear_acceleration_norm=0.3,
)
_artifact_failure_summary = _reset_validator.evaluate_reset_attempts(
    _artifact_failure_results,
    expected_reset_count=20,
    namespaces=['drone0', 'drone1'],
    target_namespace='drone0',
)
check("reset validator rejects post-reset acceleration artifacts", not _artifact_failure_summary.success and 'linear acceleration' in _artifact_failure_summary.message)

_proof_failure_results = list(_valid_reset_results)
_proof_failure_results[5] = _proof_failure_results[5].with_overrides(dynamic_artifacts_cleared=False)
_proof_failure_summary = _reset_validator.evaluate_reset_attempts(
    _proof_failure_results,
    expected_reset_count=20,
    namespaces=['drone0', 'drone1'],
    target_namespace='drone0',
)
check("reset validator rejects missing dynamic artifact proof", not _proof_failure_summary.success and 'dynamic artifacts' in _proof_failure_summary.message)

_teleport_failure_results = list(_valid_reset_results)
_teleport_failure_results[6] = _teleport_failure_results[6].with_overrides(
    post_reset_max_sample_displacement=0.5,
)
_teleport_failure_summary = _reset_validator.evaluate_reset_attempts(
    _teleport_failure_results,
    expected_reset_count=20,
    namespaces=['drone0', 'drone1'],
    target_namespace='drone0',
)
check("reset validator rejects post-reset teleport spike", not _teleport_failure_summary.success and 'teleport spike' in _teleport_failure_summary.message)

check(
    "reset validator computes max post-reset sample displacement",
    math.isclose(
        _reset_validator.max_pose_sample_displacement([
            {'position': [0.0, 0.0, 1.0]},
            {'position': [0.03, 0.04, 1.0]},
            {'position': [0.03, 0.04, 1.12]},
        ]),
        0.12,
        abs_tol=1e-6,
    ),
)
check(
    "reset validator excludes intended reset relocation from hold displacement",
    math.isclose(
        _reset_validator.max_target_hold_displacement(
            [
                {'position': [0.0, 0.0, 0.0], 'yaw': 0.0},
                {'position': [0.0, 0.0, 1.0], 'yaw': 0.0},
                {'position': [0.0, 0.0, 1.04], 'yaw': 0.0},
            ],
            target_pose=[0.0, 0.0, 1.0, 0.0],
            position_tolerance=0.15,
            yaw_tolerance=0.2,
        ),
        0.04,
        abs_tol=1e-6,
    ),
)

_obs_failure_results = list(_valid_reset_results)
_obs_failure_results[7] = _obs_failure_results[7].with_overrides(observation=[1.2, 0.0, 0.0, 0.0])
_obs_failure_summary = _reset_validator.evaluate_reset_attempts(
    _obs_failure_results,
    expected_reset_count=20,
    namespaces=['drone0', 'drone1'],
    target_namespace='drone0',
)
check("reset validator rejects unnormalized observations", not _obs_failure_summary.success and 'observation outside [-1, 1]' in _obs_failure_summary.message)

_namespace_failure_results = list(_valid_reset_results)
_namespace_failure_results[5] = _namespace_failure_results[5].with_overrides(
    other_namespace_poses={'drone1': [8.0, 0.0, 1.0, 0.0]},
)
_namespace_failure_summary = _reset_validator.evaluate_reset_attempts(
    _namespace_failure_results,
    expected_reset_count=20,
    namespaces=['drone0', 'drone1'],
    target_namespace='drone0',
)
check("reset validator rejects namespace isolation leak", not _namespace_failure_summary.success and 'namespace isolation' in _namespace_failure_summary.message)

_motion_validator_path = Path(__file__).resolve().parent / 'validate_reset_motion_live.py'
_motion_validator_spec = importlib.util.spec_from_file_location('validate_reset_motion_live', _motion_validator_path)
_motion_validator = importlib.util.module_from_spec(_motion_validator_spec)
assert _motion_validator_spec is not None and _motion_validator_spec.loader is not None
_motion_validator_spec.loader.exec_module(_motion_validator)

_actionability_success = _motion_validator.evaluate_actionability_result(
    reset_info={
        'reset_success': True,
        'reset_method': 'simulator_service',
        'reset_service_attempted': True,
        'reset_service_status': 'service_success',
    },
    distances=[2.0, 1.8, 1.6],
    positions=[[0.0, 0.0, 1.0], [0.2, 0.0, 1.0], [0.4, 0.0, 1.0]],
    command_acceptance=[True, True, True],
    min_distance_reduction=0.25,
    min_position_delta=0.1,
)
check("motion validator accepts service-backed post-reset actionability", _actionability_success['success'] is True)

_velocity_primary_failure = _motion_validator.evaluate_actionability_result(
    reset_info={
        'reset_success': True,
        'reset_method': 'velocity',
        'reset_service_attempted': False,
        'reset_service_status': 'not_attempted',
    },
    distances=[2.0, 1.5],
    positions=[[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]],
    command_acceptance=[True, True],
    min_distance_reduction=0.25,
    min_position_delta=0.1,
)
check("motion validator rejects velocity-only primary reset", _velocity_primary_failure['success'] is False and _velocity_primary_failure['reason'] == 'reset_not_service_backed')

_stuck_actionability = _motion_validator.evaluate_actionability_result(
    reset_info={
        'reset_success': True,
        'reset_method': 'simulator_service',
        'reset_service_attempted': True,
        'reset_service_status': 'service_success',
    },
    distances=[2.0, 1.95],
    positions=[[0.0, 0.0, 1.0], [0.02, 0.0, 1.0]],
    command_acceptance=[True, True],
    min_distance_reduction=0.25,
    min_position_delta=0.1,
)
check("motion validator rejects accepted but non-actionable command", _stuck_actionability['success'] is False and _stuck_actionability['reason'] == 'insufficient_physical_motion')

# ===========================================================================
# TEST 20 — Service-backed reset client contracts
# ===========================================================================
print("\n[20] Service-backed reset client contracts")

_env20 = AS2TestEnv(drone_namespace='drone20', reset_service_timeout=0.01)
check(
    "env reset service name is absolute and namespaced",
    _env20._reset_service_name() == '/drone20/platform/reset_simulator_state',
)
_env20.close()


class _UnavailableResetClient:
    def wait_for_service(self, timeout_sec=None):
        return False


_env20b = AS2TestEnv(drone_namespace='drone20b', reset_service_timeout=0.01)
_env20b._reset_service_client = _UnavailableResetClient()
_service_fallback20b = _env20b._try_service_backed_reset([0.0, 0.0, 1.0, 0.0])
check("service reset returns False when service is unavailable", _service_fallback20b is False)
check("service reset records unavailable diagnostic", _env20b._last_reset_diagnostics.get('reason') == 'service_unavailable')
_env20b.close()


class _NeverDoneFuture:
    def done(self):
        return False


class _RecordingExecutor:
    def __init__(self):
        self.spin_calls = 0
        self.shutdown_called = False

    def spin_once(self, timeout_sec=0.0):
        self.spin_calls += 1

    def shutdown(self):
        self.shutdown_called = True


_env20c = AS2TestEnv(drone_namespace='drone20c', reset_service_timeout=0.01)
_executor20c = _RecordingExecutor()
_env20c._reset_aux_executor = _executor20c
_future_completed20c = _env20c._spin_reset_future_until_complete(_NeverDoneFuture(), timeout_sec=0.01)
check("service future wait is bounded and times out", _future_completed20c is False and _executor20c.spin_calls >= 1)
_env20c.close()


class _ResetSimulatorStateRequest:
    pass


class _ResetSimulatorStateService:
    Request = _ResetSimulatorStateRequest


class _ResetSimulatorStateResponse:
    def __init__(self, success=True):
        self.success = success
        self.message = 'reset state applied' if success else 'reset rejected'
        self.position_error = 0.01
        self.yaw_error = 0.02
        self.linear_speed_norm = 0.0
        self.angular_speed_norm = 0.0


class _ImmediateResetFuture:
    def __init__(self, response):
        self._response = response

    def done(self):
        return True

    def result(self):
        return self._response


class _RecordingResetClient:
    def __init__(self, response=None):
        self.response = response if response is not None else _ResetSimulatorStateResponse()
        self.wait_timeout = None
        self.request = None

    def wait_for_service(self, timeout_sec=None):
        self.wait_timeout = timeout_sec
        return True

    def call_async(self, request):
        self.request = request
        return _ImmediateResetFuture(self.response)


class _RecordingResetNode:
    def __init__(self, node_name, response=None):
        self.node_name = node_name
        self.response = response
        self.client_service_type = None
        self.client_service_name = None
        self.client = None
        self.clients = []
        self.destroyed = False

    def create_client(self, service_type, service_name):
        self.client_service_type = service_type
        self.client_service_name = service_name
        self.client = _RecordingResetClient(self.response)
        self.clients.append((service_type, service_name, self.client))
        return self.client

    def destroy_node(self):
        self.destroyed = True


class _RecordingSingleThreadedExecutor(_RecordingExecutor):
    def __init__(self):
        super().__init__()
        self.added_nodes = []

    def add_node(self, node):
        self.added_nodes.append(node)


_created_reset_nodes20f = []
_created_reset_executors20f = []
_reset_pkg20f = types.ModuleType('as2_platform_multirotor_simulator')
_reset_srv20f = types.ModuleType('as2_platform_multirotor_simulator.srv')
_reset_srv20f.ResetSimulatorState = _ResetSimulatorStateService
_rclpy_mod20f = sys.modules['rclpy']
_original_create_node20f = getattr(_rclpy_mod20f, 'create_node', None)
_original_rclpy_executors20f = sys.modules.get('rclpy.executors')
_executors_mod20f = types.ModuleType('rclpy.executors')


def _recording_executor_factory20f():
    executor = _RecordingSingleThreadedExecutor()
    _created_reset_executors20f.append(executor)
    return executor


_executors_mod20f.SingleThreadedExecutor = _recording_executor_factory20f


def _recording_create_node20f(node_name):
    node = _RecordingResetNode(node_name)
    _created_reset_nodes20f.append(node)
    return node


sys.modules['as2_platform_multirotor_simulator'] = _reset_pkg20f
sys.modules['as2_platform_multirotor_simulator.srv'] = _reset_srv20f
sys.modules['rclpy.executors'] = _executors_mod20f
_rclpy_mod20f.create_node = _recording_create_node20f

try:
    _env20f = AS2TestEnv(
        drone_namespace='drone20f',
        reset_service_timeout=0.01,
        fixed_start_tolerance=0.12,
        reset_yaw_tolerance=0.9,
        hover_speed_threshold=0.04,
    )
    _env20f._reset_velocity_controller = lambda: True
    _env20f._wait_until_service_reset_pose_observed = lambda start_pose: True
    _env20f._confirm_post_service_command_path = lambda start_pose: True
    _service_success20f = _env20f._try_service_backed_reset([1.0, -2.0, 1.5, 0.3])
    _node20f = _created_reset_nodes20f[0] if _created_reset_nodes20f else None
    _executor20f = _created_reset_executors20f[0] if _created_reset_executors20f else None
    _reset_client_record20f = next(
        (
            record for record in (_node20f.clients if _node20f is not None else [])
            if record[1] == '/drone20f/platform/reset_simulator_state'
        ),
        None,
    )
    _reset_client20f = _reset_client_record20f[2] if _reset_client_record20f is not None else None
    _request20f = _reset_client20f.request if _reset_client20f is not None else None
    check("service reset creates an auxiliary node", _node20f is not None and _node20f.node_name == 'drone20f_reset_client')
    check("service reset adds only auxiliary node to reset executor", _executor20f is not None and _executor20f.added_nodes == [_node20f])
    check("service reset client uses generated service type", _reset_client_record20f is not None and _reset_client_record20f[0] is _ResetSimulatorStateService)
    check("service reset client uses absolute service name", _reset_client_record20f is not None)
    check("service reset sends requested pose fields", _request20f is not None and [_request20f.x, _request20f.y, _request20f.z, _request20f.yaw] == [1.0, -2.0, 1.5, 0.3])
    check("service reset sends configured tolerances", _request20f is not None and _request20f.position_tolerance == 0.12 and _request20f.yaw_tolerance == 0.9 and _request20f.linear_speed_tolerance == 0.04)
    check("service reset returns True on successful response", _service_success20f is True)
    check("service reset records success diagnostics", _env20f._last_reset_diagnostics.get('reason') == 'service_success')
    _env20f.close()
finally:
    if _original_create_node20f is None:
        try:
            delattr(_rclpy_mod20f, 'create_node')
        except AttributeError:
            pass
    else:
        _rclpy_mod20f.create_node = _original_create_node20f
    if _original_rclpy_executors20f is None:
        sys.modules.pop('rclpy.executors', None)
    else:
        sys.modules['rclpy.executors'] = _original_rclpy_executors20f


_events20g = []


class _RecordingBehaviorStop:
    def __init__(self, name):
        self.name = name

    def stop(self):
        _events20g.append(f'{self.name}.stop')
        return True


class _DroneWithActiveBehaviors:
    def __init__(self):
        self.takeoff = _RecordingBehaviorStop('takeoff')
        self.land = _RecordingBehaviorStop('land')
        self.go_to = _RecordingBehaviorStop('go_to')
        self.follow_path = _RecordingBehaviorStop('follow_path')

    def manual(self, **kw):
        pass

    def shutdown(self, **kw):
        pass


class _PreResetSpeedHandler:
    def send_speed_command_with_yaw_speed(self, **kw):
        _events20g.append(('zero_speed', list(kw.get('twist', [])), float(kw.get('yaw_speed', 999.0))))


class _SuccessfulResetClient(_RecordingResetClient):
    def call_async(self, request):
        _events20g.append('service_call')
        return super().call_async(request)


_env20g = AS2TestEnv(drone_namespace='drone20g', reset_service_timeout=0.01)
_env20g._drone = _DroneWithActiveBehaviors()
_env20g._speed_handler = _PreResetSpeedHandler()
_env20g._reset_service_type = _ResetSimulatorStateService
_env20g._reset_service_client = _SuccessfulResetClient()
_env20g._reset_aux_executor = _RecordingExecutor()
_env20g._takeoff_after_service_reset = lambda start_pose: _events20g.append('takeoff_after_service') or True
_env20g._reset_velocity_controller = lambda: _events20g.append('recreate_speed_handler') or True
_env20g._hold_service_reset_start_pose = lambda start_pose: _events20g.append('hold_start_pose') or True
_env20g._wait_until_service_reset_pose_observed = lambda start_pose: _events20g.append('pose_observed') or True
_env20g._confirm_post_service_command_path = lambda start_pose: _events20g.append('actionability_probe') or True
_service_success20g = _env20g._try_service_backed_reset([0.0, 0.0, 1.0, 0.0])


def _event_before(events, first, second):
    return first in events and second in events and events.index(first) < events.index(second)


check("service reset does not stop AS2 behavior modules before service call", not any(event.endswith('.stop') for event in _events20g if isinstance(event, str)))
_service_call_index20g = _events20g.index('service_call') if 'service_call' in _events20g else len(_events20g)
check("service reset avoids pre-service zero speed command", not any(isinstance(event, tuple) and event[0] == 'zero_speed' for event in _events20g[:_service_call_index20g]))
check("service reset recreates speed handler after success", _service_success20g is True and _event_before(_events20g, 'service_call', 'recreate_speed_handler'))
check("service reset waits for observed pose before actionability", _event_before(_events20g, 'pose_observed', 'actionability_probe'))
_env20g.close()


class _ModeRefreshResponse:
    success = True


_env20g1 = AS2TestEnv(drone_namespace='drone20g1', reset_service_timeout=0.01)
_env20g1._reset_aux_node = _RecordingResetNode('reset_aux', _ModeRefreshResponse())
_env20g1._reset_aux_executor = _RecordingExecutor()
_mode_refresh20g1 = _env20g1._refresh_controller_mode_after_service_reset()
_mode_request20g1 = _env20g1._reset_aux_node.client.request.control_mode
check("service reset refreshes controller through position mode before speed", _mode_refresh20g1 is True and _env20g1._reset_aux_node.client_service_name == '/drone20g1/controller/set_control_mode' and _mode_request20g1.reference_frame == 1)
_env20g1.close()


_events20i = []


class _OffboardResetDrone:
    position = [0.0, 0.0, 1.0]
    speed = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0]

    def arm(self, **kw):
        _events20i.append('arm')
        return True

    def offboard(self, **kw):
        _events20i.append('offboard')
        return True

    def manual(self, **kw):
        pass

    def shutdown(self, **kw):
        pass


_env20i = AS2TestEnv(drone_namespace='drone20i', reset_service_timeout=0.01)
_env20i._drone = _OffboardResetDrone()
_env20i._reset_service_type = _ResetSimulatorStateService
_env20i._reset_service_client = _SuccessfulResetClient()
_env20i._reset_aux_executor = _RecordingExecutor()
_env20i._set_platform_flying_after_service_reset = lambda: _events20i.append('platform_fsm_event') or True
_env20i._reset_velocity_controller = lambda: _events20i.append('recreate_speed_handler') or True
_env20i._confirm_post_service_command_path = lambda start_pose: _events20i.append('actionability_probe') or False
_service_success20i = _env20i._try_service_backed_reset([0.0, 0.0, 1.0, 0.0])
check("service reset does not call failing platform FSM service after failed actionability", 'platform_fsm_event' not in _events20i)
check("service reset reasserts arm before offboard", _event_before(_events20i, 'arm', 'offboard'))
check("service reset reasserts offboard before declaring command path ready", _event_before(_events20i, 'offboard', 'recreate_speed_handler'))
check("service reset fails closed when command path is not actionable", _service_success20i is False and _env20i._is_flying is False)
_env20i.close()


class _RejectedTakeoffAtResetPoseDrone:
    position = [0.0, 0.0, 1.0]
    speed = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0]

    def takeoff(self, **kw):
        return False

    def manual(self, **kw):
        pass

    def shutdown(self, **kw):
        pass


_env20i1 = AS2TestEnv(drone_namespace='drone20i1', reset_service_timeout=0.01)
_env20i1._drone = _RejectedTakeoffAtResetPoseDrone()
_env20i1._is_flying = False
_takeoff_ready20i1 = _env20i1._takeoff_after_service_reset([0.0, 0.0, 1.0, 0.0])
check("post-reset takeoff is skipped when service pose is already observed", _takeoff_ready20i1 is True and _env20i1._last_reset_diagnostics.get('post_reset_takeoff_skipped') is True)
_env20i1.close()


class _DefaultHeightTakeoffDrone:
    speed = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0]

    def __init__(self):
        self.position = [0.0, 0.0, 1.6]
        self.takeoff_calls = []

    def takeoff(self, **kw):
        self.takeoff_calls.append(dict(kw))
        self.position = [0.0, 0.0, 1.08]
        return True

    def arm(self, **kw):
        return True

    def offboard(self, **kw):
        return True

    def manual(self, **kw):
        pass

    def shutdown(self, **kw):
        pass


_env20i1b = AS2TestEnv(drone_namespace='drone20i1b', reset_service_timeout=0.01, step_duration=0.0)
_drone20i1b = _DefaultHeightTakeoffDrone()
_hold_calls20i1b = []
_env20i1b._drone = _drone20i1b
_env20i1b._reset_service_type = _ResetSimulatorStateService
_env20i1b._reset_service_client = _RecordingResetClient()
_env20i1b._reset_aux_executor = _RecordingExecutor()
_env20i1b._reset_velocity_controller = lambda: True
_env20i1b._hold_service_reset_start_pose = lambda start_pose: _hold_calls20i1b.append(list(start_pose)) or True
_env20i1b._confirm_post_service_command_path = lambda start_pose: True
_service_success20i1b = _env20i1b._try_service_backed_reset([0.0, 0.0, 1.6, 0.0])
check("service reset does not issue takeoff that would pull z toward default height", _service_success20i1b is True and _drone20i1b.takeoff_calls == [] and math.isclose(_drone20i1b.position[2], 1.6, abs_tol=1e-6))
check("service reset actively holds requested start z after controller reset", _hold_calls20i1b == [[0.0, 0.0, 1.6, 0.0]] and _env20i1b._last_reset_diagnostics.get('post_reset_controller_hold_ready') is True)
check("takeoff diagnostics expose skipped command and observed z", _env20i1b._last_reset_diagnostics.get('post_reset_takeoff_commanded_height') is None and math.isclose(_env20i1b._last_reset_diagnostics.get('post_reset_takeoff_observed_z'), 1.6, abs_tol=1e-6))
_env20i1b.close()


_events20i2 = []


class _StaleInfoOffboardResetDrone:
    position = [0.0, 0.0, 1.0]
    speed = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0]
    info = {'armed': True, 'offboard': True}

    def arm(self, **kw):
        _events20i2.append('arm')
        return True

    def offboard(self, **kw):
        _events20i2.append('offboard')
        return True

    def manual(self, **kw):
        pass

    def shutdown(self, **kw):
        pass


_env20i2 = AS2TestEnv(drone_namespace='drone20i2', reset_service_timeout=0.01)
_env20i2._drone = _StaleInfoOffboardResetDrone()
_env20i2._reset_service_type = _ResetSimulatorStateService
_env20i2._reset_service_client = _SuccessfulResetClient()
_env20i2._reset_aux_executor = _RecordingExecutor()
_env20i2._reset_velocity_controller = lambda: _events20i2.append('recreate_speed_handler') or True
_env20i2._confirm_post_service_command_path = lambda start_pose: True
_service_success20i2 = _env20i2._try_service_backed_reset([0.0, 0.0, 1.0, 0.0])
check("service reset reasserts arm/offboard even with stale true info flags", _events20i2[:3] == ['arm', 'offboard', 'recreate_speed_handler'])
check("stale info reassertion still marks command path ready", _service_success20i2 is True and _env20i2._is_flying is True)
_env20i2.close()


_env20j = AS2TestEnv(drone_namespace='drone20j', reset_service_timeout=0.01)
_env20j._drone = _OffboardResetDrone()
_env20j._reset_service_type = _ResetSimulatorStateService
_env20j._reset_service_client = _SuccessfulResetClient()
_env20j._reset_aux_executor = _RecordingExecutor()
_env20j._set_platform_flying_after_service_reset = lambda: True
_env20j._reset_velocity_controller = lambda: False
_service_success20j = _env20j._try_service_backed_reset([0.0, 0.0, 1.0, 0.0])
check("service reset falls back if post-reset speed handler is not ready", _service_success20j is False)
check("service reset records post-reset command path diagnostic", _env20j._last_reset_diagnostics.get('reason') == 'service_post_reset_command_path_failed')
_env20j.close()


_env20j2 = AS2TestEnv(drone_namespace='drone20j2', reset_service_timeout=0.01, step_duration=0.0)
_env20j2._drone = fake_drone([0.3, 0.0, 1.08], [0.0, 0.0, 0.0])
_env20j2._speed_handler = _PreResetSpeedHandler()
_env20j2._reset_service_type = _ResetSimulatorStateService
_env20j2._reset_service_client = _RecordingResetClient()
_env20j2._reset_aux_executor = _RecordingExecutor()
_env20j2._reset_velocity_controller = lambda: True
_actionability_calls20j2 = []
_env20j2._confirm_post_service_command_path = lambda start_pose: _actionability_calls20j2.append(start_pose) or True
_service_success20j2 = _env20j2._try_service_backed_reset([0.0, 0.0, 1.6, 0.0])
check("service reset fails closed when telemetry never observes reset pose", _service_success20j2 is False and _env20j2._last_reset_diagnostics.get('pose_observed') is False)
check("stale service reset telemetry does not run actionability probe", _actionability_calls20j2 == [])
check("stale service reset diagnostics expose observed pose error", _env20j2._last_reset_diagnostics.get('post_reset_observed_position_error', 0.0) > 0.0)
_env20j2.close()


_env20j3 = AS2TestEnv(drone_namespace='drone20j3', reset_service_timeout=0.01, step_duration=0.0)
_env20j3._drone = fake_drone([0.0, 0.0, 1.6], [0.12, 0.0, 0.0])
_env20j3._speed_handler = _PreResetSpeedHandler()
_env20j3._reset_service_type = _ResetSimulatorStateService
_env20j3._reset_service_client = _RecordingResetClient()
_env20j3._reset_aux_executor = _RecordingExecutor()
_env20j3._reset_velocity_controller = lambda: True
_actionability_calls20j3 = []
_env20j3._confirm_post_service_command_path = lambda start_pose: _actionability_calls20j3.append(start_pose) or True
_service_success20j3 = _env20j3._try_service_backed_reset([0.0, 0.0, 1.6, 0.0])
check("service reset fails closed while reset telemetry is still moving", _service_success20j3 is False and _env20j3._last_reset_diagnostics.get('pose_observed') is False)
check("moving reset telemetry does not run actionability probe", _actionability_calls20j3 == [])
check("moving reset telemetry exposes residual speed", _env20j3._last_reset_diagnostics.get('post_reset_observed_speed_norm', 0.0) > _env20j3.hover_speed_threshold)
_env20j3.close()


class _RejectOnceSpeedHandler:
    def __init__(self):
        self.calls = []

    def send_speed_command_with_yaw_speed(self, **kw):
        self.calls.append((list(kw.get('twist', [])), float(kw.get('yaw_speed', 999.0))))
        return False


class _AcceptingSpeedHandler(_RejectOnceSpeedHandler):
    def send_speed_command_with_yaw_speed(self, **kw):
        self.calls.append((list(kw.get('twist', [])), float(kw.get('yaw_speed', 999.0))))
        return True


class _DesiredControlMode:
    reference_frame = 0


class _ReferenceFrameRecordingSpeedHandler(_AcceptingSpeedHandler):
    def __init__(self):
        super().__init__()
        self.desired_control_mode_ = _DesiredControlMode()


_env20k0 = AS2TestEnv(drone_namespace='drone20k0')
_ref_handler20k0 = _ReferenceFrameRecordingSpeedHandler()
_env20k0._speed_handler = _ref_handler20k0
_send_success20k0 = _env20k0._send_speed_command([0.25, 0.0, 0.0], 0.0)
check("speed commands configure AS2 local ENU reference frame for earth twists", _send_success20k0 is True and _ref_handler20k0.desired_control_mode_.reference_frame == 1)
check("speed command diagnostics record twist frame", _env20k0._last_speed_command_reference_frame == 'earth')
_env20k0.close()


_env20k = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone20k',
    step_duration=0.0,
)
_inner20k = _env20k.unwrapped
_inner20k._drone = fake_drone([0.0, 1.0, 1.0], [0.0, 0.0, 0.0])
_reject_handler20k = _RejectOnceSpeedHandler()
_accept_handler20k = _AcceptingSpeedHandler()
_recovery20k = {'offboard': 0, 'controller': 0}
_inner20k._speed_handler = _reject_handler20k
_inner20k._is_flying = True
_env20k.reset()
_inner20k._drone = fake_drone([0.0, 1.0, 1.0], [0.0, 0.0, 0.0])
_inner20k._speed_handler = _reject_handler20k
_inner20k._ensure_offboard_after_reset = lambda: _recovery20k.__setitem__('offboard', _recovery20k['offboard'] + 1) or True


def _recover_controller20k():
    _recovery20k['controller'] += 1
    _inner20k._speed_handler = _accept_handler20k
    return True


_inner20k._reset_velocity_controller = _recover_controller20k
_, _, _, _, _info20k = _env20k.step(np.array([0.4, -0.2, 0.1, 0.3], dtype=np.float32))
check("step retries motion command when AS2 rejects first send", _recovery20k == {'offboard': 1, 'controller': 1})
_recovered_twist20k, _recovered_yaw20k = _accept_handler20k.calls[-1]
check(
    "step sends original action after motion path recovery",
    np.allclose(_recovered_twist20k, [0.4, -0.2, 0.1], atol=1e-6)
    and math.isclose(_recovered_yaw20k, 0.3, abs_tol=1e-6),
)
check("step reports recovered motion command accepted", _info20k.get('motion_command_accepted') is True and _info20k.get('motion_command_recovered') is True)
_env20k.close()


_original_sleep20k_step = as2_env_module.time.sleep
_sleep_calls20k_step = []


def _record_sleep20k_step(duration):
    _sleep_calls20k_step.append(float(duration))


try:
    as2_env_module.time.sleep = _record_sleep20k_step

    _env20k_step = AS2TestEnv(drone_namespace='drone20k_step', step_duration=0.12)
    _drone20k_step = fake_drone([1.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    _handler20k_step = RecordingSpeedHandler(_drone20k_step)
    _env20k_step._drone = _drone20k_step
    _env20k_step._speed_handler = _handler20k_step
    _env20k_step._is_flying = True
    _, _, _term20k_step, _trunc20k_step, _info20k_step = _env20k_step.step(
        np.array([0.4, 0.0, 0.0, 0.0], dtype=np.float32)
    )
    check("step republishes speed command across the step window", len(_handler20k_step.commands) == 3)
    check("step reports motion publication diagnostics", _info20k_step.get('motion_command_publication_count') == 3 and _info20k_step.get('motion_command_accepted_publication_count') == 3)
    check("non-terminal sustained step does not send stop command", not _term20k_step and not _trunc20k_step and 'terminal_stop_command_accepted' not in _info20k_step and not np.allclose(_handler20k_step.commands[-1][0], [0.0, 0.0, 0.0], atol=1e-6))
    _env20k_step.close()

    _env20k_zero = AS2TestEnv(drone_namespace='drone20k_zero', step_duration=0.0)
    _drone20k_zero = fake_drone([1.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    _handler20k_zero = RecordingSpeedHandler(_drone20k_zero)
    _env20k_zero._drone = _drone20k_zero
    _env20k_zero._speed_handler = _handler20k_zero
    _env20k_zero._is_flying = True
    _, _, _, _, _info20k_zero = _env20k_zero.step(np.array([0.4, 0.0, 0.0, 0.0], dtype=np.float32))
    check("zero step duration publishes exactly once", len(_handler20k_zero.commands) == 1 and _info20k_zero.get('motion_command_publication_count') == 1)
    _env20k_zero.close()

    _env20k_min = AS2TestEnv(
        drone_namespace='drone20k_min',
        step_duration=0.0,
        min_motion_command_publications=4,
    )
    _drone20k_min = fake_drone([1.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    _handler20k_min = RecordingSpeedHandler(_drone20k_min)
    _env20k_min._drone = _drone20k_min
    _env20k_min._speed_handler = _handler20k_min
    _env20k_min._is_flying = True
    _, _, _, _, _info20k_min = _env20k_min.step(np.array([0.4, 0.0, 0.0, 0.0], dtype=np.float32))
    check("step honors configured minimum motion publications", len(_handler20k_min.commands) == 4 and _info20k_min.get('motion_command_publication_count') == 4)
    _env20k_min.close()

    _env20k_short = AS2TestEnv(drone_namespace='drone20k_short', step_duration=0.01)
    _drone20k_short = fake_drone([1.0, 0.0, 1.0], [0.0, 0.0, 0.0])
    _handler20k_short = RecordingSpeedHandler(_drone20k_short)
    _env20k_short._drone = _drone20k_short
    _env20k_short._speed_handler = _handler20k_short
    _env20k_short._is_flying = True
    _, _, _, _, _info20k_short = _env20k_short.step(np.array([0.4, 0.0, 0.0, 0.0], dtype=np.float32))
    check("short step duration sleeps without extra publication", len(_handler20k_short.commands) == 1 and _info20k_short.get('motion_command_publication_count') == 1)
    _env20k_short.close()
finally:
    as2_env_module.time.sleep = _original_sleep20k_step


class _StaticAcceptingSpeedHandler:
    def __init__(self):
        self.calls = []

    def send_speed_command_with_yaw_speed(self, **kw):
        self.calls.append((list(kw.get('twist', [])), float(kw.get('yaw_speed', 999.0))))
        return True


class _DelayedProbeTelemetryDrone:
    speed = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0]

    def __init__(self):
        self.position_reads = 0

    @property
    def position(self):
        self.position_reads += 1
        if self.position_reads >= 4:
            return [0.02, 0.0, 1.0]
        return [0.0, 0.0, 1.0]

    def land(self, **kw):
        return True

    def manual(self, **kw):
        pass

    def shutdown(self, **kw):
        pass


_env20k1 = AS2TestEnv(drone_namespace='drone20k1', reset_service_timeout=0.01, step_duration=0.0)
_env20k1._drone = _DelayedProbeTelemetryDrone()
_env20k1._speed_handler = _StaticAcceptingSpeedHandler()
_delayed_probe20k1 = _env20k1._measure_post_reset_probe_effect([0.10, 0.0, 0.0], 0.10)
check("post-reset actionability probe accepts delayed telemetry movement", _delayed_probe20k1.get('actionable') is True and _delayed_probe20k1.get('forward_delta', 0.0) >= _delayed_probe20k1.get('min_forward_delta', 1.0))
_env20k1.close()


_env20l = AS2TestEnv(drone_namespace='drone20l', reset_service_timeout=0.01, step_duration=0.0)
_drone20l = fake_drone([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_static_handler20l = _StaticAcceptingSpeedHandler()
_env20l._drone = _drone20l
_env20l._speed_handler = _static_handler20l
_env20l._reset_service_type = _ResetSimulatorStateService
_env20l._reset_service_client = _RecordingResetClient()
_env20l._reset_aux_executor = _RecordingExecutor()
_env20l._set_platform_flying_after_service_reset = lambda: True
_env20l._reset_velocity_controller = lambda: setattr(_env20l, '_speed_handler', _static_handler20l) or True
_service_success20l = _env20l._try_service_backed_reset([0.0, 0.0, 1.0, 0.0])
check("service reset rejects accepted-but-stationary command path", _service_success20l is False)
check("accepted-but-stationary reset records actionability failure", _env20l._last_reset_diagnostics.get('actionability_ready') is False and _env20l._last_reset_diagnostics.get('failure_class') == 'post_reset_command_path_failed')
_env20l.close()


_env20m = AS2TestEnv(drone_namespace='drone20m', reset_service_timeout=0.01, step_duration=0.0)
_drone20m = fake_drone([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_static_handler20m = _StaticAcceptingSpeedHandler()
_moving_handler20m = RecordingSpeedHandler(_drone20m)
_env20m._drone = _drone20m
_env20m._speed_handler = _static_handler20m
_env20m._reset_service_type = _ResetSimulatorStateService
_env20m._reset_service_client = _RecordingResetClient()
_env20m._reset_aux_executor = _RecordingExecutor()
_env20m._set_platform_flying_after_service_reset = lambda: True
_env20m._reset_velocity_controller = lambda: setattr(_env20m, '_speed_handler', _static_handler20m) or True


def _recover_actionable_command_path20m():
    _env20m._speed_handler = _moving_handler20m
    return True


_env20m._recover_motion_reference_path = _recover_actionable_command_path20m
_service_success20m = _env20m._try_service_backed_reset([0.0, 0.0, 1.0, 0.0])
_actionability20m = _env20m._last_reset_diagnostics.get('post_reset_actionability', {})
check("service reset recovers accepted-but-ignored command path", _service_success20m is True and _env20m._is_flying is True)
check("recovered service reset requires measured probe motion", _actionability20m.get('recovery_ready') is True and _actionability20m.get('second_probe', {}).get('actionable') is True)
_env20m.close()


class _JumpingSpeedHandler:
    def __init__(self, drone, scale=0.2):
        self.drone = drone
        self.scale = scale
        self.calls = []

    def send_speed_command_with_yaw_speed(self, **kw):
        twist = list(kw.get('twist', [0.0, 0.0, 0.0]))
        yaw_speed = float(kw.get('yaw_speed', 0.0))
        self.calls.append((twist, yaw_speed))
        self.drone.position = [
            self.drone.position[0] + twist[0] * self.scale,
            self.drone.position[1] + twist[1] * self.scale,
            self.drone.position[2] + twist[2] * self.scale,
        ]
        return True


_env20m2 = AS2TestEnv(drone_namespace='drone20m2', reset_service_timeout=0.01, step_duration=0.0)
_drone20m2 = fake_drone([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_jumping_handler20m2 = _JumpingSpeedHandler(_drone20m2, scale=1.0)
_restore_calls20m2 = []
_env20m2._drone = _drone20m2
_env20m2._speed_handler = _jumping_handler20m2
_env20m2._reset_service_type = _ResetSimulatorStateService
_env20m2._reset_service_client = _RecordingResetClient()
_env20m2._reset_aux_executor = _RecordingExecutor()
_env20m2._reset_velocity_controller = lambda: setattr(_env20m2, '_speed_handler', _jumping_handler20m2) or True


def _restore_start_pose20m2(start_pose):
    _restore_calls20m2.append(list(start_pose))
    _drone20m2.position = list(start_pose[:3])
    _drone20m2.orientation = [0.0, 0.0, start_pose[3]]
    _env20m2._last_reset_diagnostics = {'reason': 'post_controller_hold', 'position_error': 0.0, 'yaw_error': 0.0}
    return True


_env20m2._apply_and_confirm_start_pose = _restore_start_pose20m2
_service_success20m2 = _env20m2._try_service_backed_reset([0.0, 0.0, 1.0, 0.0])
_actionability20m2 = _env20m2._last_reset_diagnostics.get('post_reset_actionability', {})
check("service reset restores start pose after successful actionability probe", _service_success20m2 is True and _restore_calls20m2 == [[0.0, 0.0, 1.0, 0.0]] and np.allclose(_drone20m2.position, [0.0, 0.0, 1.0], atol=1e-6))
check("actionability diagnostics record start pose restore", _actionability20m2.get('start_pose_restored') is True and _actionability20m2.get('start_pose_restore_method') == 'velocity_restore')
_env20m2.close()


_env20m3 = AS2TestEnv(drone_namespace='drone20m3', reset_service_timeout=0.01, step_duration=0.0)
_drone20m3 = fake_drone([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_jumping_handler20m3 = _JumpingSpeedHandler(_drone20m3, scale=1.0)
_env20m3._drone = _drone20m3
_env20m3._speed_handler = _jumping_handler20m3
_env20m3._reset_service_type = _ResetSimulatorStateService
_env20m3._reset_service_client = _RecordingResetClient()
_env20m3._reset_aux_executor = _RecordingExecutor()
_env20m3._reset_velocity_controller = lambda: setattr(_env20m3, '_speed_handler', _jumping_handler20m3) or True


def _fail_restore_start_pose20m3(start_pose):
    _env20m3._last_reset_diagnostics = {'reason': 'post_controller_reacquire_timeout'}
    raise RuntimeError('restore failed')


_env20m3._apply_and_confirm_start_pose = _fail_restore_start_pose20m3
_service_success20m3 = _env20m3._try_service_backed_reset([0.0, 0.0, 1.0, 0.0])
_actionability20m3 = _env20m3._last_reset_diagnostics.get('post_reset_actionability', {})
check("service reset fails closed when probe displacement cannot be restored", _service_success20m3 is False and _env20m3._last_reset_diagnostics.get('actionability_ready') is False)
check("failed probe restore is visible in diagnostics", _actionability20m3.get('start_pose_restored') is False and _actionability20m3.get('start_pose_restore_method') == 'velocity_restore_failed')
_env20m3.close()


_env20m4 = AS2TestEnv(
    drone_namespace='drone20m4',
    step_duration=0.0,
    fixed_start_tolerance=0.15,
    hover_speed_threshold=0.05,
)
_env20m4._drone = fake_drone([0.062, 0.0, 1.0], [0.0, 0.0, 0.0])
_env20m4._speed_handler = _StaticAcceptingSpeedHandler()
_actionability20m4 = {}
_restored20m4 = _env20m4._restore_start_pose_after_actionability_probe(
    [0.0, 0.0, 1.0, 0.0],
    _actionability20m4,
)
check("post-probe restore accepts small stable residual within fixed-start tolerance", _restored20m4 is True and _actionability20m4.get('start_pose_restored') is True)
check("post-probe restore diagnostics use fixed-start tolerance", math.isclose(_actionability20m4.get('start_pose_restore_tolerance'), 0.15, abs_tol=1e-6))
_env20m4.close()


_env20m5 = AS2TestEnv(
    drone_namespace='drone20m5',
    step_duration=0.0,
    fixed_start_tolerance=0.15,
    hover_speed_threshold=0.05,
)
_env20m5._drone = fake_drone([0.20, 0.0, 1.0], [0.0, 0.0, 0.0])
_env20m5._speed_handler = _StaticAcceptingSpeedHandler()
_env20m5._apply_and_confirm_start_pose = lambda start_pose: True
_actionability20m5 = {}
_restored20m5 = _env20m5._restore_start_pose_after_actionability_probe(
    [0.0, 0.0, 1.0, 0.0],
    _actionability20m5,
)
check("post-probe restore rejects residual beyond fixed-start tolerance", _restored20m5 is False and _actionability20m5.get('start_pose_restored') is False)
_env20m5.close()


_env20m6 = AS2TestEnv(
    drone_namespace='drone20m6',
    step_duration=0.0,
    fixed_start_tolerance=0.15,
    hover_speed_threshold=0.05,
)
_env20m6._drone = fake_drone([0.062, 0.0, 1.0], [0.08, 0.0, 0.0])
_env20m6._speed_handler = _StaticAcceptingSpeedHandler()
_env20m6._apply_and_confirm_start_pose = lambda start_pose: True
_actionability20m6 = {}
_restored20m6 = _env20m6._restore_start_pose_after_actionability_probe(
    [0.0, 0.0, 1.0, 0.0],
    _actionability20m6,
)
check("post-probe restore rejects small residual while velocity is not stable", _restored20m6 is False and _actionability20m6.get('start_pose_restored') is False)
_env20m6.close()


class _FailingBehaviorStop:
    def stop(self):
        raise RuntimeError('stop failed')


class _FailingStopDrone:
    def __init__(self):
        self.go_to = _FailingBehaviorStop()

    def manual(self, **kw):
        pass

    def shutdown(self, **kw):
        pass


class _UnexpectedResetClient(_RecordingResetClient):
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def call_async(self, request):
        self.call_count += 1
        return super().call_async(request)


_env20h = AS2TestEnv(drone_namespace='drone20h', reset_service_timeout=0.01)
_client20h = _UnexpectedResetClient()
_env20h._drone = _FailingStopDrone()
_env20h._speed_handler = _PreResetSpeedHandler()
_env20h._reset_service_type = _ResetSimulatorStateService
_env20h._reset_service_client = _client20h
_env20h._reset_velocity_controller = lambda: True
_env20h._hold_service_reset_start_pose = lambda start_pose: True
_env20h._wait_until_service_reset_pose_observed = lambda start_pose: True
_env20h._confirm_post_service_command_path = lambda start_pose: True
_service_success20h = _env20h._try_service_backed_reset([0.0, 0.0, 1.0, 0.0])
check("service reset ignores AS2 behavior stop hooks", _service_success20h is True and _env20h._last_reset_diagnostics.get('reason') == 'service_success')
check("service reset still calls service when behavior stop hook would fail", _client20h.call_count == 1)
_env20h.close()

_env20d = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone20d',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
)
_inner20d = _env20d.unwrapped
_inner20d._drone = fake_drone([2.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner20d._is_flying = True
_calls20d = {'service': 0, 'fallback': 0}
_inner20d._wait_for_hover_settle = _stub_wait_for_hover_settle
_inner20d._recover_low_altitude_hover_before_velocity_reset = lambda hover_height: True
_inner20d._reset_velocity_controller = lambda: None
_inner20d._try_service_backed_reset = lambda start_pose: _calls20d.__setitem__('service', _calls20d['service'] + 1) or True
_inner20d._apply_and_confirm_start_pose = lambda start_pose: _calls20d.__setitem__('fallback', _calls20d['fallback'] + 1) or True
_obs20d, _info20d = _env20d.reset()
check("already-flying reset bypasses simulator service for velocity reset", _calls20d == {'service': 0, 'fallback': 1})
check("already-flying velocity reset reports velocity method", _info20d.get('reset_method') == 'velocity' and _info20d.get('reset_service_attempted') is False)
check("already-flying velocity reset emits normalized obs", np.all(_obs20d >= -1.0) and np.all(_obs20d <= 1.0))
_env20d.close()

_env20m = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone20m',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    use_simulator_reset_service=True,
)
_inner20m = _env20m.unwrapped
_events20m = []
_inner20m._drone = fake_drone([0.2, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner20m._is_flying = True
_inner20m._wait_for_hover_settle = lambda: _events20m.append('wait_hover') or True
_inner20m._recover_low_altitude_hover_before_velocity_reset = lambda hover_height: _events20m.append('low_altitude_recovery') or True
_inner20m._reset_velocity_controller = lambda: _events20m.append('controller_reset') or True
_inner20m._try_service_backed_reset = lambda start_pose: _events20m.append('service_reset') or True
_inner20m._apply_and_confirm_start_pose = lambda start_pose: _events20m.append('velocity_reset') or True
_obs20m, _info20m = _env20m.reset()
check("already-flying service-enabled reset uses velocity path instead of simulator service", _events20m == ['low_altitude_recovery', 'controller_reset', 'wait_hover', 'velocity_reset'])
check("already-flying service-enabled velocity reset reports no service attempt", _info20m.get('reset_method') == 'velocity' and _info20m.get('reset_service_status') == 'not_attempted')
check("already-flying service-enabled velocity reset emits normalized obs", np.all(_obs20m >= -1.0) and np.all(_obs20m <= 1.0))
_env20m.close()

_env20m2 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone20m2',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    use_simulator_reset_service=True,
)
_inner20m2 = _env20m2.unwrapped
_events20m2 = []
_inner20m2._drone = fake_drone([0.2, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner20m2._is_flying = True
_inner20m2._last_reset_diagnostics = {'reason': 'service_rejected', 'failure_class': 'service_response_failure'}
_inner20m2._wait_for_hover_settle = lambda: _events20m2.append('wait_hover') or True
_inner20m2._recover_low_altitude_hover_before_velocity_reset = lambda hover_height: _events20m2.append('low_altitude_recovery') or True
_inner20m2._reset_velocity_controller = lambda: _events20m2.append('controller_reset') or True
_inner20m2._try_service_backed_reset = lambda start_pose: _events20m2.append('service_reset') or False
_inner20m2._apply_and_confirm_start_pose = lambda start_pose: _events20m2.append('velocity_reset') or True
_obs20m2, _info20m2 = _env20m2.reset()
check("already-flying reset ignores service helper even when service is enabled", _events20m2 == ['low_altitude_recovery', 'controller_reset', 'wait_hover', 'velocity_reset'])
check("ignored service helper leaves no service-attempt diagnostic", _info20m2.get('reset_method') == 'velocity' and _info20m2.get('reset_service_attempted') is False and _info20m2.get('reset_service_status') == 'not_attempted')
check("ignored service helper velocity reset emits normalized obs", np.all(_obs20m2 >= -1.0) and np.all(_obs20m2 <= 1.0))
_env20m2.close()

_env20n = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone20n',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    use_simulator_reset_service=True,
    use_service_reset_after_velocity_timeout=False,
)
_inner20n = _env20n.unwrapped
_events20n = []
_inner20n._drone = fake_drone([2.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner20n._is_flying = True
_inner20n._wait_for_hover_settle = lambda: True
_inner20n._recover_low_altitude_hover_before_velocity_reset = lambda hover_height: True
_inner20n._reset_velocity_controller = lambda: True
_inner20n._try_service_backed_reset = lambda start_pose: _events20n.append('service_reset') or True


def _velocity_timeout20n(start_pose):
    _inner20n._last_reset_diagnostics = {
        'reason': 'timeout',
        'phase': 'drive_xy_axis_y',
        'position_error': 1.1123,
        'initial_position_error': 8.8071,
        'best_position_error': 0.9585,
    }
    raise RuntimeError('velocity timeout')


_inner20n._apply_and_confirm_start_pose = _velocity_timeout20n
try:
    _env20n.reset()
    _disabled_fallback_failed20n = False
except RuntimeError:
    _disabled_fallback_failed20n = True
check("already-flying velocity timeout does not use service fallback when flag disabled", _disabled_fallback_failed20n and _events20n == [])
_env20n.close()

_env20o = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone20o',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    use_simulator_reset_service=True,
    use_service_reset_after_velocity_timeout=True,
)
_inner20o = _env20o.unwrapped
_events20o = []
_drone20o = fake_drone([2.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner20o._drone = _drone20o
_inner20o._is_flying = True
_inner20o._wait_for_hover_settle = lambda: True
_inner20o._recover_low_altitude_hover_before_velocity_reset = lambda hover_height: True
_inner20o._reset_velocity_controller = lambda: True


def _velocity_timeout20o(start_pose):
    _inner20o._last_reset_diagnostics = {
        'reason': 'timeout',
        'phase': 'drive_xy_axis_y',
        'position_error': 1.1123,
        'initial_position_error': 8.8071,
        'best_position_error': 0.9585,
    }
    raise RuntimeError('velocity timeout')


def _service_fallback_success20o(start_pose):
    _events20o.append('service_reset')
    _drone20o.position = list(start_pose[:3])
    _drone20o.orientation = [0.0, 0.0, start_pose[3]]
    _inner20o._speed_handler = _AcceptingSpeedHandler()
    _inner20o._last_reset_diagnostics = {'reason': 'service_success', 'position_error': 0.01, 'yaw_error': 0.02}
    _inner20o._last_reset_service_status = 'service_success'
    _inner20o._is_flying = True
    return True


_inner20o._apply_and_confirm_start_pose = _velocity_timeout20o
_inner20o._try_service_backed_reset = _service_fallback_success20o
_obs20o, _info20o = _env20o.reset()
check("enabled velocity-timeout service fallback is attempted", _events20o == ['service_reset'])
check("service fallback success exposes coherent monitor diagnostics", _info20o.get('reset_method') == 'simulator_service' and _info20o.get('reset_path') == 'velocity_timeout_service_fallback' and _info20o.get('reset_service_status') == 'service_success_after_velocity_timeout')
check("service fallback success emits normalized obs", np.all(_obs20o >= -1.0) and np.all(_obs20o <= 1.0))
_env20o.close()

_env20p = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone20p',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    use_simulator_reset_service=True,
    use_service_reset_after_velocity_timeout=True,
)
_inner20p = _env20p.unwrapped
_drone20p = fake_drone([2.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner20p._drone = _drone20p
_inner20p._is_flying = True
_inner20p._wait_for_hover_settle = lambda: True
_inner20p._recover_low_altitude_hover_before_velocity_reset = lambda hover_height: True
_inner20p._reset_velocity_controller = lambda: True
_inner20p._recover_motion_reference_path = lambda: False


def _velocity_timeout20p(start_pose):
    _inner20p._last_reset_diagnostics = {
        'reason': 'timeout',
        'phase': 'drive_xy_axis_y',
        'position_error': 1.1123,
        'initial_position_error': 8.8071,
        'best_position_error': 0.9585,
    }
    raise RuntimeError('velocity timeout')


_inner20p._apply_and_confirm_start_pose = _velocity_timeout20p


def _service_fallback_without_command_path20p(start_pose):
    _drone20p.position = list(start_pose[:3])
    _drone20p.orientation = [0.0, 0.0, start_pose[3]]
    _inner20p._speed_handler = _RejectOnceSpeedHandler()
    _inner20p._last_reset_diagnostics = {'reason': 'service_success'}
    _inner20p._last_reset_service_status = 'service_success'
    _inner20p._is_flying = True
    return True


_inner20p._try_service_backed_reset = _service_fallback_without_command_path20p
try:
    _env20p.reset()
    _command_path_failure20p = False
except RuntimeError:
    _command_path_failure20p = True
check("service fallback only reports success if post-reset command path is ready", _command_path_failure20p and _inner20p._last_reset_diagnostics.get('reason') == 'service_fallback_command_path_failed')
_env20p.close()

_env20q = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone20q',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    use_simulator_reset_service=True,
    use_service_reset_after_velocity_timeout=True,
)
_inner20q = _env20q.unwrapped
_inner20q._drone = fake_drone([2.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner20q._is_flying = True
_inner20q._wait_for_hover_settle = lambda: True
_inner20q._recover_low_altitude_hover_before_velocity_reset = lambda hover_height: True
_inner20q._reset_velocity_controller = lambda: True


def _velocity_timeout20q(start_pose):
    _inner20q._last_reset_diagnostics = {
        'reason': 'timeout',
        'phase': 'drive_xy_axis_y',
        'position_error': 1.1123,
        'initial_position_error': 8.8071,
        'best_position_error': 0.9585,
    }
    raise RuntimeError('velocity timeout')


_inner20q._apply_and_confirm_start_pose = _velocity_timeout20q


def _service_fallback_failure20q(start_pose):
    _inner20q._last_reset_diagnostics = {'reason': 'service_unavailable', 'failure_class': 'service_unavailable'}
    _inner20q._last_reset_service_status = 'service_unavailable'
    return False


_inner20q._try_service_backed_reset = _service_fallback_failure20q
try:
    _env20q.reset()
    _service_failure20q = False
except RuntimeError:
    _service_failure20q = True
check("service fallback failure fails fast with diagnostics", _service_failure20q and _inner20q._last_reset_diagnostics.get('reason') == 'service_unavailable' and 'original_velocity_reset_diagnostics' in _inner20q._last_reset_diagnostics)
_env20q.close()

_env20l = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone20l',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    use_simulator_reset_service=True,
)
_inner20l = _env20l.unwrapped
_calls20l = {'service': 0, 'fallback': 0, 'arm': 0, 'offboard': 0, 'takeoff': 0}


class _FreshResetDrone:
    position = [0.0, 0.0, 0.0]
    speed = [0.0, 0.0, 0.0]
    orientation = [0.0, 0.0, 0.0]
    def arm(self, **kw):
        _calls20l['arm'] += 1
        return True
    def offboard(self, **kw):
        _calls20l['offboard'] += 1
        return True
    def takeoff(self, **kw):
        _calls20l['takeoff'] += 1
        return True
    def land(self, **kw): return True
    def manual(self, **kw): pass
    def shutdown(self, **kw): pass


_inner20l._drone = _FreshResetDrone()
_inner20l._is_flying = False
_inner20l._wait_for_hover_settle = _stub_wait_for_hover_settle


def _failed_fresh_service20l(start_pose):
    _calls20l['service'] += 1
    _inner20l._last_reset_diagnostics = {
        'reason': 'service_post_reset_command_path_failed',
        'offboard_ready': False,
        'controller_ready': False,
    }
    return False


_inner20l._try_service_backed_reset = _failed_fresh_service20l
_inner20l._apply_and_confirm_start_pose = lambda start_pose: _calls20l.__setitem__('fallback', _calls20l['fallback'] + 1) or True
try:
    _env20l.reset()
    _fresh_failure_fast20l = False
    _fresh_failure_text20l = ''
except RuntimeError as _exc20l:
    _fresh_failure_fast20l = True
    _fresh_failure_text20l = str(_exc20l)
check("fresh service reset failure fails fast", _fresh_failure_fast20l and 'service-backed reset failed' in _fresh_failure_text20l)
check("fresh service reset failure does not enter velocity fallback", _calls20l['fallback'] == 0)
check("fresh service reset performs takeoff before service reset readiness", _calls20l['arm'] == 1 and _calls20l['offboard'] == 1 and _calls20l['takeoff'] == 1)
_env20l.close()


class _DestroyableNode:
    def __init__(self):
        self.destroyed = False
        self.destroyed_clients = []

    def destroy_client(self, client):
        self.destroyed_clients.append(client)

    def destroy_node(self):
        self.destroyed = True


_env20e = AS2TestEnv(drone_namespace='drone20e')
_executor20e = _RecordingExecutor()
_node20e = _DestroyableNode()
_client20e = object()
_rclpy_mod20e = sys.modules['rclpy']
_original_shutdown20e = getattr(_rclpy_mod20e, 'shutdown', None)
_shutdown_calls20e = []
_env20e._reset_aux_executor = _executor20e
_env20e._reset_aux_node = _node20e
_env20e._reset_service_client = _client20e
_rclpy_mod20e.shutdown = lambda: _shutdown_calls20e.append('shutdown')
try:
    _env20e.close()
finally:
    if _original_shutdown20e is None:
        try:
            delattr(_rclpy_mod20e, 'shutdown')
        except AttributeError:
            pass
    else:
        _rclpy_mod20e.shutdown = _original_shutdown20e
check("close destroys reset service client", _node20e.destroyed_clients == [_client20e])
check("close shuts down reset auxiliary executor", _executor20e.shutdown_called)
check("close destroys reset auxiliary node", _node20e.destroyed)
check("close clears reset client references", _env20e._reset_service_client is None and _env20e._reset_aux_node is None and _env20e._reset_aux_executor is None)
check("close does not call global rclpy shutdown", _shutdown_calls20e == [])

# ===========================================================================
# TEST 21 — ResetSimulatorState service interface definition
# ===========================================================================
print("\n[21] ResetSimulatorState service interface definition")

_simulator_srv = _simulator_source / 'srv' / 'ResetSimulatorState.srv'
_simulator_cmake_text = _simulator_cmake.read_text(encoding='utf-8') if _simulator_cmake.is_file() else ''
_simulator_package_text = _simulator_package_xml.read_text(encoding='utf-8') if _simulator_package_xml.is_file() else ''
_expected_srv_fields = [
    'float64 x',
    'float64 y',
    'float64 z',
    'float64 yaw',
    'float64 position_tolerance',
    'float64 yaw_tolerance',
    'float64 linear_speed_tolerance',
    'float64 angular_speed_tolerance',
    '---',
    'bool success',
    'string message',
    'float64 position_error',
    'float64 yaw_error',
    'float64 linear_speed_norm',
    'float64 angular_speed_norm',
    'float64 linear_acceleration_norm',
    'float64 angular_acceleration_norm',
    'float64 force_norm',
    'float64 torque_norm',
    'float64 motor_angular_velocity_norm',
    'float64 motor_angular_acceleration_norm',
    'bool dynamics_reset_applied',
    'bool motor_state_reset_applied',
    'bool controller_reset_applied',
    'bool imu_reset_applied',
    'bool odometry_reset_applied',
    'bool references_reset_applied',
    'bool dynamic_artifacts_cleared',
]
if _simulator_srv.is_file():
    _simulator_srv_text = _simulator_srv.read_text(encoding='utf-8')
else:
    _simulator_srv_text = ''
check("ResetSimulatorState.srv exists in overlay package", _simulator_srv.is_file())
for _field in _expected_srv_fields:
    check(f"  srv includes `{_field}`", _field in _simulator_srv_text)
check("CMake generates ResetSimulatorState interface", 'rosidl_generate_interfaces' in _simulator_cmake_text and 'srv/ResetSimulatorState.srv' in _simulator_cmake_text)
check("package.xml declares rosidl generators", '<buildtool_depend>rosidl_default_generators</buildtool_depend>' in _simulator_package_text)
check("package.xml exports rosidl runtime", '<exec_depend>rosidl_default_runtime</exec_depend>' in _simulator_package_text)
check("package.xml marks package as rosidl interface package", '<member_of_group>rosidl_interface_packages</member_of_group>' in _simulator_package_text)

# ===========================================================================
# TEST 22 — Simulator reset service server contracts
# ===========================================================================
print("\n[22] Simulator reset service server contracts")

_simulator_header = _simulator_source / 'include' / 'as2_platform_multirotor_simulator' / 'as2_platform_multirotor_simulator.hpp'
_simulator_source_cpp = _simulator_source / 'src' / 'as2_platform_multirotor_simulator.cpp'
_simulator_header_text = _simulator_header.read_text(encoding='utf-8') if _simulator_header.is_file() else ''
_simulator_source_text = _simulator_source_cpp.read_text(encoding='utf-8') if _simulator_source_cpp.is_file() else ''

check(
    "simulator header includes generated reset service type",
    '#include "as2_platform_multirotor_simulator/srv/reset_simulator_state.hpp"' in _simulator_header_text,
)
check(
    "simulator owns one reset service server member",
    'rclcpp::Service<ResetSimulatorState>::SharedPtr reset_simulator_state_srv_' in _simulator_header_text,
)
check(
    "simulator declares reset service callback",
    'void resetSimulatorStateCallback(' in _simulator_header_text,
)
check(
    "simulator creates relative platform reset service under node namespace",
    'create_service<ResetSimulatorState>(' in _simulator_source_text
    and '"platform/reset_simulator_state"' in _simulator_source_text,
)
check(
    "CMake exposes generated reset service headers to simulator targets",
    '${CMAKE_CURRENT_BINARY_DIR}/rosidl_generator_cpp' in _simulator_cmake_text,
)
check(
    "CMake orders simulator targets after generated reset service C++ headers",
    '${PROJECT_NAME}__cpp' in _simulator_cmake_text,
)
check(
    "CMake links simulator targets to reset service typesupport",
    'rosidl_get_typesupport_target' in _simulator_cmake_text and '${cpp_typesupport_target}' in _simulator_cmake_text,
)
check(
    "simulator validates finite pose/yaw target fields",
    all(token in _simulator_source_text for token in ['std::isfinite(request->x)', 'std::isfinite(request->y)', 'std::isfinite(request->z)', 'std::isfinite(request->yaw)']),
)
check(
    "simulator validates positive reset tolerances",
    all(token in _simulator_source_text for token in ['request->position_tolerance <= 0.0', 'request->yaw_tolerance <= 0.0', 'request->linear_speed_tolerance <= 0.0', 'request->angular_speed_tolerance <= 0.0']),
)
check(
    "simulator validates finite reset tolerances",
    all(token in _simulator_source_text for token in ['std::isfinite(request->position_tolerance)', 'std::isfinite(request->yaw_tolerance)', 'std::isfinite(request->linear_speed_tolerance)', 'std::isfinite(request->angular_speed_tolerance)']),
)
check(
    "simulator service request has no namespace selector field",
    'string namespace' not in _simulator_srv_text and 'drone_namespace' not in _simulator_srv_text,
)
check(
    "simulator applies atomic reset state mutation",
    all(token in _simulator_source_text for token in ['simulator_.get_dynamics().set_state(reset_state)', 'simulator_.get_imu().reset()', 'simulator_.get_inertial_odometry().reset()']),
)
check(
    "simulator zeroes acceleration, force, torque, and motor artifacts",
    all(token in _simulator_source_text for token in [
        'reset_state.kinematics.linear_acceleration = zero_vector',
        'reset_state.kinematics.angular_acceleration = zero_vector',
        'reset_state.dynamics.force = zero_vector',
        'reset_state.dynamics.torque = zero_vector',
        'reset_state.actuators.motor_angular_velocity.setZero()',
        'reset_state.actuators.motor_angular_acceleration.setZero()',
    ]),
)
check(
    "simulator clears stale control references before success",
    all(token in _simulator_source_text for token in ['simulator_.get_controller().reset_controller()', 'simulator_.set_reference_position(target_position)', 'simulator_.set_reference_velocity(zero_vector)', 'simulator_.set_reference_yaw_rate(0.0)']),
)
check(
    "simulator exposes acceleration and dynamic artifact response metrics",
    all(token in _simulator_source_text for token in [
        'response->linear_acceleration_norm',
        'response->angular_acceleration_norm',
        'response->force_norm',
        'response->torque_norm',
        'response->motor_angular_velocity_norm',
        'response->motor_angular_acceleration_norm',
    ]),
)
check(
    "simulator exposes reset proof flags for internals and estimators",
    all(token in _simulator_source_text for token in [
        'response->dynamics_reset_applied',
        'response->motor_state_reset_applied',
        'response->controller_reset_applied',
        'response->imu_reset_applied',
        'response->odometry_reset_applied',
        'response->references_reset_applied',
        'response->dynamic_artifacts_cleared',
    ]),
)
check(
    "simulator reports valid reset success only after tolerance checks",
    'response->success = response->position_error <= request->position_tolerance' in _simulator_source_text and 'reset state applied' in _simulator_source_text,
)

# ===========================================================================
# TEST 23 — AS2 overlay launch/tmux sourcing contracts
# ===========================================================================
print("\n[23] AS2 overlay launch/tmux sourcing contracts")

_as2_sim_dir = Path('/home/jordi/TFM/rl_uav_aerostack2/as2_sim')
_launch_sim = _as2_sim_dir / 'launch_sim.bash'
_tmux_config = _as2_sim_dir / 'tmuxinator' / 'aerostack2.yaml'
_launch_sim_text = _launch_sim.read_text(encoding='utf-8') if _launch_sim.is_file() else ''
_tmux_config_text = _tmux_config.read_text(encoding='utf-8') if _tmux_config.is_file() else ''

check("launch script exists", _launch_sim.is_file())
check("tmuxinator config exists", _tmux_config.is_file())
check(
    "launch script sources ROS Humble before overlay validation",
    'source "/opt/ros/humble/setup.bash"' in _launch_sim_text,
)
check(
    "launch script defaults to the AS2 overlay install setup",
    'AS2_OVERLAY_SETUP:="/home/jordi/as2_rl_ws/install/setup.bash"' in _launch_sim_text,
)
check(
    "launch script verifies forked simulator package prefix",
    'ros2 pkg prefix as2_platform_multirotor_simulator' in _launch_sim_text
    and 'AS2_OVERLAY_PREFIX_EXPECTED' in _launch_sim_text,
)
check(
    "launch script passes overlay setup to tmux panes",
    'overlay_setup_file="${AS2_OVERLAY_SETUP}"' in _launch_sim_text,
)
check(
    "launch script passes expected overlay prefix to tmux panes",
    'overlay_prefix_expected="${AS2_OVERLAY_PREFIX_EXPECTED}"' in _launch_sim_text,
)
check(
    "tmux panes source ROS and overlay before ROS commands",
    'pre_window:' in _tmux_config_text
    and 'source /opt/ros/humble/setup.bash' in _tmux_config_text
    and 'source <%= overlay_setup_file %>' in _tmux_config_text,
)
check(
    "tmux panes fail closed on stale simulator package prefix",
    'ros2 pkg prefix as2_platform_multirotor_simulator' in _tmux_config_text
    and '<%= overlay_prefix_expected %>' in _tmux_config_text,
)

# ===========================================================================
# TEST 24 — AS2 DroneInterface executor ownership discovery
# ===========================================================================
print("\n[24] AS2 DroneInterface executor ownership discovery")

_drone_interface_base = Path('/opt/ros/humble/lib/python3.10/site-packages/as2_python_api/drone_interface_base.py')
_drone_interface_base_text = _drone_interface_base.read_text(encoding='utf-8') if _drone_interface_base.is_file() else ''
_as2_env_source = Path('/home/jordi/TFM/rl_uav_aerostack2/rl_uav/envs/as2_test_env.py')
_as2_env_text = _as2_env_source.read_text(encoding='utf-8') if _as2_env_source.is_file() else ''

check("installed DroneInterfaceBase source is available", _drone_interface_base.is_file())
check(
    "DroneInterfaceBase is itself an rclpy Node",
    'class DroneInterfaceBase(Node):' in _drone_interface_base_text,
)
check(
    "DroneInterfaceBase owns a private executor",
    'self.__executor = executor()' in _drone_interface_base_text
    and 'self.__executor.add_node(self)' in _drone_interface_base_text,
)
check(
    "DroneInterfaceBase spins that executor on an internal thread",
    'self.spin_thread = threading.Thread(target=self.__auto_spin)' in _drone_interface_base_text
    and 'self.__executor.spin_once(timeout_sec=0)' in _drone_interface_base_text,
)
_env24 = AS2TestEnv(drone_namespace='drone24')
check(
    "env reset client ownership model is auxiliary node",
    _env24._reset_client_ownership_model() == 'auxiliary_node',
)
_env24.close()
check(
    "env must not create reset clients on DroneInterface",
    '_drone.create_client(' not in _as2_env_text,
)
check(
    "env must not add DroneInterface to reset auxiliary executor",
    '_reset_aux_executor.add_node(self._drone' not in _as2_env_text
    and '_reset_aux_executor.add_node(_drone' not in _as2_env_text,
)

# ===========================================================================
# TEST 25 — Exp008 training-path smoke diagnostic contracts
# ===========================================================================
print("\n[25] Exp008 training-path smoke diagnostic contracts")

_training_smoke_path = Path(__file__).resolve().parent / 'validate_exp008_training_smoke.py'
_training_smoke_spec = importlib.util.spec_from_file_location(
    'validate_exp008_training_smoke',
    _training_smoke_path,
)
_training_smoke = importlib.util.module_from_spec(_training_smoke_spec)
assert _training_smoke_spec is not None and _training_smoke_spec.loader is not None
_training_smoke_spec.loader.exec_module(_training_smoke)

_smoke_action25 = _training_smoke.build_constant_action_batch(
    num_envs=1,
    action=[0.8, 0.0, 0.0, 0.0],
)
check(
    "training smoke builds SB3 vector action batch",
    _smoke_action25.shape == (1, 4)
    and _smoke_action25.dtype == np.float32
    and np.allclose(_smoke_action25[0], [0.8, 0.0, 0.0, 0.0], atol=1e-6),
)
check(
    "training smoke computes position deltas",
    np.allclose(
        _training_smoke.position_delta([0.0, -1.0, 1.0], [0.4, -0.7, 0.9]),
        [0.4, 0.3, -0.1],
        atol=1e-6,
    ),
)

_sample25 = _training_smoke.build_step_sample(
    episode_index=0,
    step_index=1,
    requested_action=[0.5, 0.0, 0.0, 0.0],
    before_info={'position': [0.0, 0.0, 1.6], 'speed': [0.0, 0.0, 0.0], 'distance': 0.5},
    info={
        'position': [0.02, 0.0, 1.6],
        'speed': [0.1, 0.0, 0.0],
        'distance': 0.48,
        'action_sent': [0.5, 0.0, 0.0, 0.0],
        'raw_action': [0.5, 0.0, 0.0, 0.0],
        'motion_command_accepted': True,
        'motion_command_recovered': False,
        'motion_command_publication_count': 3,
        'motion_command_accepted_publication_count': 2,
        'low_altitude_guard_active': False,
    },
    reward=-0.1,
    done=False,
)
check("training smoke records requested action", _sample25['action_requested'] == [0.5, 0.0, 0.0, 0.0])
check("training smoke records env-sent action", _sample25['action_sent'] == [0.5, 0.0, 0.0, 0.0])
check("training smoke records before/after speed", _sample25['speed_before'] == [0.0, 0.0, 0.0] and _sample25['speed_after'] == [0.1, 0.0, 0.0])
check("training smoke sample keeps position delta", np.allclose(_sample25['position_delta'], [0.02, 0.0, 0.0], atol=1e-6))
check("training smoke records motion publication counts", _sample25['motion_command_publication_count'] == 3 and _sample25['motion_command_accepted_publication_count'] == 2)

_stationary_count25 = 0
_stationary_decision25a = _training_smoke.evaluate_stationary_command_window(
    motion_command_accepted=True,
    position_delta_norm=0.0,
    consecutive_accepted_without_motion=_stationary_count25,
    stationary_command_window=3,
    stationary_command_epsilon=1e-4,
)
_stationary_count25 = _stationary_decision25a['consecutive_accepted_without_motion']
check("training smoke counts accepted stationary commands", _stationary_count25 == 1)
check("training smoke waits for the stationary command window", _stationary_decision25a['early_stop_reason'] is None)
_stationary_decision25b = _training_smoke.evaluate_stationary_command_window(
    motion_command_accepted=True,
    position_delta_norm=1e-4,
    consecutive_accepted_without_motion=_stationary_count25,
    stationary_command_window=3,
    stationary_command_epsilon=1e-4,
)
_stationary_count25 = _stationary_decision25b['consecutive_accepted_without_motion']
check("training smoke treats epsilon-bound deltas as stationary", _stationary_count25 == 2)
_stationary_decision25c = _training_smoke.evaluate_stationary_command_window(
    motion_command_accepted=True,
    position_delta_norm=2e-4,
    consecutive_accepted_without_motion=_stationary_count25,
    stationary_command_window=3,
    stationary_command_epsilon=1e-4,
)
check("training smoke resets stationary count after physical motion", _stationary_decision25c['consecutive_accepted_without_motion'] == 0)
_stationary_count25 = 0
for _ in range(3):
    _stationary_decision25d = _training_smoke.evaluate_stationary_command_window(
        motion_command_accepted=True,
        position_delta_norm=0.0,
        consecutive_accepted_without_motion=_stationary_count25,
        stationary_command_window=3,
        stationary_command_epsilon=1e-4,
    )
    _stationary_count25 = _stationary_decision25d['consecutive_accepted_without_motion']
check("training smoke stops after the stationary command window", _stationary_count25 == 3)
check(
    "training smoke reports accepted-without-motion early stop reason",
    _stationary_decision25d['early_stop_reason'] == 'accepted_commands_without_physical_motion',
)

_moving_samples25 = [
    {'distance_after': 2.0, 'position_delta': [0.02, 0.0, 0.0], 'motion_command_accepted': True},
    {'distance_after': 1.6, 'position_delta': [0.4, 0.0, 0.0], 'motion_command_accepted': True},
]
_moving_summary25 = _training_smoke.summarize_episode_motion(
    episode_index=0,
    samples=_moving_samples25,
    min_distance_reduction=0.25,
    min_position_delta=0.10,
    stationary_command_epsilon=1e-4,
)
check("training smoke accepts physical movement", _moving_summary25['success'] is True)
check("training smoke reports distance reduction", math.isclose(_moving_summary25['distance_reduction'], 0.4, abs_tol=1e-6))
check("training smoke counts accepted commands", _moving_summary25['accepted_command_steps'] == 2)

_stuck_samples25 = [
    {'distance_after': 2.0, 'position_delta': [0.0, 0.0, 0.0], 'motion_command_accepted': True},
    {'distance_after': 2.0, 'position_delta': [0.0, 0.0, 0.0], 'motion_command_accepted': True},
]
_stuck_summary25 = _training_smoke.summarize_episode_motion(
    episode_index=1,
    samples=_stuck_samples25,
    min_distance_reduction=0.25,
    min_position_delta=0.10,
    stationary_command_epsilon=1e-4,
)
check("training smoke rejects accepted-but-stuck commands", _stuck_summary25['success'] is False)
check("training smoke diagnoses insufficient movement", _stuck_summary25['reason'] == 'insufficient_physical_motion')

_epsilon_summary25 = _training_smoke.summarize_episode_motion(
    episode_index=2,
    samples=[
        {'distance_after': 2.0, 'position_delta': [0.005, 0.0, 0.0], 'motion_command_accepted': True},
        {'distance_after': 2.0, 'position_delta': [0.020, 0.0, 0.0], 'motion_command_accepted': True},
    ],
    min_distance_reduction=0.25,
    min_position_delta=0.10,
    stationary_command_epsilon=0.01,
)
check("training smoke summary uses configured stationary epsilon", _epsilon_summary25['accepted_without_motion_steps'] == 1)

_reset_summary25 = _training_smoke.summarize_reset_diagnostics(
    use_simulator_reset_service=True,
    diagnostics={'reason': 'service_success', 'position_error': 0.0},
)
check("training smoke records service-backed reset path", _reset_summary25['service_path'] == 'service_success')
check("training smoke records service reset enabled", _reset_summary25['use_simulator_reset_service'] is True)
_velocity_reset_summary25 = _training_smoke.summarize_reset_diagnostics(
    use_simulator_reset_service=True,
    diagnostics={'reason': 'post_controller_hold', 'position_error': 0.02},
)
check("training smoke recognizes configured service with velocity reset fallback path", _velocity_reset_summary25['service_attempted'] is False)
check("training smoke does not require service success for velocity reset path", _velocity_reset_summary25['reset_service_success'] is True)

# ===========================================================================
# TEST 26 — PPO monitor physical-motion diagnostics
# ===========================================================================
print("\n[26] PPO monitor physical-motion diagnostics")

_env26 = gymnasium.make(
    'AS2TestEnv-v0',
    drone_namespace='drone26',
    step_duration=0.0,
    fixed_start_pose=[0.0, 0.0, 1.0, 0.0],
    use_simulator_reset_service=False,
)
_inner26 = _env26.unwrapped
_drone26 = fake_drone([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
_inner26._drone = _drone26
_inner26._speed_handler = RecordingSpeedHandler(_drone26)
_inner26._is_flying = True
_inner26._wait_for_hover_settle = _stub_wait_for_hover_settle
_inner26._reset_velocity_controller = lambda: True
_inner26._apply_and_confirm_start_pose = lambda start_pose: True
_obs26, _reset_info26 = _env26.reset()
check("reset diagnostics expose velocity reset method", _reset_info26.get('reset_method') == 'velocity')
check("reset diagnostics expose reset path", _reset_info26.get('reset_path') in {'unknown', 'post_controller_hold', 'reached'})
check("reset diagnostics expose failure class field", 'reset_failure_class' in _reset_info26)
check("reset diagnostics expose position/yaw error fields", 'reset_position_error' in _reset_info26 and 'reset_yaw_error' in _reset_info26)
check("reset diagnostics expose service status field", _reset_info26.get('reset_service_status') == 'not_attempted')
check("reset diagnostics start physical displacement at zero", math.isclose(_reset_info26.get('physical_displacement', -1.0), 0.0, abs_tol=1e-6))
check("reset diagnostics start action norms at zero", all(math.isclose(_reset_info26.get(k, -1.0), 0.0, abs_tol=1e-6) for k in ['mean_action_norm', 'max_action_norm', 'mean_xy_action_norm', 'max_xy_action_norm']))
check("reset diagnostics start last action at zero", all(math.isclose(_reset_info26.get(k, -1.0), 0.0, abs_tol=1e-6) for k in ['last_action_vx', 'last_action_vy', 'last_action_vz', 'last_action_vyaw']))

_drone26.position = [0.0, 0.0, 1.0]
_inner26._speed_handler = RecordingSpeedHandler(_drone26)
_, _, _, _, _info26a = _env26.step(np.array([1.0, 0.0, -0.4, 0.0], dtype=np.float32))
_, _, _, _, _info26b = _env26.step(np.array([1.0, 0.0, -0.4, 0.0], dtype=np.float32))
check("step diagnostics accumulate physical displacement", _info26b.get('physical_displacement', 0.0) > _info26a.get('physical_displacement', 0.0) > 0.0)
check("step diagnostics accumulate path length", _info26b.get('path_length', 0.0) >= _info26b.get('physical_displacement', 0.0))
check("step diagnostics track minimum altitude", _info26b.get('min_altitude', 9.0) < 1.0)
check("step diagnostics count motion command attempts", _info26b.get('motion_command_steps') == 2)
check("step diagnostics count accepted motion commands", _info26b.get('motion_command_accepted_steps') == 2)
check("step diagnostics report command acceptance rate", math.isclose(_info26b.get('motion_command_acceptance_rate', 0.0), 1.0, abs_tol=1e-6))
_expected_action_norm26 = math.sqrt(1.0 ** 2 + 0.4 ** 2)
check("step diagnostics report mean action norm", math.isclose(_info26b.get('mean_action_norm', 0.0), _expected_action_norm26, abs_tol=1e-6))
check("step diagnostics report max action norm", math.isclose(_info26b.get('max_action_norm', 0.0), _expected_action_norm26, abs_tol=1e-6))
check("step diagnostics report mean XY action norm", math.isclose(_info26b.get('mean_xy_action_norm', 0.0), 1.0, abs_tol=1e-6))
check("step diagnostics report max XY action norm", math.isclose(_info26b.get('max_xy_action_norm', 0.0), 1.0, abs_tol=1e-6))
check("step diagnostics report last action components", all(math.isclose(_info26b.get(k, 9.0), v, abs_tol=1e-6) for k, v in {
    'last_action_vx': 1.0,
    'last_action_vy': 0.0,
    'last_action_vz': -0.4,
    'last_action_vyaw': 0.0,
}.items()))
check("step diagnostics keep reset method/path for monitor", _info26b.get('reset_method') == 'velocity' and 'reset_path' in _info26b)
check("step diagnostics keep reset failure diagnostics for monitor", all(k in _info26b for k in ['reset_failure_class', 'reset_position_error', 'reset_yaw_error', 'reset_service_status']))
_env26.close()

# ===========================================================================
# TEST 27 — Simulator reset proof and fork repository preparation docs
# ===========================================================================
print("\n[27] Simulator reset proof and fork repository preparation docs")

_reset_contract_doc = _simulator_source / 'RESET_CONTRACT.md'
_fork_repo_doc = _simulator_source / 'GITHUB_FORK_PREP.md'
_reset_contract_text = _reset_contract_doc.read_text(encoding='utf-8') if _reset_contract_doc.is_file() else ''
_fork_repo_text = _fork_repo_doc.read_text(encoding='utf-8') if _fork_repo_doc.is_file() else ''
_fork_git_status = subprocess.run(
    ['git', '-C', str(_simulator_source), 'rev-parse', '--is-inside-work-tree'],
    capture_output=True,
    text=True,
)
_fork_has_git = _fork_git_status.returncode == 0 and _fork_git_status.stdout.strip() == 'true'

check("reset contract proof document exists", _reset_contract_doc.is_file())
check(
    "reset proof document maps dynamic artifacts to implementation lines",
    all(token in _reset_contract_text for token in [
        'linear_acceleration',
        'angular_acceleration',
        'dynamics.force',
        'dynamics.torque',
        'motor_angular_velocity',
        'motor_angular_acceleration',
        'reset_controller',
        'get_imu().reset()',
        'get_inertial_odometry().reset()',
    ]),
)
check(
    "reset proof document defines telemetry-side acceleration boundary",
    'telemetry-side proof boundary' in _reset_contract_text.lower()
    and '/imu' in _reset_contract_text
    and '/ground_truth' in _reset_contract_text,
)
check(
    "reset proof document keeps ROS-free bindings as future work only",
    'ROS-free Python bindings' in _reset_contract_text
    and 'fixed-dt simulator APIs' in _reset_contract_text
    and 'future work' in _reset_contract_text.lower(),
)
check("GitHub fork preparation document exists", _fork_repo_doc.is_file())
check(
    "fork prep records manual remote, branch, commit pin, and rollback steps",
    all(token in _fork_repo_text for token in [
        'git init',
        'git remote add origin <REMOTE_URL>',
        'git checkout -b',
        'git rev-parse HEAD',
        'rollback',
        'Do not push from automation',
    ]),
)
check(
    "fork prep records intended remote branch commit pin and rollback note placeholders",
    all(token in _fork_repo_text for token in [
        'Intended remote',
        'Intended branch',
        'Commit pin',
        'Rollback note',
        'conda activate rl_uav',
        'AS2 Multirotor Simulator',
    ]),
)
check(
    "fork prep explicitly handles local non-git state",
    (not _fork_has_git and 'not a Git repository' in _fork_repo_text)
    or (_fork_has_git and 'git status --short --branch' in _fork_repo_text),
)

# ===========================================================================
# Summary
# ===========================================================================

print()
if errors == 0:
    print("\033[92m═══ All vectorization tests passed ✓ ═══\033[0m\n")
else:
    print(f"\033[91m═══ {errors} test(s) FAILED ═══\033[0m\n")
    sys.exit(1)
