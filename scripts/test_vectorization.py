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
    pos=(1.0, 2.0, 3.0),
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
vec_env.close()

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

# Alignment test: measured velocity along +x and yaw=0 should maximize path-facing.
_pos8 = (1.0, 2.0, 3.0)
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
# TEST 14 — Single-drone smoke helper timeout contracts
# ===========================================================================
print("\n[14] Single-drone smoke helper timeout contracts")

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
# Summary
# ===========================================================================

print()
if errors == 0:
    print("\033[92m═══ All vectorization tests passed ✓ ═══\033[0m\n")
else:
    print(f"\033[91m═══ {errors} test(s) FAILED ═══\033[0m\n")
    sys.exit(1)
