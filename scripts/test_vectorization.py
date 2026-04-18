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
  8. Reward correctness — continuous distance penalty value
  9. Terminal conditions — success, out-of-bounds, max_steps

Usage:
    conda run -n rl_uav python3 scripts/test_vectorization.py
"""

__authors__ = 'Jordi'
__license__ = 'BSD-3-Clause'

import sys
import types
import logging
import math

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
# TEST 8 — Reward correctness: continuous distance penalty
# ===========================================================================
print("\n[8] Reward correctness — continuous distance penalty")
# drone at [1.0, 2.0, 3.0], target = [0, 0, 1, 0]
# d = sqrt((1-0)^2 + (2-0)^2 + (3-1)^2) = sqrt(9) = 3.0
# d_max = 5.0 * sqrt(3) ≈ 8.660 → d_norm ≈ 0.3464 → reward ≈ -0.3464
_env8 = gymnasium.make('AS2TestEnv-v0', drone_namespace='drone0', step_duration=0.0)
_inner8 = _env8.unwrapped
_inner8._drone = fake_drone([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
_inner8._speed_handler = FakeSpeedHandler()
_inner8._is_flying = True
_env8.reset()  # required by OrderEnforcing wrapper
_inner8._drone = fake_drone([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])  # re-inject after reset

_, _rew8, _term8, _trunc8, _info8 = _env8.step(_env8.action_space.sample())

_d8 = math.sqrt(1.0 + 4.0 + 4.0)
_dmax8 = _inner8.pos_limit * math.sqrt(3.0)
_exp_rew8 = -(_d8 / _dmax8)

check("reward == -d_norm (exact)",              math.isclose(_rew8, _exp_rew8, rel_tol=1e-5))
check("reward in [-1.0, 0.0]",                 -1.0 <= _rew8 <= 0.0)
check("non-terminal: terminated=False",        not _term8)
check("non-terminal: truncated=False",         not _trunc8)
check("non-terminal: terminal_reason absent",  'terminal_reason' not in _info8)
_env8.close()

# ===========================================================================
# TEST 9 — Terminal conditions: success, out-of-bounds, max_steps
# ===========================================================================
print("\n[9] Terminal conditions — success / out-of-bounds / max_steps")

# 9a — Success: drone within distance_threshold of target
# target=[0,0,1,0], drone at [0.1, 0.0, 1.0] → d=0.1 < threshold=0.5
# yaw=0.0 == target_yaw → yaw_err=0 → no yaw penalty
_env9a = gymnasium.make('AS2TestEnv-v0', drone_namespace='drone0', step_duration=0.0)
_inner9a = _env9a.unwrapped
_inner9a._drone = fake_drone([0.1, 0.0, 1.0], [0.0, 0.0, 0.0], yaw=0.0)
_inner9a._speed_handler = FakeSpeedHandler()
_inner9a._is_flying = True
_env9a.reset()
_inner9a._drone = fake_drone([0.1, 0.0, 1.0], [0.0, 0.0, 0.0], yaw=0.0)  # re-inject

_, _rew9a, _term9a, _trunc9a, _info9a = _env9a.step(_env9a.action_space.sample())

_d9a = 0.1
_dmax9a = _inner9a.pos_limit * math.sqrt(3.0)
_exp_rew9a = -(_d9a / _dmax9a) + _inner9a.success_reward  # yaw_err=0

check("success: terminated=True",              _term9a)
check("success: truncated=False",              not _trunc9a)
check("success: terminal_reason='success'",    _info9a.get('terminal_reason') == 'success')
check("success: reward = -d_norm + bonus",     math.isclose(_rew9a, _exp_rew9a, rel_tol=1e-5))
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
# Summary
# ===========================================================================
print()
if errors == 0:
    print("\033[92m═══ All vectorization tests passed ✓ ═══\033[0m\n")
else:
    print(f"\033[91m═══ {errors} test(s) FAILED ═══\033[0m\n")
    sys.exit(1)
