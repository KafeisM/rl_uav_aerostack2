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

Usage:
    conda run -n rl_uav python3 scripts/test_vectorization.py
"""

__authors__ = 'Jordi'
__license__ = 'BSD-3-Clause'

import sys
import types
import logging

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


def fake_drone(pos, vel):
    """Mock drone with fixed position and velocity."""
    class _Drone:
        position = list(pos)
        speed = list(vel)
        orientation = [0.0, 0.0, 0.0, 1.0]
        def arm(self, **kw): return True
        def offboard(self, **kw): return True
        def takeoff(self, **kw): return True
        def land(self, **kw): return True
        def manual(self, **kw): pass
        def shutdown(self, **kw): pass
    return _Drone()


class FakeSpeedHandler:
    def send_speed_command_with_yaw_angle(self, **kw): pass


def inject_mocks(vec_env, positions, velocities):
    """Inject mock drones into each sub-environment."""
    for i, env in enumerate(vec_env.envs):
        inner = env.unwrapped
        inner._drone = fake_drone(positions[i], velocities[i])
        inner._speed_handler = FakeSpeedHandler()
        inner._is_flying = True


# Test data: 4 drones with different positions and velocities
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
check("single_obs_space.shape == (6,)",            vec_env.single_observation_space.shape == (6,))
check("single_action_space.shape == (3,)",         vec_env.single_action_space.shape == (3,))
check(f"batched obs_space.shape == ({NUM_ENVS},6)", vec_env.observation_space.shape == (NUM_ENVS, 6))
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
inject_mocks(vec_env, POSITIONS, VELOCITIES)

obs_r, infos_r = vec_env.reset()
check(f"reset: obs.shape == ({NUM_ENVS}, 6)", obs_r.shape == (NUM_ENVS, 6))
check("reset: obs.dtype == float32",           obs_r.dtype == np.float32)
check("reset: infos is not empty",             len(infos_r) > 0)

inject_mocks(vec_env, POSITIONS, VELOCITIES)  # re-inject after reset
actions = vec_env.action_space.sample()
obs_s, rews, terms, truncs, infos_s = vec_env.step(actions)
check(f"step: obs.shape == ({NUM_ENVS}, 6)",      obs_s.shape == (NUM_ENVS, 6))
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
inject_mocks(vec_env, POSITIONS, VELOCITIES)
obs_r, _ = vec_env.reset()
check("reset: all obs >= -1.0", np.all(obs_r >= -1.0))
check("reset: all obs <=  1.0", np.all(obs_r <=  1.0))

# Test clipping when drone exceeds boundaries (10 m with pos_limit=5)
out_pos  = [[10.0, -10.0, 10.0]] * NUM_ENVS
out_vels = [[ 5.0,  -5.0,  5.0]] * NUM_ENVS
inject_mocks(vec_env, out_pos, out_vels)
obs_out, _ = vec_env.reset()
check("out-of-bounds clipped to ±1.0", np.all(np.abs(obs_out) == 1.0))
vec_env.close()

# ===========================================================================
# TEST 6 — Numerical normalization correctness per sub-environment
# ===========================================================================
print("\n[6] Numerical normalization correctness per sub-environment")
vec_env = gymnasium.vector.SyncVectorEnv([
    (lambda ns=f'drone{i}': gymnasium.make('AS2TestEnv-v0', drone_namespace=ns))
    for i in range(NUM_ENVS)
])
inject_mocks(vec_env, POSITIONS, VELOCITIES)
vec_env.reset()

for i in range(NUM_ENVS):
    inner = vec_env.envs[i].unwrapped
    inner._drone = fake_drone(POSITIONS[i], VELOCITIES[i])  # re-inject after reset
    expected = np.clip(np.array([
        POSITIONS[i][0]  / inner.pos_limit,
        POSITIONS[i][1]  / inner.pos_limit,
        POSITIONS[i][2]  / inner.pos_limit,
        VELOCITIES[i][0] / inner.max_vel,
        VELOCITIES[i][1] / inner.max_vel,
        VELOCITIES[i][2] / inner.max_vel,
    ], dtype=np.float32), -1.0, 1.0)
    actual = inner._get_obs()
    check(f"  sub-env[{i}] normalized obs is correct", np.allclose(actual, expected))
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
# Summary
# ===========================================================================
print()
if errors == 0:
    print("\033[92m═══ All vectorization tests passed ✓ ═══\033[0m\n")
else:
    print(f"\033[91m═══ {errors} test(s) FAILED ═══\033[0m\n")
    sys.exit(1)
