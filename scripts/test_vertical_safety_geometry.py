#!/usr/bin/env python3
"""ROS-free vertical safety geometry invariants for training configs.

Reads the actual YAML configs and instantiates AS2TestEnv with them, so the
environment's own guard/penalty formulas and bound validation are the source
of truth. Catches config edits that would make the low-altitude action guard
or the descent-penalty band overlap the nominal start/target altitude.

Usage:
    conda run -n rl_uav python scripts/test_vertical_safety_geometry.py
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_uav.envs.as2_test_env import AS2TestEnv

REPO_ROOT = Path(__file__).resolve().parents[1]

FIXED_START_CONFIGS = [
    'configs/train_ppo_phase1_exp008.yaml',
    'configs/train_ppo_phase1_exp008a.yaml',
]

BASELINE_CONFIG = 'configs/train_ppo.yaml'


class DummyDrone:
    def __init__(self, position: list[float]):
        self.position = position
        self.speed = [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, 0.0]

    def land(self, **kwargs):
        return True

    def manual(self, **kwargs):
        return None

    def shutdown(self, **kwargs):
        return None


def load_env_config(config_path: Path) -> dict:
    with config_path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    return config['environment']


def build_env_from_config(env_cfg: dict) -> AS2TestEnv:
    """Instantiate AS2TestEnv with every config key the constructor accepts."""
    accepted = set(inspect.signature(AS2TestEnv.__init__).parameters) - {'self'}
    kwargs = {key: value for key, value in env_cfg.items() if key in accepted}
    kwargs['verbose'] = False
    if isinstance(kwargs.get('height_bounds'), list):
        kwargs['height_bounds'] = tuple(kwargs['height_bounds'])
    return AS2TestEnv(**kwargs)


def assert_fixed_start_geometry(config_name: str, env: AS2TestEnv) -> None:
    floor_z, ceiling_z = env.height_bounds
    unsafe = float(env.unsafe_low_altitude_threshold)
    guard = env._low_altitude_guard_height()
    band_top = unsafe + float(env.vertical_safety_band)
    start_z = float(env.fixed_start_pose[2])
    target_z = float(env._target_pose[2])
    step_travel = 2.0 * float(env.max_vel) * float(env.step_duration)
    reset_tolerance = env._reset_position_tolerance()

    # Ordering: floor < unsafe < guard < start/target < ceiling.
    assert floor_z < unsafe, (
        f'{config_name}: unsafe threshold {unsafe} must sit above the floor {floor_z}'
    )
    assert unsafe < guard, (
        f'{config_name}: guard height {guard} must sit above unsafe threshold {unsafe}'
    )
    assert guard < start_z, (
        f'{config_name}: guard height {guard} must sit below start z {start_z}'
    )
    assert guard < target_z, (
        f'{config_name}: guard height {guard} must sit below target z {target_z}'
    )

    # (a) Guard never activates in nominal flight: clearance to the nominal
    # altitude covers two full steps of maximum descent, and even a worst-case
    # reset start (start z minus reset position tolerance) stays above it.
    assert guard + step_travel <= start_z + 1e-9, (
        f'{config_name}: guard {guard} + 2*max_vel*step_duration {step_travel} '
        f'exceeds start z {start_z}; guard would fight nominal flight'
    )
    assert guard < start_z - reset_tolerance, (
        f'{config_name}: guard {guard} overlaps worst-case reset start '
        f'{start_z - reset_tolerance}'
    )

    # (b) Guard fires with room to recover before the unsafe terminal threshold.
    assert guard - unsafe >= step_travel - 1e-9, (
        f'{config_name}: guard-to-unsafe margin {guard - unsafe} is below '
        f'2*max_vel*step_duration {step_travel}'
    )

    # (c) Unsafe threshold keeps a sane margin above the physical floor bound.
    assert unsafe - floor_z >= float(env.max_vel) * float(env.step_duration) - 1e-9, (
        f'{config_name}: unsafe {unsafe} leaves less than one step of descent '
        f'above floor {floor_z}'
    )

    # (d) Descent-penalty band must not overlap the start/target altitude.
    assert band_top < start_z, (
        f'{config_name}: penalty band top {band_top} overlaps start z {start_z}'
    )
    assert band_top < target_z, (
        f'{config_name}: penalty band top {band_top} overlaps target z {target_z}'
    )
    # Keep penalty band and guard co-located so both engage in the same zone.
    assert band_top <= guard + 1e-9, (
        f'{config_name}: penalty band top {band_top} extends above guard {guard}'
    )

    # (e) Success/OOB coherence: start and target z inside height bounds with
    # at least one step of maximum vertical travel as margin to the ceiling.
    vertical_margin = float(env.max_vel) * float(env.step_duration)
    for label, z_value in (('start', start_z), ('target', target_z)):
        assert floor_z + vertical_margin <= z_value <= ceiling_z - vertical_margin, (
            f'{config_name}: {label} z {z_value} lacks margin inside height '
            f'bounds ({floor_z}, {ceiling_z})'
        )
    # Recovery hover height must clear the guard band but stay below start z.
    recovery_height = float(env.reset_ground_recovery_height)
    assert unsafe < recovery_height < start_z, (
        f'{config_name}: reset_ground_recovery_height {recovery_height} must '
        f'lie between unsafe {unsafe} and start z {start_z}'
    )
    # (f) Start-target separation must exceed the success radius by more than
    # the in-air reset restore residual (fixed_start_tolerance plus observed
    # post-reset z settle), or episodes spawn already inside the success
    # sphere and terminate with zero motion.
    start_xyz = [float(v) for v in env.fixed_start_pose[:3]]
    target_xyz = [float(v) for v in env._target_pose[:3]]
    separation = math.dist(start_xyz, target_xyz)
    reset_residual_margin = float(env.fixed_start_tolerance) + 0.10
    assert separation - float(env.distance_threshold) > reset_residual_margin, (
        f'{config_name}: start-target separation {separation:.3f} minus '
        f'distance_threshold {env.distance_threshold} must exceed the reset '
        f'residual margin {reset_residual_margin:.3f}'
    )


def assert_guard_runtime_behavior(config_name: str, env: AS2TestEnv) -> None:
    """Exercise the actual guard/penalty code at the geometry boundaries."""
    start_z = float(env.fixed_start_pose[2])
    guard = env._low_altitude_guard_height()
    reset_tolerance = env._reset_position_tolerance()

    # Nominal altitude: level and descending-to-hover commands pass through.
    env._drone = DummyDrone([0.0, 0.0, start_z])
    assert env._apply_low_altitude_action_guard(0.0) == 0.0
    assert env._last_low_altitude_guard_active is False
    assert env._compute_vertical_safety_penalty(0.0) == 0.0
    assert env._compute_vertical_safety_penalty(-float(env.max_vel)) == 0.0

    # Worst-case reset start: guard still inactive.
    env._drone = DummyDrone([0.0, 0.0, start_z - reset_tolerance])
    assert env._apply_low_altitude_action_guard(0.0) == 0.0
    assert env._last_low_altitude_guard_active is False

    # Just inside the guard band: downward command becomes a forced climb.
    env._drone = DummyDrone([0.0, 0.0, guard - 0.01])
    guarded_vz = env._apply_low_altitude_action_guard(-float(env.max_vel))
    assert guarded_vz > 0.0, f'{config_name}: guard must force a climb below {guard}'
    assert env._last_low_altitude_guard_active is True
    assert env._compute_vertical_safety_penalty(-float(env.max_vel)) < 0.0


def assert_baseline_geometry(config_name: str, env: AS2TestEnv) -> None:
    """Baseline has no fixed start and disables the penalty band; the guard
    must still sit strictly below the target altitude."""
    floor_z, ceiling_z = env.height_bounds
    unsafe = float(env.unsafe_low_altitude_threshold)
    guard = env._low_altitude_guard_height()
    target_z = float(env._target_pose[2])

    assert floor_z < unsafe < guard < target_z, (
        f'{config_name}: expected floor {floor_z} < unsafe {unsafe} < '
        f'guard {guard} < target z {target_z}'
    )
    assert floor_z < target_z < ceiling_z, (
        f'{config_name}: target z {target_z} outside height bounds'
    )
    if float(env.vertical_safety_band) > 0.0:
        band_top = unsafe + float(env.vertical_safety_band)
        assert band_top < target_z, (
            f'{config_name}: penalty band top {band_top} overlaps target z {target_z}'
        )


def main() -> int:
    for config_name in FIXED_START_CONFIGS:
        env = build_env_from_config(load_env_config(REPO_ROOT / config_name))
        try:
            assert env.fixed_start_pose is not None, (
                f'{config_name}: expected a fixed_start_pose curriculum config'
            )
            assert_fixed_start_geometry(config_name, env)
            assert_guard_runtime_behavior(config_name, env)
        finally:
            env.close()
        print(f'✓ PASS: {config_name} vertical safety geometry is coherent')

    baseline_env = build_env_from_config(load_env_config(REPO_ROOT / BASELINE_CONFIG))
    try:
        assert_baseline_geometry(BASELINE_CONFIG, baseline_env)
    finally:
        baseline_env.close()
    print(f'✓ PASS: {BASELINE_CONFIG} guard stays below the target altitude')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
