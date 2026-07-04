#!/usr/bin/env python3
"""Checks for PPO resume-from-checkpoint support.

Covers:
    - `resolve_resume_checkpoint('latest', ...)` picks the newest checkpoint
      matching the config's checkpoint_prefix under runs/ppo.
    - Explicit checkpoint paths pass through; missing paths raise.
    - 'latest' with a prefix that matches nothing raises FileNotFoundError.
    - Real SB3 smoke test: PPO.load restores num_timesteps from the exp011
      20000-steps checkpoint against a freshly constructed AS2TestEnv.

Usage:
    conda run -n rl_uav python scripts/test_resume_checkpoint.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_uav.envs.as2_test_env import AS2TestEnv
from rl_uav.training import build_env_kwargs, resolve_resume_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP011_CONFIG = REPO_ROOT / 'configs' / 'train_ppo_phase1_exp011.yaml'
EXP011_CHECKPOINT = (
    REPO_ROOT
    / 'runs'
    / 'ppo'
    / 'ppo_phase1_exp011_safe_area_randomized_50k_20260704_041029'
    / 'checkpoints'
    / 'ppo_phase1_exp011_20000_steps.zip'
)


def load_exp011_config() -> dict:
    with EXP011_CONFIG.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def test_latest_resolves_newest_exp011_checkpoint() -> None:
    config = load_exp011_config()
    checkpoint = resolve_resume_checkpoint(
        'latest', config, output_root=REPO_ROOT / 'runs' / 'ppo'
    )
    assert checkpoint.resolve() == EXP011_CHECKPOINT.resolve(), (
        f'latest resolved {checkpoint}, expected {EXP011_CHECKPOINT}'
    )


def test_explicit_path_passthrough_and_missing_path_raises() -> None:
    config = load_exp011_config()
    checkpoint = resolve_resume_checkpoint(str(EXP011_CHECKPOINT), config)
    assert checkpoint.resolve() == EXP011_CHECKPOINT.resolve()

    try:
        resolve_resume_checkpoint('/nonexistent/checkpoint.zip', config)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('nonexistent explicit path must raise FileNotFoundError')


def test_latest_with_unmatched_prefix_raises() -> None:
    config = copy.deepcopy(load_exp011_config())
    config['training']['checkpoint_prefix'] = 'ppo_phase1_nonexistent'
    try:
        resolve_resume_checkpoint(
            'latest', config, output_root=REPO_ROOT / 'runs' / 'ppo'
        )
    except FileNotFoundError as exc:
        assert 'ppo_phase1_nonexistent' in str(exc)
    else:
        raise AssertionError('unmatched prefix must raise FileNotFoundError')


def test_ppo_load_restores_num_timesteps() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env_cfg = load_exp011_config()['environment']
    kwargs = build_env_kwargs(env_cfg)
    kwargs['verbose'] = False
    env = AS2TestEnv(**kwargs)  # no ROS at construct time
    venv = DummyVecEnv([lambda: env])
    try:
        model = PPO.load(str(EXP011_CHECKPOINT), env=venv, device='cpu')
        assert model.num_timesteps == 20000, (
            f'num_timesteps {model.num_timesteps} != 20000'
        )
    finally:
        # Avoid AS2TestEnv.close() ROS teardown: the env never initialized ROS.
        pass


def main() -> int:
    test_latest_resolves_newest_exp011_checkpoint()
    test_explicit_path_passthrough_and_missing_path_raises()
    test_latest_with_unmatched_prefix_raises()
    test_ppo_load_restores_num_timesteps()
    print('✓ PASS: resume checkpoint resolution and PPO.load contract hold')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
