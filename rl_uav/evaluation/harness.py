"""Seeded, controller-agnostic evaluation protocol for AS2TestEnv policies.

One protocol serves every controller (trained PPO policy, classical PID
baseline): episodes are generated from ``base_seed + i`` so any two
controllers can be evaluated on byte-identical start/target sequences and
compared metric by metric.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

DISTANCE_BINS: tuple[tuple[float, float], ...] = ((1.0, 2.5), (2.5, 3.5), (3.5, 5.0))


class Controller(ABC):
    """Minimal interface shared by learned policies and classical baselines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable controller identifier for reports."""

    def reset_episode(self) -> None:
        """Hook called at each episode start (e.g. clear PID integrators)."""

    @abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Map one observation to one action."""


class PPOController(Controller):
    """Deterministic wrapper over a loaded SB3 PPO model."""

    def __init__(self, model: Any, name: str = 'ppo'):
        self._model = model
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = 'cpu') -> 'PPOController':
        from stable_baselines3 import PPO

        model = PPO.load(str(path), device=device)
        return cls(model, name=f'ppo:{Path(path).stem}')

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self._model.predict(obs, deterministic=True)
        return action


@dataclass
class EpisodeRecord:
    """Per-episode evaluation result, one row of the comparison dataset."""

    episode_index: int
    seed: int
    success: bool
    terminal_reason: str
    steps: int
    sim_time_s: float
    start_pose: list[float]
    target_pose: list[float]
    start_target_distance: float
    final_distance: float
    path_length: float
    path_efficiency: Optional[float]
    terminal_yaw_error: float
    mean_action_norm: float
    max_action_norm: float
    motion_command_acceptance_rate: float
    min_altitude: float


def _info_float(info: dict, key: str, default: float = float('nan')) -> float:
    try:
        return float(info.get(key, default))
    except (TypeError, ValueError):
        return default


def run_evaluation(
    env,
    controller: Controller,
    episodes: int,
    base_seed: int,
    on_episode: Callable[[EpisodeRecord], None] | None = None,
) -> list[EpisodeRecord]:
    """Run ``episodes`` seeded episodes and collect one record per episode."""
    base_env = getattr(env, 'unwrapped', env)
    step_duration = float(getattr(base_env, 'step_duration', 0.0))
    records: list[EpisodeRecord] = []

    for index in range(episodes):
        seed = base_seed + index
        obs, info = env.reset(seed=seed)
        controller.reset_episode()
        start_pose = [float(v) for v in info['start_pose']]
        target_pose = [float(v) for v in info['target_pose']]
        start_target_distance = math.dist(start_pose[:3], target_pose[:3])

        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = controller.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1

        path_length = _info_float(info, 'path_length', 0.0)
        path_efficiency = (
            start_target_distance / path_length if path_length > 0.0 else None
        )
        record = EpisodeRecord(
            episode_index=index,
            seed=seed,
            success=bool(info.get('is_success', False)),
            terminal_reason=str(info.get('terminal_reason', 'unknown')),
            steps=steps,
            sim_time_s=steps * step_duration,
            start_pose=start_pose,
            target_pose=target_pose,
            start_target_distance=start_target_distance,
            final_distance=_info_float(info, 'final_distance'),
            path_length=path_length,
            path_efficiency=path_efficiency,
            terminal_yaw_error=_info_float(info, 'terminal_yaw_error'),
            mean_action_norm=_info_float(info, 'mean_action_norm'),
            max_action_norm=_info_float(info, 'max_action_norm'),
            motion_command_acceptance_rate=_info_float(
                info, 'motion_command_acceptance_rate'
            ),
            min_altitude=_info_float(info, 'min_altitude'),
        )
        records.append(record)
        if on_episode is not None:
            on_episode(record)

    return records


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {'mean': float('nan'), 'median': float('nan'), 'std': float('nan')}
    return {
        'mean': round(statistics.fmean(values), 4),
        'median': round(statistics.median(values), 4),
        'std': round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
    }


def summarize(records: list[EpisodeRecord]) -> dict[str, Any]:
    """Aggregate records into the comparison summary (JSON-serializable)."""
    n = len(records)
    successes = [r for r in records if r.success]

    terminal_reasons: dict[str, int] = {}
    for record in records:
        terminal_reasons[record.terminal_reason] = (
            terminal_reasons.get(record.terminal_reason, 0) + 1
        )

    distance_bins: dict[str, dict[str, Any]] = {}
    for low, high in DISTANCE_BINS:
        in_bin = [r for r in records if low <= r.start_target_distance < high]
        bin_successes = [r for r in in_bin if r.success]
        distance_bins[f'{low}-{high}'] = {
            'n': len(in_bin),
            'success_rate': (
                round(len(bin_successes) / len(in_bin), 4) if in_bin else None
            ),
            'mean_final_distance': (
                round(statistics.fmean([r.final_distance for r in in_bin]), 4)
                if in_bin else None
            ),
        }

    return {
        'episodes': n,
        'success_rate': round(len(successes) / n, 4) if n else None,
        'terminal_reasons': terminal_reasons,
        'final_distance': _stats([r.final_distance for r in records]),
        'steps': _stats([float(r.steps) for r in records]),
        'sim_time_s_success': _stats([r.sim_time_s for r in successes]),
        'path_efficiency_success': _stats(
            [r.path_efficiency for r in successes if r.path_efficiency is not None]
        ),
        'mean_action_norm': _stats([r.mean_action_norm for r in records]),
        'distance_bins': distance_bins,
    }


def write_outputs(
    records: list[EpisodeRecord],
    summary: dict[str, Any],
    output_dir: str | Path,
    metadata: dict[str, Any],
) -> None:
    """Write episodes.csv, summary.json and metadata.json under output_dir."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = [asdict(record) for record in records]
    for row in rows:
        row['start_pose'] = json.dumps(row['start_pose'])
        row['target_pose'] = json.dumps(row['target_pose'])
    with (output / 'episodes.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )
    (output / 'metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8'
    )
