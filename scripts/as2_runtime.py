#!/usr/bin/env python3
"""Small bounded helpers for AS2 simulator-dependent smoke tests."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AS2EnsureResult:
    """Result of an AS2 simulator ensure operation."""

    launched: bool
    namespaces: list[str]
    launch_script: Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tmux_session_exists(namespace: str) -> bool:
    result = subprocess.run(
        ['tmux', 'has-session', '-t', namespace],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
    )
    return result.returncode == 0


def missing_namespaces(namespaces: list[str]) -> list[str]:
    return [namespace for namespace in namespaces if not tmux_session_exists(namespace)]


def stop_tmux_session(namespace: str) -> None:
    subprocess.run(
        ['tmux', 'kill-session', '-t', namespace],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )


def drone_tmux_sessions() -> list[str]:
    result = subprocess.run(
        ['tmux', 'list-sessions', '-F', '#{session_name}'],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [name for name in result.stdout.splitlines() if name.startswith('drone')]


def ensure_as2_simulator(
    *,
    num_drones: int,
    launch_script: Path | None = None,
    launch_wait_s: float = 8.0,
    readiness_timeout_s: float = 30.0,
    force_relaunch: bool = False,
) -> AS2EnsureResult:
    """Ensure AS2 tmux sessions exist, launching the repo script if needed.

    This intentionally checks only process/session readiness. The caller's
    bounded env.reset()/step() remains the real AS2 health check.
    """

    if num_drones < 1:
        raise ValueError('num_drones must be >= 1')

    root = project_root()
    script = launch_script or root / 'as2_sim' / 'launch_sim.bash'
    namespaces = [f'drone{i}' for i in range(num_drones)]

    if not script.exists():
        raise FileNotFoundError(f'AS2 launch script not found: {script}')

    launched = False
    initial_missing = missing_namespaces(namespaces)
    extra_sessions = [name for name in drone_tmux_sessions() if name not in namespaces]
    if force_relaunch or initial_missing or extra_sessions:
        # launch_sim.bash always starts drone0..N-1. If a previous smoke left
        # only part of that set running, relaunch from a clean tmux set instead
        # of racing/duplicating an existing session name.
        sessions_to_stop = sorted(set(namespaces + extra_sessions))
        for namespace in sessions_to_stop:
            stop_tmux_session(namespace)
        if sessions_to_stop:
            time.sleep(2.0)

        subprocess.run(
            ['bash', str(script), '-n', str(num_drones)],
            cwd=str(script.parent),
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        launched = True
        time.sleep(max(0.0, launch_wait_s))

    deadline = time.time() + readiness_timeout_s
    last_missing = missing_namespaces(namespaces)
    while last_missing and time.time() < deadline:
        time.sleep(1.0)
        last_missing = missing_namespaces(namespaces)

    if last_missing:
        raise TimeoutError(
            'AS2 tmux readiness timed out; missing sessions: '
            + ', '.join(last_missing)
        )

    return AS2EnsureResult(launched=launched, namespaces=namespaces, launch_script=script)
