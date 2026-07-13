# RL UAV Aerostack2

Reinforcement-learning framework for UAV point-reaching experiments on the
[Aerostack2](https://github.com/aerostack2/aerostack2) multirotor simulator.

The repository contains the Gymnasium environment, PPO training pipeline,
seeded evaluation harness, and PID baseline used to compare a learned policy
against a conventional controller on the same randomized task protocol.

## Current status

- **Mono-drone 5 m task solved with PPO**: experiment `exp016` reached 574/629
  successes overall (91%) and 158/160 successes in the final fresh-simulator
  segment (99%), with final distance around 0.32 m.
- **Evaluation is controller-agnostic**: PPO and PID controllers can be evaluated
  with the same seeded episodes, metrics, CSV output, and JSON summaries.
- **Vectorized training support is implemented**: the environment supports
  multi-drone namespaces and `DummyVecEnv`/`SubprocVecEnv`. The final mitigation
  for stale AS2 velocity commands is the idle-command hover watchdog.
- **Known limitation**: vectorized training is treated as an engineering path
  with documented caveats, not as the final headline result. The strongest final
  result is the mono-drone PPO policy plus the shared PPO/PID evaluation stack.

## Architecture

```text
configs/*.yaml
        │  declarative env/training/evaluation settings
        ▼
scripts/train_ppo.py ───────┐
scripts/evaluate_policy.py  │
        │                   │  Gymnasium + Stable-Baselines3
        ▼                   ▼
rl_uav/envs/as2_test_env.py      rl_uav/evaluation/
        │  as2_python_api.DroneInterface / velocity commands
        ▼
AS2 Multirotor Simulator  (as2_sim/ launch + stop helpers)
```

`AS2TestEnv-v0` exposes normalized relative observations `[dx, dy, dz, dyaw]`
and velocity actions `[vx, vy, vz, vyaw]`. Episodes use randomized starts and
targets, safety/action guards, service-backed resets, monitor diagnostics, and
optional idle hover commands during reset/coordination gaps.

## Prerequisites

- ROS 2 Humble or compatible.
- Aerostack2 installed in a sourced ROS 2 workspace.
- Python 3.10+.
- A Python environment with this package installed, commonly the local
  `rl_uav` Conda environment used during development.

## Installation

```bash
git clone https://github.com/KafeisM/rl_uav_aerostack2.git
cd rl_uav_aerostack2
pip install -e .[train]
```

If you only need the base package without training extras:

```bash
pip install -e .
```

## Quick start

### 1. Launch the AS2 simulator

```bash
cd as2_sim
./launch_sim.bash
```

The simulator must be running before executing live connection, training, or
evaluation commands.

### 2. Run a connectivity smoke test

```bash
conda run -n rl_uav python3 scripts/test_connection.py
```

This connects through `DroneInterface`, arms the drone, enters offboard mode,
takes off, executes velocity commands, prints observations, lands, and shuts
down.

### 3. Validate vectorized simulator access

```bash
conda run -n rl_uav python3 scripts/validate_real_vectorized_sim.py
```

This validates the real multi-drone, one-process path used by vectorized
training experiments.

### 4. Train PPO

Dry-run the training stack without calling `learn()`:

```bash
conda run -n rl_uav python3 scripts/train_ppo.py \
  --config configs/train_ppo_phase1_exp016.yaml \
  --dry-run
```

Run training with the simulator already active:

```bash
conda run -n rl_uav python3 scripts/train_ppo.py \
  --config configs/train_ppo_phase1_exp016.yaml
```

Resume from the latest checkpoint or an explicit checkpoint:

```bash
conda run -n rl_uav python3 scripts/train_ppo.py \
  --config configs/train_ppo_phase1_exp016.yaml \
  --resume latest
```

For long runs, use the supervisor so the run can recycle the simulator and warm
start from checkpoints:

```bash
scripts/supervise_train.sh configs/train_ppo_phase1_exp016.yaml
```

### 5. Evaluate PPO and PID on the same protocol

Evaluate a trained PPO policy:

```bash
conda run -n rl_uav python3 scripts/evaluate_policy.py \
  --controller ppo \
  --model runs/ppo/exp016/model_final.zip \
  --episodes 20 \
  --seed 123
```

Evaluate the PID baseline with the same seeded task protocol:

```bash
conda run -n rl_uav python3 scripts/evaluate_policy.py \
  --controller pid \
  --episodes 20 \
  --seed 123
```

Both modes write comparable metrics through the shared evaluation harness.

### 6. Stop the simulator

```bash
cd as2_sim
./stop_sim.bash
```

## Key files

| Path | Purpose |
| --- | --- |
| `rl_uav/envs/as2_test_env.py` | Gymnasium AS2 environment, reset logic, guards, target sampling, watchdog support. |
| `rl_uav/training/ppo_setup.py` | YAML config loading, vectorized env construction, PPO setup, checkpoint resume helpers. |
| `scripts/train_ppo.py` | PPO CLI entry point for dry-runs, training, resume, and final model export. |
| `scripts/supervise_train.sh` | Long-run supervisor with preventive simulator recycle and checkpoint warm start. |
| `scripts/evaluate_policy.py` | Controller-agnostic evaluation CLI for PPO and PID. |
| `rl_uav/evaluation/harness.py` | Seeded evaluation protocol, metric aggregation, CSV/JSON output. |
| `rl_uav/evaluation/pid.py` | Conventional PID baseline controller. |
| `configs/train_ppo_phase1_exp016.yaml` | Final mono-drone full 5 m curriculum configuration. |
| `configs/train_ppo_phase1_exp018.yaml` | Vectorized retry configuration with idle-command watchdog enabled. |
| `scripts/test_evaluation_harness.py` | ROS-free tests for evaluation behavior and outputs. |
| `scripts/test_idle_watchdog.py` | Tests for the watchdog that prevents stale-command drift. |

## Project structure

```text
rl_uav_aerostack2/
├── as2_sim/                    # AS2 simulator launch/stop configuration
├── configs/                    # PPO experiment YAML files
├── rl_uav/
│   ├── envs/                   # Gymnasium AS2 environment
│   ├── evaluation/             # PPO/PID evaluation harness and controllers
│   └── training/               # SB3 setup, config, checkpoint helpers
├── scripts/                    # Training, evaluation, smoke, validation, tests
├── pyproject.toml              # Package metadata and dependencies
└── README.md
```

## Known limitations and future work

- The final documented learning result is the mono-drone PPO policy; vectorized
  training is implemented and partially validated, but still carries AS2
  coordination caveats.
- PPO and PID are evaluated through the same harness, but the exact comparison
  should be regenerated from the final local model artifacts before publishing
  final tables.
- Removing the runtime ROS dependency from the Python training interface remains
  future work.

## License

BSD-3-Clause
