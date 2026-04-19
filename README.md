# RL UAV Aerostack2

Reinforcement Learning framework for UAV training using [Aerostack2](https://github.com/aerostack2/aerostack2)'s multirotor dynamic simulator.

## Architecture

```
scripts/test_connection.py
        │  gymnasium API (reset / step / close)
        ▼
rl_uav/envs/as2_test_env.py   →   AS2TestEnv(gymnasium.Env)
        │  as2_python_api.DroneInterface (ROS2)
        ▼
AS2 Multirotor Simulator  (as2_sim/ configs + launch)
```

## Prerequisites

- **ROS2 Humble** (or compatible)
- **Aerostack2** installed in your ROS2 workspace ([installation guide](https://aerostack2.github.io/_00_getting_started/index.html))
- Python 3.10+

## Installation

1. Clone the repository:
```bash
git clone https://github.com/KafeisM/rl_uav_aerostack2.git
cd rl_uav_aerostack2
```

2. Install Python dependencies:
```bash
pip install -e .
```

## Usage

### 1. Launch the AS2 simulator

```bash
cd as2_sim
./launch_sim.bash
```

### 2. Run tests

In another terminal (with ROS2 sourced) for example:

```bash
conda run -n rl_uav python3 scripts/test_connection.py
```

This single-drone smoke test will:
- Connect to the simulator via `DroneInterface`
- Arm, set offboard mode, and take off
- Execute random velocity commands for N steps
- Print normalized relative observation `[dx, dy, dz, dyaw]` at each step
- Land and shut down

For the required real vectorized validation (4 drones, one Python process):

```bash
conda run -n rl_uav python3 scripts/validate_real_vectorized_sim.py
```

### 3. Stop the simulator

```bash
cd as2_sim
./stop_sim.bash
```

## Project Structure

```
rl_uav_aerostack2/
├── as2_sim/                    # AS2 simulator launch configuration
│   ├── config/                 # YAML configs for AS2 nodes
│   ├── tmuxinator/             # Tmuxinator session definitions
│   ├── utils/                  # Helper scripts
│   ├── launch_sim.bash         # Start simulator
│   └── stop_sim.bash           # Stop simulator
├── rl_uav/                     # RL environment package
│   ├── envs/
│   │   └── as2_test_env.py     # Test Gymnasium environment
│   └── __init__.py
├── scripts/
│   ├── test_connection.py              # Single-drone smoke connectivity test
│   ├── test_vectorization.py           # Deterministic mocked vectorization tests
│   └── validate_real_vectorized_sim.py # Real AS2 4-drone one-process validator
├── configs/                    # Training configs (future)
├── pyproject.toml
└── README.md
```

## License

BSD-3-Clause
