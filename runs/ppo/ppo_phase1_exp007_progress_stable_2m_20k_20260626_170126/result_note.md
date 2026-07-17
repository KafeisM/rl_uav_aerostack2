# Exp007 Reset Reliability Gate — 2026-06-26 17:01

Result: FAIL

Command:

```bash
conda run -n rl_uav python scripts/train_ppo.py --config configs/train_ppo_phase1_exp007.yaml
```

AS2 hygiene:

```bash
conda run -n rl_uav bash as2_sim/stop_sim.bash
conda run -n rl_uav bash as2_sim/launch_sim.bash
conda run -n rl_uav bash as2_sim/stop_sim.bash
```

Run directory: `runs/ppo/ppo_phase1_exp007_progress_stable_2m_20k_20260626_170126`

Artifacts:
- Monitor: `monitor/vec_monitor.csv`
- Checkpoint: `checkpoints/ppo_phase1_exp007_2000_steps.zip`
- Tensorboard: `tensorboard/PPO_1/`

Metrics at failure:
- Timesteps reached: 3072 in SB3 rollout table before reset exception; latest completed checkpoint at 2000 steps.
- Completed episodes in monitor: 15.
- Success rate: 0.0.
- Terminal reasons: 13 `max_steps`, 2 `out_of_bounds`, 0 `success`.
- Final distance trend: unstable/non-convergent; last five completed episodes were 7.6818, 1.4680, 3.2022, 2.0388, 2.2926 m.

Reset failure:

```text
RuntimeError: Velocity-based start pose reset timed out; refusing to fall back to blocking go_to during in-air reset.
Reset diagnostics: reason=timeout, elapsed=20.004,
target_pose=[0.0, 0.0, 1.0, 0.0],
final_pose=[-0.37730684876441956, -0.463624507188797, 0.0, -0.7220121910056138],
position_error=1.165, yaw_error=0.722,
last_command=[0.0, 0.0, 1.0, 0.722]
```

Conclusion: Exp007 remains blocked as a reset reliability gate. Do not proceed to Exp008 or serious training until the in-air reset timeout is fixed.
