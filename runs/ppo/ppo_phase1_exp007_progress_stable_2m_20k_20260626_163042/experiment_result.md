# Exp007 Reliability Gate Result

- Date: 2026-06-26
- Config: `configs/train_ppo_phase1_exp007.yaml`
- Command: `conda run -n rl_uav python scripts/train_ppo.py --config configs/train_ppo_phase1_exp007.yaml`
- Simulator restart:
  - `conda run -n rl_uav bash as2_sim/stop_sim.bash`
  - `conda run -n rl_uav bash as2_sim/launch_sim.bash`
- Run directory: `runs/ppo/ppo_phase1_exp007_progress_stable_2m_20k_20260626_163042`

## Status

FAIL — Exp007 did not pass the reset reliability gate and did not complete the intended 20k timesteps.

Training reached SB3 `total_timesteps=1664` and then stopped with:

```text
RuntimeError: Velocity-based start pose reset did not hold after controller reset. Reset diagnostics: reason=post_settle_drift, target_pose=[0.0, 0.0, 1.0, 0.0], final_pose=[-0.14037413895130157, -0.12850333750247955, 0.9856137633323669, 0.08931124962602122], position_error=0.1909, yaw_error=0.0893
```

## Metrics

- Completed monitor episodes: 8
- Completed monitor steps: 1586
- Successes: 0
- Success rate: 0.0
- Final distances: `[2.2705, 4.9662, 3.6618, 3.2242, 2.1982, 1.2995, 4.9760, 2.8100]`
- Mean final distance: 3.1758 m
- Terminal reasons: `max_steps=7`, `out_of_bounds=1`
- Checkpoints: none; failure occurred before the first 2000-step checkpoint.

## Reset Reliability Conclusion

The latest reset patch improved observability and bounded failure behavior, but the gate still fails. Resets no longer stall through `go_to`; instead, post-settle drift is detected and the run exits boundedly. Multiple post-settle drift retries occurred before the final failure.

Do not proceed to Exp008 until reset hold reliability is improved and Exp007 passes this gate.
