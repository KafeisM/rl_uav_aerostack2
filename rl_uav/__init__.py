"""RL UAV Aerostack2 — Gymnasium environments for UAV training."""

import gymnasium

# ---------------------------------------------------------------------------
# Register Gymnasium environments
# ---------------------------------------------------------------------------
# This allows creating environments via:
#     gymnasium.make("AS2TestEnv-v0", drone_namespace="drone0", ...)
#
# And vectorized environments via:
#     gymnasium.make_vec("AS2TestEnv-v0", num_envs=N, vectorization_mode="sync", ...)
# ---------------------------------------------------------------------------

gymnasium.register(
    id='AS2TestEnv-v0',
    entry_point='rl_uav.envs.as2_test_env:AS2TestEnv',
)
