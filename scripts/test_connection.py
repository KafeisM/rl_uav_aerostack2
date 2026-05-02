#!/usr/bin/env python3
"""
Test script for verifying connectivity with the AS2 multirotor simulator.

Usage:
    1. Launch the AS2 simulator:
       ./as2_sim/launch_sim.bash

    2. Run this script:
       conda run -n rl_uav python3 scripts/test_connection.py

    3. Stop the simulator:
       ./as2_sim/stop_sim.bash

The script will:
    - Create one AS2TestEnv (single-drone smoke test)
    - Reset (arm, offboard, takeoff)
    - Execute N random steps, printing normalized relative state [dx, dy, dz, dyaw]
    - Close (land, shutdown)

For 4-drone one-process real validation, use:
    conda run -n rl_uav python3 scripts/validate_real_vectorized_sim.py
"""

__authors__ = 'Jordi'
__license__ = 'BSD-3-Clause'

import argparse
import logging
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Test connectivity with the AS2 multirotor simulator')
    parser.add_argument(
        '-n', '--namespace',
        type=str,
        default='drone0',
        help='Drone namespace (default: drone0)')
    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='Number of test steps to execute (default: 50)')
    parser.add_argument(
        '--use-sim-time',
        action='store_true',
        default=False,
        help='Use simulation time')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=False,
        help='Enable verbose output')

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger('test_connection')

    # Import environment (deferred to allow --help without ROS2)
    from rl_uav.envs import AS2TestEnv

    logger.info("=" * 60)
    logger.info("AS2 Simulator Connectivity Test")
    logger.info("=" * 60)
    logger.info(f"  Drone namespace: {args.namespace}")
    logger.info(f"  Test steps: {args.steps}")
    logger.info(f"  Use sim time: {args.use_sim_time}")
    logger.info("")

    # Create environment
    logger.info("Creating AS2TestEnv...")
    env = AS2TestEnv(
        drone_namespace=args.namespace,
        use_sim_time=args.use_sim_time,
        verbose=args.verbose,
    )

    try:
        # Reset (arm, offboard, takeoff)
        logger.info("Resetting environment (arm → offboard → takeoff)...")
        obs, info = env.reset()

        logger.info(f"Initial observation: {obs}")
        logger.info(f"Initial info: {info}")
        logger.info("")

        if not info.get('reset_success', False):
            logger.error("Reset failed! Check simulator connection.")
            sys.exit(1)

        # Execute random steps
        logger.info(f"Executing {args.steps} random steps...")
        logger.info("-" * 60)

        for i in range(args.steps):
            # Random velocity command
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            # Print state every step
            rel_pos = obs[:3]
            rel_yaw = obs[3]
            logger.info(
                f"Step {i+1:3d} | "
                f"RelPos: [{rel_pos[0]:7.3f}, {rel_pos[1]:7.3f}, {rel_pos[2]:7.3f}] | "
                f"dYaw: {rel_yaw:7.3f} | "
                f"Action: [{action[0]:6.3f}, {action[1]:6.3f}, {action[2]:6.3f}, {action[3]:6.3f}]"
            )

            if terminated or truncated:
                terminal_reason = info.get('terminal_reason')
                if terminal_reason == 'success':
                    logger.info(
                        "Episode ended early with successful termination "
                        f"(reason: {terminal_reason})."
                    )
                else:
                    reason_msg = terminal_reason if terminal_reason is not None else 'unknown'
                    logger.warning(
                        "Episode terminated/truncated before completing requested "
                        f"steps (reason: {reason_msg})."
                    )
                break

        logger.info("-" * 60)
        logger.info("")

        # Summary
        logger.info("=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)
        logger.info(f"  Steps completed: {i+1}")
        logger.info(f"  Final relative state: {obs}")
        logger.info(f"  Final info: {info}")
        logger.info("")
        logger.info("Connectivity test PASSED ✓")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")

    except Exception as e:
        logger.error(f"Test FAILED: {e}", exc_info=True)

    finally:
        # Close (land, shutdown)
        logger.info("Closing environment (land → shutdown)...")
        env.close()
        logger.info("Done.")


if __name__ == '__main__':
    main()
