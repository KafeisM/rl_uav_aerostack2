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
import threading
from pathlib import Path

import numpy as np


class OperationTimeout(TimeoutError):
    """Raised when a smoke-test operation exceeds its bounded timeout."""


def run_with_timeout(func, timeout_s: float, operation: str):
    """Run a blocking simulator operation with a visible bounded timeout."""
    if timeout_s <= 0.0:
        return func()

    box: dict[str, object] = {}

    def _runner():
        try:
            box['result'] = func()
        except Exception as exc:  # noqa: BLE001 - preserve original operation failure
            box['error'] = exc

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)

    if worker.is_alive():
        raise OperationTimeout(f'{operation} exceeded {timeout_s:.1f}s')
    if 'error' in box:
        raise box['error']  # type: ignore[misc]
    return box.get('result')


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
    parser.add_argument(
        '--reset-timeout',
        type=float,
        default=120.0,
        help='Maximum seconds to wait for env.reset() before failing visibly (default: 120)')
    parser.add_argument(
        '--step-timeout',
        type=float,
        default=10.0,
        help='Maximum seconds to wait for each env.step() before failing visibly (default: 10)')
    parser.add_argument(
        '--close-timeout',
        type=float,
        default=15.0,
        help='Maximum seconds to wait for env.close() during cleanup (default: 15)')
    parser.add_argument(
        '--skip-ensure-as2',
        action='store_true',
        default=False,
        help='Do not verify/launch AS2 before running the smoke test')

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stdout,
        force=True,
    )
    logger = logging.getLogger('test_connection')

    logger.info("Starting single-drone smoke test; bounded timeouts are enabled.")

    if not args.skip_ensure_as2:
        from as2_runtime import ensure_as2_simulator

        try:
            logger.info("Ensuring AS2 simulator is running for namespace %s...", args.namespace)
            ensure_result = ensure_as2_simulator(
                num_drones=1,
                launch_script=Path(__file__).resolve().parents[1] / 'as2_sim' / 'launch_sim.bash',
                launch_wait_s=8.0,
                readiness_timeout_s=30.0,
                force_relaunch=True,
            )
            logger.info(
                "AS2 ensure complete: launched=%s namespaces=%s script=%s",
                ensure_result.launched,
                ensure_result.namespaces,
                ensure_result.launch_script,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("AS2 ensure failed: %s", e, exc_info=True)
            return 1

    # Import environment (deferred to allow --help without ROS2)
    from rl_uav.envs import AS2TestEnv

    logger.info("=" * 60)
    logger.info("AS2 Simulator Connectivity Test")
    logger.info("=" * 60)
    logger.info(f"  Drone namespace: {args.namespace}")
    logger.info(f"  Test steps: {args.steps}")
    logger.info(f"  Use sim time: {args.use_sim_time}")
    logger.info(f"  Reset timeout: {args.reset_timeout:.1f}s")
    logger.info(f"  Step timeout: {args.step_timeout:.1f}s")
    logger.info(f"  Close timeout: {args.close_timeout:.1f}s")
    logger.info("")

    # Create environment
    logger.info("Creating AS2TestEnv...")
    env = None

    try:
        env = AS2TestEnv(
            drone_namespace=args.namespace,
            use_sim_time=args.use_sim_time,
            verbose=args.verbose,
        )

        # Reset (arm, offboard, takeoff)
        logger.info("Resetting environment (arm → offboard → takeoff)...")
        obs, info = run_with_timeout(env.reset, args.reset_timeout, 'env.reset()')

        logger.info(f"Initial observation: {obs}")
        logger.info(f"Initial info: {info}")
        logger.info(
            "Reset mode=%s start_pose=%s target_pose=%s attempts=%s hover_settled=%s",
            info.get('reset_mode'),
            info.get('start_pose'),
            info.get('target_pose'),
            info.get('sample_attempts'),
            info.get('hover_settled'),
        )
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

            obs, reward, terminated, truncated, info = run_with_timeout(
                lambda action=action: env.step(action),
                args.step_timeout,
                f'env.step({i + 1})',
            )

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
        return 0

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130

    except OperationTimeout as e:
        logger.error("Test FAILED: %s", e)
        logger.error(
            "Readiness diagnostic: namespace=%s use_sim_time=%s. "
            "Check AS2 Multirotor Simulator is launched and the namespace exists.",
            args.namespace,
            args.use_sim_time,
        )
        return 1

    except Exception as e:
        logger.error(f"Test FAILED: {e}", exc_info=True)
        return 1

    finally:
        # Close (land, shutdown)
        logger.info("Closing environment (land → shutdown)...")
        if env is not None:
            try:
                run_with_timeout(env.close, args.close_timeout, 'env.close()')
            except OperationTimeout as e:
                logger.error("Cleanup timed out: %s", e)
            except Exception as e:  # noqa: BLE001
                logger.error("Cleanup failed: %s", e, exc_info=True)
        logger.info("Done.")


if __name__ == '__main__':
    raise SystemExit(main())
