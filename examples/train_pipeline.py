#!/usr/bin/env python3
"""
One-click training pipeline for PenguinChess.

Orchestrates PPO training → ELO evaluation → (optional) AlphaZero training
in a single command with real-time progress updates.

Usage:
    uv run python examples/train_pipeline.py --ppo-steps 100000 --num-envs 4
    uv run python examples/train_pipeline.py --ppo-steps 50000 --with-az --az-iterations 10
    uv run python examples/train_pipeline.py --ppo-steps 200000 --with-az --num-envs 4 --eval-episodes 200
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path so all penguinchess imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from penguinchess.training_status import update_status, clear_status


def phase_header(phase: str, msg: str) -> None:
    """Print a visually distinct phase separator."""
    print(f"\n{'=' * 60}")
    print(f"  Phase {phase}: {msg}")
    print(f"{'=' * 60}")


def _build_ppo_args(pipeline_args: argparse.Namespace) -> argparse.Namespace:
    """Build a parsed-args namespace compatible with train_ppo.train()."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--force-gpu", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)

    # Build argument list from pipeline args
    argv = [
        "--timesteps", str(pipeline_args.ppo_steps),
        "--num-envs", str(pipeline_args.num_envs),
        "--lr", str(pipeline_args.lr),
        "--batch-size", str(pipeline_args.batch_size),
        "--eval-episodes", str(pipeline_args.eval_episodes),
    ]
    if pipeline_args.force_gpu:
        argv.append("--force-gpu")
    if pipeline_args.resume_ppo:
        argv.extend(["--resume", pipeline_args.resume_ppo])

    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the full training pipeline."""
    overall_start = time.time()

    # ------------------------------------------------------------------
    # Phase 1: PPO Training
    # ------------------------------------------------------------------
    phase_header("1", "PPO Training")
    update_status(is_training=True, current_phase="ppo", progress=0.0)

    # Import here to avoid loading SB3 at module level until needed
    from examples.train_ppo import train as train_ppo

    ppo_args = _build_ppo_args(args)
    train_ppo(ppo_args)

    # ------------------------------------------------------------------
    # Phase 2: Comprehensive ELO Evaluation
    # ------------------------------------------------------------------
    phase_header("2", "Comprehensive ELO Evaluation")
    update_status(is_training=True, current_phase="elo_eval", progress=0.5)

    from examples.eval_elo import main as eval_elo_main

    # eval_elo.main() parses sys.argv directly, so we save/restore
    old_argv = sys.argv
    try:
        elo_argv = [
            "eval_elo.py",
            "--incremental",
            "--episodes", str(args.eval_episodes),
        ]
        if args.skip_az:
            elo_argv.append("--skip-az")
        if args.use_rust:
            elo_argv.append("--python-core")  # False → use rust; inverted logic
        else:
            elo_argv.append("--python-core")  # Explicitly use Python engine
        sys.argv = elo_argv
        eval_elo_main()
    finally:
        sys.argv = old_argv

    # ------------------------------------------------------------------
    # Phase 3: AlphaZero Training (optional)
    # ------------------------------------------------------------------
    if args.with_az:
        phase_header("3", "AlphaZero Training")
        update_status(is_training=True, current_phase="alphazero", progress=0.7)

        from penguinchess.ai.train_alphazero import train_alphazero

        train_alphazero(
            num_iterations=args.az_iterations,
            games_per_iter=args.az_games,
            num_simulations=args.az_simulations,
            eval_simulations=args.az_eval_simulations,
            eval_interval=args.az_eval_interval,
            eval_games=args.az_eval_games,
            batch_size=args.az_batch_size,
            lr=args.az_lr,
            resume=args.resume_az,
            parallel_workers=args.az_parallel_workers,
        )
    else:
        print("  [SKIP] AlphaZero training (use --with-az to enable)")

    # ------------------------------------------------------------------
    # Pipeline Complete
    # ------------------------------------------------------------------
    total_elapsed = time.time() - overall_start
    update_status(is_training=True, current_phase="complete", progress=1.0)
    clear_status()

    stages = ["PPO → ELO"]
    if args.with_az:
        stages.append("AlphaZero")
    print(f"\n{'=' * 60}")
    print(f"  Pipeline Complete! Total time: {total_elapsed:.0f}s")
    print(f"  Stages: {' → '.join(stages)}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PenguinChess One-Click Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python examples/train_pipeline.py --ppo-steps 100000 --num-envs 4\n"
            "  uv run python examples/train_pipeline.py --ppo-steps 50000 --with-az --az-iterations 10\n"
            "  uv run python examples/train_pipeline.py --ppo-steps 200000 --eval-episodes 200 --num-envs 4\n"
        ),
    )

    # --- PPO options ---
    ppo_group = parser.add_argument_group("PPO Training")
    ppo_group.add_argument(
        "--ppo-steps", type=int, default=100_000,
        help="Number of PPO training timesteps (default: 100000)",
    )
    ppo_group.add_argument(
        "--num-envs", type=int, default=4,
        help="Number of parallel environments for PPO (default: 4)",
    )
    ppo_group.add_argument(
        "--lr", type=float, default=3e-4,
        help="PPO learning rate (default: 3e-4)",
    )
    ppo_group.add_argument(
        "--batch-size", type=int, default=64,
        help="PPO batch size (default: 64)",
    )
    ppo_group.add_argument(
        "--force-gpu", action="store_true",
        help="Force GPU for PPO training (default: CPU)",
    )
    ppo_group.add_argument(
        "--resume-ppo", type=str, default=None,
        help="Resume PPO training from a saved model path",
    )

    # --- ELO evaluation options ---
    elo_group = parser.add_argument_group("ELO Evaluation")
    elo_group.add_argument(
        "--eval-episodes", type=int, default=100,
        help="Number of evaluation episodes per model pair (default: 100)",
    )
    elo_group.add_argument(
        "--skip-az", action="store_true",
        help="Skip AlphaZero models during ELO evaluation",
    )
    elo_group.add_argument(
        "--use-rust", action="store_true",
        help="Use Rust game engine for ELO evaluation (default: Python)",
    )

    # --- AlphaZero options ---
    az_group = parser.add_argument_group("AlphaZero Training (optional)")
    az_group.add_argument(
        "--with-az", action="store_true",
        help="Enable AlphaZero training stage",
    )
    az_group.add_argument(
        "--az-iterations", type=int, default=30,
        help="Number of AlphaZero training iterations (default: 30)",
    )
    az_group.add_argument(
        "--az-games", type=int, default=200,
        help="Number of self-play games per AZ iteration (default: 200)",
    )
    az_group.add_argument(
        "--az-simulations", type=int, default=200,
        help="MCTS simulations per move during AZ training (default: 200)",
    )
    az_group.add_argument(
        "--az-eval-simulations", type=int, default=800,
        help="MCTS simulations per move during AZ evaluation (default: 800)",
    )
    az_group.add_argument(
        "--az-eval-interval", type=int, default=10,
        help="Evaluate vs best every N AZ iterations (default: 10)",
    )
    az_group.add_argument(
        "--az-eval-games", type=int, default=200,
        help="Number of evaluation games vs best (default: 200)",
    )
    az_group.add_argument(
        "--az-batch-size", type=int, default=256,
        help="AZ training batch size (default: 256)",
    )
    az_group.add_argument(
        "--az-lr", type=float, default=1e-3,
        help="AlphaZero learning rate (default: 1e-3)",
    )
    az_group.add_argument(
        "--az-parallel-workers", type=int, default=4,
        help="Root-parallel MCTS workers for AZ (default: 4)",
    )
    az_group.add_argument(
        "--resume-az", type=str, default=None,
        help="Resume AlphaZero from a saved .pth checkpoint",
    )

    parsed = parser.parse_args()
    run_pipeline(parsed)


if __name__ == "__main__":
    main()
