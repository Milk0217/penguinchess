"""PenguinChess — Gymnasium RL environment for the Penguin Chess game."""

from penguinchess.core import PenguinChessCore, TOTAL_VALUE, HEX_COUNT
from penguinchess.env import PenguinChessEnv
from penguinchess.spaces import PenguinChessFlatObs, PenguinChessActionSpace

import gymnasium as gym
gym.register("PenguinChess-v0", entry_point="penguinchess.env:PenguinChessEnv", kwargs={"use_rust": False})
gym.register("PenguinChessRust-v0", entry_point="penguinchess.env:PenguinChessEnv", kwargs={"use_rust": True})
from penguinchess.spaces import PenguinChessFlatObs, PenguinChessActionSpace
from penguinchess.reward import compute_reward, sparse_reward, dense_reward
from penguinchess.eval_utils import (
    Agent,
    RandomAgent,
    PPOAgent,
    AlphaZeroAgent,
    MCTSAgent,
    AlphaZeroMCTSAgent,
    compete,
    compute_elo,
)

__all__ = [
    "PenguinChessCore",
    "PenguinChessEnv",
    "PenguinChessFlatObs",
    "PenguinChessActionSpace",
    "compute_reward",
    "sparse_reward",
    "dense_reward",
    "TOTAL_VALUE",
    "HEX_COUNT",
    "Agent",
    "RandomAgent",
    "PPOAgent",
    "AlphaZeroAgent",
    "MCTSAgent",
    "AlphaZeroMCTSAgent",
    "compete",
    "compute_elo",
]
