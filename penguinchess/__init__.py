"""PenguinChess — Gymnasium RL environment for the Penguin Chess game."""

from penguinchess.core import PenguinChessCore, TOTAL_VALUE, HEX_COUNT
from penguinchess.env import PenguinChessEnv
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
