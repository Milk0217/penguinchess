"""
观测空间与动作空间定义。
"""

import gymnasium as gym
import numpy as np


# =============================================================================
# 棋盘格子数量（必须与 core.py HEX_COUNT 严格一致）
# =============================================================================

N_HEX = 60          # 格子总数
N_PIECES = 6        # 棋子总数
N_FEATURES = 3      # 每格特征数: q, r, value
PIECE_FEATURES = 4  # 每棋子特征数: piece_id, q, r, s

N_PLAYER_FEATURES = 2   # current_player(0/1), phase(0/1)

OBS_FLAT_SIZE = N_HEX * N_FEATURES + N_PIECES * PIECE_FEATURES + N_PLAYER_FEATURES  # 60*3 + 6*4 + 2 = 206


# =============================================================================
# 观测空间
# =============================================================================

PenguinChessFlatObs: gym.spaces.Box = gym.spaces.Box(
    low=-1.0, high=1.0, shape=(206,), dtype=np.float32
)


# =============================================================================
# 动作空间
# =============================================================================

PenguinChessActionSpace: gym.spaces.Discrete = gym.spaces.Discrete(N_HEX)
