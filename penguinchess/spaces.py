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

OBS_FLAT_SIZE = N_HEX * N_FEATURES + N_PIECES * PIECE_FEATURES  # 60*3 + 6*4 = 204


# =============================================================================
# 观测空间
# =============================================================================

PenguinChessFlatObs: gym.spaces.Box = gym.spaces.Box(
    low=-1.0, high=1.0, shape=(OBS_FLAT_SIZE,), dtype=np.float32
)


# =============================================================================
# 动作空间
# =============================================================================

PenguinChessActionSpace: gym.spaces.Discrete = gym.spaces.Discrete(N_HEX)
