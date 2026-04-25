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

class PenguinChessActionSpace(gym.Space):
    """
    企鹅棋动作空间。
    Discrete(60)，每个动作 ID 对应 hexes 数组中的一个格子索引。

    放置阶段: 选一个空格子放置己方棋子
    移动阶段: 选一个己方棋子所在格子（环境自动执行移动）

    采样无效动作后智能体需要自行过滤，推荐使用 env.info["valid_actions"]。
    """

    def __init__(self, n: int = N_HEX):
        self.n = n
        super().__init__(shape=(), dtype=np.int64)

    def sample(self, mask: np.ndarray = None) -> int:
        if mask is None:
            # numpy.random.Generator uses integers(), older numpy uses randint()
            if hasattr(self.np_random, 'integers'):
                return self.np_random.integers(self.n)
            else:
                return self.np_random.randint(self.n)
        valid = np.where(mask == 1)[0]
        if len(valid) == 0:
            raise ValueError("mask 无效，所有动作均被禁用")
        return int(self.np_random.choice(valid))

    def contains(self, x) -> bool:
        return 0 <= x < self.n

    def to_jsonable(self, x) -> int:
        return int(x)

    def from_jsonable(self, x) -> int:
        return int(x)

    def __repr__(self) -> str:
        return f"PenguinChessActionSpace({self.n})"
