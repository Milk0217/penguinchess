"""
Gymnasium 环境注册与包装。
严格遵循 Gymnasium 0.29+ API。
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np

from penguinchess.core import PenguinChessCore
from penguinchess.spaces import (
    N_HEX,
    OBS_FLAT_SIZE,
    PenguinChessActionSpace,
    PenguinChessFlatObs,
)


# =============================================================================
# Gymnasium 环境注册
# =============================================================================

gym.register(
    id="PenguinChess-v0",
    entry_point="penguinchess.env:PenguinChessEnv",
)


# =============================================================================
# Gymnasium Env 实现
# =============================================================================

class PenguinChessEnv(gym.Env):
    """
    企鹅棋 Gymnasium 强化学习环境。

    观测空间: Box(shape=(204,), low=-1.0, high=1.0, dtype=np.float32)
        前 180 维: 棋盘（60格 × 3特征: q/8, r/8, value/3）
        后 24 维: 棋子（6个 × 4特征: id/10, q/8, r/8, s/8），已移除则 id=-1

    动作空间: Discrete(60)
        动作 ID = hexes 数组索引
        放置阶段: 选择一个空格子放置己方棋子
        移动阶段: 选择己方棋子所在格子（环境自动执行移动）

    Reward 设计（稀疏版）:
        - 放置/移动得分: +value/99
        - 游戏结束: 胜=+1, 负=-1, 平=0

    最大步数: 500
    """

    metadata = {"render_modes": ["human", "text"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, reward_type: str = "sparse"):
        super().__init__()

        self.render_mode = render_mode
        self.reward_type = reward_type

        # 观测/动作空间
        self.observation_space: gym.Space = PenguinChessFlatObs
        self.action_space: gym.Space = PenguinChessActionSpace(n=N_HEX)

        # 核心游戏逻辑
        self._game: PenguinChessCore = PenguinChessCore()

        # 内部状态
        self._obs: Optional[np.ndarray] = None
        self._winner: Optional[int] = None
        self._elapsed_steps: int = 0

        # 最大步数（超过则 truncation）
        self._max_episode_steps: int = 500

    # -------------------------------------------------------------------------
    # Gymnasium 必需方法
    # -------------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """重置环境，返回初始观测和 info dict。"""
        super().reset(seed=seed)

        self._game.reset(seed=seed)
        self._obs = self._make_obs()
        self._winner = None
        self._elapsed_steps = 0

        info = self._make_info()
        return self._obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行动作，返回 (obs, reward, terminated, truncated, info)。

        注意: 每次 step() 执行"一个完整回合"，即：
        当前玩家动作 → 状态更新 → 切换玩家（不执行对方动作）。
        这意味着一个完整 episode 包含两个玩家的多个 step。
        """
        self._elapsed_steps += 1

        # 检查 truncation（最大步数）
        truncated = self._elapsed_steps >= self._max_episode_steps

        # 检查动作合法性
        if action not in self._game.get_legal_actions():
            # 无效动作
            info = self._make_info()
            info["invalid_action"] = True
            return self._obs, -0.5, False, truncated, info

        # 执行动作
        _, reward, terminated, step_info = self._game.step(action)

        # 更新观测
        self._obs = self._make_obs()

        # 判断胜负（游戏结束时）
        if self._game._terminated:
            self._winner = self._get_winner()
            terminated = True

        # 构建 info
        info = self._make_info()
        info.update(step_info)

        return self._obs, reward, terminated, truncated, info

    def close(self) -> None:
        """清理环境资源。"""
        pass

    def render(self) -> Optional[str]:
        """渲染棋盘（文本模式）。"""
        if self.render_mode == "text" or self.render_mode == "human":
            return self._game.render()
        return None

    # -------------------------------------------------------------------------
    # 内部方法
    # -------------------------------------------------------------------------

    def _make_obs(self) -> np.ndarray:
        """将 game 观测转换为扁平 numpy 数组。"""
        obs = self._game.get_observation()

        # 棋盘: 60*3=180
        board = np.array(obs["board"], dtype=np.float32).flatten()  # (180,)

        # 棋子: 6*4=24
        pieces = np.array(obs["pieces"], dtype=np.float32).flatten()  # (24,)

        # 拼接
        flat = np.concatenate([board, pieces]).astype(np.float32)  # (204,)
        assert flat.shape == (OBS_FLAT_SIZE,), f"观测形状错误: {flat.shape}"
        return flat

    def _make_info(self) -> dict:
        """构建 info dict。"""
        legal = self._game.get_legal_actions()

        # 构造动作掩码（Discrete 空间用）
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        for a in legal:
            mask[a] = 1

        return {
            "valid_actions": legal,
            "action_mask": mask,
            "current_player": self._game.current_player,
            "phase": self._game.phase,
            "scores": list(self._game.players_scores),
            "pieces_remaining": [
                self._game._count_alive_pieces(0),
                self._game._count_alive_pieces(1),
            ],
            "episode_steps": self._elapsed_steps,
            "winner": self._winner,
        }

    def _get_winner(self) -> int | None:
        """判断当前局势的胜者。"""
        if not self._game._terminated:
            return None

        p1_score = self._game.players_scores[0]
        p2_score = self._game.players_scores[1]

        if p1_score > p2_score:
            return 0
        elif p2_score > p1_score:
            return 1
        else:
            return 2  # 平局
