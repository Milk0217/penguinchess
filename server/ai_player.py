"""
AI 对战模块 — 加载训练好的 PPO 模型并执行决策。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from stable_baselines3 import PPO

from penguinchess.core import PenguinChessCore

MODELS_DIR = Path(__file__).parent.parent / "models"


class AIPlayer:
    """AI 玩家，加载 PPO 模型并做出决策。"""

    def __init__(self, player_index: int = 1, model_path: Optional[str] = None):
        """
        Args:
            player_index: AI 控制的玩家索引（0=P1, 1=P2）
            model_path: 模型文件路径，为 None 时自动寻找最佳模型
        """
        self.player_index = player_index
        self._model: Optional[PPO] = None

        if model_path is None:
            # 自动选择最佳模型
            best_path = MODELS_DIR / "best" / "best_model.zip"
            if best_path.exists():
                model_path = str(best_path)
            else:
                # 回退到最新 checkpoint
                checkpoints = sorted(MODELS_DIR.glob("ppo_penguinchess_*.zip"))
                if checkpoints:
                    model_path = str(checkpoints[-1])

        if model_path and os.path.exists(model_path):
            self._model = PPO.load(model_path)
            print(f"[AI] Loaded model: {model_path}")
        else:
            print(f"[AI] No model found at {model_path}, AI disabled")

    def is_ready(self) -> bool:
        return self._model is not None

    def select_action(self, core: PenguinChessCore) -> Optional[int]:
        """
        根据当前游戏状态选择一个合法动作。

        Returns:
            action (hex index) 或 None（无合法动作）
        """
        if self._model is None:
            return None

        legal = core.get_legal_actions()
        if not legal:
            return None

        # 构建观测
        obs = core.get_observation()
        flat_obs = _encode_flat_obs(obs)

        # 模型推理
        action_arr, _ = self._model.predict(flat_obs, deterministic=True)
        action = int(action_arr.item()) if hasattr(action_arr, 'item') else int(action_arr)

        # 确保动作合法
        if action in legal:
            return action

        # 非法动作时回退到随机选择
        return int(np.random.choice(legal))


def _encode_flat_obs(obs: dict) -> np.ndarray:
    """将观测字典编码为扁平向量（与训练时一致）。"""
    # board: 60 hex × 3 features (q, r, value)
    board = obs["board"]
    # pieces: 6 pieces × 4 features (piece_id, q, r, s)
    pieces = obs["pieces"]

    board_flat = np.array(board, dtype=np.float32).flatten()
    pieces_flat = np.array(pieces, dtype=np.float32).flatten()
    return np.concatenate([board_flat, pieces_flat]).reshape(1, -1)
