"""
AI 对战模块 — 支持 PPO (Stable-Baselines3) 和 AlphaZero 两种模型。
自动加载最强的可用模型。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from penguinchess.core import PenguinChessCore

MODELS_DIR = Path(__file__).parent.parent / "models"


def _find_best_model() -> Optional[tuple[str, str]]:
    """
    在所有可用模型中找最强的。
    优先使用 Model Registry 中的 ELO 评分，其次用 vs_random 胜率。
    均无数据时回退到按 generation/iteration 编号取最大。

    Returns:
        (model_path, model_type) where model_type is "ppo" or "alphazero"
    """
    # 1) 尝试通过 Model Registry 按 ELO 选择
    try:
        from penguinchess.model_registry import get_best_model
        result = get_best_model(criteria="elo")
        if result:
            path, mtype = result
            full_path = str(MODELS_DIR / path)
            if os.path.exists(full_path):
                return (full_path, mtype)
    except Exception:
        pass

    # 2) 回退：按 generation/iteration 编号最大（兼容旧逻辑）
    best = None
    best_priority = -1

    # PPO models: ppo_penguinchess_gen_N.zip
    for p in sorted(MODELS_DIR.glob("ppo_penguinchess_gen_*.zip")):
        try:
            n = int(p.stem.split("_gen_")[1])
            if n > best_priority:
                best = (str(p), "ppo")
                best_priority = n
        except (IndexError, ValueError):
            pass

    # AlphaZero models: alphazero/alphazero_iter_N.pth
    for p in sorted((MODELS_DIR / "alphazero").glob("alphazero_iter_*.pth")):
        try:
            n = int(p.stem.split("_iter_")[1])
            if n > best_priority:
                best = (str(p), "alphazero")
                best_priority = n
        except (IndexError, ValueError):
            pass

    # Fallback to best directory
    if best is None:
        best_zip = MODELS_DIR / "best" / "best_model.zip"
        if best_zip.exists():
            best = (str(best_zip), "ppo")

    return best


class AIPlayer:
    """AI 玩家，支持 PPO 和 AlphaZero 两种模型。"""

    def __init__(self, player_index: int = 1, model_path: Optional[str] = None):
        self.player_index = player_index
        self._model = None
        self._model_type: Optional[str] = None

        if model_path is None:
            found = _find_best_model()
            if found:
                model_path, self._model_type = found
                print(f"[AI] Auto-selected: {Path(model_path).name} ({self._model_type})")
        else:
            self._model_type = "alphazero" if model_path.endswith(".pth") else "ppo"

        if model_path and os.path.exists(model_path):
            self._load(model_path)
        else:
            print(f"[AI] No model found, AI disabled")

    def _load(self, path: str):
        """加载模型（PPO 用 SB3，AlphaZero 用 PyTorch）。"""
        if self._model_type == "ppo":
            try:
                from stable_baselines3 import PPO
                self._model = PPO.load(path)
                print(f"[AI] Loaded PPO model: {path}")
            except Exception as e:
                print(f"[AI] Failed to load PPO model: {e}")
                self._model = None
        elif self._model_type == "alphazero":
            try:
                import torch
                from penguinchess.ai.alphazero_net import AlphaZeroNet
                net = AlphaZeroNet()
                state = torch.load(path, map_location="cpu", weights_only=True)
                net.load_state_dict(state)
                net.eval()
                self._model = net
                print(f"[AI] Loaded AlphaZero model: {path}")
            except Exception as e:
                print(f"[AI] Failed to load AlphaZero model: {e}")
                self._model = None

    def is_ready(self) -> bool:
        return self._model is not None

    def select_action(self, core: PenguinChessCore) -> Optional[int]:
        if self._model is None:
            return None

        legal = core.get_legal_actions()
        if not legal:
            return None

        if self._model_type == "ppo":
            return self._select_action_ppo(core, legal)
        else:
            return self._select_action_az(core, legal)

    def _select_action_ppo(self, core: PenguinChessCore, legal: list) -> Optional[int]:
        """PPO 推理：直接 predict 选动作。"""
        obs = core.get_observation()
        flat = _encode_flat_obs(obs)
        action_arr, _ = self._model.predict(flat, deterministic=True)
        action = int(action_arr.item()) if hasattr(action_arr, 'item') else int(action_arr)
        return action if action in legal else int(np.random.choice(legal))

    def _select_action_az(self, core: PenguinChessCore, legal: list) -> Optional[int]:
        """AlphaZero 推理：用策略网络选概率最高的合法动作。"""
        probs, value = self._model.evaluate(core)
        action = int(legal[np.argmax(probs[legal])])
        return action


def _encode_flat_obs(obs: dict) -> np.ndarray:
    """将观测字典编码为扁平向量。"""
    board = np.array(obs["board"], dtype=np.float32).flatten()
    pieces = np.array(obs["pieces"], dtype=np.float32).flatten()
    meta = np.array([float(obs["current_player"]), float(obs["phase"])], dtype=np.float32)
    return np.concatenate([board, pieces, meta]).reshape(1, -1)
