"""
PenguinChess Gymnasium RL environment with optional Rust game engine.
"""

from __future__ import annotations
from typing import Optional, Tuple, Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from penguinchess.spaces import PenguinChessFlatObs, PenguinChessActionSpace, OBS_FLAT_SIZE


class PenguinChessEnv(gym.Env):
    """
    企鹅棋 Gymnasium RL 环境。

    观测空间: Box(shape=(206,), low=-1.0, high=1.0, dtype=np.float32)
    动作空间: Discrete(60)

    Reward:
        - 放置/移动得分: +value/99 (大约 0.01–0.03)
        - 游戏结束: 胜=+1, 负=-1, 平=0
    """

    metadata = {"render_modes": ["human", "text"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None, use_rust: bool = False):
        super().__init__()
        self.render_mode = render_mode
        self.use_rust = use_rust

        self.observation_space: gym.Space = PenguinChessFlatObs
        self.action_space: gym.Space = PenguinChessActionSpace

        if use_rust:
            from penguinchess.rust_core import RustCore
            from penguinchess.rust_ffi import get_engine
            self._game = RustCore(engine=get_engine())
        else:
            from penguinchess.core import PenguinChessCore as GameCore
            self._game = GameCore()

        self._obs: Optional[np.ndarray] = None
        self._winner: Optional[int] = None
        self._elapsed_steps: int = 0
        self._max_episode_steps: int = 500
        self._terminated: bool = False

    @property
    def core(self):
        """暴露游戏核心，供需要直接访问游戏状态的 agent 使用。"""
        return self._game

    # ------------------------------------------------------------------
    # Gymnasium required
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._game.reset(seed=seed)
        self._obs = self._make_obs()
        self._winner = None
        self._elapsed_steps = 0
        self._terminated = False
        info = self._make_info()
        return self._obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self._max_episode_steps

        # Check legal action
        legal = self._game.get_legal_actions()
        if action not in legal:
            info = self._make_info()
            info["invalid_action"] = True
            return self._obs, -0.5, False, truncated, info

        # Execute
        _, reward, terminated, step_info = self._game.step(action)
        self._obs = self._make_obs()

        # Terminal reward
        if terminated:
            self._terminated = True
            self._winner = self._get_winner()
            reward = 1.0 if self._winner == 0 else (-1.0 if self._winner == 1 else 0.0)

        info = self._make_info()
        info.update(step_info)
        return self._obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self.use_rust and hasattr(self._game, 'close'):
            self._game.close()

    def render(self) -> Optional[str]:
        if self.render_mode in ("text", "human"):
            if self.use_rust:
                return self._game.to_json()
            return self._game.render()
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_obs(self) -> np.ndarray:
        obs = self._game.get_observation()
        board = np.array(obs["board"], dtype=np.float32).flatten()
        pieces = np.array(obs["pieces"], dtype=np.float32).flatten()
        meta = np.array([float(obs["current_player"]), float(obs["phase"])], dtype=np.float32)
        flat = np.concatenate([board, pieces, meta]).astype(np.float32)
        assert flat.shape == (OBS_FLAT_SIZE,), f"obs shape: {flat.shape}"
        return flat

    def _make_info(self) -> dict:
        legal = self._game.get_legal_actions()
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        for a in legal:
            mask[a] = 1

        obs = self._game.get_observation()
        pieces_arr = np.array(obs["pieces"], dtype=np.float32)
        p1_alive = int(np.sum(pieces_arr[:3, 0] >= 0))
        p2_alive = int(np.sum(pieces_arr[3:, 0] >= 0))

        return {
            "valid_actions": legal,
            "action_mask": mask,
            "current_player": self._game.current_player,
            "phase": self._game.phase,
            "scores": list(self._game.players_scores),
            "pieces_remaining": [p1_alive, p2_alive],
            "episode_steps": self._elapsed_steps,
            "winner": self._winner,
        }

    def _get_winner(self) -> int | None:
        p1, p2 = self._game.players_scores
        if p1 > p2: return 0
        if p2 > p1: return 1
        return 2
