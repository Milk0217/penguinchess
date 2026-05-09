"""
RustCore — 基于 Rust FFI 的状态游戏模拟器，duck-type 兼容 PenguinChessCore。
状态保留在 Rust 内存中，避免 JSON 序列化全状态，步骤级操作快 2x。

用法 (eval_elo.py):
    core = RustCore(engine, seed=42)
    legal = core.get_legal_actions()
    _, reward, terminated, info = core.step(action)
    print(core.players_scores, core.current_player, core.phase)

MCTS 集成:
    state_json = core.to_json()  # 全状态 JSON（用于 MCTS 搜索）
    from penguinchess.rust_ffi import mcts_search_rust
    result = mcts_search_rust(state_json, ...)
"""

from __future__ import annotations

from typing import Optional
from penguinchess.rust_ffi import get_engine, RustStatefulGame, mcts_search_rust


class RustCore:
    """Stateful Rust game wrapper, duck-type compatible with PenguinChessCore."""

    def __init__(self, seed: int = 0, engine=None):
        if engine is None:
            engine = get_engine()
        self._engine = engine
        self._game: Optional[RustStatefulGame] = None
        self._current_player: int = 0
        self._phase: str = "placement"
        self._scores: tuple = (0, 0)

    def reset(self, seed: int = 0):
        if self._game is not None:
            self._game.free()
        self._game = RustStatefulGame(self._engine, seed)
        try:
            info = self._game.get_info()
            self._sync_state(info)
        except Exception:
            pass
        return self

    def get_legal_actions(self) -> list[int]:
        return self._game.get_legal_actions()

    def step(self, action: int) -> tuple:
        """
        执行动作。
        Returns: (obs, reward, terminated, info) — 与 PenguinChessCore.step() 兼容。
        """
        r = self._game.step(action)
        self._sync_state(r)
        legal = r.get("legal_actions", [])
        info = {
            "valid_actions": legal,
            "scores": self._scores,
            "current_player": self._current_player,
            "phase": self._phase,
        }
        return None, r.get("reward", 0), r.get("terminated", False), info

    def _sync_state(self, r: dict):
        """从 step 或 get_info 返回值同步缓存的状态。"""
        if "current_player" in r:
            self._current_player = r["current_player"]
        if "phase" in r:
            self._phase = r["phase"]
        if "scores" in r:
            self._scores = tuple(r["scores"])
        # get_info 的 terminated 是 bool，step 的是带在返回值里的
        if "legal_actions" in r:
            self._legal_cache = r["legal_actions"]

    @property
    def players_scores(self) -> tuple:
        if self._game is not None:
            try:
                return tuple(self._game.get_scores())
            except Exception:
                pass
        return self._scores

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def phase(self) -> str:
        return self._phase

    def get_observation(self) -> dict:
        """PPO Agent 使用的观测。返回 {"board": ..., "pieces": ..., "current_player": ..., "phase": ...}。"""
        return self._game.get_obs()

    def to_json(self) -> str:
        """序列化为 Rust GameState JSON（供 MCTS 搜索使用）。"""
        return self._game.to_json()

    @property
    def handle(self) -> int:
        """获取底层 Rust stateful game 的句柄，用于 handle-based MCTS。"""
        if self._game is None:
            raise RuntimeError("Game not initialized. Call reset() first.")
        return self._game.handle

    def close(self):
        if self._game is not None:
            self._game.free()
            self._game = None

    def __del__(self):
        self.close()
