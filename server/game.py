"""
Flask 路由层：HTTP API 包装 PenguinChessCore。
前后端分离架构，所有游戏逻辑在后端执行，前端仅负责渲染和交互。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from penguinchess.core import PenguinChessCore, create_board_from_coords, generate_sequence


# =============================================================================
# 内存中的游戏会话存储（生产环境应换用 Redis/DB）
# =============================================================================

_sessions: Dict[str, "GameSession"] = {}
_MAX_SESSIONS = 20  # 最多保留 20 个会话


def get_session(session_id: str) -> Optional["GameSession"]:
    return _sessions.get(session_id)


def create_session(seed: Optional[int] = None, board_id: Optional[str] = None) -> "GameSession":
    """
    创建新游戏会话。

    Args:
        seed: 随机种子
        board_id: 棋盘 ID。如果为 None，使用默认棋盘。
    """
    # 清理旧会话（保留最近的）
    if len(_sessions) >= _MAX_SESSIONS:
        oldest_keys = list(_sessions.keys())[:len(_sessions) - _MAX_SESSIONS + 1]
        for k in oldest_keys:
            del _sessions[k]

    session_id = str(uuid.uuid4())[:8]

    # 获取棋盘配置
    custom_coords = None
    if board_id:
        from . import boards as board_module
        board_data = board_module.get_board(board_id)
        if board_data:
            custom_coords = board_data.get("hexes")

    session = GameSession(session_id=session_id, seed=seed, custom_coords=custom_coords)
    _sessions[session_id] = session
    return session


# =============================================================================
# 游戏会话包装
# =============================================================================

@dataclass
class GameSession:
    """
    封装 PenguinChessCore，提供：
    1. 完整的观测字典（前端渲染所需全部数据）
    2. 动作验证与执行
    3. 会话生命周期管理
    """

    session_id: str
    seed: Optional[int]
    custom_coords: Optional[List[dict]] = None

    _core: PenguinChessCore = field(init=False)
    _game_over: bool = field(default=False, init=False)
    _winner: Optional[int] = field(default=None, init=False)
    _last_action: Optional[dict] = field(default=None, init=False)

    def __post_init__(self):
        self._core = PenguinChessCore(seed=self.seed, custom_coords=self.custom_coords)
        self._core.reset(seed=self.seed)

    # -------------------------------------------------------------------------
    # 游戏控制
    # -------------------------------------------------------------------------

    def reset(self) -> dict:
        """重置游戏到初始状态。"""
        self._core.reset(seed=self.seed)
        self._game_over = False
        self._winner = None
        self._last_action = None
        return self.state()

    def step(self, action: int, piece_id: Optional[int] = None) -> dict:
        """
        执行一个动作（放置或移动）。
        action: hexes 数组索引（0~59）
        piece_id: 可选参数，指定要移动的棋子 ID（仅移动阶段有效）

        Returns: {
            "state": <current state dict>,
            "reward": <reward for acting player>,
            "invalid": <bool>,
        }
        """
        if self._game_over:
            return {
                "state": self.state(),
                "reward": 0.0,
                "invalid": True,
                "error": "game already over",
            }

        # 保存动作前的状态（用于生成历史记录）
        prev_player = self._core.current_player

        # 执行动作
        obs, reward, terminated, info = self._core.step(action, piece_id=piece_id)

        # 构建动作历史
        hex_obj = self._core._action_id_to_hex(action)
        self._last_action = {
            "player": prev_player,
            "action": action,
            "piece_id": piece_id,
            "hex": {
                "q": hex_obj.q,
                "r": hex_obj.r,
                "s": hex_obj.s,
                "value": getattr(hex_obj, "value", 0),
            },
            "phase_before": self._core.phase,
        }

        # 检查游戏结束
        if terminated:
            self._game_over = True
            s1, s2 = self._core.players_scores
            if s1 > s2:
                self._winner = 0
            elif s2 > s1:
                self._winner = 1
            else:
                self._winner = 2  # 平局

        return {
            "state": self.state(),
            "reward": reward,
            "invalid": info.get("invalid", False),
        }

    # -------------------------------------------------------------------------
    # 状态序列化（前端渲染所需完整数据）
    # -------------------------------------------------------------------------

    def state(self) -> dict:
        """
        返回前端渲染所需的完整游戏状态。
        所有计算在后端完成，前端只负责显示。
        """
        hexes = []
        for i, h in enumerate(self._core.hexes):
            hexes.append({
                "index": i,          # 动作 ID
                "q": h.q,
                "r": h.r,
                "s": h.s,
                "value": h.value,    # 1/2/3=活跃, 0=被占据, -1=已消除
            })

        pieces = []
        for p in self._core.pieces:
            pieces.append({
                "id": p.id,
                "owner": p.id % 2,  # 0=Player1, 1=Player2
                "q": p.hex.q if p.hex else None,
                "r": p.hex.r if p.hex else None,
                "s": p.hex.s if p.hex else None,
                "index": self._core._hex_to_index(p.hex) if p.hex else None,
                "alive": p.alive,
            })

        legal = self._core.get_legal_actions()

        return {
            "session_id": self.session_id,
            "hexes": hexes,
            "pieces": pieces,
            "current_player": self._core.current_player,  # 0 或 1
            "phase": self._core.phase,                    # "placement" | "movement"
            "scores": list(self._core.players_scores),   # [p1, p2]
            "legal_actions": legal,                        # [hex_index, ...]
            "game_over": self._game_over,
            "winner": self._winner,                       # None/0/1/2
            "last_action": self._last_action,
            "episode_steps": self._core._episode_steps,
        }
