"""
Flask 路由层：HTTP API 包装 PenguinChessCore。
前后端分离架构，所有游戏逻辑在后端执行，前端仅负责渲染和交互。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from penguinchess.core import PenguinChessCore, create_board_from_coords, generate_sequence, json_board_to_coords
from .logger import GameLogger


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
            # JSON 棋盘使用原始坐标，需要转换
            custom_coords = json_board_to_coords(board_data.get("hexes", []))

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
    _logger: GameLogger = field(init=False)

    def __post_init__(self):
        self._core = PenguinChessCore(seed=self.seed, custom_coords=self.custom_coords)
        self._core.reset(seed=self.seed)
        self._logger = GameLogger(self.session_id)
        self._logger.phase("placement", self.seed)

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
            self._logger.invalid(self._core.current_player, action, "game already over")
            return {
                "state": self.state(),
                "reward": 0.0,
                "invalid": True,
                "error": "game already over",
            }

        # 保存动作前的状态（用于生成历史记录和检测棋子死亡）
        prev_player = self._core.current_player
        prev_pieces_alive = {p.id: p.alive for p in self._core.pieces}
        prev_phase = self._core.phase

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
                "state": hex_obj.state,
                "points": hex_obj.points,
            },
            "phase_before": prev_phase,
        }

        # 获取当前棋子存活数
        p1_alive = sum(1 for p in self._core.pieces if p.alive and p.id % 2 == 0)
        p2_alive = sum(1 for p in self._core.pieces if p.alive and p.id % 2 == 1)

        # 记录日志
        if info.get("invalid"):
            self._logger.invalid(prev_player, action, info.get("reason", "unknown"))
        elif prev_phase == "placement":
            self._logger.action("PLACEMENT", prev_player, {
                "step": self._core._episode_steps,
                "to": (hex_obj.q, hex_obj.r, hex_obj.s),
                "hex_idx": action,
                "hex_value": hex_obj.points,
                "score": reward,
                "pieces_remaining": [p1_alive, p2_alive],
            })
        else:
            from_c = (
                self._last_action["hex"]["q"],
                self._last_action["hex"]["r"],
                self._last_action["hex"]["s"]
            )
            self._logger.action("MOVEMENT", prev_player, {
                "step": self._core._episode_steps,
                "piece_id": piece_id or 0,
                "from": from_c,
                "to": (hex_obj.q, hex_obj.r, hex_obj.s),
                "hex_idx": action,
                "hex_value": hex_obj.points,
                "score": reward,
                "pieces_remaining": [p1_alive, p2_alive],
            })

        # 检测棋子死亡
        for p in self._core.pieces:
            if prev_pieces_alive.get(p.id) and not p.alive:
                coord = (p.hex.q if p.hex else 0, p.hex.r if p.hex else 0, p.hex.s if p.hex else 0) if p.hex else (0, 0, 0)
                reason = "eliminated" if p.hex is None else "no valid moves"
                self._logger.death(p.id, p.id % 2, coord, reason)

        # 检测阶段变化
        if prev_phase == "placement" and self._core.phase == "movement":
            self._logger.phase("movement")

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
            self._logger.game_over(self._winner, (s1, s2))

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
                "state": h.state,    # 'active' | 'occupied' | 'used' | 'eliminated'
                "points": h.points,  # 1/2/3 分值（仅 active 时有效）
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
