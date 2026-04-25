"""
日志工具模块 - 为 PenguinChess 服务提供简洁的结构化日志输出
"""
from __future__ import annotations

import logging
import sys
import time
from typing import Optional

# 配置根 logger
_logger = logging.getLogger("penguinchess")
_logger.setLevel(logging.INFO)

if not _logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    _logger.addHandler(handler)


def info(msg: str):
    _logger.info(msg)


def warning(msg: str):
    _logger.warning(msg)


def error(msg: str):
    _logger.error(msg)


class GameLogger:
    """游戏日志记录器"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.step = 0

    def _p(self):
        return f"[{self.session_id[:8]}]"

    def phase(self, phase: str, seed: Optional[int] = None):
        info(f"{self._p()} {phase.upper()} (seed={seed})")

    def action(self, action_type: str, player: int, data: dict):
        """记录动作
        data: {step, piece_id, from, to, hex_idx, hex_value, score, pieces_remaining}
        """
        step = data.get("step", "?")
        pieces = data.get("pieces_remaining", [3, 3])

        if action_type == "PLACEMENT":
            to = data.get("to", (0, 0, 0))
            info(f"{self._p()} #{step:03d} | P{player+1} placed | hex={data['hex_idx']} ({to[0]:+d},{to[1]:+d},{to[2]:+d}) | val={data['hex_value']} | +{data['score']:.3f} | P1:{pieces[0]}/3 P2:{pieces[1]}/3")
        else:
            from_c = data.get("from", (0, 0, 0))
            to = data.get("to", (0, 0, 0))
            info(f"{self._p()} #{step:03d} | P{player+1} moved | piece={data['piece_id']} | ({from_c[0]:+d},{from_c[1]:+d},{from_c[2]:+d}) -> ({to[0]:+d},{to[1]:+d},{to[2]:+d}) | hex={data['hex_idx']} | +{data['score']:.3f} | P1:{pieces[0]}/3 P2:{pieces[1]}/3")

    def death(self, piece_id: int, player: int, coord: tuple, reason: str):
        warning(f"{self._p()} | P{player+1} piece DEAD | piece={piece_id} | ({coord[0]:+d},{coord[1]:+d},{coord[2]:+d}) | reason={reason}")

    def piece_moves(self, player: int, piece_id: int, coord: tuple, move_count: int):
        """记录棋子合法移动数"""
        info(f"{self._p()} | P{player+1} piece={piece_id} @ ({coord[0]:+d},{coord[1]:+d},{coord[2]:+d}) | moves={move_count}")

    def elimination(self, count: int, hexes: Optional[list] = None):
        info(f"{self._p()} | Eliminated {count} hexes (disconnected)")

    def game_over(self, winner: Optional[int], scores: tuple):
        if winner == 2 or winner is None:
            result = "DRAW"
        else:
            result = f"P{winner+1} WINS"
        info(f"{self._p()} | GAME OVER | {result} | Scores: P1={scores[0]}, P2={scores[1]}")

    def invalid(self, player: int, action: int, reason: str):
        warning(f"{self._p()} | P{player+1} INVALID action={action} | reason={reason}")


# 请求日志
_request_start_time = {}

def setup_request_logging(app):
    """为 Flask app 设置请求日志中间件"""
    from flask import request

    @app.before_request
    def before_request():
        _request_start_time["time"] = time.time()

    @app.after_request
    def after_request(response):
        start_time = _request_start_time.pop("time", None)
        if start_time is not None:
            duration_ms = (time.time() - start_time) * 1000
            status = response.status_code
            if status >= 500:
                level = logging.ERROR
            elif status >= 400:
                level = logging.WARNING
            else:
                level = logging.INFO
            _logger.log(level, f"{request.method} {request.path} | {status} | {duration_ms:.1f}ms")
        return response
