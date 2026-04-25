"""
棋盘存储管理模块
负责保存、加载、删除用户自定义棋盘
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .logger import info as log_info

# 存储目录
BOARD_DIR = Path(__file__).parent.parent / "backend_data" / "boards"


def _get_boards_dir() -> Path:
    """获取或创建棋盘存储目录"""
    BOARD_DIR.mkdir(parents=True, exist_ok=True)
    return BOARD_DIR


def list_boards() -> List[Dict]:
    """
    返回所有已保存棋盘的元数据列表（不含 hexes 数据）。
    """
    boards = []
    for f in _get_boards_dir().glob("*.json"):
        with open(f, encoding="utf-8") as fp:
            data = json.load(fp)
            boards.append({
                "id": data["id"],
                "name": data["name"],
                "hex_count": data["hex_count"],
                "created_at": data.get("created_at", ""),
            })
    # 按创建时间倒序
    boards.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return boards


def get_board(board_id: str) -> Optional[Dict]:
    """
    根据 ID 获取完整棋盘数据（包括 hexes）。
    """
    path = _get_boards_dir() / f"{board_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def save_board(board_id: str, name: str, hexes: List[Dict]) -> Dict:
    """
    保存棋盘数据到 JSON 文件。

    Returns:
        棋盘元数据 {"id", "name", "hex_count"}
    """
    data = {
        "id": board_id,
        "name": name,
        "hex_count": len(hexes),
        "created_at": datetime.now().isoformat(),
        "hexes": hexes,
    }
    path = _get_boards_dir() / f"{board_id}.json"
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    return {"id": board_id, "name": name, "hex_count": len(hexes)}


def delete_board(board_id: str) -> bool:
    """
    删除指定 ID 的棋盘文件。
    内置棋盘（default, parallelogram, hexagon）不允许删除。
    """
    # 内置棋盘不允许删除
    if board_id in ("default", "parallelogram", "hexagon"):
        return False

    path = _get_boards_dir() / f"{board_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False


def board_exists(board_id: str) -> bool:
    """检查棋盘是否存在"""
    path = _get_boards_dir() / f"{board_id}.json"
    return path.exists()


def init_builtin_boards():
    """
    初始化内置棋盘（如果不存在）。
    """
    # 平行四边形 60 格
    parallelogram_hexes = []
    for q in range(-4, 4):
        r_min = -4 if q % 2 == 0 else -3
        for r in range(r_min, 4):
            s = -q - r
            parallelogram_hexes.append({"q": q, "r": r, "s": s})

    # 正六边形 61 格 (radius=4)
    hexagon_hexes = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            s = -q - r
            if abs(s) <= 4:
                hexagon_hexes.append({"q": q, "r": r, "s": s})

    builtins = {
        "default": {
            "name": "默认棋盘",
            "hexes": parallelogram_hexes,
        },
        "parallelogram": {
            "name": "平行四边形",
            "hexes": parallelogram_hexes,
        },
        "hexagon": {
            "name": "正六边形",
            "hexes": hexagon_hexes,
        },
    }

    for board_id, data in builtins.items():
        path = _get_boards_dir() / f"{board_id}.json"
        if not path.exists():
            save_board(board_id, data["name"], data["hexes"])
            log_info(f"Created builtin board: {board_id}")


# 启动时初始化内置棋盘
init_builtin_boards()
