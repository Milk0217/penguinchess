"""
Python ↔ Rust 游戏核心桥接模块。
通过子进程调用 Rust 编译的 game_engine_cli，使用 JSON 协议通信。
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_RUST_BINARY = None


def _get_binary() -> str:
    global _RUST_BINARY
    if _RUST_BINARY is not None:
        return _RUST_BINARY
    # 寻找 Rust CLI 二进制文件（从项目根目录）
    root = Path(__file__).parent.parent
    candidates = [
        root / "game_engine" / "target" / "debug" / "game_engine_cli.exe",
        root / "game_engine" / "target" / "release" / "game_engine_cli.exe",
        root / "game_engine" / "target" / "debug" / "game_engine_cli",
        root / "game_engine" / "target" / "release" / "game_engine_cli",
    ]
    for p in candidates:
        if p.exists():
            _RUST_BINARY = str(p)
            return _RUST_BINARY
    raise FileNotFoundError(
        f"Rust game_engine_cli not found. Build with: cd game_engine && cargo build"
    )


def _call_rust(cmd: dict) -> dict:
    """发送 JSON 命令到 Rust CLI 并接收响应。"""
    binary = _get_binary()
    proc = subprocess.run(
        [binary],
        input=json.dumps(cmd),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Rust CLI error: {proc.stderr}")
    return json.loads(proc.stdout.strip())


def new_game(seed: int = 42) -> dict:
    """创建新游戏，返回完整状态。"""
    result = _call_rust({"cmd": "new_game", "seed": seed})
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "unknown error"))
    return result["state"]


def step(state: dict, action: int) -> Tuple[dict, float, bool]:
    """执行一步动作，返回 (next_state, reward, terminated)。"""
    result = _call_rust({"cmd": "step", "state": state, "action": action})
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "unknown error"))
    return result["state"], result["reward"], result["terminated"]


def legal_actions(state: dict) -> List[int]:
    """获取合法动作列表。"""
    result = _call_rust({"cmd": "legal_actions", "state": state})
    if not result.get("ok"):
        raise RuntimeError(result.get("error", "unknown error"))
    return result["actions"]
