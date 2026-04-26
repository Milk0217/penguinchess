"""
高性能 Python ↔ Rust FFI 桥接 — 使用 ctypes 直接调用 Rust cdylib。
替代 subprocess + JSON 的 rust_bridge.py，速度快 100-1000x。
"""
from __future__ import annotations

import json
import os
from ctypes import CDLL, POINTER, c_char, c_int32, c_int64, c_double, create_string_buffer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _find_dll() -> str:
    """定位已编译的 Rust cdylib 文件。"""
    root = Path(__file__).parent.parent / "game_engine"
    candidates = [
        root / "target" / "debug" / "game_engine.dll",
        root / "target" / "release" / "game_engine.dll",
        root / "target" / "debug" / "libgame_engine.so",
        root / "target" / "release" / "libgame_engine.so",
        root / "target" / "debug" / "libgame_engine.dylib",
        root / "target" / "release" / "libgame_engine.dylib",
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())
    raise FileNotFoundError(f"Rust DLL not found. Build with: cd game_engine && cargo build")


class RustEngine:
    """Rust 游戏引擎的 ctypes 封装。"""

    def __init__(self):
        dll_path = _find_dll()
        self._lib = CDLL(dll_path)

        # 配置函数签名
        self._lib.game_new.argtypes = [c_int64, POINTER(c_char), c_int32]
        self._lib.game_new.restype = c_int32

        self._lib.game_step.argtypes = [POINTER(c_char), c_int32, POINTER(c_char), c_int32]
        self._lib.game_step.restype = c_int32

        self._lib.game_evaluate.argtypes = [POINTER(c_char), POINTER(c_char), c_int32]
        self._lib.game_evaluate.restype = c_int32

        self._lib.api_version.restype = c_int32

        self._buf_size = 65536  # 64KB buffer
        self._buf_in = create_string_buffer(self._buf_size)
        self._buf_out = create_string_buffer(self._buf_size)

        # 验证 API 版本
        ver = self._lib.api_version()
        assert ver >= 1, f"API version mismatch: {ver}"

    def _call(self, func, data: str) -> dict:
        """调用 Rust 函数并返回 JSON 结果。"""
        self._buf_in.value = data.encode("utf-8")
        result = func(self._buf_in, self._buf_out, self._buf_size)
        if result != 0:
            raise RuntimeError(f"Rust call failed: {result}")
        raw = self._buf_out.value.decode("utf-8")
        return json.loads(raw)

    def new_game(self, seed: int = 42) -> dict:
        """创建新游戏，返回完整状态。"""
        self._buf_in.value = str(seed).encode("utf-8")
        result = self._lib.game_new(c_int64(seed), self._buf_out, self._buf_size)
        if result != 0:
            raise RuntimeError(f"Rust new_game failed: {result}")
        raw = self._buf_out.value.decode("utf-8")
        return json.loads(raw)

    def step(self, state_json: str, action: int) -> dict:
        """执行一步动作。"""
        self._buf_in.value = state_json.encode("utf-8")
        result = self._lib.game_step(self._buf_in, c_int32(action), self._buf_out, self._buf_size)
        if result != 0:
            raise RuntimeError(f"Rust step failed: {result}")
        raw = self._buf_out.value.decode("utf-8")
        return json.loads(raw)

    def evaluate(self, state_json: str) -> dict:
        """评估游戏状态。"""
        self._buf_in.value = state_json.encode("utf-8")
        result = self._lib.game_evaluate(self._buf_in, self._buf_out, self._buf_size)
        if result != 0:
            raise RuntimeError(f"Rust evaluate failed: {result}")
        raw = self._buf_out.value.decode("utf-8")
        return json.loads(raw)


# 单例
_engine: Optional[RustEngine] = None


def get_engine() -> RustEngine:
    global _engine
    if _engine is None:
        _engine = RustEngine()
    return _engine
