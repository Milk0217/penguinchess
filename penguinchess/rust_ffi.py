"""
高性能 Python ↔ Rust FFI 桥接 — 使用 ctypes 直接调用 Rust cdylib。
替代 subprocess + JSON 的 rust_bridge.py，速度快 100-1000x。
"""
from __future__ import annotations

import json
import os
from ctypes import CDLL, POINTER, c_char, c_int32, c_int64, c_double, create_string_buffer, CFUNCTYPE, c_void_p, memmove
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

        self._lib.mcts_search_rust.argtypes = [
            POINTER(c_char), c_int32, c_double, c_int32,
            CFUNCTYPE(c_int32, POINTER(c_char), POINTER(c_char), c_int32),
            POINTER(c_char), c_int32,
        ]
        self._lib.mcts_search_rust.restype = c_int32

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


# =============================================================================
# Rust MCTS 搜索（通过 DLL + 回调调用神经网络）
# =============================================================================

EVAL_CB = CFUNCTYPE(c_int32, c_char_p, c_char_p, c_int32)


def mcts_search_rust(
    state_json: str,
    model: Any = None,
    num_simulations: int = 800,
    c_puct: float = 1.4,
    batch_size: int = 32,
) -> dict:
    """
    使用 Rust MCTS 进行搜索，通过回调调用 PyTorch 神经网络批量推理。

    Args:
        state_json: Rust GameState 的 JSON 字符串
        model: AlphaZeroNet 实例（需要 evaluate_batch 方法）
        num_simulations: MCTS 模拟次数
        c_puct: 探索常数
        batch_size: 批量推理大小

    Returns:
        动作计数 dict: {action: visit_count}
    """
    import numpy as np
    import torch
    import ctypes

    engine = get_engine()
    net = model

    # 准备回调用缓冲区
    cb_result_buf = create_string_buffer(65536)

    @EVAL_CB
    def evaluate_cb(states_json_ptr, res_buf, buf_size):
        """Rust MCTS 调用的回调：批量评估状态。"""
        nonlocal net
        try:
            states_json_str = ctypes.cast(states_json_ptr, c_char_p).value.decode("utf-8")
            states_data = json.loads(states_json_str)

            # 构建批量观测（直接解析 JSON，无需 PenguinChessCore）
            obs_list = []
            for sd in states_data:
                cells = sd["board"]["cells"]
                # board: 60 cells × [q/8, r/8, value/3]
                board_flat = []
                for c in cells:
                    coord = c["coord"]
                    pts = c["points"] / 3.0 if c["state"] == "active" else 0.0
                    board_flat.extend([coord["q"] / 8.0, coord["r"] / 8.0, pts])

                # pieces: 6 pieces × [id/10, q/8, r/8, s/8]
                pieces_flat = []
                for p in sd["pieces"]:
                    if p["alive"] and p["hex_idx"] is not None:
                        hi = p["hex_idx"]
                        cell = cells[hi]
                        coord = cell["coord"]
                        pieces_flat.extend([p["id"] / 10.0,
                                            coord["q"] / 8.0,
                                            coord["r"] / 8.0,
                                            coord["s"] / 8.0])
                    else:
                        pieces_flat.extend([-1.0, 0.0, 0.0, 0.0])

                # meta: [current_player, phase]
                cp = float(sd.get("current_player", 0))
                ph = 0.0 if sd.get("phase") == "placement" else 1.0
                meta = [cp, ph]

                obs = np.array(board_flat + pieces_flat + meta, dtype=np.float32)
                obs_list.append(obs)

            batch = np.array(obs_list, dtype=np.float32)  # (B, 206)

            # 神经网络评估
            if net is not None and hasattr(net, 'evaluate_flat_batch'):
                logits, values = net.evaluate_flat_batch(batch)
            else:
                # 无网络时使用均匀分布
                B = len(states_data)
                logits = np.zeros((B, 60), dtype=np.float64)
                values = np.zeros(B, dtype=np.float64)

            # 返回结果
            results = []
            for i in range(len(states_data)):
                results.append({
                    "logits": logits[i].tolist(),
                    "value": float(values[i]),
                })
            result_str = json.dumps(results)
            ctypes.memmove(res_buf, result_str.encode("utf-8"), min(len(result_str), buf_size - 1))
            return 0
        except Exception as e:
            result_str = json.dumps([])
            ctypes.memmove(res_buf, result_str.encode("utf-8"), min(len(result_str), buf_size - 1))
            return -1

    # 调用 Rust MCTS
    result_buf = create_string_buffer(65536)
    try:
        engine._lib.mcts_search_rust(
            state_json.encode("utf-8"),
            c_int32(num_simulations),
            c_double(c_puct),
            c_int32(batch_size),
            evaluate_cb,
            result_buf,
            c_int32(65536),
        )
    except Exception as e:
        raise RuntimeError(f"Rust MCTS failed: {e}")

    result_str = result_buf.value.decode("utf-8")
    return json.loads(result_str)  # {action: visit_count, ...}


# 单例
_engine: Optional[RustEngine] = None


def get_engine() -> RustEngine:
    global _engine
    if _engine is None:
        _engine = RustEngine()
    return _engine
