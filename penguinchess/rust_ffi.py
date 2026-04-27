"""
高性能 Python ↔ Rust FFI 桥接 — 使用 ctypes 直接调用 Rust cdylib。
替代 subprocess + JSON 的 rust_bridge.py，速度快 100-1000x。
"""
from __future__ import annotations

import json
import os
from ctypes import CDLL, POINTER, c_char, c_char_p, c_float, c_int32, c_int64, c_double, create_string_buffer, CFUNCTYPE, c_void_p, memmove
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _find_dll() -> str:
    """定位已编译的 Rust cdylib 文件。"""
    root = Path(__file__).parent.parent / "game_engine"
    candidates = [
        root / "target" / "release" / "game_engine.dll",
        root / "target" / "release" / "libgame_engine.so",
        root / "target" / "release" / "libgame_engine.dylib",
        root / "target" / "debug" / "game_engine.dll",
        root / "target" / "debug" / "libgame_engine.so",
        root / "target" / "debug" / "libgame_engine.dylib",
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
            CFUNCTYPE(c_int32, POINTER(c_float), c_int32, POINTER(c_float), c_int32),
            POINTER(c_char), c_int32,
        ]
        self._lib.mcts_search_rust.restype = c_int32

        # Handle-based MCTS: takes handle i32 instead of JSON string
        self._lib.mcts_search_rust_handle.argtypes = [
            c_int32, c_int32, c_double, c_int32,
            CFUNCTYPE(c_int32, POINTER(c_float), c_int32, POINTER(c_float), c_int32),
            POINTER(c_char), c_int32,
        ]
        self._lib.mcts_search_rust_handle.restype = c_int32

        # Handle-based Parallel MCTS (internal thread parallelism)
        self._lib.mcts_search_rust_handle_parallel.argtypes = [
            c_int32, c_int32, c_double, c_int32, c_int32,
            CFUNCTYPE(c_int32, POINTER(c_float), c_int32, POINTER(c_float), c_int32),
            POINTER(c_char), c_int32,
        ]
        self._lib.mcts_search_rust_handle_parallel.restype = c_int32

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

# 新回调：接收 float 观测数组 (B×206)，返回 float 输出数组 (B×61)
# 零拷贝：numpy as_array 包装指针，不复制数据
EVAL_FN = CFUNCTYPE(c_int32, POINTER(c_float), c_int32, POINTER(c_float), c_int32)


def mcts_search_rust(
    state_json: str,
    model: Any = None,
    num_simulations: int = 800,
    c_puct: float = 1.4,
    batch_size: int = 32,
) -> dict:
    """
    使用 Rust MCTS 进行搜索，通过回调调用 PyTorch 神经网络批量推理。

    Rust 侧直接构建观测 float 张量 (B×206)，通过指针传给 Python。
    零拷贝 numpy wrap，完全消除 JSON 序列化开销。

    Args:
        state_json: Rust GameState 的 JSON 字符串（仅初始状态）
        model: AlphaZeroNet 实例（需要 evaluate_flat_batch 方法）
        num_simulations: MCTS 模拟次数
        c_puct: 探索常数
        batch_size: 批量推理大小

    Returns:
        动作计数 dict: {action: visit_count}
    """
    import numpy as np
    import numpy.ctypeslib as npct

    engine = get_engine()
    net = model

    @EVAL_FN
    def evaluate_fn(obs_ptr, batch_size, output_ptr, output_capacity):
        """零拷贝回调：obs_ptr → numpy → GPU inference → output_ptr"""
        nonlocal net
        try:
            B = batch_size
            # Copy data from Rust shared memory → numpy (avoids PyTorch zero-copy issues)
            obs_raw = npct.as_array(
                ctypes.cast(obs_ptr, POINTER(c_float)),
                shape=(B, 206),
            ).copy()  # .copy() ensures no shared memory with Rust

            if net is not None and hasattr(net, 'evaluate_flat_batch'):
                logits, values = net.evaluate_flat_batch(obs_raw)
            else:
                logits = np.zeros((B, 60), dtype=np.float64)
                values = np.zeros(B, dtype=np.float64)

            # Write output
            out_arr = npct.as_array(
                ctypes.cast(output_ptr, POINTER(c_float)),
                shape=(output_capacity,),
            )
            for i in range(B):
                out_arr[i * 61:(i + 1) * 61 - 1] = logits[i].astype(np.float32)
                out_arr[(i + 1) * 61 - 1] = np.float32(values[i])
            return 0
        except Exception:
            return -1

    result_buf = create_string_buffer(1_048_576)
    try:
        code = engine._lib.mcts_search_rust(
            state_json.encode("utf-8"),
            c_int32(num_simulations),
            c_double(c_puct),
            c_int32(batch_size),
            evaluate_fn,
            result_buf,
            c_int32(1_048_576),
        )
        if code != 0:
            raise RuntimeError(f"Rust MCTS returned error code: {code}")
    except Exception as e:
        raise RuntimeError(f"Rust MCTS failed: {e}")

    raw = result_buf.value
    if not raw:
        return {}
    result_str = raw.decode("utf-8")
    return json.loads(result_str)


def mcts_search_rust_handle(
    handle: int,
    model: Any = None,
    num_simulations: int = 800,
    c_puct: float = 1.4,
    batch_size: int = 32,
) -> dict:
    """
    使用 Rust MCTS 进行搜索（基于句柄，绕过 JSON 序列化）。
    与 mcts_search_rust 相同，但接受 stateful game handle 而非 JSON 字符串。

    Args:
        handle: RustStatefulGame 的句柄
        model: AlphaZeroNet 实例（需要 evaluate_flat_batch 方法）
        num_simulations: MCTS 模拟次数
        c_puct: 探索常数
        batch_size: 批量推理大小

    Returns:
        动作计数 dict: {action: visit_count}
    """
    import numpy as np
    import numpy.ctypeslib as npct

    engine = get_engine()
    net = model

    @EVAL_FN
    def evaluate_fn(obs_ptr, batch_size, output_ptr, output_capacity):
        nonlocal net
        try:
            B = batch_size
            obs_raw = npct.as_array(
                ctypes.cast(obs_ptr, POINTER(c_float)),
                shape=(B, 206),
            ).copy()

            if net is not None and hasattr(net, 'evaluate_flat_batch'):
                logits, values = net.evaluate_flat_batch(obs_raw)
            else:
                logits = np.zeros((B, 60), dtype=np.float64)
                values = np.zeros(B, dtype=np.float64)

            out_arr = npct.as_array(
                ctypes.cast(output_ptr, POINTER(c_float)),
                shape=(output_capacity,),
            )
            for i in range(B):
                out_arr[i * 61:(i + 1) * 61 - 1] = logits[i].astype(np.float32)
                out_arr[(i + 1) * 61 - 1] = np.float32(values[i])
            return 0
        except Exception:
            return -1

    result_buf = create_string_buffer(1_048_576)
    try:
        code = engine._lib.mcts_search_rust_handle(
            c_int32(handle),
            c_int32(num_simulations),
            c_double(c_puct),
            c_int32(batch_size),
            evaluate_fn,
            result_buf,
            c_int32(1_048_576),
        )
        if code != 0:
            raise RuntimeError(f"Rust MCTS handle returned error code: {code}")
    except Exception as e:
        raise RuntimeError(f"Rust MCTS handle failed: {e}")

    raw = result_buf.value
    if not raw:
        return {}
    result_str = raw.decode("utf-8")
    return json.loads(result_str)


def mcts_search_rust_handle_parallel(
    handle: int,
    model: Any = None,
    num_simulations: int = 400,
    c_puct: float = 3.0,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:
    """
    Parallel MCTS search using Rust-internal thread parallelism.
    Single FFI call replaces N independent serial MCTS calls with ThreadPoolExecutor.
    All threads run inside Rust via std::thread::scope, each cloning the GameState.

    Args:
        handle: RustStatefulGame handle
        model: AlphaZeroNet instance (needs evaluate_flat_batch method)
        num_simulations: Total MCTS simulations across all workers
        c_puct: Exploration constant
        batch_size: NN eval batch size per worker
        num_workers: Threads to spawn inside Rust

    Returns:
        {action: visit_count}
    """
    engine = get_engine()
    net = model

    @EVAL_FN
    def evaluate_fn(obs_ptr, batch_size, output_ptr, output_capacity):
        nonlocal net
        try:
            B = batch_size
            obs_raw = npct.as_array(
                ctypes.cast(obs_ptr, POINTER(c_float)),
                shape=(B, 206),
            ).copy()

            if net is not None and hasattr(net, "evaluate_flat_batch"):
                logits, values = net.evaluate_flat_batch(obs_raw)
            else:
                logits = np.zeros((B, 60), dtype=np.float64)
                values = np.zeros(B, dtype=np.float64)

            out_arr = npct.as_array(
                ctypes.cast(output_ptr, POINTER(c_float)),
                shape=(output_capacity,),
            )
            for i in range(B):
                out_arr[i * 61:(i + 1) * 61 - 1] = logits[i].astype(np.float32)
                out_arr[(i + 1) * 61 - 1] = np.float32(values[i])
            return 0
        except Exception:
            return -1

    result_buf = create_string_buffer(1_048_576)
    try:
        code = engine._lib.mcts_search_rust_handle_parallel(
            c_int32(handle),
            c_int32(num_simulations),
            c_double(c_puct),
            c_int32(batch_size),
            c_int32(num_workers),
            evaluate_fn,
            result_buf,
            c_int32(1_048_576),
        )
        if code != 0:
            raise RuntimeError(f"Rust MCTS parallel returned error code: {code}")
    except Exception as e:
        raise RuntimeError(f"Rust MCTS parallel failed: {e}")

    raw = result_buf.value
    if not raw:
        return {}
    result_str = raw.decode("utf-8")
    return json.loads(result_str)


# =============================================================================
# Stateful Game API — 状态留在 Rust，避免 JSON 序列化全状态
# =============================================================================

class RustStatefulGame:
    """Rust 有状态游戏实例。状态留在 Rust 内存中，Python 只传 handle 和 action。"""

    def __init__(self, engine: RustEngine, seed: int = 42):
        self._engine = engine
        self._handle = engine._lib.game_stateful_new(c_int64(seed))
        if self._handle < 0:
            raise RuntimeError("Rust stateful game creation failed (slots full)")
        self._buf_size = 65536  # 64KB — state JSON ~4KB, 安全余量
        self._buf = create_string_buffer(self._buf_size)

    @property
    def handle(self) -> int:
        return self._handle

    def step(self, action: int) -> dict:
        """执行动作，返回 {reward, terminated, legal_actions, scores, current_player, phase}。"""
        result = self._engine._lib.game_stateful_step(
            c_int32(self._handle), c_int32(action),
            self._buf, self._buf_size,
        )
        if result != 0:
            raise RuntimeError(f"Rust stateful step failed: {result}")
        return json.loads(self._buf.value.decode("utf-8"))

    def get_legal_actions(self) -> list[int]:
        """获取合法动作。"""
        result = self._engine._lib.game_stateful_get_legal(
            c_int32(self._handle), self._buf, self._buf_size,
        )
        if result != 0:
            raise RuntimeError(f"Rust get_legal failed: {result}")
        return json.loads(self._buf.value.decode("utf-8"))["legal_actions"]

    def to_json(self) -> str:
        """序列化当前状态为 JSON（MCTS 使用）。"""
        result = self._engine._lib.game_stateful_to_json(
            c_int32(self._handle), self._buf, self._buf_size,
        )
        if result != 0:
            raise RuntimeError(f"Rust to_json failed: {result}")
        return self._buf.value.decode("utf-8")

    def get_scores(self) -> list[int]:
        """获取双方分数。"""
        result = self._engine._lib.game_stateful_scores(
            c_int32(self._handle), self._buf, self._buf_size,
        )
        if result != 0:
            raise RuntimeError(f"Rust scores failed: {result}")
        return json.loads(self._buf.value.decode("utf-8"))["scores"]

    def get_obs(self) -> dict:
        """获取观测（PPO Agent 使用）。返回 {"board": ..., "pieces": ..., "current_player": ..., "phase": ...}。"""
        result = self._engine._lib.game_stateful_get_obs(
            c_int32(self._handle), self._buf, self._buf_size,
        )
        if result != 0:
            raise RuntimeError(f"Rust get_obs failed: {result}")
        return json.loads(self._buf.value.decode("utf-8"))

    def get_info(self) -> dict:
        """获取当前状态信息（不执行动作）。返回 {legal_actions, scores, current_player, phase, terminated}。"""
        result = self._engine._lib.game_stateful_get_info(
            c_int32(self._handle), self._buf, self._buf_size,
        )
        if result != 0:
            raise RuntimeError(f"Rust get_info failed: {result}")
        return json.loads(self._buf.value.decode("utf-8"))

    def free(self):
        """释放 Rust 端槽位。"""
        if self._handle >= 0:
            self._engine._lib.game_stateful_free(c_int32(self._handle))
            self._handle = -1

    def __del__(self):
        self.free()


# 单例
_engine: Optional[RustEngine] = None


def get_engine() -> RustEngine:
    global _engine
    if _engine is None:
        _engine = RustEngine()
    return _engine
