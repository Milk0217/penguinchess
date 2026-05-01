"""
高性能 Python ↔ Rust FFI 桥接 — 使用 ctypes 直接调用 Rust cdylib。
替代 subprocess + JSON 的 rust_bridge.py，速度快 100-1000x。
"""
from __future__ import annotations

import json
import os
import numpy as np
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
        try:
            self._lib.mcts_search_rust_handle_parallel
        except AttributeError:
            pass
        else:
            self._lib.mcts_search_rust_handle_parallel.argtypes = [
                c_int32, c_int32, c_double, c_int32, c_int32,
                CFUNCTYPE(c_int32, POINTER(c_float), c_int32, POINTER(c_float), c_int32),
                POINTER(c_char), c_int32,
            ]
            self._lib.mcts_search_rust_handle_parallel.restype = c_int32

        # AB search data generation
        try:
            self._lib.ffi_ab_generate_random_data
        except AttributeError:
            pass
        else:
            self._lib.ffi_ab_generate_random_data.argtypes = [
                c_int32, c_int32, c_int32, c_int32, c_char_p,
            ]
            self._lib.ffi_ab_generate_random_data.restype = c_int64

        # NNUE MCTS (75-dim obs, Python eval callback)
        try:
            self._lib.mcts_search_nnue_handle
        except AttributeError:
            pass
        else:
            self._lib.mcts_search_nnue_handle.argtypes = [
                c_int32, c_int32, c_double, c_int32,
                CFUNCTYPE(c_int32, POINTER(c_float), c_int32, POINTER(c_float), c_int32),
                POINTER(c_char), c_int32,
            ]
            self._lib.mcts_search_nnue_handle.restype = c_int32

        # NNUE native MCTS (Rust inference, no Python callback)
        try:
            self._lib.nnue_mcts_create
        except AttributeError:
            pass
        else:
            self._lib.nnue_mcts_create.argtypes = [POINTER(c_char), c_int32]
            self._lib.nnue_mcts_create.restype = c_int32
            self._lib.nnue_mcts_set_weights.argtypes = [c_int32, POINTER(c_float), c_int32, POINTER(c_char), c_int32]
            self._lib.nnue_mcts_set_weights.restype = c_int32
            self._lib.mcts_search_nnue_native.argtypes = [c_int32, c_int32, c_int32, c_double, POINTER(c_char), c_int32]
            self._lib.mcts_search_nnue_native.restype = c_int32

        # MCTS tree reuse
        try:
            self._lib.nnue_mcts_tree_init
        except AttributeError:
            pass
        else:
            self._lib.nnue_mcts_tree_init.argtypes = [c_int32, c_int32, c_int32, c_double, POINTER(c_char), c_int32]
            self._lib.nnue_mcts_tree_init.restype = c_int32
            self._lib.nnue_mcts_tree_step.argtypes = [c_int32, c_int32, c_int32, c_double, POINTER(c_char), c_int32]
            self._lib.nnue_mcts_tree_step.restype = c_int32
            self._lib.nnue_mcts_tree_free.argtypes = [c_int32]
            self._lib.nnue_mcts_tree_free.restype = c_int32

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
_ab_eval_callback = None  # Keep reference alive to prevent GC


def get_engine() -> RustEngine:
    global _engine
    if _engine is None:
        _engine = RustEngine()
    return _engine


def ffi_ab_init_eval_callback(model: 'NNUE', device: str = 'cuda') -> bool:
    """
    Set the batch evaluation callback for Rust Alpha-Beta search.
    Uses model.forward() on CUDA for fast batched NNUE evaluation.
    
    Must be called once before using any AB search handle.
    """
    global _ab_eval_callback
    import torch

    model.eval()
    model.to(device)

    EVAL_BATCH_STRIDE = 75  # 8 sparse + 66 dense + 1 stm
    EVAL_BATCH_CB = CFUNCTYPE(c_int32, POINTER(c_float), c_int32, POINTER(c_float))
    lib = get_engine()._lib

    @EVAL_BATCH_CB
    def _eval_batch(data_ptr, n_states, scores_ptr):
        """FFI callback: batch evaluate states for Rust AB search."""
        try:
            n = n_states
            # Read data as flat float array
            flat = np.ctypeslib.as_array(data_ptr, shape=(n * EVAL_BATCH_STRIDE,))
            flat = flat.copy()  # Copy to avoid lifetime issues
            data = flat.reshape(n, EVAL_BATCH_STRIDE)

            # Parse batch data
            sparse_batch = []
            dense_batch_np = np.empty((n, 66), dtype=np.float32)
            stm_batch = []

            for i in range(n):
                row = data[i]
                sparse = [int(row[j]) for j in range(8) if row[j] >= 0.0]
                sparse_batch.append(sparse)
                dense_batch_np[i] = row[8:74]
                stm_batch.append(int(row[74]))

            dense_t = torch.from_numpy(dense_batch_np).to(device)

            with torch.no_grad():
                values = model.forward(sparse_batch, dense_t, stm_players=stm_batch)

            scores_np = values.cpu().numpy().astype(np.float32)
            for i in range(n):
                scores_ptr[i] = scores_np[i]

            return 0
        except Exception as e:
            print(f"[EVAL ERROR] {e}")
            return -1

    lib.ffi_ab_set_eval_callback.argtypes = [EVAL_BATCH_CB]
    lib.ffi_ab_set_eval_callback.restype = c_int32

    rc = lib.ffi_ab_set_eval_callback(_eval_batch)
    _ab_eval_callback = _eval_batch  # Keep reference
    return rc == 0


# =============================================================================
# NNUE MCTS Native (Rust inference, no Python callback)
# =============================================================================

class NNUEMCTSNative:
    """Handle for Rust-native NNUE MCTS (no Python eval callback)."""

    def __init__(self, model_state: dict):
        import numpy as np
        eng = get_engine()
        buf = create_string_buffer(1024)
        rc = eng._lib.nnue_mcts_create(buf, c_int32(1024))
        result = json.loads(buf.value.decode('utf-8')) if buf.value else {}
        self._handle = result.get('handle', -1)
        if self._handle < 0:
            raise RuntimeError(f"nnue_mcts_create failed: {result}")
        self._engine = eng

        # Build flat weights from state dict
        ft_w = model_state['ft.weight'].cpu().numpy().ravel()
        ft_b = model_state['ft.bias'].cpu().numpy().ravel() if 'ft.bias' in model_state else np.zeros(64, dtype=np.float32)
        fc1_w = model_state['fc1.weight'].cpu().numpy().ravel()
        fc1_b = model_state['fc1.bias'].cpu().numpy().ravel()
        # Detect dimensions from state dict
        ft_w = model_state['ft.weight'].cpu().numpy().ravel()
        ft_dim = model_state['ft.weight'].shape[1]
        fc1_w = model_state['fc1.weight'].cpu().numpy().ravel()
        fc1_dim = model_state['fc1.weight'].shape[0]
        fc2v_w = model_state['fc2v.weight'].cpu().numpy().ravel() if 'fc2v.weight' in model_state else np.zeros(fc1_dim, dtype=np.float32)
        fc2v_b = model_state['fc2v.bias'].cpu().numpy().ravel() if 'fc2v.bias' in model_state else np.zeros(1, dtype=np.float32)
        fc2p_w = model_state['fc2p.weight'].cpu().numpy().ravel() if 'fc2p.weight' in model_state else np.zeros(fc1_dim*60, dtype=np.float32)
        fc2p_b = model_state['fc2p.bias'].cpu().numpy().ravel() if 'fc2p.bias' in model_state else np.zeros(60, dtype=np.float32)

        flat = np.concatenate([ft_w, ft_b, fc1_w, fc1_b, fc2v_w, fc2v_b, fc2p_w, fc2p_b]).astype(np.float32)
        self.set_weights(flat)

    def set_weights(self, flat: np.ndarray):
        """Update weights on existing handle."""
        import numpy as np
        ptr = flat.ctypes.data_as(POINTER(c_float))
        out2 = create_string_buffer(1024)
        rc2 = self._engine._lib.nnue_mcts_set_weights(
            c_int32(self._handle), ptr, c_int32(len(flat)), out2, c_int32(1024))
        result2 = json.loads(out2.value.decode('utf-8')) if out2.value else {}
        if not result2.get('ok', False):
            raise RuntimeError(f"nnue_mcts_set_weights failed: {result2}")

    def update_weights(self, model_state: dict):
        """Build weight array from state dict and update."""
        import numpy as np
        # Detect dimensions from model state dict
        ft_w = model_state['ft.weight'].cpu().numpy()
        ft_dim = ft_w.shape[1]  # (360, ft_dim)
        fc1_w = model_state['fc1.weight'].cpu().numpy()
        fc1_dim = fc1_w.shape[0]  # (fc1_dim, input_dim)
        fc2v_w = model_state['fc2v.weight'].cpu().numpy()
        fc2p_w = model_state['fc2p.weight'].cpu().numpy()
        fc2p_dim = fc2p_w.shape[1]  # (fc1_dim, 60)
        ft_b = model_state['ft.bias'].cpu().numpy().ravel() if 'ft.bias' in model_state else np.zeros(ft_dim, dtype=np.float32)
        fc1_b = model_state['fc1.bias'].cpu().numpy().ravel()
        fc2v_b = model_state['fc2v.bias'].cpu().numpy().ravel() if 'fc2v.bias' in model_state else np.zeros(1, dtype=np.float32)
        fc2p_b = model_state['fc2p.bias'].cpu().numpy().ravel() if 'fc2p.bias' in model_state else np.zeros(fc2p_dim, dtype=np.float32)
        flat = np.concatenate([ft_w.ravel(), ft_b, fc1_w.ravel(), fc1_b, fc2v_w.ravel(), fc2v_b, fc2p_w.ravel(), fc2p_b]).astype(np.float32)
        self.set_weights(flat)

    def search(self, game_handle: int, num_simulations: int = 200, c_puct: float = 1.4) -> dict:
        """Run Rust-native NNUE MCTS. Returns visit count dict."""
        result_buf = create_string_buffer(1_048_576)
        rc = self._engine._lib.mcts_search_nnue_native(
            c_int32(game_handle), c_int32(self._handle),
            c_int32(num_simulations), c_double(c_puct),
            result_buf, c_int32(1_048_576))
        if rc != 0:
            raise RuntimeError(f"NNUE MCTS native returned error: {rc}")
        raw = result_buf.value
        if not raw:
            return {}
        return json.loads(raw.decode('utf-8'))

    def free(self):
        """Release Rust-side handle."""
        self._engine._lib.nnue_mcts_set_weights(
            c_int32(self._handle), POINTER(c_float)(), c_int32(0),
            create_string_buffer(1024), c_int32(1024))
        self._handle = -1

    def __del__(self):
        if self._handle >= 0:
            self.free()


# =============================================================================
# MCTS Tree Reuse — 子树在 Rust 侧持久化，跨步骤复用
# =============================================================================

class MCTSTreeReuse:
    """Handle for Rust-side MCTS tree that persists across game steps."""

    def __init__(self, game_handle: int, model_native: NNUEMCTSNative,
                 num_simulations: int = 200, c_puct: float = 1.4):
        self._engine = get_engine()
        buf = create_string_buffer(1_048_576)
        rc = self._engine._lib.nnue_mcts_tree_init(
            c_int32(game_handle), c_int32(model_native._handle),
            c_int32(num_simulations), c_double(c_puct),
            buf, c_int32(1_048_576))
        if rc != 0:
            raise RuntimeError(f"nnue_mcts_tree_init failed: {rc}")
        self._handle = 0  # tree reuse handle is implicit
        self._c_puct = c_puct
        self._raw = buf.value

    @property
    def _tree_handle(self) -> int:
        # The tree is stored at index 0 in MCTS_TREES (one per game)
        return 0

    def step(self, action: int, additional_sims: int = 30) -> dict:
        """Step game, reuse tree, run additional sims. Returns visit counts."""
        buf = create_string_buffer(1_048_576)
        rc = self._engine._lib.nnue_mcts_tree_step(
            c_int32(self._tree_handle), c_int32(action),
            c_int32(additional_sims), c_double(self._c_puct),
            buf, c_int32(1_048_576))
        if rc != 0:
            raise RuntimeError(f"nnue_mcts_tree_step failed: {rc}")
        self._raw = buf.value
        raw = buf.value
        if not raw: return {}
        return json.loads(raw.decode('utf-8'))

    def free(self):
        self._engine._lib.nnue_mcts_tree_free(c_int32(self._tree_handle))

    def __del__(self):
        self.free()


# =============================================================================
# Alpha-Beta Search FFI
# =============================================================================

class AlphaBetaSearchHandle:
    """Handle for a Rust-side Alpha-Beta search instance with pre-loaded NNUE weights."""

    def __init__(self, handle: int, lib):
        self._handle = handle
        self._lib = lib

    def set_weights(self, model_state: dict) -> bool:
        """
        Set NNUE weights from a PyTorch model state dict.
        Weights are flattened into a single float array with fixed layout.
        """
        import numpy as np
        import torch

        # Build flat weight array
        ft_weight = model_state['ft.weight'].cpu().numpy().T.ravel()  # (360, 64) row-major
        ft_bias = model_state['ft.bias'].cpu().numpy().ravel()  # (64,)
        fc1_w = model_state['fc1.weight'].cpu().numpy().ravel()  # (256, 194)
        fc1_b = model_state['fc1.bias'].cpu().numpy().ravel()
        fc2_w = model_state['fc2.weight'].cpu().numpy().ravel()  # (128, 256)
        fc2_b = model_state['fc2.bias'].cpu().numpy().ravel()
        fc3_w = model_state['fc3.weight'].cpu().numpy().ravel()  # (1, 128)
        fc3_b = model_state['fc3.bias'].cpu().numpy().ravel()

        flat = np.concatenate([ft_weight, ft_bias, fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b]).astype(np.float32)
        ptr = flat.ctypes.data_as(POINTER(c_float))
        buf_size = 1024
        out = create_string_buffer(buf_size)

        rc = self._lib.ffi_ab_set_weights(
            c_int32(self._handle), ptr, c_int32(len(flat)), out, c_int32(buf_size))
        result = json.loads(out.value.decode('utf-8')) if out.value else {}
        return rc == 0 and result.get('ok', False)

    def search(self, state_json: str, max_depth: int = 6) -> dict:
        """Run Alpha-Beta search. Returns SearchResult dict."""
        buf_size = 65536
        out = create_string_buffer(buf_size)
        rc = self._lib.ffi_ab_search(
            c_int32(self._handle),
            c_char_p(state_json.encode('utf-8')),
            c_int32(max_depth),
            out, c_int32(buf_size))
        result_str = out.value.decode('utf-8') if out.value else '{}'
        return json.loads(result_str)


def mcts_search_nnue_handle(
    handle: int,
    nnue_eval_fn,
    num_simulations: int = 200,
    c_puct: float = 1.4,
    batch_size: int = 32,
) -> dict:
    """
    MCTS search using NNUE eval callback (75-dim obs).
    Uses Rust GameState handle for zero-copy.

    Args:
        handle: RustStatefulGame handle
        nnue_eval_fn: callable(sparse_batch, dense_batch, stm_list) -> (logits_60, values_1)
        num_simulations: MCTS simulations
        c_puct: exploration constant
        batch_size: batch size for NN evaluation

    Returns:
        action_counts dict: {action: visit_count}
    """
    import numpy as np
    import numpy.ctypeslib as npct

    engine = get_engine()

    @EVAL_FN
    def evaluate_fn(obs_ptr, batch_size, output_ptr, output_capacity):
        try:
            n = batch_size
            obs_np = npct.as_array(obs_ptr, shape=(n, 75)).copy()
            sparse_raw = obs_np[:, :8].astype(np.int64)
            dense = obs_np[:, 8:74]
            stm_list = [int(x) for x in obs_np[:, 74]]
            sparse_batch = [[int(x) for x in row if x >= 0] for row in sparse_raw]

            logits, values = nnue_eval_fn(sparse_batch, dense, stm_list)

            out = np.zeros((n, 61), dtype=np.float32)
            out[:, :60] = logits
            out[:, 60] = values

            out_ptr = npct.as_array(output_ptr, shape=(n * 61,))
            out_ptr[:] = out.ravel()
            return 0
        except Exception as e:
            print(f"[NNUE MCTS eval error] {e}")
            import traceback; traceback.print_exc()
            return -1

    result_buf = create_string_buffer(1_048_576)
    try:
        code = engine._lib.mcts_search_nnue_handle(
            c_int32(handle),
            c_int32(num_simulations),
            c_double(c_puct),
            c_int32(batch_size),
            evaluate_fn,
            result_buf,
            c_int32(1_048_576),
        )
        if code != 0:
            raise RuntimeError(f"NNUE MCTS returned error code: {code}")
    except Exception as e:
        raise RuntimeError(f"NNUE MCTS failed: {e}")

    raw = result_buf.value
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))



    def free(self):
        if self._handle >= 0:
            self._lib.ffi_ab_destroy(c_int32(self._handle))
            self._handle = -1

    def __del__(self):
        self.free()


def ffi_ab_create(config_json: str = '{}') -> AlphaBetaSearchHandle:
    """Create an Alpha-Beta search instance. Returns handle."""
    buf_size = 4096
    out = create_string_buffer(buf_size)
    rc = get_engine()._lib.ffi_ab_create(
        c_char_p(config_json.encode('utf-8')),
        out, c_int32(buf_size))
    result = json.loads(out.value.decode('utf-8')) if out.value else {}
    handle = result.get('handle', -1)
    if handle < 0:
        raise RuntimeError(f"ffi_ab_create failed: {result.get('error', 'unknown')} (rc={rc})")
    return AlphaBetaSearchHandle(handle, get_engine()._lib)


def ffi_ab_generate_random_data(
    ab_handle: AlphaBetaSearchHandle,
    num_games: int = 500,
    seed_offset: int = 0,
    workers: int = 4,
    output_path: str = 'data_nnue.bin',
) -> int:
    """Generate NNUE training data from random games with AB search labels.
    
    Uses Rust-native game stepping + AB search (no Python overhead).
    Returns number of positions generated.
    """
    lib = get_engine()._lib
    count = lib.ffi_ab_generate_random_data(
        c_int32(ab_handle._handle),
        c_int32(num_games),
        c_int32(seed_offset),
        c_int32(workers),
        c_char_p(output_path.encode('utf-8')),
    )
    return count


# =============================================================================
# AlphaZero Model Inference FFI (Rust-native, no Python callback)
# =============================================================================

class AZModelHandle:
    """Rust-native AlphaZero model inference handle."""

    def __init__(self, handle: int, lib):
        self._handle = handle
        self._lib = lib

    def evaluate(self, obs_batch: np.ndarray) -> tuple:
        """Run forward pass. obs_batch: (N, 206) float32. Returns (logits, values)."""
        n = obs_batch.shape[0]
        logits_out = np.empty((n, 60), dtype=np.float32)
        values_out = np.empty(n, dtype=np.float32)

        self._lib.ffi_az_evaluate(
            c_int32(self._handle),
            obs_batch.ctypes.data_as(POINTER(c_float)),
            c_int32(n),
            logits_out.ctypes.data_as(POINTER(c_float)),
            values_out.ctypes.data_as(POINTER(c_float)),
        )
        return logits_out, values_out

    def free(self):
        if self._handle >= 0:
            self._lib.ffi_az_free(c_int32(self._handle))
            self._handle = -1

    def __del__(self):
        self.free()


def ffi_az_create(arch: str = "mlp", layer_info: list = None,
                  weights: np.ndarray = None, biases: np.ndarray = None,
                  policy_idx: int = 0, value1_idx: int = 0, value2_idx: int = 0,
                  value_uses_obs: bool = False) -> AZModelHandle:
    """Create Rust AZ model handle. weights/biases are BN-folded flat arrays."""
    if layer_info is None: layer_info = []
    if weights is None: weights = np.array([], dtype=np.float32)
    if biases is None: biases = np.array([], dtype=np.float32)
    lib = get_engine()._lib

    config = {
        'arches': arch,
        'total_weights': len(weights),
        'total_biases': len(biases),
        'policy_idx': policy_idx,
        'value1_idx': value1_idx,
        'value2_idx': value2_idx,
        'value_uses_obs': value_uses_obs,
    }

    buf = create_string_buffer(4096)
    rc = lib.ffi_az_create(
        c_char_p(json.dumps(config).encode('utf-8')),
        buf, c_int32(4096))
    result = json.loads(buf.value.decode('utf-8')) if buf.value else {}
    handle = result.get('handle', -1)
    if handle < 0:
        raise RuntimeError(f"ffi_az_create failed: {result.get('error', 'unknown')} (rc={rc})")

    # Set layer info: for each layer: rows, cols, weight_offset, bias_offset, has_relu, is_residual
    layer_data = np.zeros(len(layer_info) * 6, dtype=np.int32)
    w_off = 0
    b_off = 0
    for i, (rows, cols, has_relu, is_res) in enumerate(layer_info):
        base = i * 6
        layer_data[base] = rows
        layer_data[base + 1] = cols
        layer_data[base + 2] = w_off
        layer_data[base + 3] = b_off
        layer_data[base + 4] = 1 if has_relu else 0
        layer_data[base + 5] = 1 if is_res else 0
        w_off += rows * cols
        b_off += rows

    lib.ffi_az_set_layer_info(
        c_int32(handle),
        layer_data.ctypes.data_as(POINTER(c_int32)),
        c_int32(len(layer_data)))

    if len(weights) > 0 and len(biases) > 0:
        w_ptr = weights.ctypes.data_as(POINTER(c_float))
        b_ptr = biases.ctypes.data_as(POINTER(c_float))
        lib.ffi_az_set_weights(
            c_int32(handle),
            w_ptr, c_int32(len(weights)),
            b_ptr, c_int32(len(biases)))

    return AZModelHandle(handle, lib)
