"""
AI 对战模块 — 所有推理在 Rust 端执行。
PPO (SB3) 和 AlphaZero 模型权重在 Python 加载后传给 Rust。
"""
from __future__ import annotations

import os, json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from penguinchess.core import PenguinChessCore
from penguinchess.rust_ffi import ffi_az_create, AZModelHandle

MODELS_DIR = Path(__file__).parent.parent / "models"


def _find_best_model() -> Optional[tuple[str, str]]:
    """
    在所有可用模型中找最强的。
    Returns: (model_path, model_type)  model_type="ppo" or "alphazero"
    """
    try:
        from penguinchess.model_registry import get_best_model
        result = get_best_model(criteria="elo")
        if result:
            path, mtype = result
            full_path = str(MODELS_DIR / path)
            if os.path.exists(full_path):
                return (full_path, mtype)
    except Exception:
        pass

    best = None
    best_priority = -1

    for p in sorted(MODELS_DIR.glob("ppo_penguinchess_gen_*.zip")):
        try:
            n = int(p.stem.split("_gen_")[1])
            if n > best_priority:
                best = (str(p), "ppo")
                best_priority = n
        except (IndexError, ValueError):
            pass

    az_dir = MODELS_DIR / "alphazero"
    if az_dir.exists():
        for p in sorted(az_dir.glob("alphazero*.pth")):
            stem = p.stem; parts = stem.split("_")
            iter_num = None
            if "iter" in parts:
                try:
                    idx = parts.index("iter"); iter_num = int(parts[idx + 1])
                except (IndexError, ValueError): pass
            if "best" in parts: priority = 999
            elif iter_num is not None: priority = iter_num
            elif "final" in parts: priority = 0
            else: continue
            if priority > best_priority:
                best = (str(p), "alphazero")
                best_priority = priority

    if best is None:
        best_zip = MODELS_DIR / "best" / "best_model.zip"
        if best_zip.exists():
            best = (str(best_zip), "ppo")
    return best


def _fold_bn(w: np.ndarray, b: np.ndarray,
             bn_w: np.ndarray, bn_b: np.ndarray,
             bn_mean: np.ndarray, bn_var: np.ndarray,
             eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Fold BatchNorm into preceding Linear layer."""
    scale = bn_w / np.sqrt(bn_var + eps)
    w_folded = w * scale[:, np.newaxis]
    b_folded = (b - bn_mean) * scale + bn_b
    return w_folded, b_folded


def _build_az_model(state_dict: dict) -> AZModelHandle:
    """Build Rust AZ model handle from AlphaZero PyTorch state dict."""
    from penguinchess.ai.alphazero_net import detect_net_arch
    NetClass = detect_net_arch(state_dict)
    net = NetClass()
    net.load_state_dict(state_dict)
    net.eval()

    arch_name = net.arch_name if hasattr(net, 'arch_name') else 'mlp'
    is_resnet = 'resnet' in arch_name.lower()
    device = next(net.parameters()).device

    layers = []
    flat_w = []
    flat_b = []

    def add_layer(w, b, has_relu=True, is_residual=False):
        nonlocal flat_w, flat_b, layers
        rows, cols = w.shape
        w = w.detach().cpu().numpy().astype(np.float32)
        b = b.detach().cpu().numpy().astype(np.float32)
        # Transpose to column-major for Rust
        w_t = w.T.ravel()  # (cols, rows) column-major = (rows*cols) after ravel of transposed
        # Wait, Rust expects w_t[col * rows + row] = original[row][col]
        # For matvec: y += w_t[col * rows + row] * x[col]
        # PyTorch weight shape is (rows, cols). We need (cols, rows) stored as w_t[col][row].
        w_t = np.ascontiguousarray(w.T)  # (cols, rows)
        flat_w.append(w_t.ravel())
        flat_b.append(b)
        layers.append((rows, cols, has_relu, is_residual))

    if is_resnet:
        # AlphaZeroResNet: fc1→bn1→ReLU→fc2→bn2→ReLU→+→fc3→bn3→ReLU
        w1, b1 = _fold_bn(
            state_dict['fc1.weight'].numpy(), state_dict['fc1.bias'].numpy(),
            state_dict['bn1.weight'].numpy(), state_dict['bn1.bias'].numpy(),
            state_dict['bn1.running_mean'].numpy(), state_dict['bn1.running_var'].numpy())
        add_layer(w1, b1, has_relu=True)

        w2, b2 = _fold_bn(
            state_dict['fc2.weight'].numpy(), state_dict['fc2.bias'].numpy(),
            state_dict['bn2.weight'].numpy(), state_dict['bn2.bias'].numpy(),
            state_dict['bn2.running_mean'].numpy(), state_dict['bn2.running_var'].numpy())
        add_layer(w2, b2, has_relu=False, is_residual=True)  # + identity then relu in next

        w3, b3 = _fold_bn(
            state_dict['fc3.weight'].numpy(), state_dict['fc3.bias'].numpy(),
            state_dict['bn3.weight'].numpy(), state_dict['bn3.bias'].numpy(),
            state_dict['bn3.running_mean'].numpy(), state_dict['bn3.running_var'].numpy())
        add_layer(w3, b3, has_relu=True)

        # Policy head
        add_layer(state_dict['policy_fc.weight'].numpy(), state_dict['policy_fc.bias'].numpy(), has_relu=False)
        # Value heads
        add_layer(state_dict['value_fc1.weight'].numpy(), state_dict['value_fc1.bias'].numpy(), has_relu=True)
        add_layer(state_dict['value_fc2.weight'].numpy(), state_dict['value_fc2.bias'].numpy(), has_relu=False)

        policy_idx = 3
        value1_idx = 4
        value2_idx = 5
    else:
        # AlphaZeroNet (MLP): fc1→bn1→ReLU→fc2→bn2→ReLU
        w1, b1 = _fold_bn(
            state_dict['fc1.weight'].numpy(), state_dict['fc1.bias'].numpy(),
            state_dict['bn1.weight'].numpy(), state_dict['bn1.bias'].numpy(),
            state_dict['bn1.running_mean'].numpy(), state_dict['bn1.running_var'].numpy())
        add_layer(w1, b1, has_relu=True)

        w2, b2 = _fold_bn(
            state_dict['fc2.weight'].numpy(), state_dict['fc2.bias'].numpy(),
            state_dict['bn2.weight'].numpy(), state_dict['bn2.bias'].numpy(),
            state_dict['bn2.running_mean'].numpy(), state_dict['bn2.running_var'].numpy())
        add_layer(w2, b2, has_relu=True)

        # Policy head
        add_layer(state_dict['policy_fc.weight'].numpy(), state_dict['policy_fc.bias'].numpy(), has_relu=False)
        # Value heads
        add_layer(state_dict['value_fc1.weight'].numpy(), state_dict['value_fc1.bias'].numpy(), has_relu=True)
        add_layer(state_dict['value_fc2.weight'].numpy(), state_dict['value_fc2.bias'].numpy(), has_relu=False)

        policy_idx = 2
        value1_idx = 3
        value2_idx = 4

    w_flat = np.concatenate(flat_w).astype(np.float32)
    b_flat = np.concatenate(flat_b).astype(np.float32)

    return ffi_az_create(
        arch="resnet" if is_resnet else "mlp",
        layer_info=layers,
        weights=w_flat,
        biases=b_flat,
        policy_idx=policy_idx,
        value1_idx=value1_idx,
        value2_idx=value2_idx,
        value_uses_obs=False,
    )


def _build_ppo_model(model_path: str) -> AZModelHandle:
    """Build Rust AZ model handle from PPO/SB3 policy weights."""
    from stable_baselines3 import PPO
    ppo = PPO.load(model_path)
    policy = ppo.policy
    sd = policy.state_dict()

    # Extract policy network weights
    pi_layers = []

    def add_sb3_layer(prefix, has_relu=True):
        w = sd[prefix + '.weight'].cpu().numpy().astype(np.float32)
        b = sd[prefix + '.bias'].cpu().numpy().astype(np.float32)
        w_t = np.ascontiguousarray(w.T)
        pi_layers.append((w_t.ravel(), b, has_relu))

    # Policy net: mlp_extractor.policy_net.0 → relu → mlp_extractor.policy_net.2 → relu → action_net
    add_sb3_layer('mlp_extractor.policy_net.0', has_relu=True)
    add_sb3_layer('mlp_extractor.policy_net.2', has_relu=True)
    add_sb3_layer('action_net', has_relu=False)

    # Value net: mlp_extractor.value_net.0 → relu → mlp_extractor.value_net.2 → relu → value_net
    pi_layers_val = []
    add_sb3_layer_val = lambda prefix, hr=True: pi_layers_val.append((
        np.ascontiguousarray(sd[prefix + '.weight'].cpu().numpy().T).ravel().astype(np.float32),
        sd[prefix + '.bias'].cpu().numpy().astype(np.float32), hr))
    add_sb3_layer_val('mlp_extractor.value_net.0', True)
    add_sb3_layer_val('mlp_extractor.value_net.2', True)
    add_sb3_layer_val('value_net', False)

    # Flatten
    w_flat = np.concatenate([l[0] for l in pi_layers] + [l[0] for l in pi_layers_val]).astype(np.float32)
    b_flat = np.concatenate([l[1] for l in pi_layers] + [l[1] for l in pi_layers_val]).astype(np.float32)

    # Layer info: policy trunk (3) + value trunk (3)
    layer_info = [(l[1].shape[0], sd[pr + '.weight'].shape[1], l[2], False)
                  for l, pr in zip(pi_layers, ['mlp_extractor.policy_net.0', 'mlp_extractor.policy_net.2', 'action_net'])]
    layer_info += [(l[1].shape[0], sd[pr + '.weight'].shape[1], l[2], False)
                   for l, pr in zip(pi_layers_val, ['mlp_extractor.value_net.0', 'mlp_extractor.value_net.2', 'value_net'])]

    return ffi_az_create(
        arch="mlp",
        layer_info=layer_info,
        weights=w_flat,
        biases=b_flat,
        policy_idx=2,
        value1_idx=3,
        value2_idx=5,
        value_uses_obs=True,
    )


class AIPlayer:
    """AI 玩家 — 全部推理在 Rust 端执行。"""

    def __init__(self, player_index: int = 1, model_path: Optional[str] = None):
        self.player_index = player_index
        self._az_handle: Optional[AZModelHandle] = None
        self._model_type: Optional[str] = None

        if model_path is None:
            found = _find_best_model()
            if found:
                model_path, self._model_type = found
                print(f"[AI] Auto-selected: {Path(model_path).name} ({self._model_type})")
        else:
            self._model_type = "alphazero" if model_path.endswith(".pth") else "ppo"

        if model_path and os.path.exists(model_path):
            self._load(model_path)
        else:
            print(f"[AI] No model found, AI disabled")

    def _load(self, path: str):
        try:
            if self._model_type == "ppo":
                self._az_handle = _build_ppo_model(path)
                print(f"[AI] Loaded PPO model (Rust): {path}")
            else:
                state = torch.load(path, map_location="cpu", weights_only=True)
                self._az_handle = _build_az_model(state)
                print(f"[AI] Loaded AlphaZero model (Rust): {path}")
        except Exception as e:
            print(f"[AI] Failed to load model: {e}")
            import traceback; traceback.print_exc()
            self._az_handle = None

    def is_ready(self) -> bool:
        return self._az_handle is not None

    def select_action(self, core: PenguinChessCore) -> Optional[int]:
        if self._az_handle is None:
            return None
        legal = core.get_legal_actions()
        if not legal:
            return None
        return self._select_action_rust(core, legal)

    def _select_action_rust(self, core: PenguinChessCore, legal: list) -> Optional[int]:
        """Rust 推理：encode obs → ffi_az_evaluate → argmax over legal."""
        obs = core.get_observation()
        flat = _encode_flat_obs(obs)  # (1, 206)
        logits, _ = self._az_handle.evaluate(flat)
        logits_1d = logits[0]  # (60,)
        # Mask illegal moves
        masked = logits_1d.copy()
        mask = np.zeros(60, dtype=bool)
        mask[legal] = True
        masked[~mask] = -1e9
        action = int(np.argmax(masked))
        return action if action in legal else int(np.random.choice(legal))


def _encode_flat_obs(obs: dict, target_dim: int = 272) -> np.ndarray:
    """Encode observation to flat array, padded to target_dim (272) for Rust inference."""
    board = np.array(obs["board"], dtype=np.float32).flatten()
    pieces = np.array(obs["pieces"], dtype=np.float32).flatten()
    meta = np.array([float(obs["current_player"]), float(obs["phase"])], dtype=np.float32)
    flat = np.concatenate([board, pieces, meta])  # 206-dim
    if target_dim > len(flat):
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[:len(flat)] = flat
        flat = padded
    return flat.reshape(1, -1)
