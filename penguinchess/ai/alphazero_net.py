"""
AlphaZero-style neural network with shared trunk + policy head + value head.

Intended for use as the ``evaluate_fn`` in ``mcts_search``:

.. code-block:: python

    net = AlphaZeroNet()
    net.load_state_dict(torch.load("checkpoint.pt"))
    net.eval()

    counts, root = mcts_search(state, evaluate_fn=net.evaluate)

Observation format
------------------
206-dim flat array built from ``PenguinChessCore.get_observation()``:

  - 180 = 60 hexes × [q/8, r/8, value/3]
  -  24 =  6 pieces × [id/10, q/8, r/8, s/8]
  -   2 = [current_player, phase]

Output
------
  - policy_logits : 60-dim (one per hex index; masked at inference)
  - value         : scalar in [-1, 1] (current-player perspective)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from penguinchess.core import PenguinChessCore


# =============================================================================
# Network
# =============================================================================

class AlphaZeroNet(nn.Module):
    """
    Neural network with a shared body, a policy head, and a value head.

    Architecture
    ------------
    Input (206) → FC(512) → BN → ReLU → FC(256) → BN → ReLU
        ├── Policy head: FC(256 → 60) → logits
        └── Value head: FC(256 → 128) → ReLU → FC(128 → 1) → tanh
    """

    # 架构标记，用于文件名区分和历史记录
    arch_name = "mlp"

    def __init__(self, obs_dim: int = 272, action_dim: int = 60):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # ----- shared trunk -----
        self.fc1 = nn.Linear(obs_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        # ----- policy head -----
        self.policy_fc = nn.Linear(256, action_dim)

        # ----- value head -----
        self.value_fc1 = nn.Linear(256, 128)
        self.value_fc2 = nn.Linear(128, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            Shape ``(batch, obs_dim)``.

        Returns
        -------
        policy_logits : Tensor  ``(batch, 60)``
        value         : Tensor  ``(batch, 1)``   range ≈ [-1, 1]
        """
        # Shared trunk
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)  # shape (batch, 256)

        # Policy head (raw logits, no softmax here)
        policy_logits = self.policy_fc(x)  # (batch, 60)

        # Value head
        v = F.relu(self.value_fc1(x))
        value = torch.tanh(self.value_fc2(v))  # (batch, 1) in [-1, 1]

        return policy_logits, value

    # ------------------------------------------------------------------
    # Inference convenience
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        state: PenguinChessCore,
    ) -> tuple[np.ndarray, float]:
        """
        Take a game state and return ``(policy_probs, value)``.

        Parameters
        ----------
        state : PenguinChessCore
            The current game state.  **Not** modified.

        Returns
        -------
        policy_probs : np.ndarray, shape (60,), dtype float32
            Action probabilities with **illegal actions masked to zero**.
            Sums to 1.0 over legal actions.
        value : float
            Scalar in [-1, 1] from the **current player's** perspective.
        """
        self.eval()  # ensure eval mode

        # 1. Build flat observation (206-dim)
        obs = state.get_observation()

        board = np.array(obs["board"], dtype=np.float32).flatten()    # (180,)
        pieces = np.array(obs["pieces"], dtype=np.float32).flatten()  # (24,)
        meta = np.array(
            [float(obs["current_player"]), float(obs["phase"])],
            dtype=np.float32,
        )  # (2,)
        flat = np.concatenate([board, pieces, meta]).astype(np.float32)  # (206,)

        # 2. Run through network
        device = next(self.parameters()).device
        x = torch.as_tensor(flat, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 206)
        logits, val_t = self.forward(x)
        logits_np = logits[0].cpu().numpy().astype(np.float64)
        value = float(val_t[0, 0].cpu().numpy())

        # 3. Mask illegal actions
        legal = state.get_legal_actions()
        mask = np.zeros(self.action_dim, dtype=bool)
        mask[legal] = True
        masked_logits = np.where(mask, logits_np, -1e9)

        # 4. Softmax over legal actions only
        probs = self._stable_softmax(masked_logits).astype(np.float32)

        # Safety: if masking zeroed everything (shouldn't happen), uniform
        if probs.sum() < 1e-8:
            probs[:] = 0.0
            if legal:
                probs[legal] = 1.0 / len(legal)
        elif abs(probs.sum() - 1.0) > 1e-6:
            probs = probs / probs.sum()

        return probs, value

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_batch(
        self,
        states: list,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Batch inference for MCTS — evaluate multiple states at once.

        Parameters
        ----------
        states : list[PenguinChessCore]
            List of game states to evaluate.  States are **not** modified.

        Returns
        -------
        logits : np.ndarray, shape ``(batch, 60)``, dtype float32
            Raw policy logits from the network (one per hex index).
            The caller should mask illegal actions and apply softmax.
        values : np.ndarray, shape ``(batch,)``, dtype float32
            Scalar values in [-1, 1] from each state's **current-player**
            perspective.
        """
        self.eval()
        device = next(self.parameters()).device

        # Build flat observations for all states
        obs_list = []
        for state in states:
            obs = state.get_observation()
            board = np.array(obs["board"], dtype=np.float32).flatten()      # (180,)
            pieces = np.array(obs["pieces"], dtype=np.float32).flatten()    # (24,)
            meta = np.array(
                [float(obs["current_player"]), float(obs["phase"])],
                dtype=np.float32,
            )  # (2,)
            obs_list.append(np.concatenate([board, pieces, meta]))          # (206,)

        batch = np.array(obs_list, dtype=np.float32)                        # (B, 206)

        # Forward pass (no_grad is applied by the decorator)
        x = torch.from_numpy(batch).to(device)
        logits, val_t = self.forward(x)

        # Return as numpy arrays on CPU
        logits_np = logits.cpu().numpy().astype(np.float32)                 # (B, 60)
        values_np = val_t.cpu().numpy().flatten().astype(np.float32)        # (B,)
        return logits_np, values_np

    # ------------------------------------------------------------------
    # Flat batch inference (for Rust bridge)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def evaluate_flat_batch(
        self,
        batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate a pre-built batch of flat observations (B, 206). non_blocking GPU transfer."""
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(batch).to(device, non_blocking=True)
        logits, val_t = self.forward(x)
        return logits.cpu().numpy().astype(np.float32), val_t.cpu().numpy().flatten().astype(np.float32)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stable_softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x)
        e_x = np.exp(x - x_max)
        return e_x / (e_x.sum() + 1e-30)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights to ``path``."""
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load model weights from ``path``."""
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.eval()


# =============================================================================
# ResNet-style network (with residual connections)
# =============================================================================

class _AlphaZeroResNetOriginal(nn.Module):
    """
    ResNet-style MLP with a residual skip connection around the middle layer.

    Architecture:
        Input(206) → FC(512) → BN → ReLU
            → FC(512) → BN → ReLU → + identity (residual)
            → FC(256) → BN → ReLU
        ├── Policy head: FC(256 → 60) → logits
        └── Value head:  FC(256 → 128) → ReLU → FC(128 → 1) → tanh

    The residual connection improves gradient flow and allows training
    deeper networks without vanishing/exploding gradients.

    NOTE: This is the ORIGINAL (legacy) architecture, preserved for
    backward compatibility with checkpoints saved before the configurable
    refactor.  New code should use ``AlphaZeroResNet`` (configurable).
    """

    # 架构标记，用于文件名区分
    arch_name = "resnet"

    def __init__(self, obs_dim: int = 272, action_dim: int = 60):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # ----- shared trunk with residual -----
        self.fc1 = nn.Linear(obs_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        # ----- policy head -----
        self.policy_fc = nn.Linear(256, action_dim)

        # ----- value head -----
        self.value_fc1 = nn.Linear(256, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Shared trunk
        x = F.relu(self.bn1(self.fc1(x)))       # (B, 512)
        identity = x
        x = F.relu(self.bn2(self.fc2(x)))       # (B, 512)
        x = x + identity                         # residual connection
        x = F.relu(self.bn3(self.fc3(x)))       # (B, 256)

        # Policy head
        policy_logits = self.policy_fc(x)        # (B, 60)

        # Value head
        v = F.relu(self.value_fc1(x))
        value = torch.tanh(self.value_fc2(v))    # (B, 1)

        return policy_logits, value

    # ------------------------------------------------------------------
    # Inference (identical to AlphaZeroNet)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, state) -> tuple[np.ndarray, float]:
        """Single-state evaluation. Same interface as AlphaZeroNet.evaluate."""
        self.eval()
        obs = state.get_observation()
        board = np.array(obs["board"], dtype=np.float32).flatten()
        pieces = np.array(obs["pieces"], dtype=np.float32).flatten()
        meta = np.array([float(obs["current_player"]), float(obs["phase"])], dtype=np.float32)
        flat = np.concatenate([board, pieces, meta]).astype(np.float32)
        device = next(self.parameters()).device
        x = torch.as_tensor(flat, dtype=torch.float32).unsqueeze(0).to(device)
        logits, val_t = self.forward(x)
        logits_np = logits[0].cpu().numpy().astype(np.float64)
        value = float(val_t[0, 0].cpu().numpy())
        legal = state.get_legal_actions()
        mask = np.zeros(self.action_dim, dtype=bool)
        mask[legal] = True
        masked_logits = np.where(mask, logits_np, -1e9)
        probs = self._stable_softmax(masked_logits).astype(np.float32)
        if probs.sum() < 1e-8:
            probs[:] = 0.0
            if legal:
                probs[legal] = 1.0 / len(legal)
        elif abs(probs.sum() - 1.0) > 1e-6:
            probs = probs / probs.sum()
        return probs, value

    @torch.inference_mode()
    def evaluate_batch(self, states: list) -> tuple[np.ndarray, np.ndarray]:
        """Batch inference. Same interface as AlphaZeroNet.evaluate_batch."""
        self.eval()
        device = next(self.parameters()).device
        obs_list = []
        for state in states:
            obs = state.get_observation()
            board = np.array(obs["board"], dtype=np.float32).flatten()
            pieces = np.array(obs["pieces"], dtype=np.float32).flatten()
            meta = np.array([float(obs["current_player"]), float(obs["phase"])], dtype=np.float32)
            obs_list.append(np.concatenate([board, pieces, meta]))
        batch = np.array(obs_list, dtype=np.float32)
        x = torch.from_numpy(batch).to(device)
        logits, val_t = self.forward(x)
        return logits.cpu().numpy().astype(np.float32), val_t.cpu().numpy().flatten().astype(np.float32)

    @torch.inference_mode()
    def evaluate_flat_batch(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pre-built flat observation batch. non_blocking GPU transfer."""
        self.eval()
        device = next(self.parameters()).device
        x = torch.from_numpy(batch).to(device, non_blocking=True)
        logits, val_t = self.forward(x)
        return logits.cpu().numpy().astype(np.float32), val_t.cpu().numpy().flatten().astype(np.float32)

    @staticmethod
    def _stable_softmax(x: np.ndarray) -> np.ndarray:
        x_max = np.max(x)
        e_x = np.exp(x - x_max)
        return e_x / (e_x.sum() + 1e-30)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu"))
        self.eval()

    def export_onnx(self, path: str, obs_dim: int = 272):
        """Export to ONNX for Rust-side inference (tract)."""
        self.eval()
        device = next(self.parameters()).device
        dummy = torch.randn(1, obs_dim, device=device)
        torch.onnx.export(
            self,
            dummy,
            path,
            input_names=["obs"],
            output_names=["policy_logits", "value"],
            dynamic_axes={
                "obs": {0: "batch"},
                "policy_logits": {0: "batch"},
                "value": {0: "batch"},
            },
            opset_version=19,
        )


# =============================================================================
# Architecture detection helper
# =============================================================================

class AlphaZeroResNetConfigurable(_AlphaZeroResNetOriginal):
    """
    Configurable-width AlphaZeroResNet with multiple residual blocks.

    Use `AlphaZeroResNetXL` (pre-configured) or create custom:
        net = AlphaZeroResNetConfigurable(hidden_dim=4096, num_blocks=4)

    Training memory estimate (fp16 AMP):
        params × 14 bytes (weights_fp16 + grads_fp32 + adam_fp32 + overhead)
        e.g. 320M params → ~4.5GB → fits RTX 4060 8GB
    """

    arch_name = "resnet_configurable"

    def __init__(self, obs_dim: int = 272, action_dim: int = 60,
                 hidden_dim: int = 512, num_blocks: int = 1,
                 value_dim: int = None):
        """
        value_dim: hidden size for value head (default: hidden_dim//4 for 1M/2M, hidden_dim//2 for 3M)
        """
        nn.Module.__init__(self)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        if value_dim is None:
            value_dim = max(64, hidden_dim // 4)
        self._value_dim = value_dim

        # ----- shared trunk with residual blocks -----
        self.fc_in = nn.Linear(obs_dim, hidden_dim)
        self.bn_in = nn.BatchNorm1d(hidden_dim)

        self.res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
            )
            self.res_blocks.append(block)

        self.fc_out = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_out = nn.BatchNorm1d(hidden_dim // 2)

        # ----- policy head -----
        self.policy_fc = nn.Linear(hidden_dim // 2, action_dim)

        # ----- value head -----
        self.value_fc1 = nn.Linear(hidden_dim // 2, self._value_dim)
        self.value_fc2 = nn.Linear(self._value_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Shared trunk
        x = F.relu(self.bn_in(self.fc_in(x)))

        for block in self.res_blocks:
            identity = x
            x = block[0](x)          # Linear
            x = block[1](x)          # BN
            x = block[2](x)          # ReLU
            x = block[3](x)          # Linear
            x = block[4](x)          # BN
            x = F.relu(x + identity)  # Residual + ReLU

        x = F.relu(self.bn_out(self.fc_out(x)))

        # Policy head
        policy_logits = self.policy_fc(x)

        # Value head
        v = F.relu(self.value_fc1(x))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


class AlphaZeroResNetXL(AlphaZeroResNetConfigurable):
    """313M params (hidden=8192, blocks=2). Requires 6GB GPU memory."""
    arch_name = "resnet_xl"

    def __init__(self, obs_dim: int = 272, action_dim: int = 60):
        super().__init__(obs_dim, action_dim, hidden_dim=8192, num_blocks=2)


class AlphaZeroResNet1M(AlphaZeroResNetConfigurable):
    """0.85M params. Default for backward compat with old models."""
    arch_name = "resnet_1m"

    def __init__(self, obs_dim: int = 272, action_dim: int = 60):
        super().__init__(obs_dim, action_dim, hidden_dim=512, num_blocks=1)


class AlphaZeroResNet2M(AlphaZeroResNetConfigurable):
    """1.9M params (hidden=512, blocks=3). Default for new training."""
    arch_name = "resnet_2m"

    def __init__(self, obs_dim: int = 272, action_dim: int = 60):
        super().__init__(obs_dim, action_dim, hidden_dim=512, num_blocks=3)


class AlphaZeroResNet3M(AlphaZeroResNetConfigurable):
    """
    3.0M params (hidden=512, blocks=5, value_hidden=256).
    Balanced depth/width with larger value head for intermediate-scale training.
    """
    arch_name = "resnet_3m"

    def __init__(self, obs_dim: int = 272, action_dim: int = 60):
        super().__init__(obs_dim, action_dim, hidden_dim=512, num_blocks=5, value_dim=256)


class AlphaZeroResNetXL(AlphaZeroResNetConfigurable):
    """
    313M params (hidden=8192, blocks=2).
    Requires 6GB GPU memory. For large-scale training only.
    """
    arch_name = "resnet_xl"

    def __init__(self, obs_dim: int = 272, action_dim: int = 60):
        super().__init__(obs_dim, action_dim, hidden_dim=8192, num_blocks=2)


# Backward compat aliases
AlphaZeroResNet = AlphaZeroResNet1M
AlphaZeroResNetLarge = AlphaZeroResNet3M


def detect_net_arch(state_dict) -> type:
    """
    Detect network architecture from state dict keys.

    Detection rules:
    - Has `res_blocks.0.0.weight` → JSON-able configurable
    - Has `fc3.*` keys → original ResNet classes
        - Check `fc_in.weight` / `fc1.weight` for size
    - Otherwise → AlphaZeroNet (MLP)
    """
    # New configurable architecture
    if any(k.startswith("res_blocks.") for k in state_dict.keys()):
        fc_in_w = state_dict.get("fc_in.weight")
        if fc_in_w is not None:
            h = fc_in_w.shape[0]
            if h >= 8192:
                return AlphaZeroResNetXL  # 313M params
            elif h >= 1024:
                return AlphaZeroResNet3M  # 3M params
            n_blocks = sum(1 for k in state_dict if k.startswith("res_blocks.") and k.endswith("0.weight"))
            if n_blocks >= 3:
                return AlphaZeroResNet2M  # 2M params
            return AlphaZeroResNet1M  # 1M params
        return AlphaZeroResNet2M

    # Legacy ResNet classes (fc1/bn1/fc2/bn2/fc3/bn3 keys)
    if any(k.startswith("fc3.") for k in state_dict.keys()):
        fc1_w = state_dict.get("fc1.weight")
        if fc1_w is not None and fc1_w.shape[0] >= 1024:
            return AlphaZeroResNet3M
        return _AlphaZeroResNetOriginal
    return AlphaZeroNet
