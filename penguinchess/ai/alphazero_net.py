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

    def __init__(self, obs_dim: int = 206, action_dim: int = 60):
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
        x = torch.as_tensor(flat, dtype=torch.float32).unsqueeze(0)  # (1, 206)
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
