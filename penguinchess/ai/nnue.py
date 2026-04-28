"""
NNUE (Efficiently Updatable Neural Network) for PenguinChess.

Architecture:
  Input: 360-dim sparse binary (PieceHex features)
  Feature Transformer: 360 × ft_dim = 64 (int16 accumulator)
  Dense features: 66-dim (hex values + metadata)
  FC1: (128 + 66) → 64  (128 = 2 perspectives × 64 ft_dim)
  FC2: 64 → 32
  FC3: 32 → 1  (output in [-1, 1])

During search:
  - Accumulator maintains incrementally updated (removed/added features)
  - Only recompute FC1-FC3 when accumulator changes
  - Dense features still need full recompute (they change globally)

For training:
  - Full forward pass (no accumulator optimization)
  - MSE loss on game outcomes
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from penguinchess.core import PenguinChessCore
from penguinchess.ai.sparse_features import (
    HEX_COUNT, PIECE_COUNT, PIECE_HEX_DIM, DENSE_DIM,
)
# state_to_features and extractors imported inline to avoid issues

# ─── Default sizes ────────────────────────────────────────────
FT_DIM = 64       # Feature transformer dimension
HIDDEN_DIM = 64   # Hidden layer size


class NNUEAccumulator:
    """
    Incrementally updated NNUE feature accumulator.
    
    Maintains stm (current player) and nstm (opponent) accumulators.
    When a piece moves: subtract old piece-hex feature, add new one.
    
    The accumulator represents the Feature Transformer output,
    NOT the full hidden activations. FC layers are still computed
    from scratch.
    """

    def __init__(self, ft_weight: torch.Tensor, ft_bias: torch.Tensor):
        """
        Args:
            ft_weight: (PIECE_HEX_DIM, FT_DIM) float32
            ft_bias: (FT_DIM,) float32
        """
        self.ft_weight = ft_weight.detach().cpu().numpy().astype(np.float32)
        self.ft_bias = ft_bias.detach().cpu().numpy().astype(np.float32)
        # acc[0] = stm perspective, acc[1] = nstm perspective
        self.acc = np.zeros((2, FT_DIM), dtype=np.float32)
        self.stm_player = 0

    def reset(self, sparse_indices: list[int], stm_player: int) -> None:
        """Full recompute from sparse features."""
        self.acc[:] = self.ft_bias[np.newaxis, :]
        for f in sparse_indices:
            self.acc[0] += self.ft_weight[f]
            # For nstm, the same piece is on the board but from the other
            # player's perspective. In PenguinChess, piece ownership matters,
            # so nstm uses the same sparse features (both perspectives see
            # the same board but with swapped ownership).
            # For simplicity, both perspectives share the same accumulator.
            self.acc[1] += self.ft_weight[f]
        self.stm_player = stm_player

    def apply_diff(self, removed: list[int], added: list[int]) -> None:
        """Incremental update: subtract removed features, add new ones."""
        for f in removed:
            self.acc[0] -= self.ft_weight[f]
            self.acc[1] -= self.ft_weight[f]
        for f in added:
            self.acc[0] += self.ft_weight[f]
            self.acc[1] += self.ft_weight[f]

    def get_crelu(self) -> np.ndarray:
        """
        Get concatenated CReLU(acc_stm, acc_nstm).
        Returns (128,) float32.
        """
        stm = np.clip(self.acc[0], 0.0, 127.0)
        nstm = np.clip(self.acc[1], 0.0, 127.0)
        return np.concatenate([stm, nstm]).astype(np.float32)

    def copy_from(self, other: 'NNUEAccumulator') -> None:
        """Copy state from another accumulator (for search branching)."""
        self.acc[:] = other.acc
        self.stm_player = other.stm_player


class NNUE(nn.Module):
    """
    NNUE model for PenguinChess.
    
    Training: forward(sparse_batch, dense_batch) → value predictions
    Search: evaluate_from_acc(acc, dense) or evaluate(core)
    """

    def __init__(self, ft_dim: int = FT_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        
        # Feature Transformer
        self.ft = nn.Linear(PIECE_HEX_DIM, ft_dim, bias=True)
        # Note: ft.weight shape is (ft_dim, PIECE_HEX_DIM) per nn.Linear convention.
        # We'll transpose when doing sparse gather.
        
        # Shared hidden layers after concatenating (acc_stm || acc_nstm || dense)
        total_input = ft_dim * 2 + DENSE_DIM  # 128 + 66 = 194
        self.fc1 = nn.Linear(total_input, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        # Initialize with small weights
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def ft_weight_gather(self) -> torch.Tensor:
        """
        Get ft.weight in gather-friendly shape (PIECE_HEX_DIM, ft_dim).
        nn.Linear stores weight as (out_features, in_features).
        """
        return self.ft.weight.t()  # (PIECE_HEX_DIM, ft_dim)

    @property
    def ft_bias(self) -> torch.Tensor:
        return self.ft.bias  # (ft_dim,)

    def _forward_sparse_single(
        self,
        sparse_indices: list[int],
        dense: torch.Tensor,
        stm_player: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass for a single sample using sparse gather.
        
        Args:
            sparse_indices: list of active feature indices
            dense: (DENSE_DIM,) tensor of dense features
            stm_player: 0 = P1, 1 = P2
            
        Returns:
            scalar tensor in [-1, 1]
        """
        ft_w = self.ft_weight_gather  # (360, 64)
        ft_b = self.ft_bias           # (64,)
        
        acc = ft_b.clone()
        if sparse_indices:
            idx = torch.tensor(sparse_indices, dtype=torch.long, device=ft_w.device)
            acc = acc + ft_w[idx].sum(dim=0)
        
        # Both stm and nstm use same board state
        acc_stm = F.relu(acc)
        acc_nstm = F.relu(acc)
        
        x = torch.cat([acc_stm, acc_nstm, dense], dim=-1)  # (194,)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x.squeeze(-1)

    def forward(
        self,
        sparse_batch: list[list[int]],
        dense_batch: torch.Tensor,
        stm_players: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        Batch forward pass for training.
        
        Args:
            sparse_batch: list of list[int], each is sparse indices for one sample
            dense_batch: (B, DENSE_DIM) float32 tensor
            stm_players: optional list of 0/1 values
            
        Returns:
            (B,) tensor of value predictions
        """
        B = dense_batch.shape[0]
        ft_w = self.ft_weight_gather  # (360, 64)
        ft_b = self.ft_bias           # (64,)
        device = ft_w.device
        
        # Build accumulator for each sample in batch
        acc_batch = ft_b.unsqueeze(0).expand(B, -1).clone()  # (B, 64)
        
        for i in range(B):
            idx = sparse_batch[i]
            if idx:
                idx_t = torch.tensor(idx, dtype=torch.long, device=device)
                acc_batch[i] = acc_batch[i] + ft_w[idx_t].sum(dim=0)
        
        acc_crelu = F.relu(acc_batch)  # (B, 64)
        x = torch.cat([acc_crelu, acc_crelu, dense_batch], dim=-1)  # (B, 194)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x.squeeze(-1)

    def evaluate_from_acc(
        self,
        acc: NNUEAccumulator,
        dense: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass from pre-computed accumulator (used during search).
        
        Args:
            acc: NNUEAccumulator
            dense: (DENSE_DIM,) tensor
            
        Returns:
            scalar tensor in [-1, 1]
        """
        acc_vec = torch.from_numpy(acc.get_crelu()).to(dense.device).float()
        x = torch.cat([acc_vec, dense], dim=-1)  # (194,)
        
        with torch.no_grad():
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.tanh(self.fc3(x))
        return x.squeeze(-1)

    def evaluate(self, core: PenguinChessCore) -> float:
        """
        Evaluate a game state from scratch.
        Returns value in [-1, 1] from current player's perspective.
        """
        from penguinchess.ai.sparse_features import state_to_features
        sparse, dense = state_to_features(core)
        dense_t = torch.from_numpy(dense).float()
        
        # Convert to batched format
        if not hasattr(self, '_device'):
            self._device = next(self.parameters()).device
        
        sparse_batch = [sparse]
        dense_batch = dense_t.unsqueeze(0)
        
        with torch.no_grad():
            val = self.forward(sparse_batch, dense_batch.to(self._device))
        return val.item()

    def evaluate_batch(
        self,
        states: list[tuple[list[int], np.ndarray]],
    ) -> np.ndarray:
        """
        Batch evaluate multiple states.
        
        Args:
            states: list of (sparse_indices, dense_array) tuples
            
        Returns:
            (N,) numpy array of values
        """
        if not states:
            return np.array([], dtype=np.float32)
        
        sparse_batch = [s[0] for s in states]
        dense_batch = torch.from_numpy(np.stack([s[1] for s in states])).float()
        
        with torch.no_grad():
            val = self.forward(sparse_batch, dense_batch.to(self._device))
        return val.cpu().numpy()

    def create_accumulator(self) -> NNUEAccumulator:
        """Create an accumulator for this model."""
        return NNUEAccumulator(self.ft_weight_gather, self.ft_bias)


# Architecture name for model registry
ARCH_NAME = "nnue"
