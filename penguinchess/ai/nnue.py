"""
NNUE (Efficiently Updatable Neural Network) for PenguinChess.

Architecture:
  Input: 360-dim sparse binary (PieceHex features)
  Feature Transformer: 360 × ft_dim = 64 (int16 accumulator)
  Dense features: 66-dim (hex values + metadata)
  FC1: (128 + 66) → 128  (128 = 2 perspectives × 64 ft_dim)
  FC2: 128 → 64
  FC3: 64 → 1  (output in [-1, 1])

Key: stm and nstm accumulators are split by piece ownership.
  P1 pieces use sparse indices 0-179, P2 use 180-359.
  stm = current player's pieces, nstm = opponent's pieces.
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

FT_DIM = 128
HIDDEN_DIM = 512

# P1 features: indices 0-179 (pieces 0,1,2 × 60 hexes)
# P2 features: indices 180-359 (pieces 3,4,5 × 60 hexes)
P1_FEATURE_CUTOFF = PIECE_COUNT // 2 * HEX_COUNT  # 3 * 60 = 180


def _split_stm_nstm(
    sparse_indices: list[int],
    stm_player: int,
) -> tuple[list[int], list[int]]:
    """Split sparse indices into stm and nstm by ownership."""
    if not sparse_indices:
        return [], []
    stm = []
    nstm = []
    for f in sparse_indices:
        is_p1 = f < P1_FEATURE_CUTOFF
        if stm_player == 0:  # P1 to move
            if is_p1:
                stm.append(f)
            else:
                nstm.append(f)
        else:  # P2 to move
            if is_p1:
                nstm.append(f)
            else:
                stm.append(f)
    return stm, nstm


class NNUEAccumulator:
    """
    Incrementally updated NNUE feature accumulator.
    Maintains separate stm (current player) and nstm (opponent) accumulators.
    """

    def __init__(self, ft_weight: torch.Tensor, ft_bias: torch.Tensor):
        self.ft_weight = ft_weight.detach().cpu().numpy().astype(np.float32)
        self.ft_bias = ft_bias.detach().cpu().numpy().astype(np.float32)
        self.acc = np.zeros((2, FT_DIM), dtype=np.float32)  # [0]=stm, [1]=nstm
        self.stm_player = 0

    def reset(self, sparse_indices: list[int], stm_player: int) -> None:
        """Full recompute from sparse features, split by ownership."""
        self.acc[:] = self.ft_bias[np.newaxis, :]
        stm_idx, nstm_idx = _split_stm_nstm(sparse_indices, stm_player)
        for f in stm_idx:
            self.acc[0] += self.ft_weight[f]
        for f in nstm_idx:
            self.acc[1] += self.ft_weight[f]
        self.stm_player = stm_player

    def apply_diff(self, removed: list[int], added: list[int], stm_player: int) -> None:
        """Incremental update: subtract removed features, add new ones."""
        rem_stm, rem_nstm = _split_stm_nstm(removed, stm_player)
        add_stm, add_nstm = _split_stm_nstm(added, stm_player)
        for f in rem_stm:
            self.acc[0] -= self.ft_weight[f]
        for f in rem_nstm:
            self.acc[1] -= self.ft_weight[f]
        for f in add_stm:
            self.acc[0] += self.ft_weight[f]
        for f in add_nstm:
            self.acc[1] += self.ft_weight[f]

    def get_crelu(self) -> np.ndarray:
        """Get concatenated CReLU(acc_stm, acc_nstm). Returns (128,) float32."""
        stm = np.clip(self.acc[0], 0.0, 127.0)
        nstm = np.clip(self.acc[1], 0.0, 127.0)
        return np.concatenate([stm, nstm]).astype(np.float32)

    def copy_from(self, other: 'NNUEAccumulator') -> None:
        self.acc[:] = other.acc
        self.stm_player = other.stm_player


class NNUE(nn.Module):
    """
    NNUE model for PenguinChess.
    Training: forward(sparse_batch, dense_batch, stm_players) → value predictions
    Search: evaluate_from_acc(acc, dense) or evaluate(core)
    """

    def __init__(self, ft_dim: int = FT_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.ft_dim = ft_dim

        self.ft = nn.Linear(PIECE_HEX_DIM, ft_dim, bias=True)
        total_input = ft_dim * 2 + DENSE_DIM  # 128 + 66 = 194
        self.fc1 = nn.Linear(total_input, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def ft_weight_gather(self) -> torch.Tensor:
        """(PIECE_HEX_DIM, ft_dim) for sparse gather."""
        return self.ft.weight.t()

    @property
    def ft_bias(self) -> torch.Tensor:
        return self.ft.bias

    def _forward_sparse_single(
        self,
        sparse_indices: list[int],
        dense: torch.Tensor,
        stm_player: int = 0,
    ) -> torch.Tensor:
        """Forward pass for a single sample with ownership split."""
        ft_w = self.ft_weight_gather
        ft_b = self.ft_bias
        
        stm_idx, nstm_idx = _split_stm_nstm(sparse_indices, stm_player)
        
        stm_acc = ft_b.clone()
        if stm_idx:
            idx = torch.tensor(stm_idx, dtype=torch.long, device=ft_w.device)
            stm_acc = stm_acc + ft_w[idx].sum(dim=0)
        
        nstm_acc = ft_b.clone()
        if nstm_idx:
            idx = torch.tensor(nstm_idx, dtype=torch.long, device=ft_w.device)
            nstm_acc = nstm_acc + ft_w[idx].sum(dim=0)
        
        x = torch.cat([F.relu(stm_acc), F.relu(nstm_acc), dense], dim=-1)
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
        Batch forward pass — embedding-based sparse accumulation.
        Builds padded index tensors for stm and nstm features,
        then uses ft_weight[indices].sum(dim=1) for O(1) GPU gather.
        """
        B = dense_batch.shape[0]
        device = dense_batch.device
        ft_w = self.ft_weight_gather  # (360, ft_dim)
        ft_b = self.ft_bias           # (ft_dim,)

        if stm_players is None:
            stm_players = [0] * B

        # Build padded index tensors for stm and nstm features
        # Each sample has at most 3 pieces per player
        stm_idx = torch.zeros(B, 3, dtype=torch.long, device=device)
        nstm_idx = torch.zeros(B, 3, dtype=torch.long, device=device)

        for i in range(B):
            sp = sparse_batch[i]
            if not sp:
                continue
            stm = stm_players[i]
            si = 0
            ni = 0
            for f in sp:
                is_p1 = f < P1_FEATURE_CUTOFF
                is_stm = (stm == 0 and is_p1) or (stm == 1 and not is_p1)
                if is_stm and si < 3:
                    stm_idx[i, si] = f
                    si += 1
                elif not is_stm and ni < 3:
                    nstm_idx[i, ni] = f
                    ni += 1

        # Feature Transformer: single embedding gather per perspective
        # ft_w[stm_idx]: (B, 3, ft_dim) → sum(dim=1): (B, ft_dim)
        acc_stm = ft_w[stm_idx].sum(dim=1) + ft_b  # (B, ft_dim)
        acc_nstm = ft_w[nstm_idx].sum(dim=1) + ft_b

        x = torch.cat([F.relu(acc_stm), F.relu(acc_nstm), dense_batch], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x.squeeze(-1)

    def evaluate_from_acc(
        self,
        acc: NNUEAccumulator,
        dense: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass from pre-computed accumulator (used during search)."""
        acc_vec = torch.from_numpy(acc.get_crelu()).to(dense.device).float()
        x = torch.cat([acc_vec, dense], dim=-1)
        with torch.no_grad():
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.tanh(self.fc3(x))
        return x.squeeze(-1)

    def evaluate(self, core) -> float:
        """Evaluate a game state from scratch. Returns value in [-1, 1]."""
        from penguinchess.ai.sparse_features import state_to_features
        sparse, dense = state_to_features(core)
        dense_t = torch.from_numpy(dense).float()
        if not hasattr(self, '_device'):
            self._device = next(self.parameters()).device
        stm = core.current_player if hasattr(core, 'current_player') else 0
        with torch.no_grad():
            val = self._forward_sparse_single(
                sparse, dense_t.to(self._device), stm_player=stm)
        return val.item()

    def evaluate_batch(
        self,
        states: list[tuple[list[int], np.ndarray, int]],
    ) -> np.ndarray:
        """Batch evaluate with stm_player info. Each state is (sparse, dense, stm_player)."""
        if not states:
            return np.array([], dtype=np.float32)
        sparse_batch = [s[0] for s in states]
        dense_batch = torch.from_numpy(np.stack([s[1] for s in states])).float()
        stm_batch = [s[2] if len(s) > 2 else 0 for s in states]
        with torch.no_grad():
            val = self.forward(sparse_batch, dense_batch.to(self._device), stm_players=stm_batch)
        return val.cpu().numpy()

    def create_accumulator(self) -> NNUEAccumulator:
        return NNUEAccumulator(self.ft_weight_gather, self.ft_bias)


ARCH_NAME = "nnue"
