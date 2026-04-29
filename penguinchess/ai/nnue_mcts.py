"""
NNUE-MCTS: NNUE with policy head for AlphaZero-style MCTS training.

Architecture:
  FT (360→64) + CReLU(stm||nstm) → FC1 (194→256) + CReLU
    ├─ Value Head:  FC2v (256→1) + tanh
    └─ Policy Head: FC2p (256→60)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from typing import Optional

from penguinchess.ai.sparse_features import HEX_COUNT, PIECE_COUNT, PIECE_HEX_DIM, DENSE_DIM
from penguinchess.ai.nnue import FT_DIM, HIDDEN_DIM, P1_FEATURE_CUTOFF, _split_stm_nstm, NNUEAccumulator

class NNUEMCTSModel(nn.Module):
    """
    NNUE with shared trunk + value head + policy head.
    Compatible with existing NNUE weights for the trunk.
    """

    def __init__(self, ft_dim: int = FT_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.ft_dim = ft_dim
        self.ft = nn.Linear(PIECE_HEX_DIM, ft_dim, bias=True)
        total_input = ft_dim * 2 + DENSE_DIM  # 194
        self.fc1 = nn.Linear(total_input, hidden_dim)
        self.fc2v = nn.Linear(hidden_dim, 1)          # value head
        self.fc2p = nn.Linear(hidden_dim, HEX_COUNT)  # policy head → 60 logits
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Policy head final layer: zero init for uniform prior
        nn.init.zeros_(self.fc2p.weight)
        nn.init.zeros_(self.fc2p.bias)

    @property
    def ft_weight_gather(self) -> torch.Tensor:
        return self.ft.weight.t()

    @property
    def ft_bias(self) -> torch.Tensor:
        return self.ft.bias

    def shared_trunk(self, sparse_batch, dense_batch: torch.Tensor,
                     stm_players=None):
        """
        Shared trunk: FT → CReLU → FC1 → CReLU → (policy_head, value_head)
        Accepts: sparse_batch as (B, 6) padded tensor OR list of lists.
        """
        device = dense_batch.device
        ft_w = self.ft_weight_gather
        ft_b = self.ft_bias

        # Convert list of lists to padded tensor if needed
        if isinstance(sparse_batch, list):
            B = dense_batch.shape[0]
            padded = torch.zeros(B, 6, dtype=torch.long, device=device)
            for i, sp in enumerate(sparse_batch):
                if sp:
                    padded[i, :len(sp)] = torch.tensor(sp, dtype=torch.long, device=device)
            sparse_batch = padded

        B, _ = sparse_batch.shape

        if stm_players is None:
            stm_players = torch.zeros(B, dtype=torch.long, device=device)
        elif isinstance(stm_players, list):
            stm_players = torch.tensor(stm_players, dtype=torch.long, device=device)

        # Vectorized: compute stm/nstm masks from padded sparse tensor
        is_p1 = sparse_batch < P1_FEATURE_CUTOFF  # (B, 6)
        stm_mask = (stm_players[:, None] == 0) & is_p1 | (stm_players[:, None] == 1) & ~is_p1
        is_valid = sparse_batch >= 0
        nstm_mask = ~stm_mask & is_valid  # opponent features

        # Rank features within each sample (0, 1, 2, ...)
        stm_rank = stm_mask.cumsum(dim=1).float() - 1.0
        stm_rank[~stm_mask] = 999.0
        nstm_rank = nstm_mask.cumsum(dim=1).float() - 1.0
        nstm_rank[~nstm_mask] = 999.0

        # Select first 3 features per player (sorted by rank)
        _, stm_idx_sorted = stm_rank.sort(dim=1)
        _, nstm_idx_sorted = nstm_rank.sort(dim=1)

        stm_idx = torch.gather(sparse_batch, 1, stm_idx_sorted[:, :3])
        nstm_idx = torch.gather(sparse_batch, 1, nstm_idx_sorted[:, :3])

        # Zero out invalid indices (rank >= 3 or no valid feature)
        stm_rank_top3 = stm_rank.gather(1, stm_idx_sorted[:, :3])
        nstm_rank_top3 = nstm_rank.gather(1, nstm_idx_sorted[:, :3])
        stm_idx[stm_rank_top3 >= 3.0] = 0
        nstm_idx[nstm_rank_top3 >= 3.0] = 0
        stm_idx[stm_rank_top3 < 0] = 0
        nstm_idx[nstm_rank_top3 < 0] = 0

        # Feature Transformer gather (B, 3, ft_dim) → sum → (B, ft_dim)
        acc_stm = ft_w[stm_idx.clamp(min=0)].sum(dim=1) + ft_b
        acc_nstm = ft_w[nstm_idx.clamp(min=0)].sum(dim=1) + ft_b

        h = torch.cat([F.relu(acc_stm), F.relu(acc_nstm), dense_batch], dim=-1)
        h = F.relu(self.fc1(h))

        value = torch.tanh(self.fc2v(h)).squeeze(-1)
        policy_logits = self.fc2p(h)
        return policy_logits, value, h

    def forward(self, sparse_batch: list[list[int]], dense_batch: torch.Tensor,
                stm_players: Optional[list[int]] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy_logits_60, value_1)."""
        logits, value, _ = self.shared_trunk(sparse_batch, dense_batch, stm_players)
        return logits, value

    def evaluate_mcts(self, state_features: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Single-position eval for MCTS leaf nodes.
        state_features: dict with 'sparse', 'dense', 'player' keys.
        Returns: (policy_probs_60, value).
        """
        self.eval()
        with torch.no_grad():
            sp = [state_features['sparse']]
            de = torch.from_numpy(state_features['dense']).unsqueeze(0).float()
            stm = [state_features['player']]
            logits, val = self.forward(sp, de, stm)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            value = val.item()
        return probs, value

    def load_nnue_trunk(self, nnue_state_dict: dict):
        """Load pre-trained NNUE weights for the shared trunk (FT + FC1)."""
        mapping = {
            'ft.weight': 'ft.weight', 'ft.bias': 'ft.bias',
            'fc1.weight': 'fc1.weight', 'fc1.bias': 'fc1.bias',
        }
        own = self.state_dict()
        for old_key, new_key in mapping.items():
            if old_key in nnue_state_dict and new_key in own:
                own[new_key].copy_(nnue_state_dict[old_key])
        self.load_state_dict(own)

    def make_eval_fn(self, device: str = 'cuda'):
        """
        Returns an EvalFn-compatible callback for Rust MCTS (obs_dim=75).
        Input: (batch, 75) = [8 sparse: f32, 66 dense: f32, 1 stm: f32]
        Output: (batch, 61) = [60 logits: f32, 1 value: f32]
        """
        self.to(device).eval()

        def eval_fn(obs_ptr, batch_size, output_ptr, output_capacity):
            try:
                n = batch_size
                obs_np = np.frombuffer(
                    torch.from_numpy(np.ctypeslib.as_array(obs_ptr, shape=(n, 75))).cpu().numpy()
                    if False else np.ctypeslib.as_array(obs_ptr, shape=(n, 75)),
                    dtype=np.float32
                ).reshape(n, 75)

                sparse_raw = obs_np[:, :8].astype(np.int64)
                dense = torch.from_numpy(obs_np[:, 8:74].copy()).to(device)
                stm_list = [int(x) for x in obs_np[:, 74]]
                sparse_batch = [[int(x) for x in row if x >= 0] for row in sparse_raw]

                with torch.no_grad():
                    logits, values = self.forward(sparse_batch, dense, stm_list)

                out = np.zeros((n, 61), dtype=np.float32)
                out[:, :60] = logits.cpu().numpy()
                out[:, 60] = values.cpu().numpy()

                out_ptr = np.ctypeslib.as_array(output_ptr, shape=(n * 61,))
                out_ptr[:] = out.ravel()
                return 0
            except Exception as e:
                print(f"[NNUE MCTS eval error] {e}")
                import traceback; traceback.print_exc()
                return -1

        return eval_fn
