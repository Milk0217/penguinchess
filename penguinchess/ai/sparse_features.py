"""
Sparse binary feature extraction for NNUE evaluation.

Feature encoding for PenguinChess:

PieceHex (sparse, 360 dims):
  (piece_idx × 60 + hex_idx) → active if piece_idx is alive at hex_idx
  At most 6 active at any time.
  Changes per move: 2 (old hex removed, new hex added)

Dense (66 dims):
  hex_values[0:60] — normalized hex value (0.0 = eliminated, 1/3, 2/3, 1.0)
  meta[0:6] — [scores_p1/100, scores_p2/100, phase, alive_p1/3, alive_p2/3, episode_steps/500]
"""

import json
import numpy as np
from typing import Optional

from penguinchess.core import PenguinChessCore

HEX_COUNT = 60
PIECE_COUNT = 6
PIECE_HEX_DIM = PIECE_COUNT * HEX_COUNT  # 360
DENSE_DIM = 66

# Piece index → piece ID mappings (from core.py)
# P1: [4, 6, 8], P2: [5, 7, 9]
PIECE_IDS = [4, 6, 8, 5, 7, 9]

# Phase encoding
PHASE_PLACEMENT = 0.0
PHASE_MOVEMENT = 0.5


def _piece_idx(piece_id: int) -> int:
    """Map piece ID (4..9) to index (0..5)."""
    return PIECE_IDS.index(piece_id)


def _piece_hex_feature(piece_idx: int, hex_idx: int) -> int:
    """Encode (piece_idx, hex_idx) → sparse feature index [0, 360)."""
    return piece_idx * HEX_COUNT + hex_idx


def extract_sparse(core) -> list[int]:
    """
    Extract sparse binary features from game state.
    Works with both PenguinChessCore and RustCore.
    """
    if hasattr(core, 'pieces'):
        # PenguinChessCore path
        features = []
        for piece in core.pieces:
            if piece.alive and piece.hex is not None:
                try:
                    hex_idx = core.hexes.index(piece.hex)
                except ValueError:
                    continue
                p_idx = _piece_idx(piece.id)
                features.append(_piece_hex_feature(p_idx, hex_idx))
        return features
    else:
        # RustCore path: use to_json()
        return extract_sparse_from_json(core.to_json())


def extract_sparse_from_json(json_str: str) -> list[int]:
    """Extract sparse features from RustCore JSON string."""
    data = json.loads(json_str) if isinstance(json_str, str) else json_str
    features = []
    for piece in data.get("pieces", []):
        if piece.get("alive", False) and piece.get("hex", -1) >= 0:
            pid = piece["id"]
            hex_idx = piece["hex"]
            try:
                p_idx = _piece_idx(pid)
                features.append(_piece_hex_feature(p_idx, hex_idx))
            except ValueError:
                continue
    return features


def extract_dense(core) -> np.ndarray:
    """Extract dense features, works with both PenguinChessCore and RustCore."""
    if hasattr(core, 'hexes'):
        return _extract_dense_py(core)
    else:
        return _extract_dense_rust(core)


def _extract_dense_py(core: PenguinChessCore) -> np.ndarray:
    """Extract dense features from PenguinChessCore."""
    hex_values = np.zeros(HEX_COUNT, dtype=np.float32)
    for i, h in enumerate(core.hexes):
        if h.state in ('used', 'eliminated'):
            hex_values[i] = 0.0
        elif h.state == 'occupied':
            hex_values[i] = 0.0
        else:
            hex_values[i] = h.points / 3.0

    meta = np.zeros(6, dtype=np.float32)
    meta[0] = core.players_scores[0] / 100.0
    meta[1] = core.players_scores[1] / 100.0
    meta[2] = PHASE_PLACEMENT if core.phase == 'placement' else PHASE_MOVEMENT
    meta[3] = sum(1 for p in core.pieces[:3] if p.alive) / 3.0
    meta[4] = sum(1 for p in core.pieces[3:] if p.alive) / 3.0
    meta[5] = getattr(core, '_episode_steps', 0) / 500.0

    return np.concatenate([hex_values, meta]).astype(np.float32)


def _extract_dense_rust(core) -> np.ndarray:
    """Extract dense features from RustCore."""
    import json
    data = json.loads(core.to_json())
    
    hex_values = np.zeros(HEX_COUNT, dtype=np.float32)
    cells = data.get("board", {}).get("cells", [])
    for i, cell in enumerate(cells):
        if i < HEX_COUNT:
            pts = cell.get("points", 0)
            state = cell.get("state", "")
            if state in ("used", "eliminated"):
                hex_values[i] = 0.0
            elif state == "occupied":
                hex_values[i] = 0.0
            else:
                hex_values[i] = pts / 3.0
    
    meta = np.zeros(6, dtype=np.float32)
    meta[0] = data.get("scores", [0, 0])[0] / 100.0
    meta[1] = data.get("scores", [0, 0])[1] / 100.0
    meta[2] = PHASE_PLACEMENT if data.get("phase", "placement") == "placement" else PHASE_MOVEMENT
    
    pieces = data.get("pieces", [])
    p1_alive = sum(1 for p in pieces if p.get("alive") and p["id"] % 2 == 0)
    p2_alive = sum(1 for p in pieces if p.get("alive") and p["id"] % 2 == 1)
    meta[3] = p1_alive / 3.0
    meta[4] = p2_alive / 3.0
    meta[5] = data.get("episode_steps", 0) / 500.0
    
    return np.concatenate([hex_values, meta]).astype(np.float32)


def compute_sparse_diff(
    before_sparse: list[int],
    after_sparse: list[int],
) -> tuple[list[int], list[int]]:
    """
    Compute what sparse features changed between two states.
    
    Returns (removed_features, added_features).
    removed_features: indices that were active before but not after
    added_features: indices active after but not before
    """
    before_set = set(before_sparse)
    after_set = set(after_sparse)
    removed = list(before_set - after_set)
    added = list(after_set - before_set)
    return removed, added


def state_to_features(core) -> tuple[list[int], np.ndarray]:
    """Convenience: returns (sparse_indices, dense_array). Works with both core types."""
    return extract_sparse(core), extract_dense(core)
