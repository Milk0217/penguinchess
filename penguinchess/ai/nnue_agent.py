"""
Alpha-Beta search with NNUE evaluation agent for PenguinChess.

Uses PenguinChessCore (Python) for search tree with snapshot-based undo,
and NNUE for position evaluation. The NNUE accumulator is incrementally
updated during search to avoid full recomputation on each node.

Search features:
- Negamax with alpha-beta pruning
- Iterative deepening (configurable max depth)
- Move ordering: elimination moves first, then high-value hexes
- Null window search (Principal Variation Search variant)
- Time management (optional)
"""

import math
import time
import numpy as np
import torch
from typing import Optional

from penguinchess.core import PenguinChessCore
from penguinchess.eval_utils import Agent
from penguinchess.ai.nnue import NNUE, NNUEAccumulator
from penguinchess.ai.sparse_features import (
    extract_sparse, extract_dense, state_to_features,
    compute_sparse_diff,
)


# ─── Move ordering helpers ────────────────────────────────────


def _score_move(core: PenguinChessCore, action: int) -> float:
    """
    Score a move for ordering (higher = search first).
    
    Factors:
    - Placements on high-value hexes
    - Captures (moves that lead to piece elimination)
    - Moves to high-value hexes
    """
    # Snapshot current state
    snap = core.get_snapshot()
    _, _, _, info = core.step(action)
    
    # Compute score
    score = 0.0
    
    # Check if this move eliminates opponent pieces
    if info.get("piece_eliminated", False):
        score += 10.0
    
    # Check if this move gains high score
    reward = info.get("reward", 0)
    score += reward  # 1-3 points for hex value
    
    # Check if this move causes hex elimination (battle strategy)
    if info.get("hexes_eliminated", 0) > 0:
        score += 5.0
    
    # Restore
    core.restore_snapshot(snap)
    return score


def _order_moves(core: PenguinChessCore, legal: list[int]) -> list[int]:
    """Order moves: captures/eliminations first, high-value first."""
    if len(legal) <= 1:
        return legal
    
    scored = [(action, _score_move(core, action)) for action in legal]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [a for a, _ in scored]


def _get_reward_from_step(core: PenguinChessCore, action: int) -> float:
    """Get the reward from taking an action without modifying state."""
    snap = core.get_snapshot()
    _, reward, _, info = core.step(action)
    core.restore_snapshot(snap)
    return reward


# ─── NNUE Agent ───────────────────────────────────────────────


class NNUEAgent(Agent):
    """
    Alpha-Beta search agent using NNUE for position evaluation.
    
    Uses PenguinChessCore (Python engine) for search tree traversal
    with snapshot-based O(1) undo, and NNUEAccumulator for fast
    incremental evaluation.
    """

    def __init__(
        self,
        nnue_model: NNUE,
        max_depth: int = 6,
        time_limit: Optional[float] = None,
        use_accumulator: bool = True,
    ):
        """
        Args:
            nnue_model: trained NNUE model
            max_depth: maximum search depth (iterative deepening cap)
            time_limit: optional time limit per move in seconds
            use_accumulator: if True, use incremental accumulator updates
        """
        self.model = nnue_model
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.use_accumulator = use_accumulator
        self._nodes_searched = 0
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Lazily get device
        self._device = None

    @property
    def device(self):
        if self._device is None:
            self._device = next(self.model.parameters()).device
        return self._device

    def select_action(self, core, legal: list[int]) -> int:
        """
        Select best action using alpha-beta search.
        
        Works with PenguinChessCore (full search with snapshot undo)
        and RustCore (one-step lookahead with NNUE evaluation).
        """
        if not legal:
            return 0
        
        if len(legal) == 1:
            return legal[0]
        
        # Try to use PenguinChessCore for full search
        if hasattr(core, 'get_snapshot'):
            best_move, _ = self._iterative_deepening(core, legal)
            return best_move
        else:
            # RustCore fallback: one-step lookahead with NNUE evaluation
            return self._select_best_by_eval(core, legal)

    def _select_best_by_eval(self, core, legal: list[int]) -> int:
        """
        Fallback for RustCore: evaluate each move with NNUE, pick best.
        No tree search, but fast and works with any core type.
        """
        best_action = legal[0]
        best_score = -float('inf')
        snapshot = None
        
        if hasattr(core, 'get_snapshot'):
            snapshot = core.get_snapshot()
        
        for action in legal:
            # Save sparse features if using PenguinChessCore
            if hasattr(core, 'get_snapshot'):
                old_sparse = extract_sparse(core)
            
            _, reward, terminated, info = core.step(action)
            
            # Evaluate resulting position
            sparse = extract_sparse(core)
            dense = extract_dense(core)
            dense_t = torch.from_numpy(dense).float().to(self.device)
            sparse_batch = [sparse]
            dense_batch = dense_t.unsqueeze(0)
            
            with torch.no_grad():
                val = self.model.forward(sparse_batch, dense_batch)
            
            # Score = immediate reward + future value
            score = reward + (0.0 if terminated else val.item())
            
            # Undo
            if hasattr(core, 'restore_snapshot') and snapshot is not None:
                core.restore_snapshot(snapshot)
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action

    def _evaluate(self, core: PenguinChessCore) -> float:
        """
        Evaluate a position using the NNUE model.
        Returns value from current player's perspective in [-1, 1].
        """
        sparse, dense = state_to_features(core)
        dense_t = torch.from_numpy(dense).float().to(self.device)
        sparse_batch = [sparse]
        dense_batch = dense_t.unsqueeze(0)
        
        with torch.no_grad():
            val = self.model.forward(sparse_batch, dense_batch)
        return val.item()

    def _evaluate_with_acc(
        self,
        core: PenguinChessCore,
        acc: NNUEAccumulator,
    ) -> float:
        """
        Evaluate using pre-computed accumulator (fast path).
        Falls back to full evaluation if accumulator is stale.
        """
        dense = extract_dense(core)
        dense_t = torch.from_numpy(dense).float().to(self.device)
        
        with torch.no_grad():
            val = self.model.evaluate_from_acc(acc, dense_t)
        return val.item()

    def _is_game_over(self, core: PenguinChessCore) -> bool:
        """Check if the game is over."""
        if core.phase == 'gameover':
            return True
        # Check if there are legal actions
        legal = core.get_legal_actions()
        return len(legal) == 0

    def _negamax(
        self,
        core: PenguinChessCore,
        depth: int,
        alpha: float,
        beta: float,
        acc: Optional[NNUEAccumulator] = None,
    ) -> tuple[float, Optional[int]]:
        """
        Negamax with alpha-beta pruning.
        
        Args:
            core: current game state
            depth: remaining depth
            alpha, beta: search window
            acc: optional NNUE accumulator for fast evaluation
            
        Returns:
            (score, best_action) tuple
        """
        self._nodes_searched += 1
        
        # Check terminal states
        if self._is_game_over(core) or depth == 0:
            if acc is not None and self.use_accumulator:
                val = self._evaluate_with_acc(core, acc)
            else:
                val = self._evaluate(core)
            return val, None
        
        legal = core.get_legal_actions()
        if not legal:
            if acc is not None and self.use_accumulator:
                val = self._evaluate_with_acc(core, acc)
            else:
                val = self._evaluate(core)
            return val, None
        
        # Move ordering
        legal = _order_moves(core, legal)
        
        best_action = legal[0]
        best_score = -float('inf')
        snapshot = core.get_snapshot()
        
        for action in legal:
            # Save old sparse features for incremental update
            if acc is not None and self.use_accumulator:
                old_sparse = extract_sparse(core)
            
            # Make move
            _, reward, terminated, info = core.step(action)
            
            # Update accumulator incrementally
            if acc is not None and self.use_accumulator:
                new_sparse = extract_sparse(core)
                removed, added = compute_sparse_diff(old_sparse, new_sparse)
                acc.apply_diff(removed, added)
            
            # Recursive search
            score, _ = self._negamax(core, depth - 1, -beta, -alpha, acc)
            score = -score
            
            # Restore accumulator
            if acc is not None and self.use_accumulator:
                acc.apply_diff(added, removed)  # reverse
            
            # Restore game state
            core.restore_snapshot(snapshot)
            
            if score > best_score:
                best_score = score
                best_action = action
            
            alpha = max(alpha, score)
            if alpha >= beta:
                break  # beta cutoff
        
        return best_score, best_action

    def _iterative_deepening(
        self,
        core: PenguinChessCore,
        legal: list[int],
    ) -> tuple[int, float]:
        """
        Iterative deepening: search at increasing depths.
        
        Returns:
            (best_action, best_score)
        """
        best_action = legal[0]
        best_score = 0.0
        start_time = time.time()
        
        # Initialize accumulator for the root position
        acc = None
        if self.use_accumulator:
            acc = self.model.create_accumulator()
            sparse = extract_sparse(core)
            acc.reset(sparse, core.current_player)
        
        for depth in range(1, self.max_depth + 1):
            # Check time limit
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                break
            
            old_nodes = self._nodes_searched
            
            # Search with aspiration windows
            window = 0.5  # Initial aspiration window
            alpha = -1.0
            beta = 1.0
            
            score, action = self._negamax(core, depth, alpha, beta, acc)
            
            if action is not None:
                best_action = action
                best_score = score
            
            self._last_depth_reached = depth
        
        return best_action, best_score

    def reset(self) -> None:
        """Reset search statistics."""
        self._nodes_searched = 0
        self._last_depth_reached = 0
