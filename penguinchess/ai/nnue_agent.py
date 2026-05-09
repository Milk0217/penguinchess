"""
Alpha-Beta search with NNUE evaluation agent for PenguinChess.

Optimizations for depth 10+:
  - NNUE-based move ordering (eval-based sort at root)
  - Principal Variation Search (zero-window scout)
  - Transposition Table (caches search results)
  - Aspiration windows with iterative deepening
  - Incremental NNUE accumulator

Search uses PenguinChessCore (Python) with snapshot undo.
RustCore fallback: one-step lookahead (no tree search).
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

# ─── Transposition Table ───────────────────────────────────────


class TTEntry:
    __slots__ = ("depth", "score", "flag", "best_move", "age")
    EXACT = 0
    LOWER = 1  # beta cutoff (score is lower bound)
    UPPER = 2  # alpha didn't improve (score is upper bound)


class TranspositionTable:
    """Simple hash-based transposition table with age-based replacement."""

    def __init__(self, size: int = 1 << 20):
        self.size = size
        self.mask = size - 1
        self.table: dict[int, TTEntry] = {}
        self.age = 0

    def _hash_state(self, core) -> int:
        """Compute a (near-)unique hash for the current board state."""
        if hasattr(core, 'pieces'):
            pieces = tuple(sorted(
                (p.id, p.hex.q, p.hex.r, p.hex.s)
                for p in core.pieces if p.alive and p.hex
            ))
            hexes = tuple(
                h.points if h.is_active() else -99
                for h in core.hexes
            )
        else:
            # RustCore fallback
            data = core.to_json()
            if isinstance(data, str):
                return hash(data)
            pieces = tuple(sorted(
                (p['id'], p['q'], p['r'], p['s'])
                for p in data.get('pieces', []) if p.get('alive') and p.get('hex', -1) >= 0
            ))
            hexes = tuple(
                h['value'] if h.get('value', 0) > 0 else -99
                for h in data.get('hexes', [])
            )
        h = hash((pieces, hexes, core.current_player, core.phase))
        return h

    def lookup(self, core, depth: int, alpha: float, beta: float):
        """Look up entry. Returns (score, move, flag) or None."""
        h = self._hash_state(core)
        entry = self.table.get(h)
        if entry is None or entry.depth < depth:
            return None
        if entry.flag == TTEntry.EXACT:
            return entry.score, entry.best_move, TTEntry.EXACT
        if entry.flag == TTEntry.LOWER and entry.score >= beta:
            return entry.score, entry.best_move, TTEntry.LOWER
        if entry.flag == TTEntry.UPPER and entry.score <= alpha:
            return entry.score, entry.best_move, TTEntry.UPPER
        return None

    def store(self, core, depth: int, score: float, flag: int, best_move: Optional[int]):
        h = self._hash_state(core)
        entry = TTEntry()
        entry.depth = depth
        entry.score = score
        entry.flag = flag
        entry.best_move = best_move
        entry.age = self.age
        self.table[h] = entry

    def get_best_move(self, core) -> Optional[int]:
        """Get stored best move for a position (if any)."""
        h = self._hash_state(core)
        entry = self.table.get(h)
        return entry.best_move if entry else None

    def clear(self):
        self.table.clear()

    def new_search(self):
        """Call at start of each new root search."""
        self.age += 1


# ─── Move ordering: NNUE evaluation based ──────────────────────


def _eval_order_moves(
    core: PenguinChessCore,
    legal: list[int],
    model: NNUE,
    device,
    tt_best_move: Optional[int] = None,
) -> list[int]:
    """
    Order moves by NNUE evaluation.
    TT best move first, then high-scoring moves.
    Uses incremental accumulator for fast evaluation.
    """
    if len(legal) <= 1:
        return legal

    # TT move first
    if tt_best_move is not None and tt_best_move in legal:
        moves = [tt_best_move]
        remaining = [m for m in legal if m != tt_best_move]
    else:
        moves = []
        remaining = list(legal)

    # Batch evaluate all remaining moves for ordering
    if remaining:
        snap = core.get_snapshot()
        root_sparse = extract_sparse(core)
        root_stm = core.current_player
        root_acc = model.create_accumulator()
        root_acc.reset(root_sparse, root_stm)

        scored = []
        for action in remaining:
            # Simulate move and evaluate with one-step NNUE
            old_sparse = extract_sparse(core)
            _, reward, terminated, info = core.step(action)
            new_sparse = extract_sparse(core)
            removed, added = compute_sparse_diff(old_sparse, new_sparse)
            root_acc.apply_diff(removed, added, core.current_player)
            dense = extract_dense(core)
            dense_t = torch.from_numpy(dense).float().to(device)

            with torch.no_grad():
                val = model.evaluate_from_acc(root_acc, dense_t).item()
            score = reward + (0.0 if terminated else val)

            root_acc.apply_diff(added, removed, core.current_player)  # reverse
            core.restore_snapshot(snap)

            scored.append((action, score))

        # Sort by score descending
        remaining = [a for a, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
        moves.extend(remaining)

    return moves


# ─── Fast heuristic ordering (for internal nodes, no NNUE) ─────


def _heuristic_order_moves(
    core: PenguinChessCore,
    legal: list[int],
    tt_best_move: Optional[int] = None,
) -> list[int]:
    """Fast heuristic ordering: TT first, then captures/eliminations, then value."""
    if len(legal) <= 1:
        return legal

    # TT move first
    if tt_best_move is not None and tt_best_move in legal:
        moves = [tt_best_move]
        remaining = [m for m in legal if m != tt_best_move]
    else:
        moves = []
        remaining = list(legal)

    # Quick move ordering using info from step (no NNUE eval)
    snap = core.get_snapshot()
    scored = []
    for a in remaining:
        _, reward, _, info = core.step(a)
        score = reward
        if info.get("piece_eliminated", False):
            score += 10.0
        if info.get("hexes_eliminated", 0) > 0:
            score += 5.0
        scored.append((a, score))
        core.restore_snapshot(snap)

    remaining = [a for a, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
    moves.extend(remaining)
    return moves


# ─── NNUE Agent with advanced search ───────────────────────────


class NNUEAgent(Agent):
    """
    Alpha-Beta search agent using NNUE for position evaluation.
    Optimized for depth 6-10 search with PVS + TT + NNUE move ordering.

    Placement phase: one-step lookahead.
    Movement phase: full iterative-deepening Alpha-Beta.
    """

    def __init__(
        self,
        nnue_model: NNUE,
        max_depth: int = 6,
        time_limit: Optional[float] = None,
        use_accumulator: bool = True,
        tt_size: int = 1 << 20,
    ):
        self.model = nnue_model
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.use_accumulator = use_accumulator
        self._nodes_searched = 0
        self._last_depth_reached = 0
        self.model.eval()
        self._device = None
        self.tt = TranspositionTable(tt_size)

    @property
    def device(self):
        if self._device is None:
            self._device = next(self.model.parameters()).device
        return self._device

    def select_action(self, core, legal: list[int]) -> int:
        if not legal:
            return 0
        if len(legal) == 1:
            return legal[0]

        if hasattr(core, 'get_snapshot'):
            self.tt.new_search()
            best_move, _ = self._iterative_deepening(core, legal)
            return best_move
        else:
            return self._select_best_by_eval(core, legal)

    def reset(self) -> None:
        self._nodes_searched = 0
        self._last_depth_reached = 0
        self.tt.clear()

    # ─── Fallback for RustCore ─────────────────────────────────

    def _select_best_by_eval(self, core, legal: list[int]) -> int:
        best_action = legal[0]
        best_score = -float('inf')
        snapshot = core.get_snapshot() if hasattr(core, 'get_snapshot') else None

        for action in legal:
            _, reward, terminated, info = core.step(action)
            sparse = extract_sparse(core)
            dense = extract_dense(core)
            stm = core.current_player

            with torch.no_grad():
                val = self.model._forward_sparse_single(
                    sparse, torch.from_numpy(dense).float().to(self.device),
                    stm_player=stm).item()
            score = reward + (0.0 if terminated else val)

            if snapshot is not None and hasattr(core, 'restore_snapshot'):
                core.restore_snapshot(snapshot)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    # ─── Evaluation ────────────────────────────────────────────

    def _evaluate(self, core) -> float:
        sparse, dense = state_to_features(core)
        stm = core.current_player
        with torch.no_grad():
            val = self.model._forward_sparse_single(
                sparse, torch.from_numpy(dense).float().to(self.device),
                stm_player=stm)
        return val.item()

    def _evaluate_with_acc(self, core, acc: NNUEAccumulator) -> float:
        dense = extract_dense(core)
        dense_t = torch.from_numpy(dense).float().to(self.device)
        with torch.no_grad():
            val = self.model.evaluate_from_acc(acc, dense_t)
        return val.item()

    def _is_game_over(self, core) -> bool:
        if core.phase == 'gameover':
            return True
        return len(core.get_legal_actions()) == 0

    # ─── PVS Negamax ───────────────────────────────────────────

    def _order_moves_inline(
        self,
        core,
        legal: list[int],
        depth: int,
        acc: Optional[NNUEAccumulator] = None,
        tt_move: Optional[int] = None,
    ) -> list[tuple]:
        """
        Order legal moves for search.
        
        Near leaves (depth <= 2): NNUE-based evaluation (accurate, better pruning).
        Deeper: heuristic ordering (fast, with TT best move first).
        
        Returns list of (action, reward, terminated) sorted by score.
        """
        if len(legal) <= 1:
            return [(a, 0.0, False) for a in legal]

        snapshot = core.get_snapshot()

        if depth <= 1 and acc is not None and self.use_accumulator:
            # NNUE-based ordering
            scored = []
            for a in legal:
                old_sparse = extract_sparse(core)
                _, reward, term, _ = core.step(a)
                new_sparse = extract_sparse(core)
                removed, added = compute_sparse_diff(old_sparse, new_sparse)
                acc.apply_diff(removed, added, core.current_player)

                leaf_val = self._evaluate_with_acc(core, acc)
                order_score = reward + (0.0 if term else leaf_val)

                acc.apply_diff(added, removed, core.current_player)
                core.restore_snapshot(snapshot)
                scored.append((a, reward, term, order_score))
            scored.sort(key=lambda x: x[3], reverse=True)
            return [(a, r, t) for a, r, t, _ in scored]
        else:
            # Heuristic ordering with TT best move first
            if tt_move is not None and tt_move in legal:
                ordered = [(tt_move, 0.0, False)]
                remaining = [m for m in legal if m != tt_move]
            else:
                ordered = []
                remaining = list(legal)

            # Quick heuristic: just reward from stepping
            heur = []
            for a in remaining:
                _, reward, term, info = core.step(a)
                score = reward
                if info.get("piece_eliminated", False):
                    score += 10.0
                if info.get("hexes_eliminated", 0) > 0:
                    score += 5.0
                heur.append((a, reward, term, score))
                core.restore_snapshot(snapshot)
            heur.sort(key=lambda x: x[3], reverse=True)
            ordered.extend([(a, r, t) for a, r, t, _ in heur])
            return ordered

    def _negamax(
        self,
        core,
        depth: int,
        alpha: float,
        beta: float,
        acc: Optional[NNUEAccumulator] = None,
        is_pv_node: bool = True,
    ) -> tuple[float, Optional[int]]:
        """Principal Variation Search (PVS) with depth-adaptive move ordering."""
        self._nodes_searched += 1

        # TT lookup
        tt_result = self.tt.lookup(core, depth, alpha, beta)
        if tt_result is not None:
            return tt_result[0], tt_result[1]
        tt_move = self.tt.get_best_move(core)

        # Terminal / leaf
        if self._is_game_over(core) or depth == 0:
            val = self._evaluate_with_acc(core, acc) if (acc and self.use_accumulator) else self._evaluate(core)
            self.tt.store(core, depth, val, TTEntry.EXACT, None)
            return val, None

        legal = core.get_legal_actions()
        if not legal:
            val = self._evaluate_with_acc(core, acc) if (acc and self.use_accumulator) else self._evaluate(core)
            self.tt.store(core, depth, val, TTEntry.EXACT, None)
            return val, None

        # Ordered moves: NNUE-based (shallow) or heuristic (deep)
        ordered = self._order_moves_inline(core, legal, depth, acc, tt_move)

        best_action = ordered[0][0]
        best_score = -float('inf')
        snapshot = core.get_snapshot()
        flag = TTEntry.UPPER

        for i, (action, reward, term) in enumerate(ordered):
            if acc is not None and self.use_accumulator:
                old_sparse = extract_sparse(core)

            _, reward2, term2, _ = core.step(action)

            if acc is not None and self.use_accumulator:
                new_sparse = extract_sparse(core)
                removed, added = compute_sparse_diff(old_sparse, new_sparse)
                acc.apply_diff(removed, added, core.current_player)

            if term2:
                score_after = reward2 + self._evaluate_with_acc(core, acc)
            elif depth == 1:
                score_after = reward2 + self._evaluate_with_acc(core, acc)
            else:
                if i == 0 or is_pv_node:
                    score_after, _ = self._negamax(
                        core, depth - 1, -beta, -alpha, acc, is_pv_node)
                    score_after = -score_after
                else:
                    score_after, _ = self._negamax(
                        core, depth - 1, -alpha - 1e-10, -alpha, acc, False)
                    score_after = -score_after
                    if score_after > alpha and score_after < beta:
                        score_after, _ = self._negamax(
                            core, depth - 1, -beta, -alpha, acc, True)
                        score_after = -score_after

            if acc is not None and self.use_accumulator:
                acc.apply_diff(added, removed, core.current_player)
            core.restore_snapshot(snapshot)

            if score_after > best_score:
                best_score = score_after
                best_action = action

            alpha = max(alpha, best_score)
            if alpha >= beta:
                flag = TTEntry.LOWER
                break

        self.tt.store(core, depth, best_score, flag, best_action)
        return best_score, best_action

    # ─── Iterative Deepening ───────────────────────────────────

    def _iterative_deepening(self, core, legal: list[int]) -> tuple[int, float]:
        """Iterative deepening with aspiration windows."""
        best_action = legal[0]
        best_score = 0.0
        start_time = time.time()

        # Root accumulator
        acc = None
        if self.use_accumulator:
            acc = self.model.create_accumulator()
            root_sparse = extract_sparse(core)
            acc.reset(root_sparse, core.current_player)

        # NNUE move ordering at root (depth 1 evaluation)
        legal = _eval_order_moves(core, legal, self.model, self.device, None)

        for depth in range(1, self.max_depth + 1):
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                break

            old_nodes = self._nodes_searched

            # Aspiration window centered on previous score
            if depth >= 3 and best_score != 0.0:
                window = max(0.1, 0.5 - 0.04 * depth)  # shrink window as depth grows
                alpha = max(-1.0, best_score - window)
                beta = min(1.0, best_score + window)
            else:
                alpha = -1.0
                beta = 1.0

            score, action = self._negamax(core, depth, alpha, beta, acc, is_pv_node=True)

            # Aspiration fail-low: re-search with full window
            if score <= alpha:
                score, action = self._negamax(core, depth, -1.0, beta, acc, is_pv_node=True)
            # Aspiration fail-high: re-search with full window
            if score >= beta:
                score, action = self._negamax(core, depth, alpha, 1.0, acc, is_pv_node=True)

            if action is not None:
                best_action = action
                best_score = score

            nodes_at_depth = self._nodes_searched - old_nodes
            self._last_depth_reached = depth

        return best_action, best_score
