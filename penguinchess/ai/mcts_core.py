"""
MCTS (Monte Carlo Tree Search) implementation for PenguinChess.

Supports three modes:
  1. Custom evaluate_fn(state) -> (policy_probs, value)  — direct function
  2. SB3 model with .predict()  — auto-wrapped
  3. None  — uniform random baseline (for testing)
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from penguinchess.core import PenguinChessCore


# =============================================================================
# MCTS Node
# =============================================================================

class MCTSNode:
    """MCTS tree node with PUCT selection formula."""

    __slots__ = (
        "state", "parent", "action", "children",
        "visits", "total_value", "prior",
    )

    def __init__(
        self,
        state: PenguinChessCore,
        parent: Optional[MCTSNode] = None,
        action: Optional[int] = None,
        prior: float = 0.0,
    ):
        self.state: PenguinChessCore = state
        self.parent: Optional[MCTSNode] = parent
        self.action: Optional[int] = action       # action that led to this node
        self.children: Dict[int, MCTSNode] = {}   # {action: MCTSNode}

        self.visits: int = 0
        self.total_value: float = 0.0
        self.prior: float = prior                 # prior probability from policy

    # ------------------------------------------------------------------
    # PUCT
    # ------------------------------------------------------------------

    def ucb_score(self, c_puct: float = 1.4) -> float:
        """PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N)."""
        if self.visits == 0:
            return float("inf")
        if self.parent is None:
            return 0.0
        q = self.total_value / self.visits
        u = (
            c_puct
            * self.prior
            * math.sqrt(self.parent.visits)
            / (1.0 + self.visits)
        )
        return q + u

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def expanded(self) -> bool:
        """Whether this node has been expanded (has child nodes)."""
        return len(self.children) > 0

    def value(self) -> float:
        """Average value across visits (from current-player perspective)."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def best_child(self, c_puct: float = 1.4) -> Tuple[int, MCTSNode]:
        """Select child with highest PUCT score.  Returns (action, child_node)."""
        action = max(self.children, key=lambda a: self.children[a].ucb_score(c_puct))
        return action, self.children[action]


# =============================================================================
# Observation builder
# =============================================================================

OBS_FLAT_SIZE = 206  # 60*3 + 6*4 + 2 — must match spaces.py


def _build_flat_obs(state: PenguinChessCore) -> np.ndarray:
    """Build 206-dim flat observation array from a PenguinChessCore state."""
    obs = state.get_observation()

    # Board: 60 hexes x 3 features [q/8, r/8, value/3]  → (180,)
    board = np.array(obs["board"], dtype=np.float32).flatten()

    # Pieces: 6 pieces x 4 features [id/10, q/8, r/8, s/8]  → (24,)
    pieces = np.array(obs["pieces"], dtype=np.float32).flatten()

    # Metadata: [current_player, phase]  → (2,)
    meta = np.array(
        [float(obs["current_player"]), float(obs["phase"])],
        dtype=np.float32,
    )

    flat = np.concatenate([board, pieces, meta]).astype(np.float32)
    assert flat.shape == (OBS_FLAT_SIZE,), f"obs shape mismatch: {flat.shape}"
    return flat


# =============================================================================
# Helper: numerically stable softmax
# =============================================================================

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    x_max = np.max(x)
    e_x = np.exp(x - x_max)
    return e_x / (e_x.sum() + 1e-30)


# =============================================================================
# Evaluate-function factories
# =============================================================================

def _make_uniform_evaluator() -> Callable[[PenguinChessCore], Tuple[np.ndarray, float]]:
    """Return an evaluate_fn that uses uniform policy and zero value."""
    def _evaluate(state: PenguinChessCore) -> Tuple[np.ndarray, float]:
        legal = state.get_legal_actions()
        probs = np.zeros(60, dtype=np.float32)
        if legal:
            probs[legal] = 1.0 / len(legal)
        return probs, 0.0
    return _evaluate


def _make_sb3_evaluator(
    model: Any,
) -> Callable[[PenguinChessCore], Tuple[np.ndarray, float]]:
    """
    Wrap an SB3 model (PPO, A2C) into an evaluate_fn.

    Uses ``model.policy`` to extract action logits and state value.
    Falls back to uniform policy if the internal structure is unexpected.
    """
    import torch

    def _evaluate(state: PenguinChessCore) -> Tuple[np.ndarray, float]:
        flat_obs = _build_flat_obs(state)
        obs_tensor = (
            torch.as_tensor(flat_obs, dtype=torch.float32).unsqueeze(0)
        )
        legal = state.get_legal_actions()

        try:
            with torch.no_grad():
                # ---- features extractor ----
                features = model.policy.features_extractor(obs_tensor)

                # ---- mlp extractor (shared → pi + vf latents) ----
                latent_pi, latent_vf = model.policy.mlp_extractor(features)

                # ---- action logits ----
                logits = model.policy.action_net(latent_pi)
                logits_np = logits[0].cpu().numpy().astype(np.float64)

                # ---- value ----
                value_t = model.policy.value_net(latent_vf)
                value = float(value_t[0, 0].cpu().numpy())

        except Exception:
            # Graceful fallback: uniform policy, zero value
            logits_np = np.zeros(60, dtype=np.float64)
            value = 0.0

        # Mask illegal actions before softmax
        mask = np.zeros(60, dtype=bool)
        mask[legal] = True
        masked_logits = np.where(mask, logits_np, -1e9)
        probs = _softmax(masked_logits).astype(np.float32)

        # Renormalise in case the mask eliminated all probability mass
        probs_sum = probs.sum()
        if probs_sum < 1e-8:
            probs[:] = 0.0
            if legal:
                probs[legal] = 1.0 / len(legal)
        elif abs(probs_sum - 1.0) > 1e-6:
            probs = probs / probs_sum

        return probs, value

    return _evaluate


# =============================================================================
# Terminal-state helper
# =============================================================================

def _terminal_value(state: PenguinChessCore) -> float:
    """
    Return +1/-1/0 from the perspective of ``state.current_player``.

    At a terminal node the game-over check happened *before* player switch,
    so ``current_player`` is the player who just made the move that ended
    the game.
    """
    p1, p2 = state.players_scores[0], state.players_scores[1]
    cp = state.current_player

    if p1 > p2:
        return 1.0 if cp == 0 else -1.0
    if p2 > p1:
        return 1.0 if cp == 1 else -1.0
    return 0.0  # draw


# =============================================================================
# Core MCTS search
# =============================================================================

def mcts_search(
    root_state: PenguinChessCore,
    model: Any = None,
    num_simulations: int = 800,
    c_puct: float = 1.4,
    temperature: float = 1.0,
    evaluate_fn: Optional[Callable[[PenguinChessCore], Tuple[np.ndarray, float]]] = None,
) -> Tuple[Dict[int, int], MCTSNode]:
    """
    Run MCTS search from a root state.

    Parameters
    ----------
    root_state : PenguinChessCore
        The starting game state. The function will not modify it.
    model :
        Either an SB3 model with ``.predict()``, or a callable that follows the
        ``evaluate_fn`` signature (see below).
    num_simulations : int
        Number of MCTS simulations to run (default 800).
    c_puct : float
        Exploration constant in the PUCT formula (default 1.4).
    temperature : float
        Temperature for the final action sampling (default 1.0).
    evaluate_fn : callable or None
        Custom evaluator ``(state) -> (policy_probs_60, value_scalar)``.
        Takes priority over ``model``.

    Returns
    -------
    action_counts : dict[int, int]
        ``{action: visit_count}`` at the root node.
    root : MCTSNode
        The root node (useful for debugging / inspection).
    """
    # ---- resolve evaluate_fn ----
    if evaluate_fn is not None:
        pass
    elif callable(model):
        evaluate_fn = model
    elif model is not None and hasattr(model, "predict"):
        evaluate_fn = _make_sb3_evaluator(model)
    else:
        evaluate_fn = _make_uniform_evaluator()

    # Use fast snapshot for the simulation engine (1 deepcopy total)
    root_snapshot = root_state.get_snapshot()
    sim_state = copy.deepcopy(root_state)

    root = MCTSNode(state=sim_state, parent=None, action=None, prior=0.0)

    for _ in range(num_simulations):
        node = root
        sim_state.restore_snapshot(root_snapshot)

        # ----- SELECT -----
        while node.expanded():
            action = max(node.children, key=lambda a: node.children[a].ucb_score(c_puct))
            child = node.children[action]
            _, _, terminated, _ = sim_state.step(action)
            node = child
            if terminated:
                break

        # ----- EXPAND & EVALUATE -----
        terminated = sim_state._terminated
        legal = sim_state.get_legal_actions()

        if terminated:
            value = _terminal_value(sim_state)
        elif not legal:
            value = 0.0
        else:
            policy, value = evaluate_fn(sim_state)
            if not isinstance(policy, np.ndarray) or policy.shape != (60,):
                policy = np.zeros(60, dtype=np.float32)
                if legal:
                    policy[legal] = 1.0 / len(legal)

            # Snapshot the leaf state for child creation
            leaf_snap = sim_state.get_snapshot()
            for a in legal:
                if a not in node.children:
                    sim_state.step(a)
                    child_core = copy.deepcopy(sim_state)
                    sim_state.restore_snapshot(leaf_snap)
                    node.children[a] = MCTSNode(
                        state=child_core,
                        parent=node,
                        action=a,
                        prior=float(policy[a]),
                    )

        # ----- BACKUP -----
        while node is not None:
            node.visits += 1
            node.total_value += value
            value = -value
            node = node.parent

    # ---- build result ----
    action_counts = {}
    for action, child in root.children.items():
        action_counts[action] = child.visits
    if not action_counts:
        legal = root_state.get_legal_actions()
        for a in legal:
            action_counts[a] = 1
    return action_counts, root


# =============================================================================
# Convenience: select action from visit counts
# =============================================================================

def select_action(
    action_counts: Dict[int, int],
    temperature: float = 1.0,
) -> int:
    """
    Sample an action from the visit-count distribution at the root.

    Parameters
    ----------
    action_counts : dict[int, int]
        ``{action: visit_count}`` (output of ``mcts_search``).
    temperature : float
        Temperature for exploration:
        - 0.0  → deterministic (argmax)
        - 1.0  → proportional to visit count
        - >1.0 → more uniform

    Returns
    -------
    int : selected action index.
    """
    if temperature < 1e-8:
        # Deterministic: pick most-visited
        return max(action_counts, key=action_counts.__getitem__)

    actions = list(action_counts.keys())
    counts = np.array([action_counts[a] for a in actions], dtype=np.float64)

    if temperature != 1.0:
        counts = counts ** (1.0 / temperature)

    probs = counts / counts.sum()
    return int(np.random.choice(actions, p=probs))
