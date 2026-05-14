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
# Dirichlet noise config (AlphaZero root exploration)
# Match Rust implementation: DIRICHLET_ALPHA=0.15, DIRICHLET_EPS=0.25
# =============================================================================

DIRICHLET_ALPHA: float = 0.15
DIRICHLET_EPS: float = 0.25


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
    training: bool = False,
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
    training : bool
        If True, apply Dirichlet noise to the **root** node's prior
        probabilities to encourage exploration (AlphaZero-style).
        Must be ``False`` during evaluation / inference.  Default ``False``.

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

            # --- Dirichlet noise at root (AlphaZero training exploration) ---
            if training and node.parent is None:
                noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(legal))
                for i, a in enumerate(legal):
                    child = node.children[a]
                    child.prior = (1.0 - DIRICHLET_EPS) * child.prior + DIRICHLET_EPS * float(noise[i])

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
# Batch evaluation helper
# =============================================================================


def _evaluate_and_expand_batch(
    pending: List[Tuple[MCTSNode, List[int]]],
    batch_eval_fn: Callable,
    training: bool = False,
) -> None:
    """
    Internal helper for ``mcts_search_batched``.

    Takes a list of ``(node, legal_actions)`` pairs collected during the
    SELECT phase, evaluates all of their leaf states in a **single forward
    pass**, then expands children and BACKUPs for each.

    Parameters
    ----------
    pending : list of (MCTSNode, list[int])
        Each entry is a leaf node whose ``.state`` will be evaluated, together
        with its list of legal action indices.
    batch_eval_fn : callable
        A function ``(states) -> (logits, values)`` where
        ``states`` is a list of ``PenguinChessCore`` objects.

        ``logits`` : ndarray ``(B, 60)`` — raw policy logits.
        ``values`` : ndarray ``(B,)``   -- scalar values in ``[-1, 1]``.

        Typically ``model.evaluate_batch``.
    training : bool
        If True, apply Dirichlet noise to the **root** node's priors
        to force exploration (AlphaZero-style).  Has no effect on non-root
        nodes.  Default ``False`` (safe for evaluation / inference).
    """
    # Collect leaf states for batch evaluation
    states = [node.state for node, _ in pending]

    # Batch forward pass — returns raw logits + values
    logits_batch, values_batch = batch_eval_fn(states)

    B = len(pending)
    assert logits_batch.shape == (B, 60), (
        f"Expected logits shape ({B}, 60), got {logits_batch.shape}"
    )
    assert values_batch.shape == (B,), (
        f"Expected values shape ({B},), got {values_batch.shape}"
    )

    # Expand every node and BACKUP
    for (node, legal), logits, value in zip(pending, logits_batch, values_batch):
        # --- mask illegal actions and softmax ---
        mask = np.zeros(60, dtype=bool)
        mask[legal] = True
        masked_logits = np.where(mask, logits, -1e9)
        policy = _softmax(masked_logits).astype(np.float32)

        # Safety: if masking zeroed everything, fall back to uniform
        if policy.sum() < 1e-8:
            policy[:] = 0.0
            if legal:
                policy[legal] = 1.0 / len(legal)
        elif abs(policy.sum() - 1.0) > 1e-6:
            policy = policy / policy.sum()

        # --- create children ---
        leaf_snap = node.state.get_snapshot()
        for a in legal:
            if a not in node.children:
                node.state.step(a)
                child_core = copy.deepcopy(node.state)
                node.state.restore_snapshot(leaf_snap)
                node.children[a] = MCTSNode(
                    state=child_core,
                    parent=node,
                    action=a,
                    prior=float(policy[a]),
                )

        # --- Dirichlet noise at root (AlphaZero training exploration) ---
        if training and node.parent is None:
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(legal))
            for i, a in enumerate(legal):
                child = node.children[a]
                child.prior = (1.0 - DIRICHLET_EPS) * child.prior + DIRICHLET_EPS * float(noise[i])

        # --- BACKUP ---
        n = node
        v = value
        while n is not None:
            n.visits += 1
            n.total_value += v
            v = -v
            n = n.parent


# =============================================================================
# Batched MCTS search
# =============================================================================


def mcts_search_batched(
    root_state: PenguinChessCore,
    model: Any = None,
    num_simulations: int = 800,
    c_puct: float = 1.4,
    temperature: float = 1.0,
    evaluate_fn: Optional[Callable] = None,
    batch_size: int = 32,
    training: bool = False,
) -> Tuple[Dict[int, int], MCTSNode]:
    """
    MCTS search with **batched neural-network evaluation**.

    Instead of calling ``evaluate_fn`` once per leaf (which incurs CPU↔GPU
    transfer overhead), this version collects multiple leaf states and
    evaluates them together in a single forward pass every ``batch_size``
    simulations.

    Parameters
    ----------
    root_state : PenguinChessCore
        The starting game state.  **Not** modified.
    model :
        Either an ``AlphaZeroNet`` (or any object with ``.evaluate_batch()``),
        or a callable that acts as ``batch_eval_fn`` -- see below.
    num_simulations : int
        Total number of MCTS simulations to run (default 800).
    c_puct : float
        Exploration constant in the PUCT formula (default 1.4).
    temperature : float
        Temperature for the final action sampling (default 1.0).
    evaluate_fn : callable or None
        **Batch-capable** evaluate function with the signature:

        ``evaluate_fn(states: list[PenguinChessCore]) -> (logits, values)``

        where ``logits`` is ``ndarray (B, 60)`` and ``values`` is
        ``ndarray (B,)``.  Takes priority over ``model``.

        The passed function **must** support batch -- it receives a list of
        states, not a single state.  For single-state evaluation use
        ``mcts_search`` instead.
    batch_size : int
        Maximum number of leaf states collected before a batch evaluation is
        triggered (default 32).  Larger values give better GPU utilisation but
        may slightly alter the search tree since backups are delayed.
    training : bool
        If True, apply Dirichlet noise to the **root** node's prior
        probabilities to encourage exploration (AlphaZero-style).
        Must be ``False`` during evaluation / inference.  Default ``False``.

    Returns
    -------
    action_counts : dict[int, int]
        ``{action: visit_count}`` at the root node.
    root : MCTSNode
        The root node (useful for debugging / inspection).
    """
    # ---- resolve batch_eval_fn ----
    if evaluate_fn is not None:
        batch_eval_fn = evaluate_fn
    elif model is not None and hasattr(model, "evaluate_batch"):
        batch_eval_fn = model.evaluate_batch  # type: ignore[union-attr]
    elif callable(model):
        batch_eval_fn = model
    else:
        # Nothing batch-capable → fall back to sequential MCTS
        return mcts_search(
            root_state,
            model=model,
            num_simulations=num_simulations,
            c_puct=c_puct,
            temperature=temperature,
            evaluate_fn=evaluate_fn,
            training=training,
        )

    # Fast snapshot for simulation engine
    root_snapshot = root_state.get_snapshot()
    sim_state = copy.deepcopy(root_state)

    root = MCTSNode(state=sim_state, parent=None, action=None, prior=0.0)
    pending: List[Tuple[MCTSNode, List[int]]] = []

    for _ in range(num_simulations):
        node = root
        sim_state.restore_snapshot(root_snapshot)

        # ----- SELECT -----
        while node.expanded():
            action = max(
                node.children, key=lambda a: node.children[a].ucb_score(c_puct)
            )
            child = node.children[action]
            _, _, terminated, _ = sim_state.step(action)
            node = child
            if terminated:
                break

        # ----- Check termination / collect for batch -----
        terminated = sim_state._terminated
        legal = sim_state.get_legal_actions()

        if terminated:
            value = _terminal_value(sim_state)
        elif not legal:
            value = 0.0
        else:
            pending.append((node, legal))
            value = None  # will be set during batch eval

        if terminated or not legal:
            # BACKUP immediately (cheap — no NN call needed)
            while node is not None:
                node.visits += 1
                node.total_value += value
                value = -value
                node = node.parent

        # Flush pending when batch is full
        if len(pending) >= batch_size:
            _evaluate_and_expand_batch(pending, batch_eval_fn, training=training)
            pending.clear()

    # ---- flush remaining pending nodes ----
    if pending:
        _evaluate_and_expand_batch(pending, batch_eval_fn, training=training)
        pending.clear()

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
# Root-parallelised MCTS search
# =============================================================================


def mcts_search_parallel(
    root_state: PenguinChessCore,
    model: Any = None,
    num_simulations: int = 800,
    c_puct: float = 1.4,
    temperature: float = 1.0,
    evaluate_fn: Optional[Callable] = None,
    num_workers: int = 4,
    batch_size: int = 32,
    use_batched: bool = True,
    training: bool = False,
) -> Tuple[Dict[int, int], None]:
    """
    Root-parallelised MCTS via **ensemble averaging**.

    Runs ``num_workers`` independent MCTS searches (each with
    ``num_simulations // num_workers`` simulations) and merges their root
    visit counts.  Extremely effective because different random seeds explore
    different parts of the tree.

    Parameters
    ----------
    root_state : PenguinChessCore
        Starting game state (not modified).
    model :
        Model or evaluate function passed to each worker.
    num_simulations : int
        Total simulations across all workers (default 800).
    c_puct : float
        Exploration constant.
    temperature : float
        Temperature for final action selection.
    evaluate_fn : callable or None
        Evaluate function (single-state; passed to workers).
    num_workers : int
        Number of independent searches to run (default 4).
        Each worker gets ``num_simulations // num_workers`` simulations.
    batch_size : int
        Batch size for batched workers (default 32).
    use_batched : bool
        If ``True`` (default), uses ``mcts_search_batched`` for each worker.
        If ``False``, uses the original ``mcts_search``.
    training : bool
        If True, apply Dirichlet noise to the **root** node's prior
        probabilities in each worker search (AlphaZero-style).
        Must be ``False`` during evaluation / inference.  Default ``False``.

    Returns
    -------
    merged_counts : dict[int, int]
        Merged visit counts from all workers.
    root : None
        ``None`` because there is no single root tree.
    """
    sims_per_worker = max(1, num_simulations // num_workers)

    merged_counts: Dict[int, int] = {}

    for w in range(num_workers):
        if use_batched:
            counts, _ = mcts_search_batched(
                root_state,
                model=model,
                num_simulations=sims_per_worker,
                c_puct=c_puct,
                temperature=temperature,
                evaluate_fn=evaluate_fn,
                batch_size=batch_size,
                training=training,
            )
        else:
            counts, _ = mcts_search(
                root_state,
                model=model,
                num_simulations=sims_per_worker,
                c_puct=c_puct,
                temperature=temperature,
                evaluate_fn=evaluate_fn,
                training=training,
            )
        for action, count in counts.items():
            merged_counts[action] = merged_counts.get(action, 0) + count

    return merged_counts, None


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
    if not action_counts:
        raise ValueError("select_action: empty action_counts — MCTS returned no visits")

    if temperature < 1e-8:
        return max(action_counts, key=action_counts.__getitem__)

    actions = list(action_counts.keys())
    counts = np.array([action_counts[a] for a in actions], dtype=np.float64)

    if temperature != 1.0:
        counts = counts ** (1.0 / temperature)

    probs = counts / counts.sum()
    if np.any(np.isnan(probs)) or probs.sum() == 0:
        # Fallback: uniform over all actions
        probs = np.ones_like(counts) / len(counts)
    return int(np.random.choice(actions, p=probs))
