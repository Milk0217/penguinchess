"""
Shared evaluation utilities for PenguinChess.

Provides a unified Agent interface and compete/evaluation functions
used by both PPO training and ELO evaluation scripts.

Usage:
    from penguinchess.eval_utils import (
        Agent, RandomAgent, PPOAgent, AlphaZeroAgent,
        compete, compute_elo,
    )
"""

from __future__ import annotations

from typing import Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# =============================================================================
# Agent 接口
# =============================================================================


class Agent:
    """Base class for all agents. Subclasses must implement select_action."""

    def select_action(self, core, legal: list[int]) -> int:
        raise NotImplementedError


class RandomAgent(Agent):
    """随机策略。每个实例有独立 RNG，不受全局 numpy 状态影响。"""

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def select_action(self, core, legal: list[int]) -> int:
        return int(self._rng.choice(legal)) if legal else 0


class PPOAgent(Agent):
    """PPO Agent — pre-allocated buffer + torch.from_numpy 零拷贝 + policy._predict 直调。"""

    def __init__(self, model, deterministic: bool = True):
        import torch

        self._model = model
        self._deterministic = deterministic
        self._expected_dim = model.observation_space.shape[0]
        self._obs_buf = np.zeros(self._expected_dim, dtype=np.float32)
        self._tensor = None
        self._rng = np.random.default_rng()
        self._device = next(model.policy.parameters()).device

    def select_action(self, core, legal: list[int]) -> int:
        import torch

        obs = core.get_observation()
        board = obs["board"]
        pieces = obs["pieces"]

        idx = 0
        for row in board:
            self._obs_buf[idx] = row[0]
            self._obs_buf[idx + 1] = row[1]
            self._obs_buf[idx + 2] = row[2]
            idx += 3
        for row in pieces:
            self._obs_buf[idx] = row[0]
            self._obs_buf[idx + 1] = row[1]
            self._obs_buf[idx + 2] = row[2]
            self._obs_buf[idx + 3] = row[3]
            idx += 4
        if self._expected_dim == 206:
            self._obs_buf[204] = float(obs["current_player"])
            self._obs_buf[205] = float(obs["phase"])

        if self._tensor is None:
            self._tensor = torch.from_numpy(self._obs_buf).unsqueeze(0).to(self._device)
        else:
            # numpy buffer update auto-reflects, but move to device each time
            # (zero-copy on CPU, explicit transfer on GPU)
            if str(self._device) != "cpu":
                self._tensor = torch.from_numpy(self._obs_buf).unsqueeze(0).to(self._device)

        with torch.no_grad():
            action_tensor = self._model.policy._predict(self._tensor, deterministic=self._deterministic)
            action = int(action_tensor[0].item())

        return action if action in legal else int(self._rng.choice(legal))


class AlphaZeroAgent(Agent):
    def __init__(self, model):
        self._model = model

    def select_action(self, core, legal: list[int]) -> int:
        probs, value = self._model.evaluate(core)
        return int(legal[np.argmax(probs[legal])])


class MCTSAgent(Agent):
    """Rust MCTS Agent (uniform prior) — 使用 mcts_search_rust 进行树搜索。需要 --rust-core 支持。"""

    def __init__(self, num_simulations: int = 800, c_puct: float = 3.0, batch_size: int = 32):
        self._num_simulations = num_simulations
        self._c_puct = c_puct
        self._batch_size = batch_size

    def select_action(self, core, legal: list[int]) -> int:
        from penguinchess.rust_ffi import mcts_search_rust

        state_json = core.to_json()
        result = mcts_search_rust(
            state_json,
            model=None,
            num_simulations=self._num_simulations,
            c_puct=self._c_puct,
            batch_size=self._batch_size,
        )
        if not result:
            return int(np.random.default_rng().choice(legal))
        best = max(result.items(), key=lambda x: x[1])[0]
        return int(best)


class AlphaZeroMCTSAgent(Agent):
    """Rust MCTS + AlphaZeroNet — 神经网络引导 Rust MCTS 搜索。需要 --rust-core。"""

    def __init__(self, net, num_simulations: int = 800, c_puct: float = 3.0, batch_size: int = 32):
        self._net = net
        self._num_simulations = num_simulations
        self._c_puct = c_puct
        self._batch_size = batch_size

    def select_action(self, core, legal: list[int]) -> int:
        from penguinchess.rust_ffi import mcts_search_rust

        state_json = core.to_json()
        result = mcts_search_rust(
            state_json,
            model=self._net,
            num_simulations=self._num_simulations,
            c_puct=self._c_puct,
            batch_size=self._batch_size,
        )
        if not result:
            return int(np.random.default_rng().choice(legal))
        best = max(result.items(), key=lambda x: x[1])[0]
        return int(best)


class AZMCTSWrapper(Agent):
    """Python MCTS + AlphaZero net — 使用 PenguinChessCore 的 MCTS 批处理搜索。
    用于 NNUE 训练数据生成。"""

    def __init__(self, net, num_simulations: int = 200, c_puct: float = 1.4, batch_size: int = 32):
        self.net = net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.batch_size = batch_size

    def select_action(self, core, legal: list[int]) -> int:
        if not legal:
            return 0
        if len(legal) == 1:
            return legal[0]
        from penguinchess.ai.mcts_core import mcts_search_batched

        counts, _ = mcts_search_batched(
            core, evaluate_fn=self.net.evaluate,
            num_simulations=self.num_simulations,
            c_puct=self.c_puct, batch_size=self.batch_size)
        return max(counts, key=counts.__getitem__)


# =============================================================================
# 对战引擎
# =============================================================================


def _make_core_factory(use_rust: bool) -> Callable:
    """返回一个 (seed) → core 的工厂函数。"""
    if use_rust:
        from penguinchess.rust_ffi import get_engine
        from penguinchess.rust_core import RustCore

        engine = get_engine()
        return lambda s: RustCore(engine=engine).reset(s)
    else:
        from penguinchess.core import PenguinChessCore

        return lambda s: PenguinChessCore(seed=s).reset()


def _run_series(agent_p1, agent_p2, num_games: int, core_factory: Callable,
                seed_base: int = 0, game_workers: int = 1) -> dict:
    """
    固定方向（agent_p1 始终先手）的系列对战。
    支持并行对局 (game_workers > 1)。

    seed_base: 棋盘种子偏移，确保不同系列对局使用不同棋盘。
    game_workers: 并行对局数（默认 1 = 串行）。
    返回 {"p1_win", "p2_win", "draw"} 比例。
    """
    a1 = agent_p1 if agent_p1 is not None else RandomAgent(seed=seed_base)
    a2 = agent_p2 if agent_p2 is not None else RandomAgent(seed=seed_base + 1)

    def _play_one(ep: int) -> int:
        """Play one game. Returns: 1 if a1 wins, 0 if a2 wins, 0.5 if draw."""
        core = core_factory(seed_base + ep)
        terminated = False
        while not terminated:
            legal = core.get_legal_actions()
            if not legal:
                break
            action = a1.select_action(core, legal)
            _, _, terminated, _ = core.step(action)
            if terminated:
                break
            legal = core.get_legal_actions()
            if not legal:
                break
            action = a2.select_action(core, legal)
            _, _, terminated, _ = core.step(action)
        s1, s2 = core.players_scores
        if s1 > s2:
            return 1
        elif s2 > s1:
            return 0
        else:
            return 0.5

    if game_workers <= 1:
        results = [_play_one(ep) for ep in range(num_games)]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=game_workers) as pool:
            futs = {pool.submit(_play_one, ep): ep for ep in range(num_games)}
            for fut in as_completed(futs):
                results.append(fut.result())

    p1_wins = sum(1 for r in results if r == 1)
    p2_wins = sum(1 for r in results if r == 0)
    draws = sum(1 for r in results if r == 0.5)

    n = num_games
    return {"p1_win": p1_wins / n, "p2_win": p2_wins / n, "draw": draws / n}


def compete(
    agent_p1: Agent | None,
    agent_p2: Agent | None,
    num_episodes: int,
    use_rust: bool = False,
    seed_offset: int = 0,
    game_workers: int = 1,
) -> dict:
    """
    agent_p1 与 agent_p2 对战 num_episodes 局。
    自动交换先后手（各一半），消除先手优势导致的 ELO 偏差。

    seed_offset: 棋盘种子偏移，每对模型使用不同棋盘序列。
    game_workers: 并行对局数（默认 1 = 串行）。
    None → RandomAgent。
    返回 {"p1_win", "p2_win", "draw"} 比例 (0.0~1.0)。
    """
    half = num_episodes // 2
    cf = _make_core_factory(use_rust)

    # 每对模型用独立棋盘种子，避免所有模型都在固定 N 个棋盘上比赛
    pair_seed = seed_offset * 2000

    # 方向 1: agent_p1 先手
    r1 = _run_series(agent_p1, agent_p2, half, cf, seed_base=pair_seed, game_workers=game_workers)
    # 方向 2: agent_p2 先手（交换先后手，使用相同棋盘种子确保同一棋盘先后各一次）
    r2 = _run_series(agent_p2, agent_p1, half, cf, seed_base=pair_seed, game_workers=game_workers)

    # r2 的 P1 是 agent_p2，P2 是 agent_p1
    # 合并时 r1.p1 和 r2.p2 都是 agent_p1 获胜，r1.p2 和 r2.p1 都是 agent_p2 获胜
    n = num_episodes
    return {
        "p1_win": (r1["p1_win"] * half + r2["p2_win"] * half) / n,
        "p2_win": (r1["p2_win"] * half + r2["p1_win"] * half) / n,
        "draw": (r1["draw"] * half + r2["draw"] * half) / n,
    }


# =============================================================================
# ELO 计算
# =============================================================================


def compute_elo(rating_a: float, rating_b: float, score_a: float, K: float = 32) -> tuple[float, float]:
    """
    计算 ELO 评分变化。
    score_a: agent_a 的胜率 (0~1)，平局算 0.5。
    Returns: (new_rating_a, new_rating_b)
    """
    ea = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    new_a = rating_a + K * (score_a - ea)
    new_b = rating_b + K * (ea - score_a)
    return new_a, new_b


__all__ = [
    "Agent",
    "RandomAgent",
    "PPOAgent",
    "AlphaZeroAgent",
    "MCTSAgent",
    "AlphaZeroMCTSAgent",
    "compete",
    "compute_elo",
]
