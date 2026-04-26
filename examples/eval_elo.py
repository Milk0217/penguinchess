#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ELO 评估脚本 — 支持 Python/Rust 游戏模拟 + PPO/AlphaZero/MCTS Agent。

用法:
    uv run python examples/eval_elo.py                              # 默认 1000 局/对，交换先后手
    uv run python examples/eval_elo.py --episodes 1000              # 全量评估（所有模型 Round-Robin）
    uv run python examples/eval_elo.py --episodes 1000 --incremental # 增量评估（新模型 vs 已有）
    uv run python examples/eval_elo.py --python-core                 # 使用 Python 引擎（默认 Rust）
    uv run python examples/eval_elo.py --gpu                         # PPO 使用 GPU 推理
    uv run python examples/eval_elo.py --stochastic                  # PPO 随机采样（而非 argmax）
    uv run python examples/eval_elo.py --mcts --simulations 200     # Rust MCTS 评估
"""

from __future__ import annotations

# 设置 Windows 控制台 UTF-8，避免中文乱码
import sys
if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except Exception:
        pass

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np

MODELS_DIR = Path(__file__).parent.parent / "models"


def _rust_available() -> bool:
    """检测 Rust DLL 是否可用。"""
    try:
        from penguinchess.rust_ffi import get_engine
        get_engine()
        return True
    except Exception:
        return False


def _gpu_available() -> bool:
    """检测 CUDA GPU 是否可用。"""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# =============================================================================
# Agent 接口
# =============================================================================

class Agent:
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
            return int(np.random.choice(legal))
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
            return int(np.random.choice(legal))
        best = max(result.items(), key=lambda x: x[1])[0]
        return int(best)

    def select_action_serialize(self, core, legal: list[int]) -> int:
        """For debugging: serialize state once, compare MCTS vs direct policy."""
        return self.select_action(core, legal)


# =============================================================================
# 对战引擎
# =============================================================================

def _make_core_factory(use_rust: bool):
    """返回一个 (seed) → core 的工厂函数。"""
    if use_rust:
        from penguinchess.rust_ffi import get_engine
        from penguinchess.rust_core import RustCore
        engine = get_engine()
        return lambda s: RustCore(engine=engine).reset(s)
    else:
        from penguinchess.core import PenguinChessCore
        return lambda s: PenguinChessCore(seed=s).reset()


def _run_series(agent_p1, agent_p2, num_games, core_factory, seed_base=0) -> dict:
    """
    固定方向（agent_p1 始终先手）的系列对战。
    seed_base: 棋盘种子偏移，确保不同系列对局使用不同棋盘。
    返回 {"p1_win", "p2_win", "draw"} 比例。
    """
    a1 = agent_p1 if agent_p1 is not None else RandomAgent(seed=seed_base)
    a2 = agent_p2 if agent_p2 is not None else RandomAgent(seed=seed_base + 1)

    p1_wins = p2_wins = draws = 0
    for ep in range(num_games):
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
            p1_wins += 1
        elif s2 > s1:
            p2_wins += 1
        else:
            draws += 1

    n = num_games
    return {"p1_win": p1_wins / n, "p2_win": p2_wins / n, "draw": draws / n}


def compete(agent_p1, agent_p2, num_episodes, use_rust=False, seed_offset=0):
    """
    agent_p1 与 agent_p2 对战 num_episodes 局。
    自动交换先后手（各一半），消除先手优势导致的 ELO 偏差。
    seed_offset: 棋盘种子偏移，每对模型使用不同棋盘序列。
    None → RandomAgent。
    """
    half = num_episodes // 2
    cf = _make_core_factory(use_rust)

    # 每对模型用独立棋盘种子，避免所有模型都在固定 N 个棋盘上比赛
    pair_seed = seed_offset * 2000

    # 方向 1: agent_p1 先手
    r1 = _run_series(agent_p1, agent_p2, half, cf, seed_base=pair_seed)
    # 方向 2: agent_p2 先手（交换先后手，使用相同棋盘种子确保同一棋盘先后各一次）
    r2 = _run_series(agent_p2, agent_p1, half, cf, seed_base=pair_seed)

    # r2 的 P1 是 agent_p2，P2 是 agent_p1
    # 合并时 r1.p1 和 r2.p2 都是 agent_p1 获胜，r1.p2 和 r2.p1 都是 agent_p2 获胜
    n = num_episodes
    return {
        "p1_win": (r1["p1_win"] * half + r2["p2_win"] * half) / n,
        "p2_win": (r1["p2_win"] * half + r2["p1_win"] * half) / n,
        "draw": (r1["draw"] * half + r2["draw"] * half) / n,
    }


def compute_elo(rating_a, rating_b, score_a, K=32):
    ea = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    new_a = rating_a + K * (score_a - ea)
    new_b = rating_b + K * (ea - score_a)
    return new_a, new_b


# =============================================================================
# 模型发现与加载
# =============================================================================

def discover_models() -> list[dict]:
    models = []
    for p in sorted(MODELS_DIR.glob("ppo_penguinchess_gen_*.zip")):
        try:
            n = int(p.stem.split("_gen_")[1])
        except (IndexError, ValueError):
            continue
        models.append({
            "id": f"ppo_gen_{n}", "type": "ppo",
            "file": f"ppo_penguinchess_gen_{n}.zip",
            "gen": n, "iter": None, "path": str(p),
        })
    az_dir = MODELS_DIR / "alphazero"
    if az_dir.exists():
        for p in sorted(az_dir.glob("alphazero_iter_*.pth")):
            try:
                n = int(p.stem.split("_iter_")[1])
            except (IndexError, ValueError):
                continue
            models.append({
                "id": f"az_iter_{n}", "type": "alphazero",
                "file": f"alphazero/alphazero_iter_{n}.pth",
                "gen": None, "iter": n, "path": str(p),
            })
        # 也检测 best 模型
        best_p = az_dir / "alphazero_best.pth"
        if best_p.exists() and not any(m["id"] == "az_best" for m in models):
            models.append({
                "id": "az_best", "type": "alphazero",
                "file": "alphazero/alphazero_best.pth",
                "gen": None, "iter": 999,
                "path": str(best_p),
            })
    return models


def load_agent(info: dict, use_mcts=False, mcts_simulations=800, deterministic=True, use_gpu=False) -> Agent:
    if info["type"] == "alphazero" and use_mcts:
        # AlphaZero + Rust MCTS: load neural net, guide Rust MCTS search
        import torch
        from penguinchess.ai.alphazero_net import AlphaZeroNet, AlphaZeroResNet, detect_net_arch
        state = torch.load(info["path"], map_location="cpu", weights_only=True)
        NetClass = detect_net_arch(state)
        net = NetClass()
        net.load_state_dict(state)
        net.eval()
        return AlphaZeroMCTSAgent(net, num_simulations=mcts_simulations)
    elif use_mcts:
        return MCTSAgent(num_simulations=mcts_simulations)

    if info["type"] == "ppo":
        from stable_baselines3 import PPO
        device = "cuda" if use_gpu else "cpu"
        model = PPO.load(info["path"], device=device)
        return PPOAgent(model, deterministic=deterministic)
    else:
        import torch
        from penguinchess.ai.alphazero_net import AlphaZeroNet, AlphaZeroResNet, detect_net_arch
        state = torch.load(info["path"], map_location="cpu", weights_only=True)
        NetClass = detect_net_arch(state)
        net = NetClass()
        net.load_state_dict(state)
        net.eval()
        return AlphaZeroAgent(net)


# =============================================================================
# 主流程
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PenguinChess ELO 评估")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="每对 Round-Robin 对战局数（默认 1000，交换先后手各一半）")
    parser.add_argument("--vs-episodes", type=int, default=500,
                        help="vs 随机 AI 对战局数（默认 500，独立精度更高）")
    parser.add_argument("--skip-az", action="store_true", help="跳过 AlphaZero")
    parser.add_argument("--incremental", action="store_true",
                        help="增量模式：跳过双方已有 ELO 的对局")
    parser.add_argument("--python-core", action="store_true",
                        help="使用 Python 游戏引擎（默认自动检测 Rust DLL，回退 Python）")
    parser.add_argument("--gpu", action="store_true",
                        help="PPO 使用 GPU 推理（默认 CPU）")
    parser.add_argument("--mcts", action="store_true",
                        help="使用 Rust MCTS 作为评估 Agent")
    parser.add_argument("--simulations", type=int, default=800,
                        help="MCTS 模拟次数（默认 800）")
    parser.add_argument("--stochastic", action="store_true",
                        help="PPO 使用随机采样而非 argmax（增加方差但更接近实际对局分布）")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行工作进程数（默认 1=顺序执行；>1 使用 ProcessPoolExecutor）")
    args = parser.parse_args()

    all_models = discover_models()
    if args.skip_az:
        all_models = [m for m in all_models if m["type"] != "alphazero"]

    if not all_models:
        print("未找到任何模型！")
        return

    print(f"发现 {len(all_models)} 个模型:")
    for m in all_models:
        print(f"  {m['id']}")
    # 自动检测 Rust DLL，默认启用
    args.use_rust = _rust_available() and not args.python_core
    args.rust_core = args.use_rust  # 兼容旧引用
    args.use_gpu = args.gpu
    det_str = "随机" if args.stochastic else "确定"
    gpu_str = " GPU" if args.use_gpu else ""
    print(f"引擎: {'Rust (stateful)' if args.use_rust else 'Python'}"
          f" | Agent: {'MCTS' if args.mcts else 'PPO/AZ'}"
          f" | 模式: {'增量' if args.incremental else '全量'}"
          f" | PPO: {det_str}{gpu_str}")
    print()

    # 加载模型
    print("加载模型中...")
    agents = {}
    for m in all_models:
        try:
            agents[m["id"]] = load_agent(m, use_mcts=args.mcts, mcts_simulations=args.simulations,
                                          deterministic=not args.stochastic, use_gpu=args.use_gpu)
            print(f"  [OK] {m['id']}")
        except Exception as e:
            print(f"  [FAIL] {m['id']}: {e}")

    model_ids = list(agents.keys())
    if args.mcts:
        print("  [MCTS Agent 对所有 PPO/AZ 模型逐一评估]")

    if len(model_ids) < 2:
        print("至少需要 2 个模型才能进行 ELO 评估")
        return

    from penguinchess.model_registry import (register_model as reg_reg,
                                               update_evaluation as reg_upd,
                                               get_model as reg_get)

    elo_ratings = {mid: 1200.0 for mid in model_ids}
    for mid in model_ids:
        entry = reg_get(mid)
        if entry and entry.get("eval", {}).get("elo") is not None:
            elo_ratings[mid] = entry["eval"]["elo"]

    # 注册所有模型
    for m in all_models:
        if m["id"] not in agents:
            continue
        reg_reg(model_id=m["id"], model_type=m["type"],
                file_path=m["file"], generation=m["gen"], iteration=m["iter"])

    if args.mcts:
        run_mcts_eval(args, agents, model_ids, elo_ratings, reg_get, reg_upd)
    elif args.workers > 1:
        # 并行 Round-Robin（每个 worker 自己加载模型）
        _run_round_robin_parallel(args, all_models, model_ids, elo_ratings, reg_get, reg_upd)
    else:
        # 顺序 Round-Robin
        run_round_robin(args, agents, model_ids, elo_ratings, reg_get, reg_upd)

    # vs_random（并行模式下重新加载模型做 vs_random）
    if args.workers > 1 and not args.incremental:
        _run_vs_random_parallel(args, all_models, model_ids, reg_get, reg_upd, elo_ratings)
    else:
        run_vs_random(args, agents, model_ids, reg_get, reg_upd, elo_ratings)

    agents.clear()
    print("\n完成!")


def run_round_robin(args, agents, model_ids, elo_ratings, reg_get, reg_upd):
    """标准 Round-Robin ELO 评估（支持增量和 Rust 引擎）。"""
    print(f"\n开始{'增量 ' if args.incremental else ''}Round-Robin 评估 ({args.episodes} 局/对)...")
    print(f"游戏引擎: {'Rust' if args.rust_core else 'Python'}")
    start_time = time.time()

    num_pairs = len(model_ids) * (len(model_ids) - 1) // 2
    pair_done = 0
    pair_skipped = 0

    pair_idx = 0
    for i in range(len(model_ids)):
        for j in range(i + 1, len(model_ids)):
            mid_a, mid_b = model_ids[i], model_ids[j]

            if args.incremental:
                ea = reg_get(mid_a)
                eb = reg_get(mid_b)
                if (ea and ea.get("eval", {}).get("elo") is not None and
                    eb and eb.get("eval", {}).get("elo") is not None):
                    pair_skipped += 1
                    continue

            pair_done += 1
            total = num_pairs - pair_skipped
            print(f"  [{pair_done}/{total}] {mid_a} vs {mid_b} ...", end=" ", flush=True)

            result = compete(agents[mid_a], agents[mid_b], args.episodes,
                             use_rust=args.rust_core, seed_offset=pair_idx)
            pair_idx += 1
            score_a = result["p1_win"] + 0.5 * result["draw"]
            new_a, new_b = compute_elo(elo_ratings[mid_a], elo_ratings[mid_b], score_a)
            elo_ratings[mid_a], elo_ratings[mid_b] = new_a, new_b
            print(f"胜 {result['p1_win']:.1%} 负 {result['p2_win']:.1%} 平 {result['draw']:.1%}"
                  f" | ELO: {mid_a}={new_a:.0f} {mid_b}={new_b:.0f}")

            reg_upd(mid_a, {"elo": round(elo_ratings[mid_a], 1)})
            reg_upd(mid_b, {"elo": round(elo_ratings[mid_b], 1)})

    elapsed = time.time() - start_time
    if pair_skipped > 0:
        print(f"  跳过 {pair_skipped} 个已有 ELO 的对局")
    print(f"\n=== Round-Robin 完成! 耗时: {elapsed:.0f}s ===")

    ranked = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'排名':>4} {'模型':<16} {'ELO':>6}")
    print("-" * 30)
    for rank, (mid, elo) in enumerate(ranked, 1):
        print(f"{rank:>4} {mid:<16} {elo:>6.0f}")


def run_mcts_eval(args, agents, model_ids, elo_ratings, reg_get, reg_upd):
    """MCTS 评估：每个模型 vs Rust MCTS（不是 RR）。"""
    print(f"\nMCTS 评估 ({args.simulations} sims, {args.episodes} 局/模型)...")
    start_time = time.time()
    mcts_agent = MCTSAgent(num_simulations=args.simulations)

    for i, mid in enumerate(model_ids):
        print(f"  {mid} vs MCTS ...", end=" ", flush=True)
        result = compete(agents[mid], mcts_agent, args.episodes,
                         use_rust=args.rust_core, seed_offset=i)
        # 胜率 → 近似 ELO
        score = result["p1_win"] + 0.5 * result["draw"]
        elo_ratings[mid] = 1200 + (score - 0.5) * 400
        print(f"胜 {result['p1_win']:.1%} 平 {result['draw']:.1%} 负 {result['p2_win']:.1%}"
              f" | 近似 ELO={elo_ratings[mid]:.0f}")
        reg_upd(mid, {"elo": round(elo_ratings[mid], 1),
                       "vs_mcts": {"win": result["p1_win"], "lose": result["p2_win"],
                                    "draw": result["draw"]}})

    elapsed = time.time() - start_time
    print(f"\n=== MCTS 评估完成! 耗时: {elapsed:.0f}s ===")
    ranked = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    print(f"{'排名':>4} {'模型':<16} {'ELO':>6}")
    print("-" * 30)
    for rank, (mid, elo) in enumerate(ranked, 1):
        print(f"{rank:>4} {mid:<16} {elo:>6.0f}")


def run_vs_random(args, agents, model_ids, reg_get, reg_upd, elo_ratings):
    """vs 随机 AI 评估（增量更新，独立局数）。"""
    vs_eps = args.vs_episodes
    print(f"\n--- vs 随机 AI 评估 ({vs_eps} 局，交换先后手) ---")
    for i, mid in enumerate(model_ids):
        entry = reg_get(mid)
        if entry and entry.get("eval", {}).get("vs_random") is not None:
            print(f"  [SKIP] {mid}: 已有 vs_random 数据")
            continue

        if mid not in agents:
            continue
        print(f"  {mid} vs 随机 ...", end=" ", flush=True)
        result = compete(agents[mid], None, vs_eps,
                         use_rust=args.rust_core, seed_offset=i + 10000)
        vr = {"win": result["p1_win"], "lose": result["p2_win"], "draw": result["draw"]}
        existing = (entry.get("eval") or {}) if entry else {}
        eval_data = dict(existing)
        eval_data["vs_random"] = vr
        reg_upd(mid, eval_data)
        print(f"胜 {vr['win']:.1%} 负 {vr['lose']:.1%} 平 {vr['draw']:.1%}")

    # 最终排名
    ranked = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'='*50}")
    print(f"{'排名':>4} {'模型':<16} {'ELO':>6}  {'vs随机':>8}")
    print("-" * 45)
    for rank, (mid, elo) in enumerate(ranked, 1):
        entry = reg_get(mid)
        vr = (entry.get("eval") or {}).get("vs_random", {}) if entry else {}
        wr = vr.get("win", 0)
        print(f"{rank:>4} {mid:<16} {elo:>6.0f}  {wr:.0%}")
    print(f"{'='*50}")


# =============================================================================
# 并行 Round-Robin
# =============================================================================

def _worker_run_pair(info_a, info_b, num_episodes, use_rust, seed_offset,
                     use_mcts, mcts_simulations, deterministic, use_gpu=False) -> dict:
    """模块级 worker：加载模型，运行 compete，返回结果。"""
    agent_a = load_agent(info_a, use_mcts=use_mcts,
                         mcts_simulations=mcts_simulations,
                         deterministic=deterministic,
                         use_gpu=use_gpu) if info_a else None
    agent_b = load_agent(info_b, use_mcts=use_mcts,
                         mcts_simulations=mcts_simulations,
                         deterministic=deterministic,
                         use_gpu=use_gpu) if info_b else None
    return compete(agent_a, agent_b, num_episodes,
                   use_rust=use_rust, seed_offset=seed_offset)


def _run_round_robin_parallel(args, all_models, model_ids, elo_ratings, reg_get, reg_upd):
    """并行 Round-Robin：每个 pair 在独立进程中运行。"""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"\n开始{'增量 ' if args.incremental else ''}Round-Robin 评估 "
          f"({args.episodes} 局/对, {args.workers} workers)...")
    print(f"游戏引擎: {'Rust' if args.rust_core else 'Python'}")
    start_time = time.time()

    # 构建 model_info 查找表
    info_map = {m["id"]: m for m in all_models}

    # 构建 pairs 列表
    pairs = []
    skipped = 0
    for i in range(len(model_ids)):
        for j in range(i + 1, len(model_ids)):
            mid_a, mid_b = model_ids[i], model_ids[j]

            if args.incremental:
                ea = reg_get(mid_a)
                eb = reg_get(mid_b)
                if (ea and ea.get("eval", {}).get("elo") is not None and
                    eb and eb.get("eval", {}).get("elo") is not None):
                    skipped += 1
                    continue

            pairs.append((mid_a, mid_b))

    num_pairs = len(pairs)
    if num_pairs == 0:
        print(f"  跳过 {skipped} 个已有 ELO 的对局")
        elapsed = time.time() - start_time
        print(f"\n=== Round-Robin 完成! 耗时: {elapsed:.0f}s ===")
        return

    print(f"  共 {num_pairs} 对 ({skipped} 跳过)")

    # 提交到进程池
    pool = ProcessPoolExecutor(max_workers=args.workers)
    futures = {}
    try:
        for pair_idx, (mid_a, mid_b) in enumerate(pairs):
            future = pool.submit(
                _worker_run_pair,
                info_map[mid_a], info_map[mid_b],
                args.episodes, args.rust_core, pair_idx * 2000,
                args.mcts, args.simulations, not args.stochastic,
                args.use_gpu,
            )
            futures[future] = (mid_a, mid_b)

        done = 0
        for future in as_completed(futures):
            mid_a, mid_b = futures[future]
            done += 1
            try:
                result = future.result()
            except Exception as e:
                print(f"\n  [FAIL] {mid_a} vs {mid_b}: {e}")
                continue

            score_a = result["p1_win"] + 0.5 * result["draw"]
            new_a, new_b = compute_elo(elo_ratings[mid_a], elo_ratings[mid_b], score_a)
            elo_ratings[mid_a], elo_ratings[mid_b] = new_a, new_b
            print(f"  [{done}/{num_pairs}] {mid_a} vs {mid_b}: "
                  f"胜 {result['p1_win']:.1%} 负 {result['p2_win']:.1%} 平 {result['draw']:.1%}"
                  f" | ELO: {mid_a}={new_a:.0f} {mid_b}={new_b:.0f}")

            reg_upd(mid_a, {"elo": round(elo_ratings[mid_a], 1)})
            reg_upd(mid_b, {"elo": round(elo_ratings[mid_b], 1)})

    except KeyboardInterrupt:
        print("\n[中断] 正在取消剩余任务...")
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=False)

    elapsed = time.time() - start_time
    print(f"\n=== Round-Robin 完成! 耗时: {elapsed:.0f}s ===")

    ranked = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'排名':>4} {'模型':<16} {'ELO':>6}")
    print("-" * 30)
    for rank, (mid, elo) in enumerate(ranked, 1):
        print(f"{rank:>4} {mid:<16} {elo:>6.0f}")


def _run_vs_random_parallel(args, all_models, model_ids, reg_get, reg_upd, elo_ratings):
    """并行 vs_random。"""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    vs_eps = args.vs_episodes
    print(f"\n--- vs 随机 AI 评估 ({vs_eps} 局, {args.workers} workers) ---")

    info_map = {m["id"]: m for m in all_models}
    pending = []
    for i, mid in enumerate(model_ids):
        entry = reg_get(mid)
        if entry and entry.get("eval", {}).get("vs_random") is not None:
            print(f"  [SKIP] {mid}: 已有 vs_random 数据")
            continue
        if mid not in info_map:
            continue
        pending.append((mid, i))

    if not pending:
        # 打印最终排名
        _print_final_ranking(elo_ratings, reg_get, model_ids)
        return

    pool = ProcessPoolExecutor(max_workers=args.workers)
    futures = {}
    try:
        for mid, i in pending:
            future = pool.submit(
                _worker_run_pair,
                info_map[mid], None,
                vs_eps, args.rust_core, i + 10000,
                False, 0, True, False,
            )
            futures[future] = mid

        done = 0
        for future in as_completed(futures):
            mid = futures[future]
            done += 1
            try:
                result = future.result()
            except Exception as e:
                print(f"\n  [FAIL] {mid}: {e}")
                continue

            vr = {"win": result["p1_win"], "lose": result["p2_win"], "draw": result["draw"]}
            entry = reg_get(mid)
            existing = (entry.get("eval") or {}) if entry else {}
            eval_data = dict(existing)
            eval_data["vs_random"] = vr
            reg_upd(mid, eval_data)
            print(f"  [{done}/{len(pending)}] {mid}: 胜 {vr['win']:.1%} 负 {vr['lose']:.1%} 平 {vr['draw']:.1%}")

    except KeyboardInterrupt:
        print("\n[中断] 正在取消剩余任务...")
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=False)

    _print_final_ranking(elo_ratings, reg_get, model_ids)


def _print_final_ranking(elo_ratings, reg_get, model_ids):
    """打印最终 ELO 排名。"""
    ranked = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'='*50}")
    print(f"{'排名':>4} {'模型':<16} {'ELO':>6}  {'vs随机':>8}")
    print("-" * 45)
    for rank, (mid, elo) in enumerate(ranked, 1):
        entry = reg_get(mid)
        vr = (entry.get("eval") or {}).get("vs_random", {}) if entry else {}
        wr = vr.get("win", 0)
        print(f"{rank:>4} {mid:<16} {elo:>6.0f}  {wr:.0%}")
    print(f"{'='*50}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
