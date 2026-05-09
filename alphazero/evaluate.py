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
    uv run python examples/eval_elo.py --nnue-depth 6               # NNUE Alpha-Beta 深度 6
"""

from __future__ import annotations

from penguinchess._compat import ensure_utf8_stdout
ensure_utf8_stdout()

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
# Agent 接口（从 eval_utils 导入）
# =============================================================================

from penguinchess.eval_utils import (
    Agent,
    RandomAgent,
    PPOAgent,
    AlphaZeroAgent,
    MCTSAgent,
    AlphaZeroMCTSAgent,
    compete,
    compute_elo,
    _make_core_factory,
)

# =============================================================================
# 模型发现与加载
# =============================================================================

def _detect_az_arch(pth_path: str) -> str:
    """从文件名或 state_dict 检测 AZ 模型架构。"""
    stem = Path(pth_path).stem
    parts = stem.split("_")
    # 新命名: alphazero_resnet_iter_10 → parts = [alphazero, arch, ...]
    #          alphazero_resnet_xl_iter_10 → parts = [alphazero, resnet, xl, ...]
    if len(parts) >= 3 and parts[1] in ("resnet", "mlp"):
        if len(parts) >= 4 and parts[2] in ("xl", "large", "configurable"):
            return f"{parts[1]}_{parts[2]}"
        return parts[1]
    # 旧命名: alphazero_iter_10 → 加载 state_dict 检测
    try:
        import torch
        from penguinchess.ai.alphazero_net import detect_net_arch
        state = torch.load(pth_path, map_location="cpu", weights_only=True)
        cls = detect_net_arch(state)
        return "resnet" if cls.__name__ == "AlphaZeroResNet" else "mlp"
    except Exception:
        return "legacy"


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
        # 扫描所有 AZ 模型文件（新命名 + 旧命名）
        for p in sorted(az_dir.glob("alphazero_*.pth")):
            stem = p.stem
            parts = stem.split("_")
            arch = _detect_az_arch(str(p))

            # 解析多段式架构名: alphazero_resnet_xl_iter_N.pth
            # parts = ["alphazero", "resnet", "xl", "iter", "N"]
            # Single-part arch: parts[1]=resnet/mlp; 2-part: parts[1]_parts[2]
            p1 = parts[1] if len(parts) > 1 else ""
            p2 = parts[2] if len(parts) > 2 else ""
            p3 = parts[3] if len(parts) > 3 else ""
            _2part_archs = {"xl", "large", "configurable"}

            def _is_iter(p) -> bool: return p in ("iter", "best", "final")

            # Single arch + iter: alphazero_resnet_iter_10 (parts=4)
            if p1 in ("resnet", "mlp") and p2 == "iter" and len(parts) == 4:
                try:
                    n = int(p3)
                except (ValueError, IndexError):
                    continue
                mid = f"az_{arch}_iter_{n}"
                models.append({
                    "id": mid, "type": "alphazero", "arch": arch,
                    "file": f"alphazero/{stem}.pth",
                    "gen": None, "iter": n, "path": str(p),
                })
            # 2-part arch + iter: alphazero_resnet_xl_iter_10 (parts=5)
            elif p1 == "resnet" and p2 in _2part_archs and p3 == "iter" and len(parts) == 5:
                try:
                    n = int(parts[4])
                except (ValueError, IndexError):
                    continue
                mid = f"az_{arch}_iter_{n}"
                models.append({
                    "id": mid, "type": "alphazero", "arch": arch,
                    "file": f"alphazero/{stem}.pth",
                    "gen": None, "iter": n, "path": str(p),
                })
            # Single arch + best/final: alphazero_resnet_best.pth
            elif p1 in ("resnet", "mlp") and p2 in ("best", "final") and len(parts) == 3:
                mid = f"az_{arch}_{p2}"
                if not any(m["id"] == mid for m in models):
                    models.append({
                        "id": mid, "type": "alphazero", "arch": arch,
                        "file": f"alphazero/{stem}.pth",
                        "gen": None, "iter": 999 if p2 == "best" else None,
                        "path": str(p),
                    })
            # 2-part arch + best/final: alphazero_resnet_xl_best.pth
            elif p1 == "resnet" and p2 in _2part_archs and p3 in ("best", "final") and len(parts) == 4:
                mid = f"az_{arch}_{p3}"
                if not any(m["id"] == mid for m in models):
                    models.append({
                        "id": mid, "type": "alphazero", "arch": arch,
                        "file": f"alphazero/{stem}.pth",
                        "gen": None, "iter": 999 if p3 == "best" else None,
                        "path": str(p),
                    })
            # 旧命名: alphazero_best.pth
            elif stem == "alphazero_best":
                if not any(m["id"] == "az_best" for m in models):
                    models.append({
                        "id": "az_best", "type": "alphazero", "arch": arch,
                        "file": f"alphazero/{stem}.pth",
                        "gen": None, "iter": 999,
                        "path": str(p),
                    })
            # 旧命名: alphazero_iter_N.pth
            elif p1 == "iter" and len(parts) == 3:
                try:
                    n = int(p2)
                except (ValueError, IndexError):
                    continue
                models.append({
                    "id": f"az_iter_{n}", "type": "alphazero", "arch": arch,
                    "file": f"alphazero/{stem}.pth",
                    "gen": None, "iter": n, "path": str(p),
                })
            # 旧命名: alphazero_final.pth
            elif stem == "alphazero_final":
                if not any(m["id"] == "az_final" for m in models):
                    models.append({
                        "id": "az_final", "type": "alphazero", "arch": arch,
                        "file": f"alphazero/{stem}.pth",
                        "gen": None, "iter": None,
                        "path": str(p),
                    })
    return models


def discover_nnue_models() -> list[dict]:
    """Discover NNUE models in models/nnue/ directory."""
    nnue_dir = MODELS_DIR / "nnue"
    if not nnue_dir.exists():
        return []
    models = []
    for p in sorted(nnue_dir.glob("nnue_gen_*.pt")):
        try:
            n = int(p.stem.split("_gen_")[1])
        except (IndexError, ValueError):
            continue
        models.append({
            "id": f"nnue_gen_{n}", "type": "nnue",
            "file": f"nnue/{p.name}",
            "gen": n, "iter": None, "path": str(p),
        })
    return models


def load_agent(info: dict, use_mcts=False, mcts_simulations=800, deterministic=True, use_gpu=False) -> Agent:
    if info["type"] == "nnue":
        # NNUE + Alpha-Beta Agent
        import torch
        from penguinchess.ai.nnue import NNUE
        from penguinchess.ai.nnue_agent import NNUEAgent
        state = torch.load(info["path"], map_location="cpu", weights_only=True)
        model = NNUE()
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        model.eval()
        if use_gpu and torch.cuda.is_available():
            model = model.cuda()
        depth = getattr(load_agent, 'nnue_depth', 4)
        return NNUEAgent(model, max_depth=depth)

    if info["type"] == "alphazero":
        # AZ 模型默认使用 MCTS 搜索（它的核心优势所在）
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
                        help="非 AZ 模型也使用 Rust MCTS 评估（默认 AZ 已启用 MCTS，PPO 使用自身策略）")
    parser.add_argument("--simulations", type=int, default=800,
                        help="MCTS 模拟次数（默认 800）")
    parser.add_argument("--stochastic", action="store_true",
                        help="PPO 使用随机采样而非 argmax（增加方差但更接近实际对局分布）")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行工作进程数（默认 1=顺序执行；>1 使用 ProcessPoolExecutor）")
    parser.add_argument("--game-workers", type=int, default=8,
                        help="每场比赛并行对局数（默认 8，大幅加速 ELO 评估）")
    parser.add_argument("--nnue-depth", type=int, default=4,
                        help="NNUE Alpha-Beta 搜索深度（默认 4）")
    args = parser.parse_args()

    # 将 nnue_depth 设置为 load_agent 的类属性（在 load_agent 内部读取）
    load_agent.nnue_depth = args.nnue_depth

    all_models = discover_models()
    all_models.extend(discover_nnue_models())
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
    has_nnue = any(m["type"] == "nnue" for m in all_models)
    nnue_str = f" | NNUE: AB({args.nnue_depth})" if has_nnue else ""
    print(f"引擎: {'Rust (stateful)' if args.use_rust else 'Python'}"
          f" | AZ: MCTS({args.simulations} sims){nnue_str}"
          f" | PPO: {'自身策略'}{gpu_str}"
          f" | 模式: {'增量' if args.incremental else '全量'}"
          f" | {'确定' if not args.stochastic else '随机'}")
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
                file_path=m["file"], generation=m["gen"],
                iteration=m["iter"], arch=m.get("arch"))

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
                             use_rust=args.rust_core, seed_offset=pair_idx,
                             game_workers=args.game_workers)
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
                         use_rust=args.rust_core, seed_offset=i,
                         game_workers=args.game_workers)
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
                         use_rust=args.rust_core, seed_offset=i + 10000,
                         game_workers=args.game_workers)
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
    """并行 Round-Robin：每个 pair 在独立进程中运行，ELO 按固定顺序计算。

    双阶段策略保证结果确定性：
    Phase 1 (并行): 运行所有 compete()，收集原始胜负结果
    Phase 2 (顺序): 按 pair_idx 固定顺序计算 ELO，确保结果与顺序模式一致
    """
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

    # =====================================================================
    # Phase 1: 并行运行所有对战（昂贵的部分，ProcessPoolExecutor）
    # =====================================================================
    pool = ProcessPoolExecutor(max_workers=args.workers)
    futures = {}
    try:
        for pair_idx, (mid_a, mid_b) in enumerate(pairs):
            # NOTE: seed_offset = pair_idx (不是 pair_idx * 2000)
            # compete() 内部已经做了 seed_offset * 2000
            future = pool.submit(
                _worker_run_pair,
                info_map[mid_a], info_map[mid_b],
                args.episodes, args.rust_core, pair_idx,
                args.mcts, args.simulations, not args.stochastic,
                args.use_gpu,
            )
            futures[future] = (mid_a, mid_b, pair_idx)

        # 等待所有任务完成，不在此阶段计算 ELO
        for future in as_completed(futures):
            pass

        # =====================================================================
        # Phase 2: 收集原始结果（无 ELO 计算）
        # =====================================================================
        results: dict[int, tuple[str, str, dict | None]] = {}
        for future, (mid_a, mid_b, pair_idx) in futures.items():
            try:
                results[pair_idx] = (mid_a, mid_b, future.result())
            except Exception as e:
                results[pair_idx] = (mid_a, mid_b, None)
                print(f"\n  [FAIL] {mid_a} vs {mid_b}: {e}")

        # =====================================================================
        # Phase 3: 按 pair_index 固定顺序计算 ELO（确定性的关键！）
        # =====================================================================
        for pair_idx in sorted(results.keys()):
            mid_a, mid_b, result = results[pair_idx]
            if result is None:
                continue

            score_a = result["p1_win"] + 0.5 * result["draw"]
            new_a, new_b = compute_elo(elo_ratings[mid_a], elo_ratings[mid_b], score_a)
            elo_ratings[mid_a], elo_ratings[mid_b] = new_a, new_b
            print(f"  [{pair_idx + 1}/{num_pairs}] {mid_a} vs {mid_b}: "
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
