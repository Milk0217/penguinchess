#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 设置 Windows 控制台 UTF-8 编码，避免中文乱码
import sys
if sys.platform == "win32":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
    except Exception:
        pass
"""
AlphaZero 自对弈训练 — 使用 MCTS + 神经网络进行自我对弈学习。

训练流程:
  1. 当前网络 vs 自己（MCTS 搜索选择动作）
  2. 保存 (state, MCTS_policy, winner) 到经验池
  3. 从经验池采样训练网络
  4. 定期与历史最佳对战，若胜率 > 55% 则更新最佳

运行方式:
    uv run python penguinchess/ai/train_alphazero.py                        # 默认训练
    uv run python penguinchess/ai/train_alphazero.py --iterations 20        # 20 迭代
    uv run python penguinchess/ai/train_alphazero.py --resume models/alphazero/alphazero_iter_10.pth
"""

import os
import sys
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from penguinchess.core import PenguinChessCore
from penguinchess.ai.mcts_core import mcts_search, mcts_search_batched, select_action
from penguinchess.ai.alphazero_net import AlphaZeroNet

MODELS_DIR = Path(__file__).parent.parent.parent / "models"
ALPHAZERO_DIR = MODELS_DIR / "alphazero"
ALPHAZERO_DIR.mkdir(exist_ok=True)
DATA_DIR = MODELS_DIR / "alphazero_data"
DATA_DIR.mkdir(exist_ok=True)


# =============================================================================
# Python → Rust JSON 状态转换（用于 Rust MCTS）
# =============================================================================

def _core_to_rust_json(core) -> str:
    """
    将 PenguinChessCore 状态转换为 Rust GameState JSON 字符串。
    供 mcts_search_rust 使用。
    """
    import json

    cells = []
    for idx, h in enumerate(core.hexes):
        state_map = {
            "active": "active", "occupied": "occupied",
            "used": "used", "eliminated": "eliminated",
        }
        pts = h.points if h.state in ("active", "occupied") else 0
        cells.append({
            "coord": {"q": h.q, "r": h.r, "s": h.s},
            "state": state_map.get(h.state, "eliminated"),
            "points": pts,
        })

    pieces = []
    for p in core.pieces:
        hex_idx = None
        hex_val = 0
        if p.hex:
            hex_idx = core._hex_map.get((p.hex.q, p.hex.r, p.hex.s))
            hex_val = p.hex.points
        pieces.append({
            "id": p.id,
            "alive": p.alive,
            "hex_idx": hex_idx,
            "hex_value": hex_val,
        })

    phase_str = "placement" if core.phase == core.PHASE_PLACEMENT else "movement"

    state = {
        "board": {"cells": cells},
        "pieces": pieces,
        "scores": list(core.players_scores),
        "current_player": core.current_player,
        "phase": phase_str,
        "terminated": core._terminated,
        "episode_steps": core._episode_steps,
        "placement_count": sum(1 for p in core.pieces if p.hex is not None),
    }

    return json.dumps(state)


# =============================================================================
# 自对弈数据收集
# =============================================================================

def self_play_game(
    net: AlphaZeroNet,
    num_simulations: int = 200,
    temperature: float = 1.0,
    temp_threshold: int = 30,
) -> list:
    """
    使用神经网络 + MCTS 进行一次自对弈。

    Returns:
        [(state_snapshot, policy_target_60, current_player), ...] × steps
    """
    core = PenguinChessCore()
    core.reset(seed=None)
    game_data = []

    # 检测 Rust MCTS 是否可用
    _use_rust_mcts = False
    try:
        from penguinchess.rust_ffi import mcts_search_rust, get_engine
        get_engine()
        _use_rust_mcts = True
    except Exception:
        pass

    for step in range(500):
        # 温度退火
        t = temperature if step < temp_threshold else 0.1

        if _use_rust_mcts:
            # Rust MCTS（快 10-100x，自动利用 GPU 批处理）
            state_json = _core_to_rust_json(core)
            raw_counts = mcts_search_rust(
                state_json,
                model=net,
                num_simulations=num_simulations,
                c_puct=1.4,
                batch_size=128,
            )
            counts = {int(k): v for k, v in raw_counts.items()}
        else:
            # Python MCTS（回退）
            counts, root = mcts_search_batched(
                core, model=None, num_simulations=num_simulations,
                temperature=t, evaluate_fn=net.evaluate_batch,
                batch_size=128,
            )

        total = sum(counts.values())
        policy = np.zeros(60, dtype=np.float32)
        if t > 0:
            for a, c in counts.items():
                policy[a] = c ** (1.0 / t)
        else:
            best_cnt = max(counts.values())
            for a, c in counts.items():
                policy[a] = 1.0 if c == best_cnt else 0.0
        policy /= policy.sum()

        game_data.append((core.get_snapshot(), policy, core.current_player))

        action = select_action(counts, temperature=t)

        obs, reward, terminated, info = core.step(action)
        if terminated:
            break

    winner = _get_winner(core)
    result = []
    for snap, policy, cp in game_data:
        if winner == 2:
            value = 0.0
        else:
            value = 1.0 if cp == winner else -1.0
        result.append((snap, policy, value))

    return result, winner


def _get_winner(core: PenguinChessCore) -> int:
    """0=P1, 1=P2, 2=Draw."""
    s0, s1 = core.players_scores
    if s0 > s1:
        return 0
    elif s1 > s0:
        return 1
    return 2


# =============================================================================
# 模型评估（best_net 对比）
# =============================================================================

def _evaluate_models(
    net_a: AlphaZeroNet,
    net_b: AlphaZeroNet,
    num_games: int = 200,
    num_simulations: int = 800,
) -> float:
    """
    net_a vs net_b 使用 Rust MCTS 对战。

    双方都用 MCTS + 各自的神经网络引导搜索。
    net_a 始终先手（P1）。

    Returns:
        net_a 的胜率 (win + 0.5*draw) / games
    """
    net_a.eval()
    net_b.eval()

    # Try to use Rust MCTS for speed; fall back to Python MCTS
    _use_rust_mcts = False
    try:
        from penguinchess.rust_ffi import mcts_search_rust, get_engine
        from penguinchess.rust_core import RustCore
        _engine = get_engine()
        _use_rust_mcts = True
    except Exception:
        pass

    a_wins = b_wins = draws = 0

    for ep in range(num_games):
        if _use_rust_mcts:
            core = RustCore(engine=_engine).reset(ep)
        else:
            core = PenguinChessCore(seed=ep)
            core.reset()

        terminated = False
        current_net = net_a  # P1 uses net_a

        while not terminated:
            legal = core.get_legal_actions()
            if not legal:
                break

            if _use_rust_mcts:
                # Rust MCTS (returns {str_action: visit_count})
                state_json = core.to_json()
                raw_counts = mcts_search_rust(
                    state_json,
                    model=current_net,
                    num_simulations=num_simulations,
                    c_puct=1.4,
                    batch_size=128,
                )
                counts = {int(k): v for k, v in raw_counts.items()}
            else:
                # Python MCTS (fallback)
                counts, _ = mcts_search_batched(
                    core, model=None, num_simulations=num_simulations,
                    temperature=0.1, evaluate_fn=current_net.evaluate_batch,
                    batch_size=32,
                )

            if not counts:
                break
            action = max(counts, key=counts.__getitem__)

            _, _, terminated, _ = core.step(action)

            # Switch net for next player
            current_net = net_b if current_net is net_a else net_a

        s1, s2 = core.players_scores
        if s1 > s2:
            a_wins += 1
        elif s2 > s1:
            b_wins += 1
        else:
            draws += 1

    if hasattr(core, 'close'):
        core.close()

    score = (a_wins + 0.5 * draws) / num_games
    return score


# =============================================================================
# 训练循环
# =============================================================================

def train_alphazero(
    num_iterations: int = 100,
    games_per_iter: int = 200,
    num_simulations: int = 200,
    eval_simulations: int = 800,
    eval_interval: int = 10,
    eval_games: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    l2_reg: float = 1e-4,
    resume: str | None = None,
):
    """AlphaZero 自对弈训练主循环。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_gpu = torch.cuda.is_available()
    if is_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"设备: {device} ({gpu_name})")
    else:
        print(f"设备: {device}")

    # 预估每迭代时间（基于实际测速 ~110000 sim/s on GPU）
    est_game_s = max(1, games_per_iter * num_simulations * 14 / 110000)
    est_iter_s = est_game_s + 5
    est_total_s = est_iter_s * num_iterations
    print(f"配置: games={games_per_iter}, sims={num_simulations}, "
          f"eval_sims={eval_simulations}, eval_interval={eval_interval}")
    print(f"预估: 每迭代 ~{est_iter_s:.0f}s, 总计 ~{est_total_s//60:.0f}m{est_total_s%60:.0f}s")
    print(f"      (自对弈 ~{est_game_s:.0f}s + 训练 ~5s) × {num_iterations} 迭代")
    print()

    net = AlphaZeroNet().to(device)

    # 加载已有模型（续训）
    if resume:
        if os.path.exists(resume):
            state = torch.load(resume, map_location=device, weights_only=True)
            net.load_state_dict(state)
            print(f"续训练模型: {resume}")
        else:
            print(f"模型文件不存在: {resume}")

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=l2_reg)

    # best_net 跟踪
    best_state = copy.deepcopy(net.state_dict())
    best_iter = 0
    best_win_rate = 0.0
    print(f"初始 best = iter_{best_iter}（训练起点）")
    _start_time = time.time()

    data_buffer = []
    MAX_BUFFER = 100000  # 增大经验池支持更多数据

    for iteration in range(1, num_iterations + 1):
        print(f"\n=== 迭代 {iteration}/{num_iterations} ===")

        # ----- 自对弈 -----
        t0 = time.time()
        iter_data = []
        game_results = []
        for g in range(games_per_iter):
            game_data, winner = self_play_game(net, num_simulations=num_simulations)
            iter_data.extend(game_data)
            game_results.append(winner)

        t1 = time.time()
        win_rate = game_results.count(0) / len(game_results)
        draw_rate = game_results.count(2) / len(game_results)
        print(f"对弈: {games_per_iter}局 ({t1-t0:.0f}s)  P1胜: {win_rate:.0%}  平: {draw_rate:.0%}")
        print(f"  平均步数: {sum(len(g)//3 for g in [iter_data])/len(game_results):.0f}")

        # 加入经验池
        data_buffer.extend(iter_data)
        if len(data_buffer) > MAX_BUFFER:
            data_buffer = data_buffer[-MAX_BUFFER:]
        print(f"  经验池: {len(data_buffer)} 条")

        # ----- 训练 -----
        t2 = time.time()
        net.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = max(1, len(data_buffer) // batch_size)

        for _ in range(num_batches):
            batch = np.random.choice(len(data_buffer), min(batch_size, len(data_buffer)), replace=False)
            snapshots = [data_buffer[i][0] for i in batch]
            policy_targets = [data_buffer[i][1] for i in batch]
            value_targets = [data_buffer[i][2] for i in batch]

            obs_batch = []
            for snap in snapshots:
                core = PenguinChessCore()
                core.reset(seed=None)
                core.restore_snapshot(snap)
                obs = core.get_observation()
                board = np.array(obs["board"], dtype=np.float32).flatten()
                pieces = np.array(obs["pieces"], dtype=np.float32).flatten()
                meta = np.array([float(obs["current_player"]), float(obs["phase"])], dtype=np.float32)
                flat = np.concatenate([board, pieces, meta])
                obs_batch.append(flat)

            obs_tensor = torch.from_numpy(np.array(obs_batch, dtype=np.float32)).to(device)
            policy_tensor = torch.from_numpy(np.array(policy_targets, dtype=np.float32)).to(device)
            value_tensor = torch.from_numpy(np.array(value_targets, dtype=np.float32)).to(device)

            optimizer.zero_grad()
            policy_logits, values = net(obs_tensor)
            policy_loss = F.cross_entropy(policy_logits, policy_tensor)
            value_loss = F.mse_loss(values.squeeze(-1), value_tensor)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        t3 = time.time()
        print(f"训练: {num_batches} batches ({t3-t2:.0f}s)  "
              f"Loss={total_loss/num_batches:.4f}  "
              f"P={total_policy_loss/num_batches:.4f}  "
              f"V={total_value_loss/num_batches:.4f}")

        # ----- 定期保存迭代模型 -----
        if iteration % 10 == 0:
            path = str(ALPHAZERO_DIR / f"alphazero_iter_{iteration}.pth")
            torch.save(net.state_dict(), path)
            print(f"模型已保存: {path}")

        # ----- 评估 vs best_net -----
        if iteration % eval_interval == 0:
            t4 = time.time()
            print(f"  评估: current vs best (iter_{best_iter}) "
                  f"({eval_games}局, {eval_simulations} sims)...", end=" ", flush=True)

            # 构建 best_net 副本（CPU 评估）
            best_net = AlphaZeroNet().to("cpu")
            best_net.load_state_dict(best_state)
            best_net.eval()

            net_cpu = AlphaZeroNet().to("cpu")
            net_cpu.load_state_dict(net.state_dict())
            net_cpu.eval()

            wr = _evaluate_models(net_cpu, best_net,
                                  num_games=eval_games,
                                  num_simulations=eval_simulations)
            t5 = time.time()
            print(f"胜率: {wr:.1%} ({t5-t4:.0f}s)")

            if wr > 0.55:
                best_state = copy.deepcopy(net.state_dict())
                best_iter = iteration
                best_win_rate = wr
                best_path = str(ALPHAZERO_DIR / "alphazero_best.pth")
                torch.save(best_state, best_path)
                print(f"  [BEST] 新 best! iter_{iteration} 胜率 {wr:.1%}")

                # 注册到 Model Registry
                try:
                    from penguinchess.model_registry import register_model, update_evaluation
                    register_model("alphazero_best", "alphazero",
                                   "alphazero/alphazero_best.pth", iteration=iteration)
                    update_evaluation("alphazero_best", {
                        "elo": round(1200 + (wr - 0.5) * 400, 1),
                        "vs_best_prev": {"win": wr, "lose": 1 - wr, "draw": 0.0},
                    })
                except Exception:
                    pass
            else:
                print(f"  保持当前 best (iter_{best_iter}, 胜率 {best_win_rate:.1%})")

    # ----- 训练结束：保存最终模型 -----
    final_path = str(ALPHAZERO_DIR / "alphazero_final.pth")
    torch.save(net.state_dict(), final_path)
    _elapsed = time.time() - _start_time
    print(f"\n训练结束! 最终 best = iter_{best_iter} (胜率 {best_win_rate:.1%})")
    print(f"  总耗时: {_elapsed//60:.0f}m{_elapsed%60:.0f}s")
    print(f"  每迭代平均: {_elapsed/num_iterations:.0f}s")
    print(f"  best 模型: {ALPHAZERO_DIR / 'alphazero_best.pth'}")
    print(f"  最终模型: {final_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AlphaZero 训练")
    parser.add_argument("--iterations", type=int, default=100, help="迭代次数")
    parser.add_argument("--games", type=int, default=100, help="每迭代对局数（默认 100）")
    parser.add_argument("--simulations", type=int, default=400,
                        help="训练 MCTS 模拟次数（默认 400）")
    parser.add_argument("--eval-simulations", type=int, default=800,
                        help="评估 MCTS 模拟次数（默认 800）")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="评估间隔迭代数（默认 10）")
    parser.add_argument("--eval-games", type=int, default=200,
                        help="评估局数（默认 200）")
    parser.add_argument("--batch-size", type=int, default=256, help="训练批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--resume", type=str, default=None, help="续训练模型路径")
    args = parser.parse_args()

    train_alphazero(
        num_iterations=args.iterations,
        games_per_iter=args.games,
        num_simulations=args.simulations,
        eval_simulations=args.eval_simulations,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=args.resume,
    )
