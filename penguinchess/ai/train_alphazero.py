#!/usr/bin/env python3
"""
AlphaZero 自对弈训练 — 使用 MCTS + 神经网络进行自我对弈学习。

训练流程:
  1. 当前网络 vs 自己（MCTS 搜索选择动作）
  2. 保存 (state, MCTS_policy, winner) 到经验池
  3. 从经验池采样训练网络
  4. 定期与历史最佳对战，若胜率 > 55% 则更新最佳

运行方式:
    uv run python penguinchess/ai/train_alphazero.py
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
# 自对弈数据收集
# =============================================================================

def self_play_game(
    net: AlphaZeroNet,
    num_simulations: int = 200,
    temperature: float = 1.0,
    temp_threshold: int = 30,  # 前 30 步使用温度
) -> list:
    """
    使用神经网络 + MCTS 进行一次自对弈。
    
    Returns:
        [(state_snapshot, policy_target_60, current_player), ...] × steps
    """
    core = PenguinChessCore()
    core.reset(seed=None)
    game_data = []

    for step in range(500):
        # 温度退火
        t = temperature if step < temp_threshold else 0.1

        # MCTS 搜索（使用批处理推理优化）
        counts, root = mcts_search_batched(
            core, model=None, num_simulations=num_simulations,
            temperature=t, evaluate_fn=net.evaluate_batch,
            batch_size=32,
        )

        # 策略目标 = MCTS 访问次数的分布
        total = sum(counts.values())
        policy = np.zeros(60, dtype=np.float32)
        for a, c in counts.items():
            policy[a] = c ** (1.0 / t) if t > 0 else (1.0 if c == max(counts.values()) else 0.0)
        policy /= policy.sum()

        # 保存训练数据（使用快照）
        game_data.append((core.get_snapshot(), policy, core.current_player))

        # 选择动作
        action = select_action(counts, temperature=t)

        # 执行动作
        obs, reward, terminated, info = core.step(action)
        if terminated:
            break

    # 计算每步的价值目标（基于最终胜负）
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
# 训练循环
# =============================================================================

def train_alphazero(
    num_iterations: int = 100,
    games_per_iter: int = 50,
    num_simulations: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    l2_reg: float = 1e-4,
):
    """AlphaZero 自对弈训练主循环。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    net = AlphaZeroNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=l2_reg)

    data_buffer = []
    MAX_BUFFER = 50000

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
        num_batches = max(1, len(data_buffer) // batch_size)

        for _ in range(num_batches):
            batch = np.random.choice(len(data_buffer), min(batch_size, len(data_buffer)), replace=False)
            snapshots = [data_buffer[i][0] for i in batch]
            policy_targets = [data_buffer[i][1] for i in batch]
            value_targets = [data_buffer[i][2] for i in batch]

            # 从快照恢复并编码为批次
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

        t3 = time.time()
        print(f"训练: {num_batches} batches ({t3-t2:.0f}s)  Loss: {total_loss/num_batches:.4f}")

        # ----- 保存 -----
        if iteration % 10 == 0:
            path = str(ALPHAZERO_DIR / f"alphazero_iter_{iteration}.pth")
            torch.save(net.state_dict(), path)
            print(f"模型已保存: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AlphaZero 训练")
    parser.add_argument("--iterations", type=int, default=100, help="迭代次数")
    parser.add_argument("--games", type=int, default=50, help="每迭代对局数")
    parser.add_argument("--simulations", type=int, default=50, help="MCTS 模拟次数（训练用 50，评估用 800）")
    parser.add_argument("--batch-size", type=int, default=256, help="训练批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    args = parser.parse_args()

    train_alphazero(
        num_iterations=args.iterations,
        games_per_iter=args.games,
        num_simulations=args.simulations,
        batch_size=args.batch_size,
        lr=args.lr,
    )
