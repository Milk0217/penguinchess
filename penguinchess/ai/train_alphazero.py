#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    uv run python penguinchess/ai/train_alphazero.py --resume models/alphazero/alphazero_resnet_iter_10.pth
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
from penguinchess.ai.alphazero_net import AlphaZeroNet, AlphaZeroResNet, detect_net_arch
from penguinchess._compat import ensure_utf8_stdout
from penguinchess.training_status import update_status as _update_ts, clear_status as _clear_ts
ensure_utf8_stdout()

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
# 观测编码工具（无快照 → 存 flat obs，训练时零 CPU 解码）
# =============================================================================

def _encode_flat_obs(core) -> np.ndarray:
    """将核心状态编码为 206-dim float32 数组。"""
    obs = core.get_observation()
    board = np.array(obs["board"], dtype=np.float32).flatten()
    pieces = np.array(obs["pieces"], dtype=np.float32).flatten()
    meta = np.array([float(obs["current_player"]), float(obs["phase"])], dtype=np.float32)
    return np.concatenate([board, pieces, meta]).astype(np.float32)


# =============================================================================
# 自对弈数据收集
# =============================================================================

def self_play_game(
    net: AlphaZeroNet,
    num_simulations: int = 200,
    temperature: float = 1.0,
    temp_threshold: int = 30,
    parallel_workers: int = 1,
) -> list:
    """
    使用神经网络 + MCTS 进行一次自对弈。
    parallel_workers > 1 时启用根并行 MCTS（集成平均，提升探索质量）。
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
        t = temperature if step < temp_threshold else 0.1

        if _use_rust_mcts:
            # === Rust MCTS（根并行支持）===
            n_workers = max(1, parallel_workers)
            sims_per = max(1, num_simulations // n_workers)
            all_counts = {}

            # 动态 batch_size：确保至少 4 次回调 / 搜索
            bs = min(256, max(32, sims_per // 4))
            for w in range(n_workers):
                state_json = _core_to_rust_json(core)
                raw = mcts_search_rust(
                    state_json, model=net,
                    num_simulations=sims_per,
                    c_puct=3.0, batch_size=bs,
                )
                for k, v in raw.items():
                    key = int(k)
                    all_counts[key] = all_counts.get(key, 0) + v

            counts = all_counts
        else:
            # === Python MCTS（支持根并行）===
            if parallel_workers > 1:
                from penguinchess.ai.mcts_core import mcts_search_parallel
                counts, _ = mcts_search_parallel(
                    core, model=None,
                    num_simulations=num_simulations,
                    temperature=t,
                    evaluate_fn=net.evaluate_batch,
                    num_workers=parallel_workers,
                    batch_size=128,
                )
            else:
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

        # 存 flat obs 替代 snapshot（训练时零 CPU 解码）
        flat_obs = _encode_flat_obs(core)
        game_data.append((flat_obs, policy, core.current_player))

        action = select_action(counts, temperature=t)

        obs, reward, terminated, info = core.step(action)
        if terminated:
            break

    winner = _get_winner(core)
    result = []
    for flat_obs, policy, cp in game_data:
        if winner == 2:
            value = 0.0
        else:
            value = 1.0 if cp == winner else -1.0
        result.append((flat_obs, policy, value))

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
    parallel_workers: int = 4,
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
                # Rust MCTS (根并行集成平均)
                n_workers = max(1, parallel_workers)
                sims_per = max(1, num_simulations // n_workers)
                all_c = {}
                bs = min(256, max(32, sims_per // 4))
                for w in range(n_workers):
                    state_json = core.to_json()
                    raw = mcts_search_rust(
                        state_json, model=current_net,
                        num_simulations=sims_per,
                        c_puct=3.0, batch_size=bs,
                    )
                    for k, v in raw.items():
                        key = int(k)
                        all_c[key] = all_c.get(key, 0) + v
                counts = all_c
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
# 资源监控器
# =============================================================================

class ResourceMonitor:
    """轻量级 CPU/GPU/RAM 监控，每 5 迭代刷新一次 nvidia-smi。"""

    def __init__(self):
        self._peak_gpu_util = 0
        self._peak_gpu_mem = 0
        self._peak_ram = 0
        self._peak_cpu = 0
        self._counter = 0
        self._gpu_name = ""
        self._gpu_total_mem = 0
        try:
            import torch
            if torch.cuda.is_available():
                self._gpu_name = torch.cuda.get_device_name(0)
                self._gpu_total_mem = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        except Exception:
            pass
        # 首次调用采集基线
        self._sample()

    def _sample(self):
        """采集当前资源数据。"""
        import psutil, torch
        # CPU
        self._cpu = psutil.cpu_percent(interval=0)
        # RAM
        mem = psutil.virtual_memory()
        self._ram_used = mem.used / (1024**3)
        self._ram_total = mem.total / (1024**3)
        self._ram_percent = mem.percent
        self._peak_cpu = max(self._peak_cpu, self._cpu)
        self._peak_ram = max(self._peak_ram, self._ram_used)
        # GPU
        if torch.cuda.is_available():
            self._gpu_mem = torch.cuda.memory_allocated(0) // (1024**2)
            self._peak_gpu_mem = max(self._peak_gpu_mem, self._gpu_mem)
            # GPU 利用率每 5 次采一次（nvidia-smi 调用较慢）
            if self._counter % 5 == 0:
                try:
                    import subprocess
                    r = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=2,
                    )
                    self._gpu_util = int(r.stdout.strip())
                    self._peak_gpu_util = max(self._peak_gpu_util, self._gpu_util)
                except Exception:
                    self._gpu_util = 0

    def summary_line(self) -> str:
        """单行资源摘要。"""
        self._counter += 1
        self._sample()
        parts = []
        if self._gpu_name:
            parts.append(f"GPU {self._gpu_util}% {self._gpu_mem}/{self._gpu_total_mem}MB")
        parts.append(f"CPU {self._cpu}% RAM {self._ram_used:.1f}GB")
        return " | ".join(parts)

    def header(self, config_str: str) -> str:
        """起始总览。"""
        lines = []
        lines.append("=" * 65)
        lines.append(" PenguinChess AlphaZero 训练")
        lines.append("=" * 65)
        if self._gpu_name:
            lines.append(f" GPU:    {self._gpu_name}")
        lines.extend(config_str.split("\n"))
        lines.append("-" * 65)
        return "\n".join(lines)

    def footer(self, elapsed: float) -> str:
        """结束时峰值总结。"""
        lines = []
        lines.append("=" * 65)
        lines.append(f" 训练完成! 总耗时 {elapsed//60:.0f}m{elapsed%60:.0f}s")
        if self._gpu_name:
            lines.append(f" GPU 峰值: {self._peak_gpu_util}% | {self._peak_gpu_mem}/{self._gpu_total_mem} MB")
        lines.append(f" CPU 峰值: {self._peak_cpu}% | RAM 峰值: {self._peak_ram:.1f}GB")
        lines.append("=" * 65)
        return "\n".join(lines)


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
    parallel_workers: int = 4,
):
    """AlphaZero 自对弈训练主循环。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_gpu = torch.cuda.is_available()

    # 初始化资源监控
    monitor = ResourceMonitor()

    # 配置摘要
    total_sims = num_simulations * parallel_workers
    config_lines = [
        f" 网络:    AlphaZeroResNet",
        f" 设备:    {str(device).upper()}",
        f" 配置:    games={games_per_iter}, sims={num_simulations}×{parallel_workers}并行={total_sims}总",
        f"          eval_sims={eval_simulations}, eval_interval={eval_interval}",
        f"          batch_size={batch_size}, lr={lr}",
    ]

    print(monitor.header("\n".join(config_lines)))

    # 使用 ResNet 架构；续训时自动检测旧模型架构
    if resume and os.path.exists(resume):
        state = torch.load(resume, map_location=device, weights_only=True)
        NetClass = detect_net_arch(state)
        net = NetClass().to(device)
        net.load_state_dict(state)
        print(f" 续训: {resume} ({NetClass.__name__})")
    else:
        net = AlphaZeroResNet().to(device)
        if resume:
            print(f" 模型不存在: {resume}")
        else:
            print(f" 新网络: AlphaZeroResNet ({sum(p.numel() for p in net.parameters()):,} 参数)")

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=l2_reg)

    # best_net 跟踪
    best_state = copy.deepcopy(net.state_dict())
    best_iter = 0
    best_win_rate = 0.0
    print(f" 初始 best = iter_{best_iter}")
    _start_time = time.time()

    # AMP (Automatic Mixed Precision)
    use_amp = is_gpu
    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
        print(f" AMP:    fp16 mixed precision 启用")

    # TensorBoard 日志
    from torch.utils.tensorboard import SummaryWriter
    tb_log_dir = str(Path(__file__).parent.parent.parent / "models" / "logs" / "alphazero")
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    print(f" TensorBoard: {tb_log_dir}")

    # 表头
    iter_avg_time = 0.0
    print(f"\n{'迭代':>6}  │ 对弈   训练   总耗时   Loss     P-Loss   V-Loss   资源")
    print("-" * 65)

    data_buffer = []
    MAX_BUFFER = 100000

    for iteration in range(1, num_iterations + 1):
        iter_start = time.time()

        # ----- 自对弈 -----
        t0 = time.time()
        iter_data = []
        game_results = []
        for g in range(games_per_iter):
            game_data, winner = self_play_game(
                net, num_simulations=num_simulations,
                parallel_workers=parallel_workers,
            )
            iter_data.extend(game_data)
            game_results.append(winner)

        t1 = time.time()
        game_time = t1 - t0
        win_rate = game_results.count(0) / len(game_results)
        draw_rate = game_results.count(2) / len(game_results)

        # 加入经验池
        data_buffer.extend(iter_data)
        if len(data_buffer) > MAX_BUFFER:
            data_buffer = data_buffer[-MAX_BUFFER:]

        # ----- 训练 -----
        t2 = time.time()
        net.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = max(1, len(data_buffer) // batch_size)

        for _ in range(num_batches):
            batch = np.random.choice(len(data_buffer), min(batch_size, len(data_buffer)), replace=False)
            # flat obs already pre-encoded — zero CPU decode needed
            obs_batch = np.array([data_buffer[i][0] for i in batch], dtype=np.float32)
            policy_targets = np.array([data_buffer[i][1] for i in batch], dtype=np.float32)
            value_targets = np.array([data_buffer[i][2] for i in batch], dtype=np.float32)

            obs_tensor = torch.from_numpy(obs_batch).to(device, non_blocking=True)
            policy_tensor = torch.from_numpy(policy_targets).to(device, non_blocking=True)
            value_tensor = torch.from_numpy(value_targets).to(device, non_blocking=True)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    policy_logits, values = net(obs_tensor)
                    policy_loss = F.cross_entropy(policy_logits, policy_tensor)
                    value_loss = F.mse_loss(values.squeeze(-1), value_tensor)
                    loss = policy_loss + value_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
        train_time = t3 - t2
        iter_elapsed = time.time() - iter_start
        total_elapsed = time.time() - _start_time

        # 更新 ETA
        if iteration <= 3:
            iter_avg_time = iter_elapsed
        else:
            iter_avg_time = iter_avg_time * 0.8 + iter_elapsed * 0.2
        eta_remaining = (num_iterations - iteration) * iter_avg_time

        avg_loss = total_loss / max(1, num_batches)
        avg_p = total_policy_loss / max(1, num_batches)
        avg_v = total_value_loss / max(1, num_batches)

        # 单行迭代摘要
        res_str = monitor.summary_line()
        print(f"  {iteration:>3d}/{num_iterations:<3d}  │ "
              f"{game_time:>4.0f}s {train_time:>4.0f}s "
              f"{total_elapsed:>5.0f}s  "
              f"{avg_loss:.4f} {avg_p:.4f} {avg_v:.4f}  │ {res_str}")

        # TensorBoard 日志
        tb_writer.add_scalar("loss/total", avg_loss, iteration)
        tb_writer.add_scalar("loss/policy", avg_p, iteration)
        tb_writer.add_scalar("loss/value", avg_v, iteration)
        tb_writer.add_scalar("performance/game_time", game_time, iteration)
        tb_writer.add_scalar("performance/train_time", train_time, iteration)
        tb_writer.add_scalar("performance/win_rate", win_rate, iteration)
        tb_writer.add_scalar("performance/draw_rate", draw_rate, iteration)

        # 更新训练状态（供前端仪表盘使用）
        _update_ts(
            is_training=True,
            current_phase="alphazero",
            iteration=iteration,
            total_iterations=num_iterations,
            avg_loss=avg_loss,
            win_rate=win_rate,
        )

        # ----- 定期保存迭代模型 -----
        if iteration % 10 == 0:
            arch_tag = net.arch_name  # "mlp" 或 "resnet"
            path = str(ALPHAZERO_DIR / f"alphazero_{arch_tag}_iter_{iteration}.pth")
            torch.save(net.state_dict(), path)

        # ----- 评估 vs best_net -----
        if iteration % eval_interval == 0:
            t4 = time.time()
            print(f"  ├─ 评估 vs best(iter_{best_iter}) "
                  f"({eval_games}局, {eval_simulations} sims)...", end=" ", flush=True)

            # 构建 best_net 副本（CPU 评估）
            # 自动检测 best_net 和 current_net 的架构
            BestClass = detect_net_arch(best_state)
            best_net = BestClass().to("cpu")
            best_net.load_state_dict(best_state)
            best_net.eval()

            current_state = net.state_dict()
            CurrClass = detect_net_arch(current_state)
            net_cpu = CurrClass().to("cpu")
            net_cpu.load_state_dict(current_state)
            net_cpu.eval()

            wr = _evaluate_models(net_cpu, best_net,
                                  num_games=eval_games,
                                  num_simulations=eval_simulations,
                                  parallel_workers=parallel_workers)
            t5 = time.time()
            print(f"胜率: {wr:.1%} ({t5-t4:.0f}s)")

            # TensorBoard 记录评估结果
            tb_writer.add_scalar("eval/win_rate_vs_best", wr, iteration)
            tb_writer.add_scalar("eval/eval_time", t5 - t4, iteration)

            if wr > 0.55:
                best_state = copy.deepcopy(net.state_dict())
                best_iter = iteration
                best_win_rate = wr
                arch_tag = net.arch_name  # "mlp" 或 "resnet"
                best_path = str(ALPHAZERO_DIR / f"alphazero_{arch_tag}_best.pth")
                torch.save(best_state, best_path)
                print(f"  [BEST] 新 best! iter_{iteration} 胜率 {wr:.1%} ({arch_tag})")

                # 注册到 Model Registry
                try:
                    from penguinchess.model_registry import register_model, update_evaluation
                    model_id = f"az_{arch_tag}_best"
                    rel_path = f"alphazero/alphazero_{arch_tag}_best.pth"
                    register_model(model_id, "alphazero",
                                   rel_path, iteration=iteration)
                    update_evaluation(model_id, {
                        "elo": round(1200 + (wr - 0.5) * 400, 1),
                        "vs_best_prev": {"win": wr, "lose": 1 - wr, "draw": 0.0},
                    })
                except Exception:
                    pass
            else:
                print(f"  保持当前 best (iter_{best_iter}, {best_win_rate:.1%})")

    # ----- 训练结束：保存最终模型 -----
    arch_tag = net.arch_name
    final_path = str(ALPHAZERO_DIR / f"alphazero_{arch_tag}_final.pth")
    torch.save(net.state_dict(), final_path)
    _elapsed = time.time() - _start_time
    print(f"\n{monitor.footer(_elapsed)}")
    print(f"  best 模型: {ALPHAZERO_DIR / f'alphazero_{arch_tag}_best.pth'} (iter_{best_iter}, {arch_tag})")
    print(f"  最终模型: {final_path}")

    # TensorBoard 关闭
    tb_writer.close()
    _clear_ts()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AlphaZero 训练")
    parser.add_argument("--iterations", type=int, default=100, help="迭代次数")
    parser.add_argument("--games", type=int, default=200, help="每迭代对局数（默认 200）")
    parser.add_argument("--simulations", type=int, default=200,
                        help="训练 MCTS 模拟次数（默认 200，配合 --parallel-workers 总模拟量 = simulations × workers）")
    parser.add_argument("--eval-simulations", type=int, default=800,
                        help="评估 MCTS 模拟次数（默认 800）")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="评估间隔迭代数（默认 10）")
    parser.add_argument("--eval-games", type=int, default=200,
                        help="评估局数（默认 200）")
    parser.add_argument("--batch-size", type=int, default=256, help="训练批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--parallel-workers", type=int, default=4,
                        help="根并行 MCTS workers（默认 4，每个 worker 获得 simulations/workers 次模拟）")
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
        parallel_workers=args.parallel_workers,
    )
