#!/usr/bin/env python3
"""
PPO 训练脚本 — 使用 Stable-Baselines3 训练 PenguinChess 智能体。

本机配备 NVIDIA GPU 时，PyTorch 会自动使用 GPU 加速训练（`device="auto"`）。
可通过 `nvidia-smi` 确认 GPU 状态。

运行方式:
    cd /mnt/e/programming/penguinchess
    uv run python examples/train_ppo.py              # 默认训练 100k 步
    uv run python examples/train_ppo.py --timesteps 500000  # 训练 500k 步
    uv run python examples/train_ppo.py --evaluate-only     # 仅评估已有模型

GPU 说明:
    MLP 策略网络在 GPU 上的加速效果有限，小规模训练时 CPU 可能更快。
    如需使用 GPU 训练，脚本默认 device="auto" 会自动检测。
    也可强制使用 CPU: --device cpu
"""

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from penguinchess.env import PenguinChessEnv

MODELS_DIR = Path(__file__).parent.parent / "models"
LOGS_DIR = MODELS_DIR / "logs"
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# =============================================================================
# 环境工厂
# =============================================================================

def make_env(rank: int = 0, seed: int = 0) -> callable:
    """创建 PenguinChess 环境实例（用于 VecEnv 的工厂函数）。"""
    def _init():
        env = gym.make("PenguinChess-v0")
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def create_vec_env(num_envs: int = 1, seed: int = 0) -> VecNormalize:
    """创建向量化环境（自动选择并行加速或单进程）。"""
    if num_envs > 1:
        vec_env = SubprocVecEnv([make_env(i, seed) for i in range(num_envs)])
    else:
        vec_env = DummyVecEnv([make_env(0, seed)])
    return VecNormalize(vec_env, norm_obs=True, norm_reward=False)


# =============================================================================
# 对战/评估工具
# =============================================================================

def compete(
    model_p1: PPO | None,
    model_p2: PPO | None,
    num_episodes: int = 100,
    deterministic: bool = True,
) -> dict:
    """
    model_p1 与 model_p2 对战 num_episodes 局。
    
    如果某个 model 为 None，则该方使用随机策略。
    
    Returns:
        {"p1_win": float, "p2_win": float, "draw": float,
         "p1_avg_reward": float, "p2_avg_reward": float}
    """
    env = gym.make("PenguinChess-v0")
    p1_wins = 0; p2_wins = 0; draws = 0
    p1_total_r = 0.0; p2_total_r = 0.0

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        terminated = False; truncated = False

        while not terminated and not truncated:
            # P1 选动作
            if model_p1:
                act_arr, _ = model_p1.predict(obs, deterministic=deterministic)
                action = int(act_arr.item())
            else:
                legal = info.get("valid_actions", [])
                action = int(np.random.choice(legal)) if legal else 0
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated: break

            # P2 选动作
            if model_p2:
                act_arr, _ = model_p2.predict(obs, deterministic=deterministic)
                action = int(act_arr.item())
            else:
                legal = info.get("valid_actions", [])
                action = int(np.random.choice(legal)) if legal else 0
            obs, reward, terminated, truncated, info = env.step(action)

        winner = info.get("winner")
        if winner is None and truncated:
            # 截断时按分数判胜负
            scores = info.get("scores", [0, 0])
            if scores[0] > scores[1]:
                winner = 0
            elif scores[1] > scores[0]:
                winner = 1
            else:
                winner = 2
        if winner == 0: p1_wins += 1
        elif winner == 1: p2_wins += 1
        elif winner == 2: draws += 1

    env.close()
    n = num_episodes
    return {
        "p1_win": p1_wins / n, "p2_win": p2_wins / n, "draw": draws / n,
    }


# =============================================================================
# 主训练流程
# =============================================================================

def get_next_gen() -> int:
    """返回下一个可用的 generation 编号。"""
    existing = [int(p.stem.split("_gen_")[1]) for p in MODELS_DIR.glob("ppo_penguinchess_gen_*.zip") if "_gen_" in p.stem]
    return (max(existing) + 1) if existing else 1


def compute_elo(rating_a: float, rating_b: float, score_a: float, K: float = 32) -> tuple:
    """计算 ELO 评分变化。score_a: P1 胜率 (0~1)。"""
    ea = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    new_a = rating_a + K * (score_a - ea)
    new_b = rating_b + K * (ea - score_a)
    return new_a, new_b


def train(args):
    """训练 PPO 模型。"""
    print("=" * 60)
    print("PPO 训练 - PenguinChess")
    print("=" * 60)

    device = "cpu"  # MLP 策略在 CPU 更快
    if args.force_gpu:
        device = "cuda"

    # 创建并行环境（加速数据收集）
    num_envs = args.num_envs
    vec_env = create_vec_env(num_envs, seed=0)
    print(f"环境: {num_envs} 个并行, 设备: {device}")

    # 学习率衰减
    lr_schedule = args.lr

    # 模型配置
    if args.resume and os.path.exists(args.resume):
        print(f"续训练模型: {args.resume}")
        model = PPO.load(args.resume, env=vec_env, device=device,
                         learning_rate=lr_schedule,
                         n_steps=args.n_steps // max(num_envs, 1),
                         batch_size=args.batch_size,
                         n_epochs=args.n_epochs,
                         gamma=args.gamma,
                         clip_range=args.clip_range,
                         ent_coef=args.ent_coef,
                         vf_coef=0.5,
                         max_grad_norm=0.5,
                         tensorboard_log=str(LOGS_DIR),
                         verbose=1)
    else:
        model = PPO("MlpPolicy", vec_env,
                     learning_rate=lr_schedule,
                     n_steps=args.n_steps // max(num_envs, 1),
                     batch_size=args.batch_size,
                     n_epochs=args.n_epochs,
                     gamma=args.gamma, gae_lambda=0.95,
                     clip_range=args.clip_range,
                     ent_coef=args.ent_coef,
                     vf_coef=0.5, max_grad_norm=0.5,
                     tensorboard_log=str(LOGS_DIR), verbose=1, device=device)

    # 训练
    print(f"\n开始训练 {args.timesteps} 步...")
    start_time = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    elapsed = time.time() - start_time
    print(f"训练完成! 耗时: {elapsed:.1f}s ({args.timesteps / elapsed:.0f} steps/s)")

    # 保存为新一代
    gen = get_next_gen()
    gen_path = str(MODELS_DIR / f"ppo_penguinchess_gen_{gen}.zip")
    model.save(gen_path)
    print(f"已保存: gen_{gen}")

    # ===== 对战评估 & ELO =====
    print(f"\n--- Gen {gen} 对战评估 ---")

    # 1) 对随机 AI
    r = compete(model, None, args.eval_episodes)
    print(f"vs 随机 AI:  胜 {r['p1_win']:.1%}  负 {r['p2_win']:.1%}  平 {r['draw']:.1%}")

    # 2) 对前几代（交叉评估）
    elo_ratings = {g: 1200.0 for g in range(1, gen + 1)}  # 初始 ELO 1200
    eval_models = {}
    for g in range(1, gen + 1):
        p = MODELS_DIR / f"ppo_penguinchess_gen_{g}.zip"
        if p.exists():
            eval_models[g] = PPO.load(str(p), device="cpu")

    # 当前代与所有前代对战
    total_wins = 0
    total_games = 0
    for g in range(1, gen):
        if g in eval_models:
            r2 = compete(model, eval_models[g], args.eval_episodes)
            score_a = r2['p1_win'] + 0.5 * r2['draw']
            elo_ratings[gen], elo_ratings[g] = compute_elo(elo_ratings[gen], elo_ratings[g], score_a)
            total_wins += r2['p1_win']
            total_games += 1
            rival = f"gen_{g}"
            if g == gen - 1:
                print(f"vs {rival}:      胜 {r2['p1_win']:.1%}  负 {r2['p2_win']:.1%}  平 {r2['draw']:.1%}  ELO={elo_ratings[gen]:.0f}")
            else:
                print(f"vs {rival}:      胜 {r2['p1_win']:.1%}  负 {r2['p2_win']:.1%}  平 {r2['draw']:.1%}")

    if total_games > 0 and total_wins / total_games > 0.5:
        print("[OK] 超越前代平均值!")
    else:
        print("[..] 未超越前代平均值")

    print(f"ELO: {elo_ratings[gen]:.0f}")
    vec_env.close()


# =============================================================================
# 评估已有模型
# =============================================================================

def evaluate(args):
    """加载已有模型，与随机 AI 和上一代对战。"""
    model_path = args.model_path or str(MODELS_DIR / "ppo_penguinchess_final.zip")
    if not os.path.exists(model_path):
        # 尝试最新的 generation
        gens = sorted(MODELS_DIR.glob("ppo_penguinchess_gen_*.zip"))
        if gens:
            model_path = str(gens[-1])
        else:
            print(f"模型文件不存在: {model_path}")
            return

    print(f"加载模型: {model_path}")
    model = PPO.load(model_path, device="auto")

    gen_str = "（最终版）" if "gen_" not in model_path else f"（gen_{model_path.split('_gen_')[1].split('.')[0]}）"
    print(f"\n评估 {gen_str}，{args.eval_episodes} 局:")

    # vs 随机
    r = compete(model, None, args.eval_episodes)
    print(f"vs 随机 AI:  胜 {r['p1_win']:.1%}  负 {r['p2_win']:.1%}  平 {r['draw']:.1%}")

    # vs 前代（如果有）
    if "gen_" in model_path:
        n = int(model_path.split("_gen_")[1].split(".")[0])
        prev_path = MODELS_DIR / f"ppo_penguinchess_gen_{n-1}.zip"
        if prev_path.exists():
            prev = PPO.load(str(prev_path), device="auto")
            r2 = compete(model, prev, args.eval_episodes)
            print(f"vs gen_{n-1}:         胜 {r2['p1_win']:.1%}  负 {r2['p2_win']:.1%}  平 {r2['draw']:.1%}")

    # 验收标准
    if r['p1_win'] > 0.80:
        print("\n✅ 通过验收: 对随机策略胜率 > 80%")
    else:
        print(f"\n❌ 未通过验收: 对随机策略胜率 {r['p1_win']:.1%} < 80%")


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PenguinChess PPO 训练")
    parser.add_argument("--timesteps", type=int, default=100_000, help="训练步数")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--n-steps", type=int, default=2048, help="每次更新步数")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--n-epochs", type=int, default=10, help="每次更新 epoch 数")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip 范围")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="熵正则系数")
    parser.add_argument("--num-envs", type=int, default=1, help="并行环境数（加速训练）")
    parser.add_argument("--force-gpu", action="store_true", help="强制使用 GPU（默认 CPU）")
    parser.add_argument("--eval-episodes", type=int, default=100, help="评估局数")
    parser.add_argument("--evaluate-only", action="store_true", help="仅评估，不训练")
    parser.add_argument("--model-path", type=str, default=None, help="模型路径（评估时使用）")
    parser.add_argument("--resume", type=str, default=None, help="继续训练的模型路径")

    args = parser.parse_args()

    if args.evaluate_only:
        evaluate(args)
    else:
        train(args)
