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
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from penguinchess.env import PenguinChessEnv

MODELS_DIR = Path(__file__).parent.parent / "models"
LOGS_DIR = MODELS_DIR / "logs"
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# =============================================================================
# 环境工厂
# =============================================================================

def make_env(seed: int = 0) -> gym.Env:
    """创建 PenguinChess 环境实例（用于 DummyVecEnv）。"""
    def _init():
        env = gym.make("PenguinChess-v0")
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


# =============================================================================
# 评估：与随机 AI 对战
# =============================================================================

def evaluate_against_random(
    model: PPO,
    num_episodes: int = 100,
    render: bool = False,
) -> dict:
    """
    评估 PPO 模型与随机策略的对战胜率。
    
    Returns:
        {"win_rate": float, "draw_rate": float, "loss_rate": float, "avg_reward": float}
    """
    env = gym.make("PenguinChess-v0")
    wins = 0
    draws = 0
    losses = 0
    total_reward = 0.0

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        terminated = False
        truncated = False
        episode_reward = 0.0

        while not terminated and not truncated:
            # PPO 扮演 P1（玩家 0）
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

            # 随机 AI 扮演 P2（玩家 1）
            legal = info.get("valid_actions", [])
            random_action = np.random.choice(legal) if legal else 0
            obs, reward, terminated, truncated, info = env.step(random_action)
            episode_reward += reward

        total_reward += episode_reward
        winner = info.get("winner")
        if winner == 0:
            wins += 1
        elif winner == 2:
            draws += 1
        elif winner == 1:
            losses += 1

    env.close()

    return {
        "win_rate": wins / num_episodes,
        "draw_rate": draws / num_episodes,
        "loss_rate": losses / num_episodes,
        "avg_reward": total_reward / num_episodes,
    }


# =============================================================================
# 主训练流程
# =============================================================================

def train(args):
    """训练 PPO 模型。"""
    print("=" * 60)
    print("PPO 训练 - PenguinChess")
    print("=" * 60)

    # 创建环境
    vec_env = DummyVecEnv([make_env(seed=0)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    # 模型配置：从零开始或续训
    if args.resume and os.path.exists(args.resume):
        print(f"续训练模型: {args.resume}")
        model = PPO.load(args.resume, env=vec_env, device="auto",
                         learning_rate=args.lr,
                         n_steps=args.n_steps,
                         batch_size=args.batch_size,
                         n_epochs=args.n_epochs,
                         gamma=args.gamma,
                         clip_range=args.clip_range,
                         ent_coef=0.01,
                         vf_coef=0.5,
                         max_grad_norm=0.5,
                         tensorboard_log=str(LOGS_DIR),
                         verbose=1)
    else:
        model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=args.clip_range,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(LOGS_DIR),
        verbose=1,
        device="auto",
    )

    # 回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.timesteps // 10, 1000),
        save_path=str(MODELS_DIR),
        name_prefix="ppo_penguinchess",
    )

    eval_env = DummyVecEnv([make_env(seed=42)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=0.8, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=max(args.timesteps // 20, 500),
        best_model_save_path=str(MODELS_DIR / "best"),
        verbose=1,
    )

    # 训练
    print(f"\n开始训练 {args.timesteps} 步...")
    start_time = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=False,
    )
    elapsed = time.time() - start_time
    print(f"训练完成! 耗时: {elapsed:.1f}s ({args.timesteps / elapsed:.0f} steps/s)")

    # 保存最终模型
    final_path = str(MODELS_DIR / "ppo_penguinchess_final.zip")
    model.save(final_path)
    print(f"最终模型已保存: {final_path}")

    # 评估
    print("\n评估中...")
    results = evaluate_against_random(model, num_episodes=args.eval_episodes)
    print(f"  胜率: {results['win_rate']:.1%}")
    print(f"  平局率: {results['draw_rate']:.1%}")
    print(f"  负率: {results['loss_rate']:.1%}")
    print(f"  平均奖励: {results['avg_reward']:.3f}")

    vec_env.close()
    eval_env.close()


# =============================================================================
# 仅评估
# =============================================================================

def evaluate(args):
    """加载已训练的模型并评估。"""
    model_path = args.model_path or str(MODELS_DIR / "ppo_penguinchess_final.zip")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    print(f"加载模型: {model_path}")
    model = PPO.load(model_path)

    results = evaluate_against_random(model, num_episodes=args.eval_episodes)
    print(f"\n评估结果 ({args.eval_episodes} 局):")
    print(f"  胜率: {results['win_rate']:.1%}")
    print(f"  平局率: {results['draw_rate']:.1%}")
    print(f"  负率: {results['loss_rate']:.1%}")
    print(f"  平均奖励: {results['avg_reward']:.3f}")

    # 验收标准
    if results["win_rate"] > 0.80:
        print("\n✅ 通过验收: 对随机策略胜率 > 80%")
    else:
        print(f"\n❌ 未通过验收: 胜率 {results['win_rate']:.1%} < 80%")


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
    parser.add_argument("--eval-episodes", type=int, default=100, help="评估局数")
    parser.add_argument("--evaluate-only", action="store_true", help="仅评估，不训练")
    parser.add_argument("--model-path", type=str, default=None, help="模型路径（评估时使用）")
    parser.add_argument("--resume", type=str, default=None, help="继续训练的模型路径，例如 models/ppo_penguinchess_50000_steps.zip")

    args = parser.parse_args()

    if args.evaluate_only:
        evaluate(args)
    else:
        train(args)
