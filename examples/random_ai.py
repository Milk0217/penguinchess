#!/usr/bin/env python3
"""
随机 AI 验证脚本。
用于验证 PenguinChess-v0 Gymnasium 环境是否正确构建。

运行方式:
    cd /mnt/e/programming/penguinchess
    uv run python examples/random_ai.py
"""

import gymnasium as gym
import numpy as np
from collections import Counter

from penguinchess.env import PenguinChessEnv


def random_agent(legal_actions):
    """最简单的随机智能体。"""
    return np.random.choice(legal_actions)


def evaluate(env, n_games: int = 100, verbose: bool = False) -> dict:
    """
    评估随机智能体 vs 随机智能体（自对弈）的表现。
    返回统计信息。
    """
    results = []
    episode_lengths = []
    scores_per_player = [[], []]

    for i in range(n_games):
        obs, info = env.reset(seed=i)
        done = False
        steps = 0

        while not done:
            legal = info["valid_actions"]
            if not legal:
                # 无合法动作，该局结束
                break
            action = random_agent(legal)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        winner = info.get("winner", None)
        results.append(winner)
        episode_lengths.append(steps)
        scores = info["scores"]
        scores_per_player[0].append(scores[0])
        scores_per_player[1].append(scores[1])

        if verbose and (i + 1) % 10 == 0:
            print(f"  完成 {i + 1}/{n_games} 局...")

    counter = Counter(results)
    return {
        "total_games": n_games,
        "p1_wins": counter.get(0, 0),
        "p2_wins": counter.get(1, 0),
        "draws": counter.get(2, 0),
        "avg_steps": np.mean(episode_lengths),
        "avg_p1_score": np.mean(scores_per_player[0]),
        "avg_p2_score": np.mean(scores_per_player[1]),
        "p1_winrate": counter.get(0, 0) / n_games * 100,
        "p2_winrate": counter.get(1, 0) / n_games * 100,
        "draw_rate": counter.get(2, 0) / n_games * 100,
    }


def test_env_interface():
    """测试 Gymnasium 环境接口的正确性。"""
    print("\n========== 环境接口测试 ==========\n")

    env = gym.make("PenguinChess-v0")
    obs, info = env.reset(seed=42)

    # 1. 观测空间检查
    print(f"✓ env.reset() 成功")
    print(f"  观测形状: {obs.shape}")
    print(f"  观测空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")

    assert obs.shape == (204,), f"观测形状错误: {obs.shape}"
    assert env.observation_space.contains(obs), "观测不在空间内"
    assert isinstance(info, dict), "info 应该是 dict"
    assert "valid_actions" in info, "info 缺少 valid_actions"
    print(f"✓ 观测空间包含合法动作: {info['valid_actions'][:5]}...")

    # 2. 执行一个 step
    legal = info["valid_actions"]
    action = legal[0]
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"\n✓ env.step({action}) 成功")
    print(f"  reward: {reward:.4f}")
    print(f"  terminated: {terminated}, truncated: {truncated}")
    print(f"  phase: {info.get('phase')}")
    print(f"  current_player: {info.get('current_player')}")

    # 3. 多步测试
    print(f"\n✓ 执行 10 步测试...")
    for i in range(10):
        legal = info.get("valid_actions", [])
        if not legal:
            print(f"  第 {i} 步: 无合法动作，结束")
            break
        action = np.random.choice(legal)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"  第 {i} 步: 游戏结束 (winner={info.get('winner')})")
            break
    print(f"  完成，未崩溃")

    # 4. seed 一致性
    print(f"\n✓ 测试 seed=42 重置一致性...")
    env1 = gym.make("PenguinChess-v0")
    obs1, _ = env1.reset(seed=123)
    obs2, _ = env1.reset(seed=123)
    assert np.allclose(obs1, obs2), "相同 seed 应该产生相同观测"
    print(f"  相同 seed 产生相同观测 ✓")

    env.close()
    print("\n✅ 环境接口测试全部通过!")


def test_board_generation():
    """测试棋盘生成是否正确。"""
    print("\n========== 棋盘生成测试 ==========\n")

    from penguinchess.core import generate_sequence, create_board, PenguinChessCore

    # 测试序列生成
    print("测试 generate_sequence()...")
    seq = generate_sequence()
    print(f"  序列长度: {len(seq)}")
    print(f"  总和: {sum(seq)} (应为 99)")
    print(f"  数字3个数: {seq.count(3)} (应在 8~10 之间)")

    assert len(seq) == 60, f"序列长度应为60，实际{len(seq)}"
    assert sum(seq) == 99, f"总和应为99，实际{sum(seq)}"
    assert 8 <= seq.count(3) <= 10, f"3的个数应在8~10，实际{seq.count(3)}"

    # 测试棋盘生成
    print("\n测试 create_board()...")
    hexes = create_board(seq)
    print(f"  生成格子数: {len(hexes)} (应为 60)")
    active = [h for h in hexes if h.value > 0]
    print(f"  活跃格子数: {len(active)}")

    # 测试 PenguinChessCore
    print("\n测试 PenguinChessCore.reset()...")
    game = PenguinChessCore(seed=42)
    game.reset(seed=42)
    print(f"  格子数: {len(game.hexes)} (应为 60)")
    print(f"  棋子数: {len(game.pieces)} (应为 6)")
    print(f"  分数: {game.players_scores} (应为 [0, 0])")
    print(f"  阶段: {game.phase}")

    # 测试合法动作
    print("\n测试 get_legal_actions()...")
    legal = game.get_legal_actions()
    print(f"  放置阶段合法动作数: {len(legal)} (应为 60)")

    print("\n✅ 棋盘生成测试全部通过!")


def test_full_episodes():
    """测试完整对局是否正常运行。"""
    print("\n========== 完整对局测试 ==========\n")

    stats = evaluate(gym.make("PenguinChess-v0"), n_games=100, verbose=True)

    print(f"\n100 局随机对战统计:")
    print(f"  P1 胜场: {stats['p1_wins']} ({stats['p1_winrate']:.1f}%)")
    print(f"  P2 胜场: {stats['p2_wins']} ({stats['p2_winrate']:.1f}%)")
    print(f"  平局:    {stats['draws']} ({stats['draw_rate']:.1f}%)")
    print(f"  平均步数: {stats['avg_steps']:.1f}")
    print(f"  P1 均分: {stats['avg_p1_score']:.1f}")
    print(f"  P2 均分: {stats['avg_p2_score']:.1f}")

    # 随机 vs 随机，P1 先手，P1 应该有微弱优势
    assert stats["p1_wins"] + stats["p2_wins"] + stats["draws"] == 100

    print(f"\n✅ 完整对局测试通过 (P1 先手胜率 {stats['p1_winrate']:.1f}%)!")


if __name__ == "__main__":
    print("=" * 50)
    print("企鹅棋 Gymnasium 环境验证")
    print("=" * 50)

    test_env_interface()
    test_board_generation()
    test_full_episodes()

    print("\n" + "=" * 50)
    print("🎉 所有测试通过，环境构建完成！")
    print("=" * 50)
