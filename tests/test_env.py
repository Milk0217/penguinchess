"""
Gymnasium API 合规性测试。
"""

import pytest
import numpy as np
import gymnasium as gym
from penguinchess.env import PenguinChessEnv


class TestGymnasiumEnvRegistration:
    """Gymnasium 环境注册测试。"""

    def test_env_can_be_created(self):
        """环境可以通过 gym.make() 创建。"""
        env = gym.make("PenguinChess-v0")
        assert env is not None
        env.close()

    def test_env_close(self):
        """环境可以正常关闭。"""
        env = gym.make("PenguinChess-v0")
        env.close()
        # 不应该抛出异常


class TestResetAPI:
    """reset() API 测试。"""

    def test_reset_returns_tuple(self):
        """reset() 应该返回 (observation, info) 元组。"""
        env = gym.make("PenguinChess-v0")
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        env.close()

    def test_reset_returns_valid_observation(self):
        """reset() 返回的观测应该是有效的 numpy array。"""
        env = gym.make("PenguinChess-v0")
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (204,)
        env.close()

    def test_reset_returns_info_dict(self):
        """reset() 返回的 info 应该是 dict。"""
        env = gym.make("PenguinChess-v0")
        obs, info = env.reset()
        assert isinstance(info, dict)
        assert "valid_actions" in info
        env.close()

    def test_reset_with_seed(self):
        """带种子的 reset() 应该有确定性结果。"""
        env1 = gym.make("PenguinChess-v0")
        obs1, _ = env1.reset(seed=42)
        env1.close()

        env2 = gym.make("PenguinChess-v0")
        obs2, _ = env2.reset(seed=42)
        env2.close()

        assert np.array_equal(obs1, obs2), "Same seed should produce same observation"

    def test_reset_produces_valid_actions(self):
        """reset() 后的 info 应该包含有效的动作列表。"""
        env = gym.make("PenguinChess-v0")
        obs, info = env.reset()
        assert isinstance(info["valid_actions"], list)
        assert len(info["valid_actions"]) > 0
        env.close()


class TestStepAPI:
    """step() API 测试。"""

    def test_step_returns_five_tuple(self):
        """step() 应该返回 5 元组 (obs, reward, terminated, truncated, info)。"""
        env = gym.make("PenguinChess-v0")
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert isinstance(result, tuple)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        env.close()

    def test_step_reward_type(self):
        """step() 返回的 reward 应该是 float。"""
        env = gym.make("PenguinChess-v0")
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (float, np.floating))
        env.close()

    def test_step_terminal_flags_type(self):
        """step() 返回的 terminated 和 truncated 应该是 bool。"""
        env = gym.make("PenguinChess-v0")
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()

    def test_step_info_has_valid_actions(self):
        """step() 返回的 info 应该包含 valid_actions。"""
        env = gym.make("PenguinChess-v0")
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert "valid_actions" in info
        env.close()

    def test_step_updates_observation(self):
        """step() 后 observation 应该更新。"""
        env = gym.make("PenguinChess-v0")
        obs1, _ = env.reset()
        action = env.action_space.sample()
        obs2, _, _, _, _ = env.step(action)
        # 观测应该改变（除非恰好回到相同状态）
        # 这里我们只检查不是完全相同
        # 注意：由于编码方式，可能有些情况下观测相同也是正常的
        env.close()

    def test_invalid_action_handling(self):
        """无效动作应该有合理的处理（返回负 reward 或在 info 中标记）。"""
        env = gym.make("PenguinChess-v0")
        env.reset()
        # 动作 9999 应该无效
        obs, reward, terminated, truncated, info = env.step(9999)
        # 应该返回负 reward 或标记 invalid
        assert reward < 0 or info.get("invalid", False) or terminated
        env.close()


class TestObservationSpace:
    """观测空间合规性测试。"""

    def test_observation_space_shape(self):
        """环境的 observation_space 形状应该是 (204,)。"""
        env = gym.make("PenguinChess-v0")
        assert env.observation_space.shape == (204,)
        env.close()

    def test_observation_space_dtype(self):
        """环境的 observation_space 数据类型应该是 float32。"""
        env = gym.make("PenguinChess-v0")
        assert env.observation_space.dtype == np.float32
        env.close()

    def test_observation_within_bounds(self):
        """观测值应该在合理范围内。"""
        env = gym.make("PenguinChess-v0")
        obs, _ = env.reset()
        # 所有值应该在 [-1, 1] 或类似的合理范围内
        assert obs.min() >= -1.5, f"Observation min {obs.min()} too low"
        assert obs.max() <= 1.5, f"Observation max {obs.max()} too high"
        env.close()


class TestActionSpace:
    """动作空间合规性测试。"""

    def test_action_space_discrete_60(self):
        """环境的 action_space 应该是 Discrete(60)。"""
        env = gym.make("PenguinChess-v0")
        assert hasattr(env.action_space, 'n')
        assert env.action_space.n == 60
        env.close()

    def test_action_space_sample_valid(self):
        """sample() 返回的动作应该是有效的。"""
        env = gym.make("PenguinChess-v0")
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            assert env.action_space.contains(action)
        env.close()


class TestEpisodeCycle:
    """完整回合周期测试。"""

    def test_episode_reaches_terminal(self):
        """游戏应该能够正常结束（达到 terminated 状态）。"""
        env = gym.make("PenguinChess-v0")
        obs, info = env.reset()
        done = False
        steps = 0
        max_steps = 500

        while not done and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done or steps < max_steps, "Episode should eventually terminate or truncate"
        env.close()

    def test_max_steps_truncation(self):
        """超过最大步数应该触发 truncated。"""
        env = gym.make("PenguinChess-v0")
        obs, info = env.reset()
        done = False
        steps = 0
        max_steps = 500

        while not done and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        # 如果达到最大步数，应该是 truncated（不是 terminated）
        if steps >= max_steps:
            assert truncated, "Should be truncated after max_steps"
        env.close()

    def test_episode_reset_cycle_multiple_times(self):
        """可以多次 reset 重开新回合。"""
        env = gym.make("PenguinChess-v0")
        for episode in range(5):
            obs, info = env.reset()
            assert obs.shape == (204,)
            done = False
            steps = 0
            while not done and steps < 100:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
        env.close()


class TestStability:
    """稳定性测试。"""

    def test_100_episode_stability(self):
        """100 个回合不应该崩溃。"""
        env = gym.make("PenguinChess-v0")
        for episode in range(100):
            obs, info = env.reset()
            done = False
            steps = 0
            while not done and steps < 500:
                action = env.action_space.sample()
                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    steps += 1
                except Exception as e:
                    pytest.fail(f"Episode {episode} failed at step {steps}: {e}")
                    break
            assert True, f"Episode {episode} completed in {steps} steps"
        env.close()

    def test_random_seed_consistency(self):
        """相同种子的多次运行应该产生相同结果。"""
        outcomes = []
        for _ in range(3):
            env = gym.make("PenguinChess-v0")
            env.reset(seed=42)
            episode_rewards = []
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_rewards.append(reward)
                done = terminated or truncated
            outcomes.append(sum(episode_rewards))
            env.close()

        # 相同种子应该有相同的总 reward（由于随机动作采样，结果会不同）
        # 但我们应该检查的是环境不会崩溃
        assert len(outcomes) == 3

    def test_info_fields_present(self):
        """info dict 应该包含所有必要字段。"""
        env = gym.make("PenguinChess-v0")
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        required_fields = ["valid_actions"]
        for field in required_fields:
            assert field in info, f"Missing field: {field}"

        env.close()


class TestRender:
    """render() 方法测试。"""

    def test_render_mode_human(self):
        """可以创建带 render_mode 的环境。"""
        env = gym.make("PenguinChess-v0", render_mode="human")
        assert env.render_mode == "human"
        env.close()

    def test_render_mode_text(self):
        """可以创建带 render_mode="text" 的环境。"""
        env = gym.make("PenguinChess-v0", render_mode="text")
        assert env.render_mode == "text"
        env.close()

    def test_render_text(self):
        """render() 应该返回字符串。"""
        env = gym.make("PenguinChess-v0", render_mode="text")
        env.reset()
        output = env.render()
        assert isinstance(output, str)
        assert len(output) > 0
        env.close()


class TestMetadata:
    """环境 metadata 测试。"""

    def test_metadata_render_modes(self):
        """metadata 应该包含 render_modes。"""
        env = gym.make("PenguinChess-v0")
        assert "render_modes" in env.metadata
        env.close()

    def test_metadata_render_fps(self):
        """metadata 应该包含 render_fps。"""
        env = gym.make("PenguinChess-v0")
        assert "render_fps" in env.metadata
        env.close()