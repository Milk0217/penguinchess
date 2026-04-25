import sys
sys.path.insert(0, r'E:\programming\penguinchess')

from penguinchess.env import PenguinChessEnv
import time

print("Testing PenguinChessEnv...")

# Test basic functionality
env = PenguinChessEnv()
obs, info = env.reset()
print(f'reset() OK, legal_actions={len(info["valid_actions"])}')

# Test a few steps
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
        print(f'Episode finished at step {i+1}, reset.')

print('10 random steps OK')

# Test 100 episodes
start = time.perf_counter()
for ep in range(100):
    obs, info = env.reset()
    done = False
    steps = 0
    while not done and steps < 500:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
elapsed = time.perf_counter() - start
print(f'100 episodes in {elapsed:.3f}s, avg {elapsed*10:.1f}ms per episode')

# Test get_legal_actions performance
start = time.perf_counter()
for _ in range(10000):
    legal = env._game.get_legal_actions()
elapsed = time.perf_counter() - start
print(f'10000 get_legal_actions() in {elapsed:.3f}s, avg {elapsed*100:.3f}ms per call')

env.close()
print('All tests passed!')