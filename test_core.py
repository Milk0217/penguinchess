from penguinchess.env import PenguinChessEnv
import time

env = PenguinChessEnv()
obs, info = env.reset()
print(f'reset OK, {len(info["valid_actions"])} legal actions')

# Test 100 episodes
for ep in range(100):
    obs, info = env.reset()
    done = False
    steps = 0
    while not done and steps < 500:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

print('100 episodes completed')

# Test get_legal_actions performance
start = time.perf_counter()
for _ in range(10000):
    legal = env._game.get_legal_actions()
elapsed = time.perf_counter() - start
print(f'10000 get_legal_actions() in {elapsed:.3f}s, avg {elapsed*100:.4f}ms per call')

env.close()
print('All tests passed!')