# PenguinChess — Agent 开发指南

## 一、项目目标

本项目有两个核心方向：

1. **Gymnasium 环境** — 构建严格遵循 Farama Gymnasium 接口的企鹅棋 RL 环境
2. **强 AI 训练** — 基于该环境训练高胜率 RL 智能体

---

## 二、架构概览

```
penguinchess/
├── main.py                          # Flask Web 服务器（人类对战）
├── templates/index.html             # Web UI
├── statics/                         # Web 前端 JS/CSS
│   ├── main.js                      # 游戏逻辑（Web 版本）
│   ├── board.js                     # Hex 棋盘类
│   ├── piece.js                     # 棋子类
│   ├── player.js                   # 玩家类
│   └── config.js                   # 配置常量
│
├── penguinchess/                    # RL 环境模块
│   ├── __init__.py
│   ├── env.py                      # ★ Gymnasium 环境核心（待实现）
│   ├── spaces.py                   # ★ 观测空间 / 动作空间定义（待实现）
│   ├── reward.py                   # ★ Reward shaping 函数（待实现）
│   └── wrapped.py                  # ★ Gymnasium 包装器（待实现）
│
└── examples/                        # 训练示例
    ├── train_ppo.py                # ★ PPO 训练（待实现）
    ├── train_sac.py                # ★ SAC 训练（待实现）
    ├── selfplay.py                 # ★ Self-play 训练（待实现）
    └── evaluate.py                 # ★ 评估工具（待实现）
```

---

## 三、游戏规则（完整参考）

> **权威规则文档**: `docs/RULES.md` 是游戏的 source of truth。所有代码必须精确遵循该文档。
> 本章节仅作概览和开发参考，细节以 RULES.md 为准。

### 3.1 棋盘

- 60 个六边形格子，平行四边形排布
- 立方体坐标 (q, r, s)，约束 q + r + s = 0
- q ∈ [-4, 3]，r 随行变化
- 每个格子值 ∈ {1, 2, 3}，总和 = 99
- value > 0 → 活跃；value ≤ 0 → 已消除

### 3.2 放置阶段

| 步骤 | 说明 |
|------|------|
| 顺序 | Player 1 先手，双方**严格交替**各放 3 个棋子（6 步） |
| 棋子 ID | Player 1: {4, 6, 8}，Player 2: {5, 7, 9} |
| 计分 | 放置时立即获得该格子分值 |
| 结束条件 | 双方都放满 3 个棋子后（步骤 6 完成后）→ 进入移动阶段 |
| 切换逻辑 | 每步检查当前玩家是否已放满：未满则切换到对方，已满则不切换 |

### 3.3 移动阶段

| 步骤 | 说明 |
|------|------|
| 顺序 | Player 1 先手，双方轮流移动 |
| 方向 | 棋子可沿 q轴、r轴、s轴 之一移动 |
| 约束 | 目标格子必须与当前位置在相同轴上坐标相同 |
| 阻挡 | 其他棋子（任意方）、已消除格子、中间格子不空 |
| 计分 | 移动到新格子后获得该格子分值 |
| 消除 | 每次移动后，不与任何棋子连通（6方向邻接）的格子被消除 |
| 销毁 | 无合法移动的棋子被移除 |

### 3.4 胜负条件

**胜负由最终分数决定**，而非棋子存活数量。分数更高者获胜，分数相同则平局。

游戏结束的触发条件（满足任一即结束）：

| 触发条件 | 处理 |
|----------|------|
| 一方棋子全部被移除 | 幸存者获得所有剩余格子分值 |
| 所有活跃格子均被消除 | 平局（双方都有棋子但无路可走） |
| 双方棋子同时被移除 | 平局 |

游戏结束时，比较双方最终得分（放置阶段得分 + 移动阶段得分 + 剩余格子分），分数高者获胜。

---

## 四、Gymnasium 环境规范

### 4.1 环境注册名

```python
gymnasium.make("PenguinChess-v0")
```

### 4.2 观测空间（Observation Space）

```python
# Dict 空间，包含以下键
{
    "board": Box(shape=(60, 3), low=0, high=99, dtype=np.float32),
        # shape=(60, 3): [q, r, value] 每行一个格子
        # value = 0 表示已消除格子
        # value = -1 表示无棋子，1/2/3 表示 P1/P2/无棋子但有值
        # 最终设计待实现时确定

    "pieces": Box(shape=(6, 4), low=-10, high=10, dtype=np.float32),
        # shape=(6, 4): [piece_id, q, r, s] 每行一个棋子
        # piece_id = -1 表示该棋子已被移除
        # 前3行 P1 棋子，后3行 P2 棋子

    "current_player": Box(shape=(1,), low=0, high=1, dtype=np.int8),
        # 0 = Player 1, 1 = Player 2

    "phase": Box(shape=(1,), low=0, high=1, dtype=np.int8),
        # 0 = 放置阶段, 1 = 移动阶段

    "scores": Box(shape=(2,), low=0, high=99, dtype=np.float32),
        # [player1_score, player2_score]
}
```

> **设计备注**: 实际观测空间设计需在 `env.py` 实现阶段根据游戏平衡性调优。可以先用简单的扁平化表示（board 展平成 60×3，pieces 展平成 24），稳定后再优化为 Dict 结构以提升 RL 效率。

### 4.3 动作空间（Action Space）

```python
# MultiDiscrete 或 Discrete，取决于设计选择

# 方案 A: Discrete（简单，60 个格子）
#   - 放置阶段: 60 个格子 ID（无效位置自动过滤）
#   - 移动阶段: 60 个格子 ID（无效移动自动过滤）

# 方案 B: MultiDiscrete（表达更丰富）
#   - axis: 选择移动轴（0=q, 1=r, 2=s）
#   - steps: 沿该轴移动的步数（+1 / -1）

# 推荐先实现方案 A，验证训练可行后再升级
Discrete(60)  # 或 MultiDiscrete([3, 2])
```

### 4.4 Step 返回值

```python
observation,    # 见上文观测空间
reward,         # float，本步回报
terminated,     # bool，游戏结束（胜负已定）
truncated,      # bool，是否被截断（如超过最大步数）
info            # dict，额外信息（见下文）
```

### 4.5 info 字段

```python
{
    "valid_actions": List[int],
        # 当前所有合法动作 ID，供智能体过滤用

    "current_player": int,
        # 当前玩家 0/1

    "phase": str,
        # "placement" | "movement"

    "scores": List[float],
        # [player1_score, player2_score]

    "pieces_remaining": List[int],
        # [player1_pieces, player2_pieces]

    "winner": int | None,
        # 游戏结束时: 0=Player1胜, 1=Player2胜, 2=平局
}
```

### 4.6 Reward 设计

| 阶段 | 事件 | Reward |
|------|------|--------|
| 任意 | 获得格子分值 | +分值/99（归一化） |
| 任意 | 对手棋子被移除 | +20 |
| 任意 | 自己的棋子被移除 | -20 |
| 任意 | 格子被消除（己方连通） | +0.5 |
| 移动阶段 | 移动到高价值格子 | +额外奖励 |
| 移动阶段 | 阻断对手连通 | +额外奖励 |
| 任意 | 胜利 | +100 |
| 任意 | 失败 | -100 |
| 任意 | 平局 | 0 |

> **设计备注**: Reward 设计对 RL 训练效果影响极大。初期用简单的稀疏 reward（胜负 ±1）也能收敛，但密集 reward 能显著加速训练。建议先实现稀疏版本，建立 baseline，再逐步加入 shaping reward。

### 4.7 最大步数

```python
max_episode_steps = 500  # 足够覆盖最长可能游戏
```

### 4.8 种子（Seed）

严格遵循 Gymnasium 规范：

```python
env.reset(seed=42)
env.action_space.seed(42)   # 动作空间也要设置种子
```

---

## 五、训练架构

### 5.1 推荐算法

| 算法 | 适用场景 | 推荐库 |
|------|----------|--------|
| **PPO** |首选，样本效率高，稳定性好 | Stable-Baselines3 / tianshou |
| **SAC** | 连续动作（若改用连续移动距离） | Stable-Baselines3 |
| **AlphaZero** | 完美信息博弈，自对弈 | 自主实现 |
| **DQN** | 简单基线，快速验证环境 | Stable-Baselines3 |

**推荐从 PPO 开始**，因为：
- 离散动作空间天然适配 PPO
- 社区成熟，文档丰富
- 超参数鲁棒

### 5.2 Self-Play 训练

```python
# examples/selfplay.py 核心循环

while training:
    # 当前策略与历史策略对战
    opponent = select_opponent(population, method="ELO_weighted")

    # 收集 self-play 对局数据
    for _ in range(num_selfplay_games):
        env = PenguinChessEnv()
        trajectory = collect_trajectory(env, current_policy, opponent)

    # 更新策略
    current_policy = ppo_update(trajectory, current_policy)

    # 评估 & 加入种群
    if should_eval(current_policy, population):
        elo_rating = evaluate(current_policy, population)
        population.add(current_policy, elo_rating)
```

### 5.3 评估指标

| 指标 | 说明 |
|------|------|
| 对随机策略胜率 | 基线 > 95% 才算环境可用 |
| 对简单贪心胜率 | 贪心 = 每步选最高分值移动 |
| ELO 评分 | Self-play 过程中跟踪 |
| 对抗特定对手胜率 | 测试泛化能力 |

---

## 六、开发路线

### Phase 1: Gymnasium 环境核心

**目标**: 实现 `penguinchess/env.py`，通过 Gymnasium API 完整覆盖游戏逻辑。

**交付物**:
- `penguinchess/spaces.py` — 观测/动作空间定义
- `penguinchess/env.py` — 核心环境类，完整实现 `reset() / step() / close() / seed()`
- `penguinchess/reward.py` — reward shaping 函数

**验收标准**:
```python
env = gymnasium.make("PenguinChess-v0")
obs, info = env.reset(seed=42)
assert env.observation_space.contains(obs)
action = env.action_space.sample(info["valid_actions"])
obs, reward, terminated, truncated, info = env.step(action)
assert not (terminated and truncated)  # 不能同时为 True
```

**预计工作量**: 300-500 行 Python

---

### Phase 2: 包装器与工具

**目标**: 提供实用的高层工具。

**交付物**:
- `penguinchess/wrapped.py` — `ObservationHistoryWrapper`（历史帧叠放）、`RewardMaskingWrapper`（过滤无效动作）
- `penguinchess/env.py` 添加 `get_reward_if_action()` 辅助方法（用于 MCTS 等树搜索）
- `examples/evaluate.py` — 胜率评估工具，支持自定义对手

**预计工作量**: 200-300 行 Python

---

### Phase 3: 基础 RL 训练

**目标**: 用 PPO 训练出一个能稳定击败随机策略的智能体。

**交付物**:
- `examples/train_ppo.py` — 完整可运行的 PPO 训练脚本
- `penguinchess/models/ppo.py` — PPO 模型定义（如果用 SB3 则不需要）
- 训练 checkpoint（保存到 `runs/` 或 `models/`）

**验收标准**:
- 训练 100k 步后，对随机策略胜率 > 80%
- 训练 500k 步后，对随机策略胜率 > 95%

**预计工作量**: 200 行 Python（主要在 examples/）

---

### Phase 4: Self-Play 训练框架

**目标**: 实现完整的自我对弈训练循环。

**交付物**:
- `examples/selfplay.py` — Self-play 主循环
- `examples/train_alphazero.py` — AlphaZero 风格的 MCTS + 神经网络训练
- ELO 评分系统
- 早停与最佳模型保存

**验收标准**:
- Self-play 训练 1M 步后，智能体能在 100 局内击败随机策略
- 可视化训练曲线（win rate / ELO over time）

**预计工作量**: 500-800 行 Python

---

### Phase 5: 评估与可视化

**目标**: 完整的训练仪表盘和分析工具。

**交付物**:
- TensorBoard /Weights & Biases 集成
- 棋盘热力图（展示智能体的价值估计）
- 对局回放播放器（Web 端复用现有 replay 系统）
- 与人类对战的能力（Flask API 暴露 RL 智能体作为对手）

---

## 七、Web 对战与 RL 环境的代码共享

为避免重复实现游戏逻辑，游戏核心规则（判定胜负、计算合法动作、消除格子）应在 **Python 端统一实现**，Web 前端通过 HTTP 调用或共享规则模块。

**推荐方案**: Python 核心 `penguinchess/core.py` 包含所有游戏规则，Web 前端 `statics/` 中的 JS 通过 `/api/` 路由调用。

```python
# penguinchess/core.py（待实现）
class PenguinChessCore:
    def reset(self, seed=None): ...
    def get_legal_actions(self): ...
    def step(self, action): ...
    def get_observation(self): ...
    def check_game_over(self): ...

# Flask API 暴露给 Web
@app.route("/api/reset", methods=["POST"])
def api_reset():
    game = PenguinChessCore()
    return jsonify(game.get_observation())

@app.route("/api/step", methods=["POST"])
def api_step(action):
    obs, reward, done = game.step(action)
    return jsonify({"obs": obs, "reward": reward, "done": done})
```

---

## 八、文件依赖关系

```
env.py              # 导入 core.py / spaces.py
spaces.py           # 定义 ObservationSpace / ActionSpace
reward.py           # 被 env.py 调用
wrapped.py          # gymnasium.Wrapper 子类，包装 env.py
core.py             # 游戏规则核心（未来与 Web 共享）

main.py (Flask)     # 可选：通过 API 调用 core.py
examples/train_*.py # 导入 env.py / wrapped.py
```

---

## 九、已知问题（Web 版）

> 以下问题已在 Web 版中修复或待修复，RL 环境实现时需确保对应逻辑一致。

- `gameovercheck()` 棋子归属判断：应用 `Math.floor(piece.id / 2) % 2` 而非 `piece.id % 2`
- `replayHistory()` 参数遮蔽全局变量：`history` → `historyRecord`
- `aftergame()` 结算逻辑：应包含胜者公告和重新开始按钮
- 魔法数字集中在 `config.js` 中管理

---

## 十、运行测试

```bash
# Web 对战
uv run main.py

# RL 环境自检
uv run python -c "
import gymnasium as gym
from penguinchess.env import PenguinChessEnv
env = gym.make('PenguinChess-v0')
obs, info = env.reset()
print('OK: env.reset() works')
env.close()
"
```
