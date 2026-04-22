# 企鹅棋 PenguinChess

双人六边形棋盘对战游戏，同时作为强化学习研究平台。

## 项目目标

本项目有两个核心方向：

### 目标 1：Gymnasium 标准强化学习环境

构建严格遵循 [Gymnasium](https://gymnasium.farama.org/)（强化学习标准接口）规范的企鹅棋环境，使 AI 智能体可以通过标准 RL 接口与游戏交互：

```python
import gymnasium as gym
from penguinchess.env import PenguinChessEnv

env = gym.make("PenguinChess-v0")
observation, info = env.reset()

while not terminated and not truncated:
    action = your_agent.select_action(observation)   # 智能体决策
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

### 目标 2：企鹅棋强 AI 训练

基于目标 1 构建的 Gymnasium 环境，训练能在企鹅棋中达到高胜率的强化学习智能体：

- 使用 PPO、SAC、AlphaZero 等算法训练
- 支持 self-play 自对弈训练
- 提供训练可视化与胜率评估工具

---

## 游戏规则

### 棋盘

- 60 个六边形格子，使用立方体坐标 (q, r, s)，其中 q + r + s = 0
- 每个格子随机分配值 1、2 或 3，总和固定为 99
- 格子分为活跃（value > 0）和已消除（value ≤ 0）两种状态

### 放置阶段

1. 玩家轮流放置棋子，Player 1 先手
2. 每位玩家放置 3 个棋子（Player 1: ID 4,6,8；Player 2: ID 5,7,9）
3. 放置时获得该格子的分值

### 移动阶段

1. 双方轮流移动（Player 1 先手）
2. 棋子可沿 q轴、r轴 或 s轴 方向移动到相邻同轴格子
3. 移动被以下情况阻挡：其他棋子、已消除格子、中间格子有障碍
4. 移动到新格子后获得该格子分值
5. 每次移动后，与所有棋子失去连接的格子被消除
6. 无合法移动的棋子被移除

### 胜负

- **游戏结束时**，比较双方最终得分（放置得分 + 移动得分 + 剩余格子分），分数高者获胜
- **游戏结束触发条件**：一方棋子全被移除，或所有活跃格子均被消除

---

## 快速开始

### Web 对战（人类对战）

```bash
cd /mnt/e/programming/penguinchess
uv run main.py
# 访问 http://localhost:8080
```

### Gymnasium 环境（AI 对战）

```bash
cd /mnt/e/programming/penguinchess
uv run python -c "
import gymnasium as gym
from penguinchess.env import PenguinChessEnv

env = gym.make('PenguinChess-v0')
obs, info = env.reset()
print('Observation space:', env.observation_space)
print('Action space:', env.action_space)
env.close()
"
```

### 训练示例

```bash
cd /mnt/e/programming/penguinchess
uv run python examples/train_ppo.py
```

---

## 项目结构

```
penguinchess/
├── main.py                     # Flask 入口（Web 界面）
├── pyproject.toml              # Python 项目配置 + 依赖
├── templates/
│   └── index.html              # Web 对战前端 HTML
├── statics/                     # Web 前端静态资源
│   ├── main.js                 # 游戏主逻辑（Web 版）
│   ├── board.js                # 棋盘类（Hex）
│   ├── piece.js                # 棋子类
│   ├── player.js               # 玩家类
│   ├── config.js               # 全局配置常量
│   └── style.css               # 样式
└── penguinchess/               # RL 环境模块（待实现）
    ├── __init__.py
    ├── env.py                  # Gymnasium 环境（待实现）
    ├── wrapped.py              # 包装器（待实现）
    └── models/                 # 训练模型（待实现）
        └── ppo.py
```

---

## 开发路线

| 阶段 | 内容 | 状态 |
|------|------|------|
| **Phase 0** | Web 对战基础游戏（当前） | 进行中 |
| **Phase 1** | Gymnasium 环境核心（`env.py`） | 待开始 |
| **Phase 2** | Gymnasium 包装器（reward shaping 等） | 待开始 |
| **Phase 3** | 基础 RL 训练 pipeline | 待开始 |
| **Phase 4** | Self-play 训练框架 | 待开始 |
| **Phase 5** | 评估与可视化工具 | 待开始 |

详细说明见 [AGENTS.md](./AGENTS.md)。

---

## 技术栈

- **Web 前端**: Vanilla JavaScript / CSS（无框架）
- **Web 后端**: Python 3.11+ / Flask / uv
- **RL 环境**: Python / Gymnasium
- **RL 训练**: PyTorch / Stable-Baselines3 / tianshou

---

## 许可证

MIT
