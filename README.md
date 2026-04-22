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

详细规则请参阅 [docs/RULES.md](docs/RULES.md)（权威规则文档）。

### 快速概览

- **棋盘**: 60 个六边形格子，总分 99 分
- **放置阶段**: Player 1 先手，双方严格交替各放 3 个棋子（6 步），放置时获得格子分值
- **移动阶段**: 双方轮流移动棋子（同轴方向），移动后获得目标格子分值
- **连通性消除**: 每次移动后，不与任何棋子连通的格子被消除
- **胜负**: 比较双方最终得分（放置 + 移动 + 剩余格子分），分数高者胜

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

[GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html)
