# 企鹅棋 PenguinChess

双人六边形棋盘对战游戏，同时作为强化学习研究平台。

## 项目目标

### 目标 1：高性能游戏引擎（Rust）

用 Rust 实现游戏核心逻辑，提供极低延迟（<1ms）的游戏推理性能，支持高并发（>10000 连接）。

详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

### 目标 2：强化学习研究与训练（Python）

构建基于 [Gymnasium](https://gymnasium.farama.org/) 标准的企鹅棋环境，训练能在游戏中达到高胜率的 RL 智能体。

```python
import gymnasium as gym
from penguinchess.env import PenguinChessEnv

env = gym.make("PenguinChess-v0")
obs, info = env.reset()

while not terminated and not truncated:
    action = your_agent.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

### 目标 3：高平局率 AI

在达成高胜率 AI 后，进一步训练追求**高平局率**的策略 AI（防守型/和棋型 AI）。目标是在无法获胜的情况下最大化平局概率，研究防守策略与博弈平衡。

提供直观的浏览器对战界面，人类玩家对抗 AI 智能体。

---

## 游戏规则

详细规则请参阅 [docs/RULES.md](docs/RULES.md)。

### 快速概览

- **棋盘**: 60 个六边形格子，10×3 + 20×2 + 30×1 = 100 分
- **放置阶段**: Player 1 先手，双方严格交替各放 3 个棋子（6 步），放置时获得格子分值
- **移动阶段**: 双方轮流移动棋子（同轴方向），移动后获得目标格子分值
- **连通性消除**: 每次移动后，不与任何棋子连通的格子被消除
- **胜负**: 比较双方最终得分（放置 + 移动 + 剩余格子分），分数高者胜

---

## 快速开始

### 启动 Web 对战（当前 Python 后端）

```bash
cd /mnt/e/programming/penguinchess
source .venv/bin/activate
python server/app.py
# 访问 http://localhost:8080
```

### 启动开发前端（Vite 热重载）

```bash
cd frontend
npm install
npm run dev
# 访问 http://localhost:5173
```

### Gymnasium 环境（AI 训练）

```bash
cd /mnt/e/programming/penguinchess
source .venv/bin/activate
python -c "
import gymnasium as gym
from penguinchess.env import PenguinChessEnv

env = gym.make('PenguinChess-v0')
obs, info = env.reset()
print('Observation space:', env.observation_space)
print('Action space:', env.action_space)
env.close()
"
```

### 运行测试

```bash
cd /mnt/e/programming/penguinchess
source .venv/bin/activate
pytest tests/ -q
```

---

## 项目结构

```
penguinchess/
├── server/                     # Flask HTTP 服务层（当前）
│   ├── app.py                 # 路由、会话管理、静态托管
│   └── game.py                # Game 会话封装
│
├── penguinchess/               # Python 游戏核心
│   ├── core.py               # PenguinChessCore（棋盘/棋子/规则）
│   ├── env.py                # Gymnasium 环境（AI 训练接口）
│   ├── reward.py             # 奖励函数
│   ├── spaces.py             # Action / Observation Space
│   └── random_ai.py          # 随机 AI 基准
│
├── statics/                    # 原始前端（Vanilla JS）
│
├── frontend/                   # React 前端（开发中）
│   └── src/
│       ├── App.tsx           # 主应用状态机
│       ├── Board.tsx         # 六边形棋盘渲染
│       └── api.ts            # 后端 API 客户端
│
└── docs/
    ├── RULES.md              # 游戏规则（权威文档）
    └── ARCHITECTURE.md       # 架构设计与演进路线
```

详细架构说明见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

---

## 技术栈

| 层级 | 当前 | 未来 |
|------|------|------|
| 游戏核心 | Python | Rust |
| HTTP 框架 | Flask | Axum / Actix-web |
| 前端 | React + TypeScript | React + TypeScript |
| AI 训练 | Python Gymnasium + PyTorch（CPU/GPU 自动检测） | Python Gymnasium + PyTorch |
| 后端加速 | — | Rust（训练规模扩大后迁移游戏核心） |

---

## 许可证

[GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html)
