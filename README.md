# 企鹅棋 PenguinChess

双人六边形棋盘对战游戏，同时作为强化学习研究平台。

## 项目目标

### 目标 1：高性能游戏引擎（Rust ✅ 已实现）

Rust 游戏核心已实现，编译为 cdylib 通过 FFI 被 Python 调用。单步性能 0.014ms（release 构建），相比 Python 版本加速约 **11.5 倍**。

详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) 和 `game_engine/`。

### 目标 2：强化学习研究与训练（Python ✅ 已实现）

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

### 安装依赖

```bash
# Python 依赖（使用 uv，推荐）
uv sync

# 或使用 pip
pip install -e ".[dev]"

# 前端依赖
cd frontend
bun install   # 或 npm install
cd ..

# Rust 游戏引擎（可选，用于加速）
cd game_engine
cargo build --release
cd ..
```

### 启动 Web 对战

**方式 A：一键启动（后端 + 前端开发服务器）**
```bash
uv run python start_all.py
# 后端 http://localhost:8080
# 前端 http://localhost:5173
```

**方式 B：前后端分离开发**
```bash
# 终端 1 - 后端
uv run python server/app.py

# 终端 2 - 前端（热重载）
cd frontend && bun run dev
```

### Gymnasium 环境（AI 训练）

```bash
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

### AI 训练

```bash
# PPO 训练（100k 步，4 并行环境）
uv run python examples/train_ppo.py --timesteps 100000 --num-envs 4

# AlphaZero 训练（30 轮，每轮 200 局）
uv run python -m penguinchess.ai.train_alphazero --iterations 30

# ELO 评估
uv run python examples/eval_elo.py --episodes 50
```

### 运行测试

```bash
uv run python -m pytest tests/ -q
```

---

## 项目结构

```
penguinchess/
├── server/                     # Flask HTTP 服务层
│   ├── app.py                # 路由、会话管理、静态托管
│   ├── game.py               # Game 会话封装
│   └── boards.py             # 棋盘存储管理
│
├── game_engine/                # Rust 游戏核心（cdylib）
│   └── src/
│       ├── board.rs           # 六边形棋盘数据结构
│       ├── rules.rs           # 游戏规则引擎
│       ├── ffi.rs             # C FFI 导出
│       ├── mcts_rs.rs         # Rust MCTS 搜索
│       ├── net_infer.rs       # ONNX 推理（预留）
│       └── lib.rs             # 模块入口
│
├── penguinchess/               # Python 游戏核心 & AI
│   ├── core.py               # PenguinChessCore（棋盘/棋子/规则）
│   ├── env.py                # Gymnasium 环境（AI 训练接口）
│   ├── reward.py             # 奖励函数
│   ├── spaces.py             # Action / Observation Space
│   ├── model_registry.py     # Model Registry（ELO 持久化）
│   ├── rust_ffi.py           # Rust ctypes FFI 桥接
│   ├── rust_core.py          # RustCore 包装（duck-type 兼容 core.py）
│   ├── rust_bridge.py        # Rust 统一桥接层
│   └── ai/                   # AI 算法模块
│       ├── alphazero_net.py   # AlphaZero 神经网络（可配置 ResNet）
│       │   ├── AlphaZeroResNet         # 550K 参数，~25MB GPU
│       │   ├── AlphaZeroResNetLarge    # 3M 参数，~74MB GPU
│       │   ├── AlphaZeroResNetXL       # 581M 参数，~3GB GPU
│       │   └── AlphaZeroResNetConfigurable # 自定义 hidden_dim + num_blocks
│       ├── mcts_core.py       # MCTS 搜索核心
│       └── train_alphazero.py # AlphaZero 训练脚本
│
├── examples/                   # 训练与评估示例
│   ├── random_ai.py           # 随机 AI 基准
│   ├── train_ppo.py           # PPO 训练脚本
│   └── eval_elo.py            # ELO 评估工具
│
├── models/                     # 训练产出的模型
│   ├── ppo_penguinchess_gen_*.zip  # PPO 各代模型
│   ├── alphazero/             # AlphaZero 模型
│   └── model_registry.json    # ELO 评分注册表
│
├── frontend/                   # React + TypeScript 前端
│   └── src/
│       ├── App.tsx           # 主应用状态机
│       ├── api.ts            # 后端 API 客户端
│       └── board/            # 棋盘可视化系统
│
├── statics/                    # 原始前端（Vanilla JS）
│
├── backend_data/
│   └── boards/               # 已保存的棋盘 JSON
│
└── docs/
    ├── RULES.md              # 游戏规则（权威文档）
    ├── ARCHITECTURE.md       # 架构设计与演进路线
    ├── BOARD_EDITOR.md       # 棋盘编辑器设计
    ├── MCTS_PLAN.md          # MCTS + 神经网络方案
    └── OPTIMIZATION.md       # 性能优化分析
```

详细架构说明见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

---

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 游戏核心（Python） | Python 3.11 | `penguinchess/core.py`，可靠原型 |
| 游戏核心（Rust 加速） | Rust cdylib | `game_engine/`，`--release` 构建，快 11.5x |
| HTTP 框架 | Flask | 当前后端，稳定可用 |
| 前端 | React 18 + TypeScript + Vite | 带棋盘可视化系统 |
| AI 训练 | Python Gymnasium + PyTorch + SB3 | PPO + AlphaZero 双线路 |
| Rust MCTS 搜索 | `mcts_rs.rs` | 通过回调调用 Python 神经网络 |

---

## 许可证

[GNU General Public License v3.0 (GPL-3.0)](https://www.gnu.org/licenses/gpl-3.0.en.html)
