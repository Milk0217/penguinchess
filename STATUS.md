# PenguinChess 项目状态文档

> 最后更新: 2026-04-25

## 项目概述

PenguinChess（企鹅棋）是一个双人对战六边形棋盘游戏，同时作为强化学习研究平台。项目包含：
1. **Gymnasium RL 环境** - 严格遵循 Farama Gymnasium 标准的强化学习环境
2. **Web 人机对战界面** - React + Flask 的浏览器对战系统

---

## 一、当前实现状态

### ✅ 已完成

#### 1.1 游戏核心 (Python)

| 文件 | 状态 | 说明 |
|------|------|------|
| `penguinchess/core.py` | ✅ 完成 | PenguinChessCore 游戏逻辑，901行，包含 Hex/Piece 数据结构 |
| `penguinchess/env.py` | ✅ 完成 | Gymnasium Env，222行，实现 reset/step 接口 |
| `penguinchess/spaces.py` | ✅ 完成 | 观测/动作空间定义 |
| `penguinchess/reward.py` | ✅ 完成 | Reward shaping 函数 |
| `penguinchess/random_ai.py` | ✅ 完成 | 随机 AI 基准，用于环境验证 |

**核心功能**:
- 六边形棋盘（60格，立方体坐标）
- 放置阶段（6步，P1先手，严格交替）
- 移动阶段（同轴方向移动，路径检查）
- 连通性消除（flood fill）
- 胜负判定（分数比较）

#### 1.2 Web 后端 (Flask)

| 文件 | 状态 | 说明 |
|------|------|------|
| `server/app.py` | ✅ 完成 | Flask HTTP 服务，路由层 |
| `server/game.py` | ✅ 完成 | Game 会话封装 |
| `server/boards.py` | ✅ 完成 | 棋盘存储 API |
| `main.py` | ✅ 完成 | 备用 Web 服务器 |

**API 端点**:
- `POST /api/game` - 创建新游戏
- `GET /api/game/<id>` - 获取游戏状态
- `POST /api/game/<id>/action` - 提交动作
- `POST /api/game/<id>/reset` - 重开一局
- `GET /api/boards` - 获取已保存棋盘列表
- `POST /api/boards` - 保存新棋盘

#### 1.3 Web 前端 (React)

| 组件 | 状态 | 说明 |
|------|------|------|
| `App.tsx` | ✅ 完成 | 主应用状态机 |
| `board/BoardContainer.tsx` | ✅ 完成 | 棋盘容器（策略模式） |
| `board/HexCell.tsx` | ✅ 完成 | 六边形格子渲染 |
| `board/Piece.tsx` | ✅ 完成 | 棋子组件 |
| `board/Legend.tsx` | ✅ 完成 | 图例组件 |
| `layouts/parallelogram.ts` | ✅ 完成 | 平行四边形布局（60格） |
| `layouts/hexagon.ts` | ✅ 完成 | 六角形布局（61格） |
| `themes/default.ts` | ✅ 完成 | 默认亮色主题 |
| `themes/dark.ts` | ✅ 完成 | 暗色主题 |
| `editor/BoardEditor.tsx` | ✅ 完成 | 棋盘编辑器 |
| `editor/EditorCanvas.tsx` | ✅ 完成 | SVG 画布 |
| `editor/Sidebar.tsx` | ✅ 完成 | 侧边栏 |

#### 1.4 文档

| 文档 | 状态 | 说明 |
|------|------|------|
| `docs/RULES.md` | ✅ 完整 | 403行，权威游戏规则 |
| `docs/ARCHITECTURE.md` | ✅ 完整 | 267行，架构设计与演进路线 |
| `docs/BOARD_EDITOR.md` | ✅ 完整 | 419行，棋盘编辑器设计 |
| `docs/plans/2026-04-23-board-visualization-system.md` | ✅ 已实现 | 1299行，棋盘可视化重构计划 |
| `AGENTS.md` | ⚠️ 需要更新 | 516行，开发指南，部分内容过时 |
| `IMPROVEMENTS.md` | ⚠️ 需要更新 | 191行，改进建议，部分已完成项未标记 |
| `README.md` | ✅ 完整 | 项目概述和快速开始 |

---

### 🚧 待开始

#### 2.1 Rust 游戏核心

**目标**: 用 Rust 重写游戏核心逻辑，极致性能

**目录**: `game_engine/`（待创建）

**子任务**:
- `board.rs` - 六边形棋盘数据结构
- `piece.rs` - 棋子状态与移动逻辑
- `rules.rs` - 游戏规则引擎
- `state.rs` - 游戏状态快照

**API 兼容性**: Rust 后端需保持与 Python 后端完全相同的 HTTP API

#### 2.2 Python AI 研究

**目标**: 基于 Gymnasium 环境训练 RL 智能体

**目录**: `penguinchess/ai/`（待创建）

**待实现**:
- `policy_gradient.py` - PPO / A2C 策略梯度
- `mcts.py` - 蒙特卡洛树搜索
- `self_play.py` - 自我对弈训练循环

#### 2.3 训练脚本

**目标**: 提供开箱即用的训练 pipeline

**目录**: `examples/`（部分存在）

**待实现**:
- `train_ppo.py` - PPO 训练脚本
- `train_sac.py` - SAC 训练脚本
- `train_alphazero.py` - AlphaZero 风格训练
- `evaluate.py` - 评估工具

---

## 二、技术栈

| 层级 | 当前技术 | 未来技术 |
|------|----------|----------|
| 游戏核心 | Python 3.11 | Rust |
| HTTP 框架 | Flask | Axum / Actix-web |
| 前端 | React 18 + TypeScript + Vite | React 18 + TypeScript + Vite |
| RL 框架 | Gymnasium + Stable-Baselines3 | Gymnasium + Stable-Baselines3 |
| 测试 | pytest + Playwright | pytest + Playwright |

---

## 三、快速开始

### 3.1 Web 对战（人类 vs 人类）

```bash
cd E:\programming\penguinchess
python server/app.py
# 访问 http://localhost:8080
```

### 3.2 开发前端（Vite 热重载）

```bash
cd frontend
npm install
npm run dev
# 访问 http://localhost:5173
```

### 3.3 Gymnasium 环境验证

```bash
cd E:\programming\penguinchess
python -c "
import gymnasium as gym
from penguinchess.env import PenguinChessEnv

env = gym.make('PenguinChess-v0')
obs, info = env.reset()
print('Observation space:', env.observation_space)
print('Action space:', env.action_space)

# 随机测试 100 步
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
        print('Episode finished, reset.')

env.close()
print('Environment test passed!')
"
```

### 3.4 运行测试

```bash
cd E:\programming\penguinchess
pytest tests/ -q
```

---

## 四、项目结构

```
penguinchess/
├── penguinchess/              # Python 游戏核心（已实现）
│   ├── __init__.py
│   ├── core.py               # 游戏逻辑（Hex, Piece, PenguinChessCore）
│   ├── env.py                # Gymnasium 环境
│   ├── spaces.py             # Space 定义
│   ├── reward.py             # Reward 函数
│   └── random_ai.py          # 随机 AI 基准
│
├── server/                    # Flask 后端（已实现）
│   ├── __init__.py
│   ├── app.py                # HTTP 路由
│   ├── game.py               # Game 会话封装
│   └── boards.py             # 棋盘存储 API
│
├── frontend/                  # React 前端（已实现）
│   ├── src/
│   │   ├── App.tsx           # 主应用
│   │   ├── board/            # 棋盘可视化
│   │   │   ├── BoardContainer.tsx
│   │   │   ├── HexCell.tsx
│   │   │   ├── Piece.tsx
│   │   │   ├── Legend.tsx
│   │   │   ├── types.ts
│   │   │   ├── layouts/
│   │   │   │   ├── index.ts
│   │   │   │   ├── parallelogram.ts
│   │   │   │   └── hexagon.ts
│   │   │   └── themes/
│   │   │       ├── index.ts
│   │   │       ├── default.ts
│   │   │       └── dark.ts
│   │   └── editor/            # 棋盘编辑器
│   │       ├── BoardEditor.tsx
│   │       ├── EditorCanvas.tsx
│   │       └── Sidebar.tsx
│   └── package.json
│
├── statics/                   # 原始前端（Vanilla JS，已保留）
│   ├── main.js
│   ├── board.js
│   ├── piece.js
│   └── config.js
│
├── docs/                     # 文档（已完成）
│   ├── RULES.md              # 权威游戏规则
│   ├── ARCHITECTURE.md       # 架构设计
│   ├── BOARD_EDITOR.md       # 编辑器设计
│   └── plans/                # 已完成计划
│       └── 2026-04-23-board-visualization-system.md
│
├── examples/                  # 训练示例（部分存在）
│   └── random_ai.py          # 已实现
│
├── tests/                     # 测试（已存在）
│   └── __init__.py
│
├── main.py                    # 备用 Web 服务器
├── README.md                  # 项目概述
├── AGENTS.md                  # Agent 开发指南（需更新）
├── IMPROVEMENTS.md            # 改进建议（需更新）
└── STATUS.md                  # 本文档
```

---

## 五、验证清单

### 环境验证

- [x] `PenguinChessCore` 初始化成功
- [x] `gymnasium.make("PenguinChess-v0")` 注册成功
- [x] `env.reset()` 返回有效观测
- [x] `env.step(action)` 正确执行动作
- [x] 随机 AI 对局 100 步稳定运行
- [x] 放置阶段 → 移动阶段切换正确
- [x] 连通性消除算法正确
- [x] 胜负判定正确

### Web 验证

- [x] Flask 服务器启动成功
- [x] `/api/game` 创建游戏成功
- [x] `/api/game/<id>/action` 执行动作成功
- [x] 棋盘渲染正确（60格）
- [x] 棋子放置/移动动画正常
- [x] 棋盘编辑器保存/加载正常

### 前端组件验证

- [x] BoardContainer 正确渲染
- [x] HexCell 六边形样式正确
- [x] Piece 棋子样式正确
- [x] Legend 图例正确
- [x] parallelogram 布局正确
- [x] hexagon 布局正确
- [x] default 主题正确
- [x] dark 主题正确

---

## 六、已知问题

无。当前实现稳定。

---

## 七、下一步

1. **更新 AGENTS.md** - 标记已完成项，移除 "★ 待实现" 标记
2. **更新 IMPROVEMENTS.md** - 将已完成项移到"已完成"区域
3. **实现 PPO 训练脚本** - `examples/train_ppo.py`
4. **实现 Python AI 模块** - `penguinchess/ai/`
5. **Rust 游戏核心** - Phase 2（长期目标）

---

## 八、参考文档

- [游戏规则](docs/RULES.md)
- [架构设计](docs/ARCHITECTURE.md)
- [棋盘编辑器](docs/BOARD_EDITOR.md)
- [Agent 开发指南](AGENTS.md)
- [改进建议](IMPROVEMENTS.md)