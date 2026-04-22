# PenguinChess 架构文档

## 概述

企鹅棋（PenguinChess）是一款基于六边形格子的双人对战棋类游戏。本文档描述当前架构设计与未来演进路线。

---

## 当前架构（Python 后端原型）

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 (React)                          │
│  http://localhost:8080 → Flask 静态托管 / Vite 开发服务器     │
│  职责：棋盘渲染、用户交互、状态展示                             │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP JSON API
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    后端 (Python / Flask)                     │
│                                                             │
│  server/app.py      HTTP 路由层（会话管理、静态文件）           │
│  server/game.py     Game 会话封装（PenguinChessCore 包装）    │
│                                                             │
│  penguinchess/                                             │
│    core.py          游戏核心逻辑（棋盘、棋子、规则）            │
│    env.py           Gymnasium 强化学习环境接口                 │
│    reward.py        奖励函数定义                              │
│    spaces.py        Gymnasium Action/Observation Space      │
│    random_ai.py     随机 AI（验证环境正确性）                  │
│                                                             │
│  职责：游戏逻辑、合法性检查、AI 训练环境                        │
└─────────────────────────────────────────────────────────────┘
```

### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/game` | 创建新游戏，返回初始状态 |
| `GET` | `/api/game/<id>` | 获取指定游戏状态 |
| `POST` | `/api/game/<id>/action` | 提交动作（放置/移动） |
| `POST` | `/api/game/<id>/reset` | 重开一局（相同会话） |

所有 API 响应均为完整游戏状态，前端无状态。

### 目录结构

```
penguinchess/
├── server/                  # Flask HTTP 服务层
│   ├── app.py               # 路由、会话管理、静态托管
│   └── game.py              # Game 会话封装
│
├── penguinchess/            # Python 游戏核心
│   ├── __init__.py
│   ├── core.py              # PenguinChessCore（棋盘/棋子/规则）
│   ├── env.py               # Gymnasium 环境（AI 训练接口）
│   ├── reward.py            # 奖励函数
│   ├── spaces.py            # Action / Observation Space
│   └── random_ai.py         # 随机 AI 基准
│
├── statics/                 # 原始前端（Vanilla JS，已有）
│   ├── main.js / board.js / piece.js / config.js
│   └── style.css
│
├── frontend/                # React 前端（开发中）
│   ├── src/
│   │   ├── App.tsx          # 主应用状态机
│   │   ├── Board.tsx        # 六边形棋盘渲染
│   │   ├── api.ts           # 后端 API 客户端
│   │   └── index.css        # Tailwind CSS
│   └── vite.config.ts
│
└── tests/                   # pytest 测试套件
```

---

## 未来架构（Rust 后端）

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 (React)                          │
│              与 Python 后端共用同一套 React 前端               │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP JSON API
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    后端 (Rust / Axum)                       │
│                                                             │
│  game_engine/         游戏核心（棋盘、棋子、规则）             │
│    ├── board.rs       六边形棋盘数据结构与操作                │
│    ├── piece.rs       棋子状态与移动逻辑                      │
│    ├── rules.rs       游戏规则引擎                          │
│    └── state.rs       游戏状态快照与历史记录                  │
│                                                             │
│  api/                 HTTP API 层                           │
│    ├── routes.rs      路由定义（与 Python 版完全兼容）          │
│    ├── session.rs     会话管理                              │
│    └── middleware.rs  CORS、日志等中间件                     │
│                                                             │
│  ai/                  AI 推理接口层                          │
│    ├── interface.rs   AI 决策抽象接口                       │
│    └── python_ai.rs   Python AI 进程通信（保留）              │
│                                                             │
│  性能目标：单请求 < 1ms，支持 > 10000 并发连接                 │
└─────────────────────────────────────────────────────────────┘
```

### 为什么要用 Rust

| 维度 | Python | Rust |
|------|--------|------|
| 推理性能 | ~10-100μs/步 | ~0.1-1μs/步（100x 提升） |
| 并发能力 | GIL 限制，多线程受限 | 零成本抽象，无 GIL |
| 内存占用 | 数百 MB（解释器） | 数 MB（原生二进制） |
| 冷启动 | 即时 | 即时（无 JVM/解释器） |
| AI 训练 | 原生支持 NumPy/PyTorch | 通过 Python 进程通信 |

Rust 将接管所有**游戏逻辑**和**高性能推理**部分。Python 专注于**AI 训练**和**策略研究**。

### Rust ↔ Python 通信设计

Rust 后端保留 Python AI 的集成方式：

```
┌─────────────────┐     IPC / Unix Socket     ┌─────────────────┐
│  Rust 后端      │ ◄──────────────────────► │  Python 训练器   │
│  (游戏逻辑)     │   发送棋盘状态            │  (PyTorch)       │
│                 │   接收 AI 动作            │                  │
│  ai/python_ai   │   JSON / protobuf       │  策略网络         │
│  .rs             │                         │  蒙特卡洛树搜索   │
└─────────────────┘                           └─────────────────┘
```

Python 训练环境（`penguinchess/env.py`）保持不变，Rust 后端通过子进程调用 Python AI 进行人机对战。

---

## AI 开发路线图

```
阶段 1 (当前)                    阶段 2                    阶段 3
───────────────────────    ────────────────────    ─────────────────────
Python Gymnasium 环境         Rust 游戏核心             Rust AI 推理
+ 随机/规则 AI               + Python AI 进程通信     + 快速策略网络
验证游戏逻辑正确性            保持 AI 开发灵活性        极致性能人机对战
```

### Python AI 层（长期保留）

以下代码模块**不受 Rust 重写影响**，将永久保留：

```
penguinchess/
├── env.py          # Gymnasium 接口，AI 训练入口（不变）
├── reward.py       # 奖励函数研究（不变）
├── spaces.py       # Space 定义（不变）
└── ai/             # 新增：Python AI 算法研究目录
    ├── policy_gradient.py   # PPO / A2C 策略梯度
    ├── mcts.py              # 蒙特卡洛树搜索
    └── self_play.py         # 自我对弈训练循环
```

Rust 后端的 `ai/python_ai.rs` 通过进程间通信调用上述 Python 模块，实现高性能推理。

---

## 游戏规则

详见 [docs/RULES.md](RULES.md)。

---

## 快速开始

### 启动当前版本（Python 后端）

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

### 运行 AI 训练

```bash
cd /mnt/e/programming/penguinchess
source .venv/bin/activate
python -c "
from penguinchess.env import PenguinChessEnv
env = PenguinChessEnv()
obs, info = env.reset()
for _ in range(1000):
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc:
        obs, info = env.reset()
"
```

### 运行测试

```bash
cd /mnt/e/programming/penguinchess
source .venv/bin/activate
pytest tests/ -q
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 游戏核心（当前） | Python 3.11 |
| 游戏核心（未来） | Rust（待实现） |
| HTTP 框架（当前） | Flask |
| HTTP 框架（未来） | Axum / Actix-web |
| 前端 | React 18 + TypeScript + Vite + Tailwind CSS v4 |
| AI 训练 | Python Gymnasium + PyTorch + Stable-Baselines3 |
| 测试 | pytest + Playwright |
