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

#### 游戏 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/game` | 创建新游戏，返回初始状态 |
| `GET` | `/api/game/<id>` | 获取指定游戏状态 |
| `POST` | `/api/game/<id>/action` | 提交动作（放置/移动） |
| `POST` | `/api/game/<id>/ai_move` | AI 自动执行一次移动 |
| `POST` | `/api/game/<id>/reset` | 重开一局（相同会话） |
| `GET` | `/api/health` | 健康检查 |

#### 模型 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/models` | 获取所有可用模型及评估数据（ELO/胜率） |
| `GET` | `/api/models/best` | 获取当前最优模型信息（基于 ELO） |

#### 棋盘 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/boards` | 获取所有已保存棋盘列表 |
| `POST` | `/api/boards` | 保存新棋盘（编辑器调用） |
| `GET` | `/api/boards/<id>` | 获取指定棋盘完整数据 |
| `DELETE` | `/api/boards/<id>` | 删除棋盘 |

所有 API 响应均为完整游戏状态，前端无状态。

### 目录结构

```
penguinchess/
├── server/                  # Flask HTTP 服务层
│   ├── app.py               # 路由、会话管理、静态托管
│   ├── game.py              # Game 会话封装
│   └── boards.py            # 棋盘存储管理
│
├── game_engine/              # Rust 游戏核心（cdylib）
│   └── src/
│       ├── board.rs          # 六边形棋盘
│       ├── rules.rs          # 游戏规则引擎
│       ├── ffi.rs            # C FFI 导出（有状态/无状态 API）
│       ├── mcts_rs.rs        # Rust MCTS 搜索
│       ├── net_infer.rs      # ONNX 推理（预留）
│       ├── lib.rs            # 模块入口
│       └── bin/cli.rs        # 独立 CLI 二进制
│
├── penguinchess/            # Python 游戏核心 & AI
│   ├── __init__.py
│   ├── core.py              # PenguinChessCore（棋盘/棋子/规则）
│   ├── env.py               # Gymnasium 环境（AI 训练接口）
│   ├── reward.py            # 奖励函数
│   ├── spaces.py            # Action / Observation Space
│   ├── model_registry.py    # Model Registry（ELO 持久化）
│   ├── rust_ffi.py          # Rust ctypes FFI 桥接
│   ├── rust_core.py         # RustCore 包装（duck-type 兼容 core.py）
│   ├── rust_bridge.py       # Rust 统一桥接层
│   └── ai/                  # AI 算法模块
│       ├── __init__.py
│       ├── alphazero_net.py # AlphaZero 神经网络
│       ├── mcts_core.py     # MCTS 搜索核心
│       └── train_alphazero.py # AlphaZero 训练脚本
│
├── examples/                # 训练与评估示例
│   ├── random_ai.py         # 随机 AI 基准
│   ├── train_ppo.py         # PPO 训练脚本
│   └── eval_elo.py          # ELO 评估工具
│
├── models/                  # 训练产出的模型
│   ├── ppo_penguinchess_gen_*.zip  # PPO 各代模型
│   ├── alphazero/           # AlphaZero 模型
│   └── model_registry.json  # ELO 评分注册表
│
├── statics/                 # 原始前端（Vanilla JS）
│
├── frontend/                # React 前端
│   ├── src/
│   │   ├── App.tsx          # 主应用状态机
│   │   ├── api.ts           # 后端 API 客户端
│   │   ├── board/           # 棋盘可视化系统（策略模式）
│   │   │   ├── BoardContainer.tsx  # 棋盘容器
│   │   │   ├── HexCell.tsx   # 六边形格子
│   │   │   ├── Piece.tsx     # 棋子渲染
│   │   │   ├── Legend.tsx    # 图例
│   │   │   ├── types.ts      # 类型定义
│   │   │   ├── layouts/      # 布局策略
│   │   │   └── themes/       # 主题策略
│   │   └── editor/          # 棋盘编辑器
│   │       ├── BoardEditor.tsx
│   │       ├── EditorCanvas.tsx
│   │       ├── Sidebar.tsx
│   │       └── templates.ts
│   └── vite.config.ts
│
├── backend_data/
│   └── boards/              # 已保存的棋盘 JSON
│       ├── parallelogram.json
│       ├── hexagon.json
│       ├── default.json
│       └── custom-*.json    # 用户自定义棋盘
│
├── docs/                    # 文档
│   ├── RULES.md             # 游戏规则（权威文档）
│   ├── ARCHITECTURE.md      # 本文件
│   ├── BOARD_EDITOR.md      # 棋盘编辑器设计
│   ├── MCTS_PLAN.md         # MCTS + 神经网络方案
│   └── OPTIMIZATION.md      # 性能优化分析
│
├── start_all.py             # 一键启动脚本
└── tests/                   # pytest 测试套件
```

---

## Rust 游戏引擎（已实现）

`game_engine/` 是一个 Rust cdylib 库，已编译为 `game_engine.dll`，通过 ctypes FFI 被 Python 调用。

```
┌──────────────────────────────────────────────────────────────────┐
│                          Python 层                               │
│                                                                  │
│  penguinchess/                                                   │
│    core.py        ← Python 游戏核心（仍可用，与 Rust 并行）       │
│    rust_ffi.py    ← ctypes 直接调用 Rust DLL（快 11.5x）          │
│    rust_core.py   ← RustCore 包装，duck-type 兼容 core.py        │
│                                                                  │
│  examples/                                                        │
│    eval_elo.py    ← ELO 评估，支持 --rust-core / --incremental    │
│    train_ppo.py   ← PPO 训练，训练后自动写 Model Registry         │
│                                                                  │
└───────────────────────┬──────────────────────────────────────────┘
                        │ ctypes FFI (零拷贝、状态保持)
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                       Rust 层 (game_engine/)                     │
│                                                                  │
│  src/                                                             │
│    board.rs        ← 六边形棋盘数据结构（Hex, Board, Piece）      │
│    rules.rs        ← 完整游戏规则引擎（273 行）                    │
│    ffi.rs          ← C FFI 导出函数：                             │
│                      · game_stateful_new(seed) → handle          │
│                      · game_stateful_step(handle, action) → JSON │
│                      · game_stateful_get_legal(handle) → [int]   │
│                      · game_stateful_get_obs(handle) → dict      │
│                      · game_stateful_free(handle)                │
│                      + 旧版 stateless API (game_new/step/eval)   │
│    mcts_rs.rs      ← Rust MCTS 搜索（UCB/批量回调）               │
│    lib.rs          ← 模块入口                                    │
│    bin/cli.rs      ← 独立 CLI 二进制                              │
│                                                                  │
│  构建: cargo build --release → target/release/game_engine.dll    │
└──────────────────────────────────────────────────────────────────┘
```

### 性能基准（release 构建）

使用 `--release` 优化编译后的基准测试结果（Intel CPU, 单线程）:

| 测试项 | Python | Rust (release) | 加速比 |
|--------|--------|---------------|--------|
| `core.step()` 单步 | 0.159 ms | **0.014 ms** | **11.5x** |
| 完整一局（随机走法） | 6.9 ms | **1.0 ms** | **6.9x** |
| PPO 推理单次 | 0.201 ms | — | — |
| ELO 评估 55对×50局 | ~19s* | **~3s*** | **~6.9x** |

> \* 仅含游戏模拟时间，模型推理时间另计。实际 ELO 评估受模型推理（~70%）和游戏模拟（~30%）共同影响。

| 构建配置 | 单步耗时 | 对比 Python |
|----------|---------|-------------|
| Python core.step() | 0.159 ms | 1.0x |
| Rust debug 构建 | 0.080 ms | **2.0x** |
| Rust release 构建 | 0.014 ms | **11.5x** |

### 有状态 FFI 设计（关键优化）

旧版 `game_step()` 每次调用需要完整序列化 GameState（~8KB JSON）：

```
Python 序列化 State → JSON → ctypes → Rust 反序列化 → step → 序列化 → JSON → 返回
```

新版有状态 API（`game_stateful_*`）将 GameState 保留在 Rust 内存中：

```
Python 传 (handle, action) → Rust 直接操作内存 → 返回小 JSON (~200B)
```

结果：单步耗时从 **2.018 ms**（旧版 FFI）降至 **0.014 ms**（有状态 release），**144x 改进**。

### 增量 ELO 模式

`--incremental` 跳过双方已有 ELO 的对局：

```bash
# 全量评估（首次）
uv run python examples/eval_elo.py --episodes 50 --rust-core

# 增量评估（训练新模型后）
uv run python examples/eval_elo.py --episodes 50 --incremental --rust-core
```

训练 gen_N 时只需跑 N-1 对新局（~10s），而非 N×(N-1)/2 对（~60s）。

### 组件与文件对应

| 职责 | 文件 | 说明 |
|------|------|------|
| 游戏核心（Python 版） | `penguinchess/core.py` | 仍可用，与 Rust 并行维护 |
| 游戏核心（Rust 版） | `game_engine/src/rules.rs` + `board.rs` | 编译为 DLL |
| Rust FFI 桥接 | `penguinchess/rust_ffi.py` | ctypes 封装，含 `RustEngine` + `RustStatefulGame` |
| RustCore 包装 | `penguinchess/rust_core.py` | duck-type 兼容 `PenguinChessCore` |
| Rust 统一桥接层 | `penguinchess/rust_bridge.py` | Rust 桥接自动选择 |
| Rust MCTS 搜索 | `game_engine/src/mcts_rs.rs` | 通过回调批量调用 Python 神经网络 |
| Rust ONNX 推理（预留） | `game_engine/src/net_infer.rs` | 未来原生推理 |
| Model Registry | `penguinchess/model_registry.py` | JSON 持久化，ELO 排序 |
| ELO 评估 | `examples/eval_elo.py` | 增量/全量，Python/Rust，PPO/AZ/MCTS |
| PPO 训练 | `examples/train_ppo.py` | 训练后自动写入 Registry |
| AlphaZero 训练 | `penguinchess/ai/train_alphazero.py` | MCTS + ResNet 自对弈 |
| AlphaZero 神经网络 | `penguinchess/ai/alphazero_net.py` | ResNet 策略-价值网络 |
| MCTS 核心（Python） | `penguinchess/ai/mcts_core.py` | Python MCTS 搜索节点 |

### 构建 Rust

```bash
cd game_engine
cargo build --release    # 优化构建，产出 target/release/game_engine.dll
```

`rust_ffi.py` 自动按 `release` → `debug` 顺序搜索 DLL。如果找不到，回退到 Python `PenguinChessCore`。

---

## AI 开发进展

### 已完成里程碑

```
✅ 阶段 1: Python Gymnasium 环境          → 已完成，生产可用
✅ 阶段 2: Rust 游戏核心 (FFI cdylib)     → 已完成，快 11.5x
✅ 阶段 3: MCTS + AlphaZero 训练管道      → 已完成（Rust MCTS 搜索 + Python 神经网络）
✅ 阶段 4: PPO + AlphaZero 双线路训练     → 已完成
⬜ 阶段 5: Rust ONNX 原生推理 (net_infer) → 预留，待实施
```

### Python AI 层（长期保留）

以下代码模块**不受 Rust 重写影响**，将永久保留：

```
penguinchess/
├── env.py                     # Gymnasium 接口，AI 训练入口（不变）
├── reward.py                  # 奖励函数研究（不变）
├── spaces.py                  # Space 定义（不变）
├── model_registry.py          # Model Registry（ELO 持久化）
├── rust_ffi.py                # Rust ctypes FFI 桥接
├── rust_core.py               # RustCore 包装（duck-type 兼容 core.py）
├── rust_bridge.py             # Rust 统一桥接层
└── ai/                        # Python AI 算法模块
    ├── alphazero_net.py       # AlphaZero 神经网络
    ├── mcts_core.py           # MCTS 搜索核心
    └── train_alphazero.py     # AlphaZero 训练脚本
```

Rust 后端的 `mcts_rs.rs` 通过回调调用 Python 神经网络进行批量评估。`net_infer.rs` 预留用于未来 ONNX 原生推理，届时可绕过 Python 实现极致性能。

---

## 游戏规则

详见 [docs/RULES.md](RULES.md)。

---

## 快速开始

### 启动当前版本（Python 后端）

```bash
uv run python server/app.py
# 访问 http://localhost:8080
```

### 启动开发前端（Vite 热重载）

```bash
cd frontend && bun run dev
# 访问 http://localhost:5173
```

### 运行 AI 训练

```bash
uv run python -c "
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
uv run python -m pytest tests/ -q
```

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 游戏核心（Python 版） | Python 3.11 (`penguinchess/core.py`) |
| 游戏核心（Rust 加速） | Rust cdylib (`game_engine/`, `--release` 构建) |
| HTTP 框架 | Flask（当前）/ Axum（计划中） |
| 前端 | React 18 + TypeScript + Vite + Tailwind CSS v4 |
| AI 训练 | Python Gymnasium + PyTorch + Stable-Baselines3 |
| 测试 | pytest + Playwright |
